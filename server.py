"""Generic WebSocket inference server for any LeRobot policy.

Point `CKPT` at a trained checkpoint's `pretrained_model/` directory and run.
The same code serves ACT, Diffusion, pi0, SmolVLA, ... — everything policy-specific
(architecture, normalization stats, action chunking) lives in the checkpoint and is
loaded through LeRobot's own factories, so nothing here needs to change per policy.

Wire protocol (msgpack via serialization.pack/unpack), matching the existing ACT server:
    client -> {"images": {<cam_name>: HWC uint8 array, ...}, "state": 1-D array, "task": optional str}
    server -> {"action": 1-D array}

Inference uses `policy.select_action`, which returns ONE action per call and internally
caches an action chunk, only re-planning every `n_action_steps`. So calling it every
control step is cheap most of the time, and the client runs closed-loop (recommended for
diffusion). The policy's observation/action queues are reset on each new connection.
"""

import asyncio
import os
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from serialization import pack, unpack
from torchvision.transforms.functional import resize

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

# ----------------------------------------------------------------------------- config
CKPT = Path(
    os.environ.get(
        "CKPT",
        "/home/ubuntu/workspace/lerobot/outputs/train/"
        "piper_randomized_500_diffusion_policy/checkpoints/last/pretrained_model",
    )
)
DEVICE = os.environ.get("DEVICE", "cuda")
# bf16 autocast on CUDA: faster, negligible accuracy impact for inference. Set USE_AMP=0 to disable.
USE_AMP = os.environ.get("USE_AMP", "1") == "1" and DEVICE == "cuda"

# --------------------------------------------------------------------------- load policy
# The checkpoint's config.json carries input/output features, so the policy loads
# standalone — no dataset or env needed.
config = PreTrainedConfig.from_pretrained(CKPT)
config.pretrained_path = str(CKPT)
config.device = DEVICE
# Disable torch.compile for serving. A checkpoint trained with `compile_model=true` carries
# that flag in its config; if left on, the policy wraps submodules in torch.compile at load
# time, which renames their state-dict keys (`..._orig_mod...`) and no longer matches the
# checkpoint's plain keys -> load-time key mismatch. (It also adds autotune startup cost and
# cudagraph fragility we don't want at inference.)
if hasattr(config, "compile_model"):
    config.compile_model = False
policy = get_policy_class(config.type).from_pretrained(CKPT, config=config).to(DEVICE).eval()

# Pre/post-processors handle (un)normalization, device placement, etc. Loaded from the
# same checkpoint so stats always match the trained policy.
preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=config,
    pretrained_path=str(CKPT),
    preprocessor_overrides={"device_processor": {"device": DEVICE}},
)

# Derive what observations this policy expects straight from its config.
IMAGE_SPECS = {  # lerobot key -> (H, W) the model was trained on
    key: tuple(ft.shape[-2:]) for key, ft in config.input_features.items() if ft.type is FeatureType.VISUAL
}
STATE_KEY = next(
    (key for key, ft in config.input_features.items() if ft.type is FeatureType.STATE), None
)

print(f"Loaded {config.type} policy from {CKPT}")
print(f"  cameras: {[k.removeprefix('observation.images.') for k in IMAGE_SPECS]}")
print(f"  image size: {set(IMAGE_SPECS.values())}  state key: {STATE_KEY}  amp: {USE_AMP}")

app = FastAPI()
# select_action is stateful (per-episode queues), so serialize requests on one policy.
infer_lock = asyncio.Lock()


def build_observation(data: dict) -> dict:
    """Turn a raw client payload into a model-ready observation batch.

    Images are resized to the resolution the policy was trained on (the checkpoint has
    no resize step of its own), scaled to [0, 1], and laid out CHW with a batch dim.
    Normalization and device transfer are done afterwards by the preprocessor.
    """
    obs = {}
    for key, (h, w) in IMAGE_SPECS.items():
        cam = key.removeprefix("observation.images.")
        if cam not in data["images"]:
            raise KeyError(f"Missing camera '{cam}'. Policy expects: {list(IMAGE_SPECS)}")
        # .copy(): msgpack hands back a read-only buffer-backed array.
        img = torch.from_numpy(data["images"][cam].copy()).permute(2, 0, 1).float() / 255.0  # HWC -> CHW
        obs[key] = resize(img, [h, w], antialias=True).unsqueeze(0)

    if STATE_KEY is not None:
        state = torch.as_tensor(data["state"], dtype=torch.float32)
        obs[STATE_KEY] = state.unsqueeze(0)

    # Some policies/processors read these; harmless for the rest.
    obs["task"] = data.get("task", "")
    obs["robot_type"] = data.get("robot_type", "")
    return obs


@torch.inference_mode()
def infer(data: dict):
    obs = build_observation(data)
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if USE_AMP
        else torch.autocast(device_type="cpu", enabled=False)
    )
    with autocast:
        action = postprocessor(policy.select_action(preprocessor(obs)))
    return action.squeeze(0).cpu().numpy()


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    # New connection == new episode: clear the policy's observation/action queues.
    policy.reset()
    try:
        while True:
            data = unpack(await ws.receive_bytes())
            async with infer_lock:
                action = await asyncio.to_thread(infer, data)
            await ws.send_bytes(pack({"action": action}))
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, workers=1)
