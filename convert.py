"""Convert a RoboTwin 2.0 dataset into the LeRobot v3.0 dataset format.

RoboTwin 2.0 stores each episode as a single HDF5 file under ``<src>/data/episodeN.hdf5``
together with a language-instruction JSON under ``<src>/instructions/episodeN.json``.
This script reads those, maps them onto the LeRobot v3 schema, and writes a new
LeRobot dataset via ``lerobot.datasets.LeRobotDataset.create`` / ``add_frame`` /
``save_episode`` / ``finalize``.

Run it once per source dataset, e.g.::

    uv run python convert.py \
        --src /home/ubuntu/workspace/RoboTwin/data/piper_clean_50 \
        --repo-id local/piper_clean_50 \
        --root ./output/piper_clean_50

Conventions (verified against the RoboTwin repo and confirmed with the user):

* Robot is a *bimanual* Piper, so ``observation.state`` and ``action`` are both the
  14-dim ``joint_action/vector`` = ``[left_arm(6), left_gripper(1), right_arm(6),
  right_gripper(1)]``. ``action[t] == state[t]`` (absolute joint-position control), the
  standard RoboTwin convention used by their RDT / DexVLA loaders.
* Three camera streams are converted: ``head_camera``, ``left_camera`` (left wrist),
  ``right_camera`` (right wrist), renamed to ``head`` / ``left_wrist`` / ``right_wrist``.
* The per-episode ``task`` string is the first "seen" instruction.
* Camera intrinsics/extrinsics, end-effector poses, third-view and point clouds are not
  converted (LeRobot has no standard slot for them).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import h5py
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# RoboTwin camera key -> LeRobot image-feature suffix.
CAMERA_MAP = {
    "head_camera": "head",
    "left_camera": "left_wrist",
    "right_camera": "right_wrist",
}

# Joint layout of joint_action/vector (14 dims).
STATE_NAMES = (
    [f"left_joint_{i}" for i in range(1, 7)]
    + ["left_gripper"]
    + [f"right_joint_{i}" for i in range(1, 7)]
    + ["right_gripper"]
)

IMG_H, IMG_W = 240, 320
FALLBACK_TASK = "bimanual manipulation"


def list_episodes(src: Path) -> list[tuple[int, Path]]:
    """Return ``(episode_index, hdf5_path)`` pairs sorted by episode index."""
    paths = (src / "data").glob("episode*.hdf5")
    out = []
    for p in paths:
        m = re.search(r"episode(\d+)\.hdf5$", p.name)
        if m:
            out.append((int(m.group(1)), p))
    out.sort(key=lambda t: t[0])
    return out


def _decode_rgb_stream(raw) -> np.ndarray:
    """Decode a RoboTwin JPEG-byte rgb dataset into an (N, H, W, 3) uint8 RGB array."""
    frames = []
    for buf in raw:
        arr = np.frombuffer(buf, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR, (H, W, 3)
        if img is None:
            raise ValueError("Failed to decode an RGB frame")
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return np.stack(frames, axis=0)


def load_episode(hdf5_path: Path) -> dict:
    """Load one RoboTwin episode into numpy arrays keyed by LeRobot feature name."""
    with h5py.File(hdf5_path, "r") as f:
        state = f["joint_action/vector"][:].astype(np.float32)  # (N, 14)
        n = state.shape[0]
        images = {}
        for cam_key, suffix in CAMERA_MAP.items():
            rgb = _decode_rgb_stream(f[f"observation/{cam_key}/rgb"][:])
            if rgb.shape[0] != n:
                raise ValueError(
                    f"{hdf5_path.name}: {cam_key} has {rgb.shape[0]} frames, expected {n}"
                )
            if rgb.shape[1:3] != (IMG_H, IMG_W):
                raise ValueError(
                    f"{hdf5_path.name}: {cam_key} frame size {rgb.shape[1:3]} != ({IMG_H}, {IMG_W})"
                )
            images[suffix] = rgb
    return {"state": state, "action": state, "images": images, "num_frames": n}


def load_instruction(src: Path, ep_idx: int) -> str:
    """Return the first 'seen' language instruction for an episode."""
    path = src / "instructions" / f"episode{ep_idx}.json"
    if not path.exists():
        return FALLBACK_TASK
    data = json.loads(path.read_text())
    seen = data.get("seen") or []
    return seen[0] if seen else FALLBACK_TASK


def build_features() -> dict:
    features = {
        "observation.state": {"dtype": "float32", "shape": (14,), "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": (14,), "names": STATE_NAMES},
    }
    for suffix in CAMERA_MAP.values():
        features[f"observation.images.{suffix}"] = {
            "dtype": "video",
            "shape": (IMG_H, IMG_W, 3),
            "names": ["height", "width", "channels"],
        }
    return features


def convert(src: Path, repo_id: str, root: Path, fps: int, max_episodes: int | None) -> None:
    episodes = list_episodes(src)
    if max_episodes is not None:
        episodes = episodes[:max_episodes]
    if not episodes:
        raise SystemExit(f"No episodes found under {src / 'data'}")

    print(f"Converting {len(episodes)} episode(s) from {src} -> {root}")
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=build_features(),
        root=root,
        robot_type="piper_dual",
        use_videos=True,
    )

    for n, (ep_idx, hdf5_path) in enumerate(episodes):
        ep = load_episode(hdf5_path)
        task = load_instruction(src, ep_idx)
        for i in range(ep["num_frames"]):
            frame = {
                "observation.state": ep["state"][i],
                "action": ep["action"][i],
                "task": task,
            }
            for suffix, rgb in ep["images"].items():
                frame[f"observation.images.{suffix}"] = rgb[i]
            dataset.add_frame(frame)
        dataset.save_episode()
        print(f"  [{n + 1}/{len(episodes)}] episode{ep_idx}: {ep['num_frames']} frames")

    dataset.finalize()
    print(f"Done. total_episodes={dataset.meta.total_episodes} total_frames={dataset.meta.total_frames}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a RoboTwin 2.0 dataset to LeRobot v3.")
    parser.add_argument("--src", type=Path, required=True, help="RoboTwin dataset dir (contains data/, instructions/)")
    parser.add_argument("--repo-id", required=True, help="LeRobot repo id, e.g. local/piper_clean_50")
    parser.add_argument("--root", type=Path, required=True, help="Output directory for the LeRobot dataset")
    parser.add_argument("--fps", type=int, default=15, help="Frames per second (default: 15)")
    parser.add_argument("--max-episodes", type=int, default=None, help="Limit number of episodes (smoke test)")
    args = parser.parse_args()
    convert(args.src, args.repo_id, args.root, args.fps, args.max_episodes)


if __name__ == "__main__":
    main()
