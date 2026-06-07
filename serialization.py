import msgpack
import numpy as np


def encode(obj):
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }

    return obj


def decode(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(
            buffer=obj[b"data"],
            dtype=np.dtype(obj[b"dtype"]),
            shape=obj[b"shape"],
        )

    return obj


def pack(payload):
    return msgpack.packb(payload, default=encode, use_bin_type=True)


def unpack(blob):
    return msgpack.unpackb(blob, object_hook=decode, raw=False)
