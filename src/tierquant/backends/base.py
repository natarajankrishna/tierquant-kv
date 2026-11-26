from __future__ import annotations

import os
from abc import ABC, abstractmethod

import torch


class Backend(ABC):
    name = "base"

    @abstractmethod
    def put(self, key: str, buf: bytes) -> None: ...

    @abstractmethod
    def get_to_gpu(self, key: str, nbytes: int, device: str = "cuda") -> torch.Tensor: ...

    @abstractmethod
    def delete(self, key: str) -> None: ...

    def close(self) -> None:
        pass


def _to_gpu(raw: bytes, device: str) -> torch.Tensor:
    t = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    return t.to(device, non_blocking=False)


class DramBackend(Backend):
    """Pinned host memory -> GPU (the DRAM tier)."""
    name = "dram"

    def __init__(self):
        self.d = {}

    def put(self, key, buf):
        t = torch.frombuffer(bytearray(buf), dtype=torch.uint8).clone()
        self.d[key] = t.pin_memory() if torch.cuda.is_available() else t

    def get_to_gpu(self, key, nbytes, device="cuda"):
        return self.d[key].to(device, non_blocking=False)

    def delete(self, key):
        self.d.pop(key, None)


class NvmeFileBackend(Backend):
    """Plain POSIX files; data goes NVMe -> page cache/host buffer -> GPU (CPU bounce buffer)."""
    name = "nvme"

    def __init__(self, root: str, drop_cache: bool = True):
        self.root, self.drop_cache = root, drop_cache
        os.makedirs(root, exist_ok=True)

    def _p(self, key):
        return os.path.join(self.root, key)

    def put(self, key, buf):
        with open(self._p(key), "wb") as f:
            f.write(buf)
            f.flush()
            os.fsync(f.fileno())
        if self.drop_cache and hasattr(os, "posix_fadvise"):
            fd = os.open(self._p(key), os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)

    def get_to_gpu(self, key, nbytes, device="cuda"):
        fd = os.open(self._p(key), os.O_RDONLY)
        try:
            if self.drop_cache and hasattr(os, "posix_fadvise"):
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)   # cold read
            raw = os.read(fd, nbytes)
            while len(raw) < nbytes:
                raw += os.read(fd, nbytes - len(raw))
        finally:
            os.close(fd)
        return _to_gpu(raw, device)

    def delete(self, key):
        try:
            os.remove(self._p(key))
        except FileNotFoundError:
            pass


class RedisBackend(Backend):
    name = "redis"

    def __init__(self, url="redis://localhost:6379"):
        import redis
        self.r = redis.Redis.from_url(url)

    def put(self, key, buf):
        self.r.set(key, buf)

    def get_to_gpu(self, key, nbytes, device="cuda"):
        return _to_gpu(self.r.get(key), device)

    def delete(self, key):
        self.r.delete(key)


class MinioBackend(Backend):
    name = "minio"

    def __init__(self, endpoint="localhost:9000", access_key="minioadmin", secret_key="minioadmin",
                 bucket="kvcache", secure=False):
        from minio import Minio
        self.c, self.bucket = Minio(endpoint, access_key, secret_key, secure=secure), bucket
        if not self.c.bucket_exists(bucket):
            self.c.make_bucket(bucket)

    def put(self, key, buf):
        import io
        self.c.put_object(self.bucket, key, io.BytesIO(buf), len(buf))

    def get_to_gpu(self, key, nbytes, device="cuda"):
        r = self.c.get_object(self.bucket, key)
        try:
            raw = r.read()
        finally:
            r.close()
            r.release_conn()
        return _to_gpu(raw, device)

    def delete(self, key):
        self.c.remove_object(self.bucket, key)


class GdsBackend(Backend):
    """GPUDirect Storage through kvikio (cuFile). Reads land directly in GPU memory.

    Requires: NVIDIA GDS driver (nvidia-fs), a supported filesystem (ext4/xfs on NVMe), `kvikio`.
    kvikio falls back to POSIX compat mode if GDS is unavailable; check `kvikio.defaults` and
    `gdscheck -p`, otherwise you are benchmarking a bounce buffer and not GDS.
    """
    name = "gds"

    def __init__(self, root: str):
        import kvikio  # noqa: F401
        self.root = root
        os.makedirs(root, exist_ok=True)

    def _p(self, key):
        return os.path.join(self.root, key)

    def put(self, key, buf):
        import kvikio
        import numpy as np
        import cupy as cp
        arr = cp.asarray(np.frombuffer(buf, dtype=np.uint8))
        with kvikio.CuFile(self._p(key), "w") as f:
            f.write(arr)

    def get_to_gpu(self, key, nbytes, device="cuda"):
        import kvikio
        import cupy as cp
        out = torch.empty(nbytes, dtype=torch.uint8, device=device)
        with kvikio.CuFile(self._p(key), "r") as f:
            n = f.read(cp.asarray(out))   # zero-copy view of the torch buffer
        assert n == nbytes, f"short read {n}/{nbytes}"
        return out

    def delete(self, key):
        try:
            os.remove(self._p(key))
        except FileNotFoundError:
            pass
