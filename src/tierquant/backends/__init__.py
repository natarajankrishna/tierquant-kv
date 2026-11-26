"""Byte-level storage backends used by benchmarks/bench_backends.py.

All backends implement: put(key, buf: bytes-like), get_to_gpu(key, nbytes, device) -> torch.uint8 tensor
on the GPU, delete(key). `get_to_gpu` measures the *time to GPU-resident bytes*, which is what matters for TTFT.
"""
from .base import Backend  # noqa: F401
