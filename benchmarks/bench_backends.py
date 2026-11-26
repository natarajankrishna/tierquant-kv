"""Experiment E3: time-to-GPU-resident-bytes for each backend and stored precision.

    python benchmarks/bench_backends.py --backends nvme,gds,redis,minio --nvme-dir /mnt/nvme/kv \
        --blocks 32 --out results/backends.json

Payload sizes = one 256-token block of the target model at each precision (random bytes, so
compressors in the stack cannot cheat). Also fits t = latency + bytes/bw per backend, the inputs
the cost model needs (written under "fit").
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch

from tierquant.policy import bytes_ratio


def build(name, a):
    from tierquant.backends import base
    if name == "dram":
        return base.DramBackend()
    if name == "nvme":
        return base.NvmeFileBackend(a.nvme_dir)
    if name == "gds":
        return base.GdsBackend(os.path.join(a.nvme_dir, "gds"))
    if name == "redis":
        return base.RedisBackend(a.redis_url)
    if name == "minio":
        return base.MinioBackend(a.minio_endpoint)
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", default="nvme")
    ap.add_argument("--nvme-dir", default="/tmp/tierquant_bench")
    ap.add_argument("--redis-url", default="redis://localhost:6379")
    ap.add_argument("--minio-endpoint", default="localhost:9000")
    ap.add_argument("--kv-bytes-per-token", type=int, default=131072)
    ap.add_argument("--block-tokens", type=int, default=256)
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=32)
    ap.add_argument("--reads", type=int, default=64)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    assert torch.cuda.is_available(), "needs a GPU: the metric is time to GPU-resident bytes"
    fp16_bytes = a.kv_bytes_per_token * a.block_tokens
    results, fit = [], {}
    for name in a.backends.split(","):
        be = build(name, a)
        pts = []
        for prec in ("fp16", "fp8", "int8", "int4"):
            nbytes = int(fp16_bytes * bytes_ratio(prec, a.group))
            payload = np.random.default_rng(0).integers(0, 255, nbytes, dtype=np.uint8).tobytes()
            keys = [f"{prec}_{i}" for i in range(a.blocks)]
            t0 = time.perf_counter()
            for k in keys:
                be.put(k, payload)
            wr = (time.perf_counter() - t0) / a.blocks
            lat = []
            for _ in range(a.reads):
                k = random.choice(keys)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                x = be.get_to_gpu(k, nbytes)
                torch.cuda.synchronize()
                lat.append(time.perf_counter() - t0)
                assert x.numel() == nbytes
            for k in keys:
                be.delete(k)
            lat = np.array(lat[1:])  # drop warm-up
            row = dict(backend=name, precision=prec, nbytes=nbytes, p50_ms=float(np.percentile(lat, 50) * 1e3),
                       p99_ms=float(np.percentile(lat, 99) * 1e3), read_GBps=float(nbytes / lat.mean() / 1e9),
                       write_GBps=float(nbytes / wr / 1e9))
            results.append(row)
            pts.append((nbytes, float(np.median(lat))))
            print(row)
        xs, ys = zip(*pts)
        slope, icpt = np.polyfit(xs, ys, 1)
        fit[name] = dict(bw=float(1 / slope) if slope > 0 else None, latency=float(max(icpt, 0)))
        be.close()
    if a.out:
        json.dump(dict(rows=results, fit=fit), open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
