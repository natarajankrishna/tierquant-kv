"""Experiment E0: codec microbenchmark (GPU if available). Reports compression ratio, error,
encode/decode throughput per precision. Use it to sanity-check the byte ratios in policy.py.

    python benchmarks/bench_quant.py --layers 32 --heads 8 --head-dim 128 --tokens 256
"""
import argparse
import json
import time

import torch

from tierquant.quant import encode_block, rel_error


def timeit(fn, dev, reps=5):
    fn()
    if dev == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn()
    if dev == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    shape = (a.heads, a.tokens, a.head_dim)
    ks = [torch.randn(shape, device=dev, dtype=torch.float16) for _ in range(a.layers)]
    vs = [torch.randn(shape, device=dev, dtype=torch.float16) for _ in range(a.layers)]
    raw = sum(k.numel() + v.numel() for k, v in zip(ks, vs)) * 2
    rows = []
    for prec in ("fp16", "fp8", "int8", "int4"):
        try:
            te, enc = timeit(lambda: encode_block(ks, vs, [prec] * a.layers, a.group, a.group), dev)
        except RuntimeError as e:
            print(prec, "skipped:", e)
            continue
        td, (dk, dv) = timeit(lambda: enc.decode(), dev)
        err = sum(rel_error(x, y) for x, y in zip(ks, dk)) / a.layers
        rows.append(dict(precision=prec, device=dev, bytes_ratio=round(enc.nbytes() / raw, 4),
                         rel_err_k=round(err, 5), encode_GBps=round(raw / te / 1e9, 2),
                         decode_GBps=round(raw / td / 1e9, 2)))
    for r in rows:
        print(r)
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=2)
    print("NOTE: random Gaussian data understates real KV outliers; use eval/quality_eval.py for quality.")


if __name__ == "__main__":
    main()
