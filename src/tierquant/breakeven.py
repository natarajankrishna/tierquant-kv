"""Load-vs-recompute cost model (torch-free).

load(n)      = L0 + n * kv_bytes * ratio / bw  +  n * kv_bytes / dequant_Bps   (dequant only if ratio < 1)
recompute(n) = a*n + c*n^2          (prefill of n tokens from scratch; c captures attention cost)

A block set is loaded only if load(n) < recompute(n). Break-even = smallest n where that holds.
Parameters come from `benchmarks/calibrate.py` + `benchmarks/bench_backends.py`.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from typing import Dict, Optional

from .policy import bytes_ratio


@dataclass
class HardwareProfile:
    kv_bytes_per_token: int = 131072            # Llama-3-8B fp16: 32 layers*8 kv heads*128*2*2B
    tier_bw: Dict[str, float] = field(default_factory=dict)       # bytes/s, to GPU-resident
    tier_latency: Dict[str, float] = field(default_factory=dict)  # fixed seconds per load
    dequant_Bps: float = 400e9                  # fp16 output bytes/s of the dequant kernel
    prefill_a: float = 5e-5                     # s/token (linear part)
    prefill_c: float = 2e-10                    # s/token^2 (attention part)

    @classmethod
    def from_json(cls, path: str) -> "HardwareProfile":
        with open(path) as f:
            return cls(**json.load(f))

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)


def example_profile() -> HardwareProfile:
    """ILLUSTRATIVE numbers only (A100-class, PCIe4 NVMe). Calibrate before reporting anything."""
    return HardwareProfile(
        tier_bw={"dram": 22e9, "nvme": 6e9, "gds": 6.5e9, "redis": 1.5e9, "minio": 1.0e9},
        tier_latency={"dram": 2e-4, "nvme": 1e-3, "gds": 8e-4, "redis": 4e-3, "minio": 2e-2},
    )


def load_time(p: HardwareProfile, tier: str, n_tokens: int, ratio: float = 1.0) -> float:
    raw = n_tokens * p.kv_bytes_per_token
    t = p.tier_latency.get(tier, 0.0) + raw * ratio / p.tier_bw[tier]
    if ratio < 1.0:
        t += raw / p.dequant_Bps
    return t


def recompute_time(p: HardwareProfile, n_tokens: int) -> float:
    return p.prefill_a * n_tokens + p.prefill_c * n_tokens ** 2


def should_load(p: HardwareProfile, tier: str, n_tokens: int, ratio: float = 1.0) -> bool:
    return load_time(p, tier, n_tokens, ratio) < recompute_time(p, n_tokens)


def breakeven_tokens(p: HardwareProfile, tier: str, ratio: float = 1.0,
                     max_tokens: int = 1_000_000) -> Optional[int]:
    """Smallest n with load(n) < recompute(n); None if loading never wins up to max_tokens.

    f(n)=recompute-load is convex with f(0)<0 when latency>0, so bisection is valid.
    """
    f = lambda n: recompute_time(p, n) - load_time(p, tier, n, ratio)
    if f(max_tokens) <= 0:
        return None
    lo, hi = 1, max_tokens
    if f(lo) > 0:
        return 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    return hi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=None, help="profile.json from calibrate.py (default: illustrative)")
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    prof = HardwareProfile.from_json(args.profile) if args.profile else example_profile()
    if not args.profile:
        print("WARNING: using ILLUSTRATIVE profile; run benchmarks/calibrate.py for real numbers.\n")
    rows = []
    for tier in prof.tier_bw:
        for prec in ("fp16", "fp8", "int8", "int4"):
            r = bytes_ratio(prec, args.group)
            n = breakeven_tokens(prof, tier, r)
            rows.append(dict(tier=tier, precision=prec, bytes_ratio=round(r, 3),
                             breakeven_tokens=n if n is not None else "never"))
    print(f"{'tier':8s} {'prec':6s} {'ratio':>6s} {'break-even tokens':>18s}")
    for r in rows:
        print(f"{r['tier']:8s} {r['precision']:6s} {r['bytes_ratio']:6.3f} {str(r['breakeven_tokens']):>18s}")
    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
