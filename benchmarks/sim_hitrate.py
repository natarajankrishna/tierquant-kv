"""Experiment E1: hit rate / modelled prefill vs precision map under a fixed byte budget (no GPU).

    python benchmarks/sim_hitrate.py --workload zipf_rag --out results/sim_hitrate.csv

The latency column uses breakeven.HardwareProfile (illustrative unless you pass --profile from
calibrate.py), so treat it as a model. Hit-rate numbers depend only on the trace and sizes.
"""
import argparse
import csv

from tierquant import workloads
from tierquant.breakeven import HardwareProfile, example_profile
from tierquant.policy import PrecisionMap
from tierquant.sim import run_trace

GB = 1e9
MAPS = {
    "fp16/fp16/fp16 (baseline)": dict(tiers={"hbm": "fp16", "dram": "fp16", "nvme": "fp16"}),
    "fp16/fp8/fp8": dict(tiers={"hbm": "fp16", "dram": "fp8", "nvme": "fp8"}),
    "fp16/fp8/int8": dict(tiers={"hbm": "fp16", "dram": "fp8", "nvme": "int8"}),
    "fp16/fp8/int4": dict(tiers={"hbm": "fp16", "dram": "fp8", "nvme": "int4"}),
    "fp16/int8/int4": dict(tiers={"hbm": "fp16", "dram": "int8", "nvme": "int4"}),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", default="zipf_rag", choices=["zipf_rag", "multi_turn"])
    ap.add_argument("--hbm-gb", type=float, default=10)
    ap.add_argument("--dram-gb", type=float, default=32)
    ap.add_argument("--nvme-gb", type=float, default=150)
    ap.add_argument("--block-tokens", type=int, default=256)
    ap.add_argument("--protect-layers", type=int, default=2)
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--requests", type=int, default=5000)
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--sessions", type=int, default=3000, help="multi_turn only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    prof = HardwareProfile.from_json(a.profile) if a.profile else example_profile()
    kw = dict(n_requests=a.requests, alpha=a.alpha, seed=a.seed) if a.workload == "zipf_rag" else dict(n_sessions=a.sessions, seed=a.seed)
    trace = workloads.make(a.workload, **kw)
    caps = {"hbm": a.hbm_gb * GB, "dram": a.dram_gb * GB, "nvme": a.nvme_gb * GB}

    rows = []
    for name, spec in MAPS.items():
        pm = PrecisionMap(**spec, protect_first_layers=a.protect_layers, key_group=a.group,
                          value_group=a.group, name=name)
        r = run_trace(trace, capacities_bytes=caps, pmap=pm, block_tokens=a.block_tokens,
                      profile=prof)
        rows.append(dict(map=name, **{k: round(v, 4) for k, v in r.items()}))
    cols = list(rows[0])
    print(" | ".join(cols))
    for r in rows:
        print(" | ".join(str(r[c]) for c in cols))
    if a.out:
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
