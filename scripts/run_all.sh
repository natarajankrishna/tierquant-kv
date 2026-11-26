#!/usr/bin/env bash
# Run order for the full study. Steps needing a GPU/servers are marked.
set -euo pipefail
export PYTHONPATH=src:$PYTHONPATH
MODEL=${MODEL:-meta-llama/Llama-3.1-8B}
mkdir -p results

pytest -q                                                       # unit tests
python benchmarks/sim_hitrate.py --out results/sim_hitrate.csv   # E1  (CPU only)
python benchmarks/bench_quant.py --out results/codec.json        # E0  (GPU)
python benchmarks/bench_backends.py --backends dram,nvme,gds,redis,minio \
       --nvme-dir /mnt/nvme/kv --out results/backends.json       # E3  (GPU + services)
python benchmarks/calibrate.py --model "$MODEL" --backends-json results/backends.json   # profile.json
python -m tierquant.breakeven --profile results/profile.json --out results/breakeven.csv  # E4
python benchmarks/sim_hitrate.py --profile results/profile.json --out results/sim_hitrate_calibrated.csv
(cd eval && python quality_eval.py --model "$MODEL" --out ../results/quality.json)       # E5
(cd eval && python needle.py --model "$MODEL" --out ../results/needle.json)              # E5b
# E2 needs live servers: see docs/experiments.md (launch each variant, run benchmarks/run_ttft.py)
python scripts/plot_results.py
