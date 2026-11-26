# tierquant-kv

**Precision-descending KV-cache tiering for LLM inference.**
KV blocks are stored at full precision in GPU HBM, at lower precision in host DRAM, and at the lowest
precision on NVMe/remote storage. Demotion re-encodes a block; promotion decodes it. A quality-budget
controller picks the per-tier precision, and a load-vs-recompute rule falls back to prefill when
loading a block would be slower than recomputing it.

```
        HBM (fp16)  --evict-->  DRAM (fp8)  --evict-->  NVMe / remote (int4)
            ^                      |                        |
            +---- decode + promote-+------------------------+
   recompute instead if  load(n) + dequant(n)  >=  prefill(n)
```

## Why
Existing tiered KV caches (LMCache, Dynamo KVBM, SGLang HiCache, Mooncake, ...) move blocks between tiers
unchanged. Tier capacity and load time are both functions of **bytes**, so lowering precision on demotion
should multiply effective capacity, shorten loads, and move the load-vs-recompute break-even, at a quality
cost that this repo measures and budgets rather than assuming. See `docs/related_work.md` for the prior art
you must check before claiming novelty (CacheGen is the closest).

## Research questions
1. How much quality (perplexity, needle accuracy) do you trade for how much capacity / hit rate / TTFT?
2. Where does load-vs-recompute break even per tier and precision, and does quantisation move it?
3. Can a controller choose per-tier precision automatically under an explicit quality budget?
4. `single_shot` vs `stacked` re-quantisation: how fast does error compound?

## What is in the repo
```
src/tierquant/   quant.py  policy.py  manager.py  breakeven.py  sim.py  workloads.py
                 roundtrip.py  block.py  lmcache_adapter.py (sketch)  backends/{dram,nvme,redis,minio,gds}
benchmarks/      bench_quant  bench_backends  calibrate  sim_hitrate  run_ttft
eval/            quality_eval (ppl / top-1 / KL)   needle (needle-in-haystack)
configs/ scripts/ docs/ tests/
```

## Quick start
```bash
pip install -e ".[dev]"            # numpy + pyyaml (torch-free modules)
pip install torch                  # for codecs/manager/benchmarks
pytest -q

# CPU-only experiments
PYTHONPATH=src python benchmarks/sim_hitrate.py            # E1: hit rate vs precision map
PYTHONPATH=src python -m tierquant.breakeven               # E4 with an ILLUSTRATIVE profile

# GPU experiments
pip install -e ".[bench,eval]"
python benchmarks/bench_quant.py                           # E0
python benchmarks/bench_backends.py --backends dram,nvme --nvme-dir /mnt/nvme/kv --out results/backends.json
python benchmarks/calibrate.py --model meta-llama/Llama-3.1-8B --backends-json results/backends.json
(cd eval && python quality_eval.py && python needle.py)    # E5
```
Full run order: `scripts/run_all.sh`. Experiment definitions and the claim -> evidence table: `docs/experiments.md`.

## Using the manager
```python
from tierquant.policy import PrecisionMap
from tierquant.manager import TieredKVManager

pm = PrecisionMap(tiers={"hbm": "fp16", "dram": "fp8", "nvme": "int4"}, protect_first_layers=2, sink_tokens=4)
mgr = TieredKVManager({"hbm": 10 << 30, "dram": 32 << 30, "nvme": 150 << 30}, pm, n_layers=32,
                      nvme_dir="/mnt/nvme/kv", requant="single_shot")
mgr.put(block_id, ks, vs, start_token=0)      # ks/vs: lists of [H, T, D] per layer
hit = mgr.get(block_id)                       # (ks, vs, meta) or None (miss or recompute-is-cheaper)
```
Controller:
```python
from tierquant.policy import QualityBudgetController
ctl = QualityBudgetController(delta_table={"fp16": 0, "fp8": .3, "int8": .1, "int4": 1.5}, budget=1.0)
pm = ctl.choose(hit_share={"dram": .3, "nvme": .3}, capacity_bytes={"dram": 32e9, "nvme": 150e9})
```
(delta values above are placeholders; use `results/quality_delta.json` from `eval/quality_eval.py`.)

## Simulator sanity output (NOT a hardware result)
`benchmarks/sim_hitrate.py` defaults: synthetic Zipf(0.9) RAG trace, 10 GB HBM / 32 GB DRAM / 150 GB NVMe,
256-token blocks, Llama-3-8B KV size. It shows only that the *mechanism* (more blocks per byte -> fewer
evictions) behaves as expected; real hit rates depend on the real workload.

| map | prefix hit rate | modelled prefill (s, illustrative profile) |
|---|---|---|
| fp16/fp16/fp16 | 0.417 | 0.64 |
| fp16/fp8/fp8 | 0.500 | 0.53 |
| fp16/fp8/int4 | 0.566 | 0.47 |

With the illustrative profile the break-even for **NVMe/GDS is only tens of tokens** (loading almost always
beats prefill for a GQA model with fast NVMe); it moves substantially only for slow tiers (Redis/MinIO).
Your measured profile decides what is true on your hardware; report that, whatever it shows.

## License
MIT
