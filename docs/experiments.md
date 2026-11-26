# Experiments

Each experiment says what to run, what to report, and what claim it can support. Do not put a number on a
CV or paper that is not in `results/`.

| ID | Question | Command | Needs |
|---|---|---|---|
| E0 | Do codec sizes/throughput match the model? | `python benchmarks/bench_quant.py` | GPU |
| E1 | Hit rate vs precision map at fixed bytes | `python benchmarks/sim_hitrate.py [--workload multi_turn]` | CPU |
| E2 | Real TTFT (P50/P99) vs baseline | `benchmarks/run_ttft.py` against each server variant | vLLM + LMCache |
| E3 | Backend load latency / bandwidth per precision | `python benchmarks/bench_backends.py --backends dram,nvme,gds,redis,minio` | GPU, NVMe, (GDS), Redis, MinIO |
| E4 | Break-even context length per tier/precision | `python -m tierquant.breakeven --profile results/profile.json` | after calibrate.py |
| E5 | Quality loss per precision config | `cd eval && python quality_eval.py` | GPU, model weights |
| E5b | Needle-in-haystack from lossy cache | `cd eval && python needle.py` | GPU |
| E6 | Controller: pick map under a budget | `QualityBudgetController` on E5 deltas + E1 hit shares | after E1, E5 |

## Methodology rules
- Warm-up: exclude the first requests (`--warmup`); report cache-cold and steady-state separately.
- Baselines for E2: (a) vLLM GPU prefix cache only, (b) LMCache fp16 tiers, (c) tierquant. Same trace, same QPS, same seed.
- Run >= 3 seeds; report P50/P99 with the spread.
- NVMe reads: keep `drop_cache=True` (cold reads) or you are measuring RAM. Check GDS is active (`gdscheck -p`, kvikio compat mode off), or the "GDS" numbers are a bounce buffer.
- Quality: always report perplexity delta *and* a task metric (needle accuracy; LongBench if you build the harness).
- E1/E4 use a model; label them "modelled" and re-run with the calibrated profile.

## E2 variants
```
scripts/launch_vllm_lmcache.sh $MODEL                                # (a) baseline
scripts/launch_vllm_lmcache.sh $MODEL configs/lmcache_baseline.yaml  # (b) fp16 tiers
# (c) tierquant: LMCache + TierQuantSerde on L2/L3 (docs/design.md; adapter is a sketch)
python benchmarks/run_ttft.py --model $MODEL --qps 2 --tag baseline --out results/ttft_baseline.json
```

## Expected shape of results (hypotheses to test, not results)
- E1: quantised tiers raise capacity ~1.8x (fp8) to ~3x (fp8/int4 blend) and raise hit rate when the working set exceeds fp16 capacity; no gain when everything already fits.
- E3/E4: for a model with GQA (128 KB/token for Llama-3-8B) and fast NVMe, loading beats recomputing at almost any realistic context, so the break-even shifts mostly for *slow* tiers (Redis/MinIO over a network) or models with larger KV per token (MHA). Expect the break-even story to be tier-dependent; if NVMe shows no shift, report that.
- E5: 8-bit near-lossless; int4 depends on protection and group size; the safe floor for the lowest tier may be int8 for your model.

## Claim -> evidence
| Claim you may want to make | Evidence file |
|---|---|
| Effective capacity ~Nx | `results/codec.json` (bytes_ratio) + `sim_hitrate*.csv` (capacity_blocks) |
| Hit rate X% -> Y% | `sim_hitrate*.csv` (modelled) or LMCache hit counters during E2 |
| P50/P99 TTFT -x% | `ttft_*.json` summaries |
| <= k points quality loss | `quality.json`, `needle.json` (LongBench: only if you add the harness) |
| Break-even moves from A to B tokens | `breakeven.csv` from a *calibrated* profile, plus measured crossover from E2 |
