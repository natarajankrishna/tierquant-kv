# Design

## Problem
Tiered KV caches (HBM -> DRAM -> NVMe/remote) move blocks between tiers unchanged. Tier capacity
and load time are both set by *bytes*, so storing blocks at lower precision in lower tiers should
(a) multiply effective capacity, (b) shorten loads, and (c) move the load-vs-recompute break-even.
The price is quantisation error, which this project budgets and measures instead of switching
compression on globally.

## Components
| Module | Role |
|---|---|
| `quant.py` | fp16 / fp8(e4m3) / int8 / int4 codecs. Keys per-channel, values per-token (KIVI-style grouping). Sink tokens and first N layers stay fp16. |
| `policy.py` | `PrecisionMap` (tier -> precision, monotone non-increasing) and `QualityBudgetController`. |
| `manager.py` | HBM/DRAM/NVMe block store, LRU, demotion = encode, promotion = decode, fallback-to-recompute rule. |
| `breakeven.py` | `load(n) = L0 + n*bytes*ratio/bw (+ dequant)` vs `recompute(n) = a n + c n^2`. |
| `sim.py`, `workloads.py` | Trace-driven simulator (CPU only) for hit rate / modelled prefill vs precision map. |
| `roundtrip.py` | Encode->decode a HuggingFace cache to measure quality loss of a lossy tier. |
| `lmcache_adapter.py` | Sketch of wiring the codec into LMCache as a per-tier serde. **Unverified**. |
| `backends/` | DRAM / NVMe / Redis / MinIO / GDS byte stores for E3. |

## Decisions (and the ablation that justifies each)
| Decision | Choice | Ablation |
|---|---|---|
| Key/value grouping | keys per-channel, values per-token | quality_eval with swapped axes (add a config) |
| Group size | 64 default | `int4_g32` / `int4_g64` / `int4_g128` configs |
| Precision map | fp16/fp8/int4 default | `benchmarks/sim_hitrate.py` + `quality_eval.py` |
| Protection | first 2 layers + 4 sink tokens | `int4` vs `int4_noprotect` |
| Re-quantisation | `single_shot` (lossy data never re-encoded) | `stacked`: error compounds; compare `est_error`, n_lossy |
| Quantise where | on the device holding the block (GPU on HBM eviction) | CPU quantisation timing (`quantize_s`) |
| Fallback | recompute when modelled load >= recompute | with/without `use_fallback` in sim |

## single_shot vs stacked
`stacked`: each demotion decodes and re-encodes; the NVMe copy has been quantised twice (fp16->fp8->int4).
`single_shot`: on HBM eviction of a lossless block, every lower tier's copy is encoded from the original,
so each has exactly one conversion. Promotion *copies up* and keeps the lower copy; a promoted
(already lossy) block that is evicted again is dropped rather than re-encoded. Cost: extra NVMe write
traffic (`nvme_write_bytes`), because a copy is written even if the DRAM copy would have been hit first.

## Quality accounting
`BlockMeta.n_lossy` and `est_error` travel with every copy. `est_error` is the measured relative error of
the conversion when `measure_error=True` (debug; costs a decode). The controller uses *offline-measured*
per-precision deltas weighted by the share of hits each tier serves; it does not guess quality online.

## Honest status
Implemented: codecs, policy + controller, manager (synchronous), cost model, simulator, workloads, quality
and needle evals, backend and TTFT benchmarks. **Not implemented** (stretch / needed for end-to-end claims):
- Async demotion on a side CUDA stream, pinned staging, layer-wise load/prefill overlap.
- GDS as the manager's NVMe tier (GDS exists only in the backend benchmark).
- LMCache integration (adapter is a sketch) and Dynamo router integration.
- INT8 KV transfer between prefill and decode workers.
- LongBench harness (use the official scorers; hook `roundtrip_legacy_cache` between prefill and decode).
- The torch-dependent code has unit tests but they have to be run on your machine (`pytest`).
