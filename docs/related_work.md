# Related work (verify before citing; written from memory)
- **LMCache**: tiered KV offload/reuse for vLLM/SGLang; storage backends incl. CPU, disk, Redis, remote; CacheGen-style serialisation for network backends. Check whether per-tier dtype/quantisation options exist in your pinned version.
- **CacheGen** (SIGCOMM '24): KV compression for network streaming. Closest prior art for "compress KV for transfer". Difference here: precision is *tier-coupled at demotion time* under an explicit quality budget, with a load-vs-recompute fallback.
- **KIVI, KVQuant, QServe, Atom**: KV quantisation inside the GPU cache. This repo reuses the key-per-channel / value-per-token grouping idea.
- **AttentionStore / CachedAttention** (ATC '24), **Mooncake**, **NVIDIA Dynamo KVBM + KV-aware router**, **SGLang HiCache**, **llm-d**, **AIBrix**: tiered/distributed KV caching and cache-aware routing, generally without per-tier precision.
- **ShadowKV, InfiniGen, Quest**: importance-based offload/selection, orthogonal to precision.

Before claiming novelty: search arXiv / GitHub / LMCache + Dynamo release notes for "tiered KV cache quantization",
"demotion quantization KV offload", "precision-adaptive KV hierarchy". Frame the contribution as
(1) tier-coupled precision + quality-budget controller and (2) an empirical load-vs-recompute study,
not as "a hierarchical KV cache".
