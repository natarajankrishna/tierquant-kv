"""Experiment E2/E4: open-loop TTFT benchmark against an OpenAI-compatible vLLM server.

Launch a server variant first (scripts/launch_vllm_lmcache.sh), then:

    python benchmarks/run_ttft.py --url http://localhost:8000 --model meta-llama/Llama-3.1-8B \
        --workload zipf_rag --qps 2 --requests 400 --tag fp16-tiers --out results/ttft_fp16.json

Prompts are token-id lists (no tokenizer needed): a document's prefix is a deterministic stream
derived from its key, so repeated requests share an exact token prefix.
"""
import argparse
import asyncio
import json
import time
import zlib

import httpx
import numpy as np

from tierquant import workloads


def stream_ids(key: str, n: int, vocab_hi: int) -> list:
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    return rng.integers(1000, vocab_hi, size=n).tolist()


def build_prompt(r, vocab_hi, uid):
    if r.store_tokens == r.prefix_tokens:     # RAG: shared doc + unique question
        return stream_ids(r.prefix_key, r.prefix_tokens, vocab_hi) + stream_ids(f"q{uid}", r.suffix_tokens, vocab_hi)
    return stream_ids(r.prefix_key, r.prefix_tokens + r.suffix_tokens, vocab_hi)   # multi-turn (approximation)


async def one(client, url, model, prompt, t_start, results, meta):
    await asyncio.sleep(max(0.0, t_start - time.perf_counter()))
    t0 = time.perf_counter()
    ttft, err = None, None
    try:
        async with client.stream("POST", f"{url}/v1/completions", json=dict(
                model=model, prompt=prompt, max_tokens=1, temperature=0, stream=True)) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.startswith("data:") and "[DONE]" not in line:
                    ttft = time.perf_counter() - t0
                    break
    except Exception as e:  # noqa
        err = repr(e)
    results.append(dict(ttft=ttft, err=err, **meta))


async def main_async(a):
    kw = dict(n_requests=a.requests, qps=a.qps, seed=a.seed) if a.workload == "zipf_rag" else dict(seed=a.seed)
    reqs = workloads.make(a.workload, **kw)[:a.requests]
    t_base = time.perf_counter() + 1.0
    t0 = reqs[0].t
    results = []
    limits = httpx.Limits(max_connections=a.max_conns)
    async with httpx.AsyncClient(timeout=600, limits=limits) as client:
        tasks = [one(client, a.url, a.model, build_prompt(r, a.vocab_hi, i), t_base + (r.t - t0),
                     results, dict(prefix_key=r.prefix_key, prefix_tokens=r.prefix_tokens, idx=i))
                 for i, r in enumerate(reqs)]
        await asyncio.gather(*tasks)
    return results


def summarize(res, warmup):
    ok = [r for r in sorted(res, key=lambda r: r["idx"]) if r["ttft"] is not None and r["idx"] >= warmup]
    def stats(rs):
        x = np.array([r["ttft"] for r in rs])
        return dict(n=len(x), mean=float(x.mean()), p50=float(np.percentile(x, 50)),
                    p99=float(np.percentile(x, 99))) if len(x) else {}
    out = dict(overall=stats(ok), errors=sum(1 for r in res if r["err"]))
    for name, lo, hi in (("<=8K", 0, 8192), ("8-16K", 8193, 16384), (">16K", 16385, 10**9)):
        out[name] = stats([r for r in ok if lo <= r["prefix_tokens"] <= hi])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--workload", default="zipf_rag", choices=["zipf_rag", "multi_turn"])
    ap.add_argument("--qps", type=float, default=2.0)
    ap.add_argument("--requests", type=int, default=400)
    ap.add_argument("--warmup", type=int, default=50, help="requests excluded from stats (cache fill)")
    ap.add_argument("--vocab-hi", type=int, default=30000)
    ap.add_argument("--max-conns", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = asyncio.run(main_async(a))
    s = summarize(res, a.warmup)
    print(json.dumps(s, indent=2))
    if a.out:
        json.dump(dict(tag=a.tag, args=vars(a), summary=s, raw=res), open(a.out, "w"))


if __name__ == "__main__":
    main()
