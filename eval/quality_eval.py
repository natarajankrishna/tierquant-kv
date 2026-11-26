"""Experiment E5: quality cost of loading a cached prefix from a lossy tier.

For each window: prefill prefix (P tokens) exactly -> encode/decode the KV with a precision config
(what a lower tier would return) -> score the next S tokens. Reports perplexity delta, top-1
agreement with the exact-cache run, and KL(exact || quantised). Also writes a delta table for the
controller (configs/quality_delta.example.json format).

    python eval/quality_eval.py --model meta-llama/Llama-3.1-8B --windows 20 --out results/quality.json
"""
import argparse
import json
import math

import torch
import torch.nn.functional as F

from common import fresh_cache, load, to_legacy, wikitext_ids
from tierquant.policy import PrecisionMap
from tierquant.roundtrip import roundtrip_legacy_cache

CONFIGS = {
    "fp8": dict(p="fp8", protect=2, sink=4),
    "int8": dict(p="int8", protect=2, sink=4),
    "int4": dict(p="int4", protect=2, sink=4),            # protected layers + sink tokens
    "int4_noprotect": dict(p="int4", protect=0, sink=0),   # ablation: what protection buys
    "int4_g32": dict(p="int4", protect=2, sink=4, group=32),
    "int4_g128": dict(p="int4", protect=2, sink=4, group=128),
}


def pmap_for(c):
    g = c.get("group", 64)
    return PrecisionMap(tiers={"hbm": "fp16", "nvme": c["p"]}, protect_first_layers=c["protect"],
                        sink_tokens=c["sink"], key_group=g, value_group=g)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--prefix", type=int, default=3584)
    ap.add_argument("--suffix", type=int, default=512)
    ap.add_argument("--windows", type=int, default=20)
    ap.add_argument("--block-tokens", type=int, default=256)
    ap.add_argument("--out", default="results/quality.json")
    a = ap.parse_args()
    tok, model = load(a.model)
    P, S = a.prefix, a.suffix
    ids = wikitext_ids(tok, a.windows * (P + S + 1))
    agg = {k: dict(nll=0.0, agree=0.0, kl=0.0, ratio=0.0) for k in ["exact", *CONFIGS]}
    n_tok = 0
    for w in range(a.windows):
        win = ids[w * (P + S + 1):(w + 1) * (P + S + 1)].unsqueeze(0).cuda()
        pre = model(win[:, :P], use_cache=True)
        legacy = to_legacy(pre.past_key_values)
        suf_in, tgt = win[:, P:P + S], win[:, P + 1:P + S + 1]

        def score(leg):
            lg = model(suf_in, past_key_values=fresh_cache(leg), use_cache=True).logits.float()
            return lg, F.cross_entropy(lg[0], tgt[0], reduction="sum").item()

        ex_logits, ex_nll = score(legacy)
        agg["exact"]["nll"] += ex_nll
        ex_lp = F.log_softmax(ex_logits[0], -1)
        for name, c in CONFIGS.items():
            rt, ratio = roundtrip_legacy_cache(legacy, pmap_for(c), "nvme", a.block_tokens)
            lg, nll = score(rt)
            lp = F.log_softmax(lg[0], -1)
            agg[name]["nll"] += nll
            agg[name]["agree"] += (lg[0].argmax(-1) == ex_logits[0].argmax(-1)).float().sum().item()
            agg[name]["kl"] += F.kl_div(lp, ex_lp, log_target=True, reduction="sum").item()
            agg[name]["ratio"] += ratio
        n_tok += S
        print(f"window {w + 1}/{a.windows}", flush=True)
    ppl_exact = math.exp(agg["exact"]["nll"] / n_tok)
    rows = {"exact": dict(ppl=ppl_exact)}
    for name in CONFIGS:
        ppl = math.exp(agg[name]["nll"] / n_tok)
        rows[name] = dict(ppl=ppl, delta_ppl_pct=100 * (ppl / ppl_exact - 1),
                          top1_agreement=agg[name]["agree"] / n_tok, kl=agg[name]["kl"] / n_tok,
                          bytes_ratio=agg[name]["ratio"] / a.windows)
    print(json.dumps(rows, indent=2))
    json.dump(dict(args=vars(a), rows=rows), open(a.out, "w"), indent=2)
    delta = {"fp16": 0.0, **{p: rows[p]["delta_ppl_pct"] for p in ("fp8", "int8", "int4")}}
    json.dump(dict(units="percent perplexity increase", delta=delta),
              open(a.out.replace(".json", "_delta.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
