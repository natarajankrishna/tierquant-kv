"""Experiment E5b: needle-in-a-haystack with the whole context served from a lossy tier.

    python eval/needle.py --model meta-llama/Llama-3.1-8B --ctx 4096,8192,16384 --depths 0.1,0.5,0.9
"""
import argparse
import json
import random

import torch

from common import greedy, load, to_legacy, wikitext_ids
from quality_eval import CONFIGS, pmap_for
from tierquant.roundtrip import roundtrip_legacy_cache


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--ctx", default="4096,8192")
    ap.add_argument("--depths", default="0.1,0.5,0.9")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--configs", default="fp8,int8,int4,int4_noprotect")
    ap.add_argument("--out", default="results/needle.json")
    a = ap.parse_args()
    tok, model = load(a.model)
    hay = wikitext_ids(tok, max(int(c) for c in a.ctx.split(",")) + 1)
    random.seed(0)
    q = tok("\n\nQuestion: What is the secret passcode?\nAnswer: The secret passcode is",
            return_tensors="pt", add_special_tokens=False).input_ids.cuda()
    res = {}
    for ctx in map(int, a.ctx.split(",")):
        for depth in map(float, a.depths.split(",")):
            for trial in range(a.trials):
                code = f"{random.randint(1000, 9999)}-{random.choice(['ALPHA', 'DELTA', 'OMEGA', 'SIGMA'])}"
                needle = tok(f"\nThe secret passcode is {code}.\n", add_special_tokens=False).input_ids
                base = hay[trial * 7:trial * 7 + ctx - len(needle)].tolist()
                pos = int(depth * len(base))
                ids = torch.tensor([base[:pos] + needle + base[pos:]]).cuda()
                legacy = to_legacy(model(ids, use_cache=True).past_key_values)
                variants = {"exact": legacy}
                for name in a.configs.split(","):
                    variants[name] = roundtrip_legacy_cache(legacy, pmap_for(CONFIGS[name]), "nvme")[0]
                for name, leg in variants.items():
                    ok = code in greedy(model, tok, leg, q, 16)
                    res.setdefault(name, {}).setdefault(f"{ctx}@{depth}", []).append(ok)
            print(ctx, depth, {n: sum(v[f'{ctx}@{depth}']) / a.trials for n, v in res.items()}, flush=True)
    json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    main()
