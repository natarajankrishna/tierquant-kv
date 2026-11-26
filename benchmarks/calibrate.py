"""Build results/profile.json (HardwareProfile) from measurements.

    python benchmarks/calibrate.py --model meta-llama/Llama-3.1-8B --backends-json results/backends.json

Measures: prefill time vs length (fits a*n + c*n^2), dequant throughput of the codec on this GPU,
and KV bytes/token from the model config; merges bandwidth/latency fits from bench_backends.py.
"""
import argparse
import json
import time

import numpy as np
import torch

from tierquant.breakeven import HardwareProfile
from tierquant.quant import encode_block


def measure_prefill(model, lengths, vocab):
    ts = []
    for n in lengths:
        ids = torch.randint(1000, min(vocab, 30000), (1, n), device=model.device)
        with torch.no_grad():
            model(ids, use_cache=True)             # warm-up
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(ids, use_cache=True)
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
        print(f"prefill n={n}: {ts[-1]:.3f}s")
    c, a, _ = np.polyfit(lengths, ts, 2)
    return float(max(a, 1e-7)), float(max(c, 0.0))


def measure_dequant(layers, heads, hd, tokens=256, group=64):
    ks = [torch.randn(heads, tokens, hd, device="cuda", dtype=torch.float16) for _ in range(layers)]
    vs = [torch.randn(heads, tokens, hd, device="cuda", dtype=torch.float16) for _ in range(layers)]
    enc = encode_block(ks, vs, ["int4"] * layers, group, group)
    enc.decode()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(5):
        enc.decode()
    torch.cuda.synchronize()
    return layers * heads * tokens * hd * 2 * 2 / ((time.perf_counter() - t0) / 5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--backends-json", default=None)
    ap.add_argument("--lengths", default="1024,2048,4096,8192,16384")
    ap.add_argument("--out", default="results/profile.json")
    a = ap.parse_args()
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(a.model)
    L, kvh = cfg.num_hidden_layers, getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    hd = cfg.hidden_size // cfg.num_attention_heads
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16, device_map="cuda")
    pa, pc = measure_prefill(model, [int(x) for x in a.lengths.split(",")], cfg.vocab_size)
    prof = HardwareProfile(kv_bytes_per_token=L * kvh * hd * 2 * 2, prefill_a=pa, prefill_c=pc,
                           dequant_Bps=measure_dequant(L, kvh, hd))
    if a.backends_json:
        fit = json.load(open(a.backends_json))["fit"]
        prof.tier_bw = {k: v["bw"] for k, v in fit.items() if v["bw"]}
        prof.tier_latency = {k: v["latency"] for k, v in fit.items()}
    prof.to_json(a.out)
    print(json.dumps(prof.__dict__, indent=2))


if __name__ == "__main__":
    main()
