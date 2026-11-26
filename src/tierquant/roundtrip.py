"""Simulate "this prefix was cached in tier T and loaded back": encode -> decode a legacy HF cache.

legacy: tuple/list over layers of (k, v), each [1, H, T, D]. Blocks of `block_tokens` are encoded
independently (as they would be in the cache); block 0 keeps `sink_tokens` in fp16.
"""
from __future__ import annotations

from typing import Tuple

import torch

from .policy import PrecisionMap
from .quant import encode_layer


def roundtrip_legacy_cache(legacy, pmap: PrecisionMap, tier: str, block_tokens: int = 256):
    L = len(legacy)
    precs = pmap.layer_precisions(tier, L)
    out, enc_bytes, raw_bytes = [], 0, 0
    for l, (k, v) in enumerate(legacy):
        assert k.shape[0] == 1, "batch size 1 only"
        T = k.shape[2]
        ks, vs = [], []
        for b0 in range(0, T, block_tokens):
            kb, vb = k[0, :, b0:b0 + block_tokens], v[0, :, b0:b0 + block_tokens]
            enc = encode_layer(kb, vb, precs[l], pmap.key_group, pmap.value_group,
                               sink_tokens=pmap.sink_tokens if b0 == 0 else 0)
            dk, dv = enc.decode()
            ks.append(dk)
            vs.append(dv)
            enc_bytes += enc.nbytes()
            raw_bytes += (kb.numel() + vb.numel()) * kb.element_size()
        out.append((torch.cat(ks, 1).unsqueeze(0).to(k.dtype), torch.cat(vs, 1).unsqueeze(0).to(v.dtype)))
    return tuple(out), enc_bytes / max(raw_bytes, 1)
