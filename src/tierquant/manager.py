"""TieredKVManager: HBM -> DRAM -> NVMe block cache with precision descending by tier.

Eviction is LRU per tier (swap in a different policy by overriding `_victim`). Demotion encodes
the block to the destination tier's precision. Two re-quantisation policies:

  single_shot : lossy data is never re-encoded. On HBM eviction of a lossless (n_lossy == 0) block
                every lower tier's copy is encoded directly from the original (each copy has
                n_lossy == 1). Promotion copies up and keeps the lower copy, so a promoted block
                that is evicted again is simply dropped from HBM. Costs extra NVMe write traffic.
  stacked     : each demotion decodes and re-encodes into the next tier (n_lossy grows, errors
                compound). Cheaper in writes; included as the ablation.

Fallback rule: if a profile is given and loading from tier T is modelled as slower than recomputing,
`get` returns None (counted as `fallback_recompute`).

Not implemented here (see docs/design.md): async demotion on a side CUDA stream, pinned-memory
staging, GPUDirect Storage for the NVMe tier, and layer-wise overlap with suffix prefill.
"""
from __future__ import annotations

import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .block import BlockMeta
from .breakeven import HardwareProfile, should_load
from .policy import TIER_ORDER, PrecisionMap
from .quant import EncodedBlock, encode_block, rel_error


@dataclass
class _Entry:
    meta: BlockMeta
    payload: object          # EncodedBlock (hbm/dram) or file path (nvme)
    nbytes: int


def _sync(dev: str):
    if dev.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


class TieredKVManager:
    def __init__(self, capacities: Dict[str, int], pmap: PrecisionMap, n_layers: int,
                 device: str = "cuda", nvme_dir: str = "/tmp/tierquant_nvme",
                 requant: str = "single_shot", profile: Optional[HardwareProfile] = None,
                 measure_error: bool = False):
        assert requant in ("single_shot", "stacked")
        self.tiers = [t for t in TIER_ORDER if t in capacities]
        self.cap, self.pmap, self.n_layers = dict(capacities), pmap, n_layers
        self.device, self.nvme_dir = device, nvme_dir
        self.requant, self.profile, self.measure_error = requant, profile, measure_error
        os.makedirs(nvme_dir, exist_ok=True)
        self.store: Dict[str, OrderedDict] = {t: OrderedDict() for t in self.tiers}
        self.used = {t: 0 for t in self.tiers}
        self.stats = dict(hits={t: 0 for t in self.tiers}, misses=0, fallback_recompute=0,
                          quantize_s=0.0, dequantize_s=0.0, nvme_write_bytes=0, nvme_read_bytes=0,
                          demotions=0, drops=0)

    # ------------------------------------------------------------------ helpers
    def _kw(self):
        return dict(key_group=self.pmap.key_group, value_group=self.pmap.value_group)

    def _encode_for(self, tier: str, ks, vs, start_token: int) -> EncodedBlock:
        sink = self.pmap.sink_tokens if start_token == 0 else 0
        return encode_block(ks, vs, self.pmap.layer_precisions(tier, self.n_layers),
                            sink_tokens=sink, **self._kw())

    def _place(self, tier: str, meta: BlockMeta, enc: EncodedBlock) -> None:
        bid = meta.block_id
        if tier == "nvme":
            path = os.path.join(self.nvme_dir, f"{bid}.pt")
            torch.save(enc.to("cpu"), path)
            self.stats["nvme_write_bytes"] += enc.nbytes()
            payload = path
        elif tier == "dram":
            payload = enc.to("cpu")
        else:
            payload = enc.to(self.device)
        meta.precision = self.pmap.tiers[tier]
        self.store[tier][bid] = _Entry(meta, payload, enc.nbytes())
        self.store[tier].move_to_end(bid)
        self.used[tier] += enc.nbytes()
        self._make_room(tier)

    def _remove(self, tier: str, bid: str) -> Optional[_Entry]:
        e = self.store[tier].pop(bid, None)
        if e is not None:
            self.used[tier] -= e.nbytes
            if tier == "nvme" and os.path.exists(e.payload):
                os.remove(e.payload)
        return e

    def _load_payload(self, tier: str, e: _Entry) -> EncodedBlock:
        if tier == "nvme":
            self.stats["nvme_read_bytes"] += e.nbytes
            return torch.load(e.payload, weights_only=False)
        return e.payload

    def _make_room(self, tier: str) -> None:
        while self.used[tier] > self.cap[tier] and self.store[tier]:
            bid = next(iter(self.store[tier]))          # LRU victim
            self._evict(tier, bid)

    def _below(self, tier: str) -> Optional[str]:
        i = self.tiers.index(tier)
        return self.tiers[i + 1] if i + 1 < len(self.tiers) else None

    def _evict(self, tier: str, bid: str) -> None:
        e = self._remove(tier, bid)
        meta = e.meta
        lower = self.tiers[self.tiers.index(tier) + 1:]
        if not lower:
            self.stats["drops"] += 1
            return
        t0 = time.perf_counter()
        if self.requant == "single_shot":
            if meta.n_lossy > 0:
                self.stats["drops"] += 1        # never re-encode lossy data
                return
            ks, vs = self._load_payload(tier, e).decode()   # lossless original (tier == hbm)
            _sync(self.device)
            for t in lower:                      # inclusive write-behind, each copy encoded once
                if bid in self.store[t]:
                    continue
                enc = self._encode_for(t, ks, vs, meta.start_token)
                m = BlockMeta(bid, meta.n_tokens, meta.start_token, n_lossy=1,
                              est_error=self._err(ks, vs, enc))
                self._place(t, m, enc)
        else:                                    # stacked
            dst = lower[0]
            ks, vs = self._load_payload(tier, e).decode()
            enc = self._encode_for(dst, ks, vs, meta.start_token)
            m = BlockMeta(bid, meta.n_tokens, meta.start_token, n_lossy=meta.n_lossy + 1,
                          est_error=meta.est_error + self._err(ks, vs, enc))
            self._place(dst, m, enc)
        _sync(self.device)
        self.stats["quantize_s"] += time.perf_counter() - t0
        self.stats["demotions"] += 1

    def _err(self, ks, vs, enc: EncodedBlock) -> float:
        if not self.measure_error:
            return 0.0
        dk, dv = enc.decode()
        return sum(rel_error(a, b) for a, b in zip(ks, dk)) / len(ks)

    # ------------------------------------------------------------------ public API
    def put(self, block_id: str, ks: Sequence[torch.Tensor], vs: Sequence[torch.Tensor],
            start_token: int = 0) -> None:
        """Insert a freshly computed (lossless) block into HBM."""
        for t in self.tiers:                       # a recomputed block supersedes stale copies
            self._remove(t, block_id)
        meta = BlockMeta(block_id, ks[0].shape[1], start_token)
        enc = encode_block([k.to(self.device) for k in ks], [v.to(self.device) for v in vs],
                           ["fp16"] * self.n_layers)
        self._place("hbm", meta, enc)

    def contains(self, block_id: str) -> Optional[str]:
        for t in self.tiers:
            if block_id in self.store[t]:
                return t
        return None

    def get(self, block_id: str) -> Optional[Tuple[List[torch.Tensor], List[torch.Tensor], BlockMeta]]:
        tier = self.contains(block_id)
        if tier is None:
            self.stats["misses"] += 1
            return None
        e = self.store[tier][block_id]
        if tier != "hbm" and self.profile is not None and tier in self.profile.tier_bw:
            if not should_load(self.profile, tier, e.meta.n_tokens, self.pmap.ratio(tier, self.n_layers)):
                self.stats["fallback_recompute"] += 1
                return None
        self.stats["hits"][tier] += 1
        e.meta.touch()
        self.store[tier].move_to_end(block_id)
        t0 = time.perf_counter()
        enc = self._load_payload(tier, e)
        ks, vs = enc.to(self.device).decode()
        _sync(self.device)
        if tier != "hbm":
            self.stats["dequantize_s"] += time.perf_counter() - t0
            self._promote(e.meta, ks, vs, from_tier=tier)
        return ks, vs, e.meta

    def _promote(self, meta: BlockMeta, ks, vs, from_tier: str) -> None:
        m = BlockMeta(meta.block_id, meta.n_tokens, meta.start_token, meta.n_lossy, meta.est_error)
        enc = encode_block(ks, vs, ["fp16"] * self.n_layers)
        if self.requant == "stacked":
            self._remove(from_tier, meta.block_id)       # move up (exclusive)
        self._place("hbm", m, enc)                       # single_shot: copy up, keep lower copy

    def summary(self) -> Dict:
        s = dict(self.stats)
        s["used_bytes"] = dict(self.used)
        s["blocks"] = {t: len(self.store[t]) for t in self.tiers}
        return s
