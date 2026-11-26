"""Trace-driven hierarchy simulator (torch-free, no GPU needed).

Answers "how much does precision-per-tier buy in hit rate and modelled prefill time under a
fixed byte budget?" Exclusive hierarchy: a block lives in exactly one tier; hits are promoted to
HBM; HBM/DRAM evictions (LRU) demote one tier down; NVMe evictions drop the block.
Sizes use PrecisionMap.ratio(); latency uses breakeven.HardwareProfile (model, not measurement).
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Iterable, List, Optional

from .breakeven import HardwareProfile, load_time, recompute_time, should_load
from .policy import TIER_ORDER, PrecisionMap


class HierarchySimulator:
    def __init__(self, capacities_bytes: Dict[str, float], pmap: PrecisionMap,
                 block_tokens: int = 256, kv_bytes_per_token: int = 131072, n_layers: int = 32,
                 profile: Optional[HardwareProfile] = None, use_fallback: bool = True):
        self.tiers = [t for t in TIER_ORDER if t in capacities_bytes]
        self.cap = dict(capacities_bytes)
        self.pmap, self.bt, self.n_layers = pmap, block_tokens, n_layers
        self.profile = profile
        self.use_fallback = use_fallback and profile is not None
        fp16_block = block_tokens * kv_bytes_per_token
        self.size = {t: fp16_block * pmap.ratio(t, n_layers) for t in self.tiers}
        self.store: Dict[str, OrderedDict] = {t: OrderedDict() for t in self.tiers}
        self.used = {t: 0.0 for t in self.tiers}
        self.stats = dict(prefix_tokens=0, hit_tokens=0, fallback_tokens=0, requests=0,
                          modelled_prefill_s=0.0, dropped_blocks=0)
        self.tier_hit_tokens = {t: 0 for t in self.tiers}

    # -- internals -------------------------------------------------------------------------
    def _find(self, key) -> Optional[str]:
        for t in self.tiers:
            if key in self.store[t]:
                return t
        return None

    def _remove(self, key) -> None:
        t = self._find(key)
        if t:
            del self.store[t][key]
            self.used[t] -= self.size[t]

    def _insert(self, tier: str, key) -> None:
        self.store[tier][key] = True
        self.store[tier].move_to_end(key)
        self.used[tier] += self.size[tier]
        i = self.tiers.index(tier)
        while self.used[tier] > self.cap[tier] and self.store[tier]:
            victim, _ = self.store[tier].popitem(last=False)
            self.used[tier] -= self.size[tier]
            if i + 1 < len(self.tiers):
                self._insert(self.tiers[i + 1], victim)
            else:
                self.stats["dropped_blocks"] += 1

    # -- API -------------------------------------------------------------------------------
    def request(self, prefix_key: str, prefix_tokens: int, suffix_tokens: int, store_tokens: int) -> None:
        n_pref_blocks = prefix_tokens // self.bt
        hits: List = []
        for i in range(n_pref_blocks):
            key = (prefix_key, i)
            t = self._find(key)
            if t is None:
                break
            hits.append((key, t))

        # fallback rule: if loading a tier's tokens costs more than recomputing, truncate there
        if self.use_fallback and hits:
            per_tier: Dict[str, int] = {}
            for _, t in hits:
                per_tier[t] = per_tier.get(t, 0) + self.bt
            for t, n in per_tier.items():
                if t == "hbm":
                    continue
                if not should_load(self.profile, t, n, self.pmap.ratio(t, self.n_layers)):
                    cut = next(i for i, (_, tt) in enumerate(hits) if tt == t)
                    self.stats["fallback_tokens"] += (len(hits) - cut) * self.bt
                    hits = hits[:cut]
                    break

        load_s = 0.0
        loaded: Dict[str, int] = {}
        for key, t in hits:
            loaded[t] = loaded.get(t, 0) + self.bt
            self.tier_hit_tokens[t] += self.bt
            self._remove(key)
        for t, n in loaded.items():
            if t != "hbm" and self.profile:
                load_s += load_time(self.profile, t, n, self.pmap.ratio(t, self.n_layers))
        for key, _ in hits:
            self._insert("hbm", key)

        hit_tokens = len(hits) * self.bt
        for i in range(len(hits), store_tokens // self.bt):
            key = (prefix_key, i)
            self._remove(key)           # recomputed blocks supersede any stale lower copy
            self._insert("hbm", key)

        self.stats["requests"] += 1
        self.stats["prefix_tokens"] += prefix_tokens
        self.stats["hit_tokens"] += hit_tokens
        if self.profile:
            new = (prefix_tokens - hit_tokens) + suffix_tokens
            self.stats["modelled_prefill_s"] += load_s + (
                recompute_time(self.profile, hit_tokens + new) - recompute_time(self.profile, hit_tokens))

    def summary(self) -> Dict[str, float]:
        s = self.stats
        pt = max(s["prefix_tokens"], 1)
        out = dict(prefix_hit_rate=s["hit_tokens"] / pt,
                   fallback_frac=s["fallback_tokens"] / pt,
                   mean_modelled_prefill_s=s["modelled_prefill_s"] / max(s["requests"], 1),
                   dropped_blocks=s["dropped_blocks"])
        for t in self.tiers:
            out[f"hit_{t}"] = self.tier_hit_tokens[t] / pt
        out["capacity_blocks"] = sum(self.cap[t] / self.size[t] for t in self.tiers)
        return out


def run_trace(requests: Iterable, **sim_kwargs) -> Dict[str, float]:
    sim = HierarchySimulator(**sim_kwargs)
    for r in requests:
        sim.request(r.prefix_key, r.prefix_tokens, r.suffix_tokens, r.store_tokens)
    return sim.summary()
