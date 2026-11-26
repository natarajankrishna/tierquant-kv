"""Precision maps and the quality-budget controller (torch-free).

A PrecisionMap says which precision each tier stores. The controller picks the map
that maximises effective capacity subject to an expected-degradation budget.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

PRECISIONS = ("fp16", "fp8", "int8", "int4")
_BITS = {"fp16": 16, "fp8": 8, "int8": 8, "int4": 4}
TIER_ORDER = ("hbm", "dram", "nvme")


def element_bytes(precision: str, group: int = 64) -> float:
    """Average stored bytes per KV element, including fp16 scale+zero per group."""
    if precision == "fp16":
        return 2.0
    if precision == "fp8":
        return 1.0  # per-tensor fp32 scale is negligible
    if precision == "int8":
        return 1.0 + 4.0 / group
    if precision == "int4":
        return 0.5 + 4.0 / group
    raise ValueError(f"unknown precision {precision}")


def bytes_ratio(precision: str, group: int = 64) -> float:
    """Stored bytes relative to fp16."""
    return element_bytes(precision, group) / 2.0


@dataclass
class PrecisionMap:
    tiers: Dict[str, str] = field(default_factory=lambda: {"hbm": "fp16", "dram": "fp8", "nvme": "int4"})
    protect_first_layers: int = 2     # these layers stay fp16 in every tier
    sink_tokens: int = 4              # first tokens of a sequence stay fp16 in every tier
    key_group: int = 64               # token-group size for per-channel key quantisation
    value_group: int = 64             # channel-group size for per-token value quantisation
    name: str = ""

    def __post_init__(self):
        for t, p in self.tiers.items():
            if p not in PRECISIONS:
                raise ValueError(f"bad precision {p} for tier {t}")
        if self.tiers.get("hbm", "fp16") != "fp16":
            raise ValueError("HBM must be fp16: it feeds attention directly")
        order = [self.tiers[t] for t in TIER_ORDER if t in self.tiers]
        for a, b in zip(order, order[1:]):
            if _BITS[b] > _BITS[a]:
                raise ValueError(f"precision must not increase down the hierarchy: {order}")

    def layer_precisions(self, tier: str, n_layers: int) -> List[str]:
        p = self.tiers[tier]
        return ["fp16" if i < self.protect_first_layers else p for i in range(n_layers)]

    def ratio(self, tier: str, n_layers: Optional[int] = None) -> float:
        """Average stored-bytes ratio (vs fp16) for this tier, accounting for protected layers."""
        r = bytes_ratio(self.tiers[tier], min(self.key_group, self.value_group))
        if n_layers:
            k = min(self.protect_first_layers, n_layers)
            return (k * 1.0 + (n_layers - k) * r) / n_layers
        return r

    def label(self) -> str:
        return self.name or "/".join(self.tiers[t] for t in TIER_ORDER if t in self.tiers)


@dataclass
class QualityBudgetController:
    """Pick the precision map that maximises effective capacity under a quality budget.

    delta_table[p]  : measured quality degradation (e.g. accuracy points or ppl delta)
                      when a cached prefix is loaded at precision p. delta['fp16'] = 0.
                      Produce it with eval/quality_eval.py.
    budget          : maximum expected degradation, in the same units as delta_table.
    hit_share[tier] : fraction of cache hits served from each tier (measure online).
    Expected degradation of a map = sum_tier hit_share[tier] * delta[precision(tier)].
    Objective: sum_tier capacity_bytes[tier] / ratio(tier)  (fp16-equivalent bytes).
    """
    delta_table: Dict[str, float]
    budget: float
    candidates: Dict[str, Sequence[str]] = field(default_factory=lambda: {
        "dram": ("fp16", "fp8", "int8"),
        "nvme": ("fp16", "fp8", "int8", "int4"),
    })
    protect_first_layers: int = 2
    sink_tokens: int = 4
    group: int = 64

    def expected_degradation(self, pmap: PrecisionMap, hit_share: Dict[str, float]) -> float:
        d = 0.0
        for tier, share in hit_share.items():
            p = pmap.tiers.get(tier, "fp16")
            d += share * self.delta_table.get(p, 0.0)
        return d

    def effective_capacity(self, pmap: PrecisionMap, capacity_bytes: Dict[str, float]) -> float:
        return sum(cap / bytes_ratio(pmap.tiers.get(t, "fp16"), self.group)
                   for t, cap in capacity_bytes.items())

    def choose(self, hit_share: Dict[str, float], capacity_bytes: Dict[str, float]) -> PrecisionMap:
        tiers = [t for t in ("dram", "nvme") if t in self.candidates]
        best, best_key = None, None
        for combo in itertools.product(*(self.candidates[t] for t in tiers)):
            m = {"hbm": "fp16", **dict(zip(tiers, combo))}
            try:
                pm = PrecisionMap(tiers=m, protect_first_layers=self.protect_first_layers,
                                  sink_tokens=self.sink_tokens, key_group=self.group,
                                  value_group=self.group)
            except ValueError:
                continue  # violates monotonic precision rule
            deg = self.expected_degradation(pm, hit_share)
            if deg > self.budget:
                continue
            key = (self.effective_capacity(pm, capacity_bytes), -deg)
            if best_key is None or key > best_key:
                best, best_key = pm, key
        if best is None:  # nothing fits the budget -> full precision everywhere
            best = PrecisionMap(tiers={"hbm": "fp16", **{t: "fp16" for t in tiers}},
                                protect_first_layers=self.protect_first_layers,
                                sink_tokens=self.sink_tokens, key_group=self.group,
                                value_group=self.group, name="fp16-fallback")
        return best
