"""KV-block codecs: fp16 (raw), fp8 (e4m3, per-tensor scale), int8 / int4 (asymmetric, grouped).

Layout convention: one layer's keys/values are [H, T, D] (kv-heads, tokens, head_dim).
Keys are quantised per-channel (statistics over a group of tokens), values per-token
(statistics over a group of channels), following the KIVI observation that key outliers are
channel-wise and value outliers are token-wise.

Sink tokens (first `sink_tokens` of a sequence) and protected layers stay fp16.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import torch

_FP8 = getattr(torch, "float8_e4m3fn", None)
_FP8_MAX = 448.0


# ---------------------------------------------------------------- packed tensor types
@dataclass
class Raw:
    data: torch.Tensor

    def nbytes(self) -> int:
        return self.data.numel() * self.data.element_size()

    def decode(self) -> torch.Tensor:
        return self.data

    def apply(self, fn: Callable) -> "Raw":
        return Raw(fn(self.data))


@dataclass
class FP8:
    data: torch.Tensor      # float8_e4m3fn
    scale: torch.Tensor     # fp32 scalar
    dtype: torch.dtype

    def nbytes(self) -> int:
        return self.data.numel() * self.data.element_size() + self.scale.numel() * 4

    def decode(self) -> torch.Tensor:
        return (self.data.to(torch.float32) * self.scale).to(self.dtype)

    def apply(self, fn: Callable) -> "FP8":
        return FP8(fn(self.data), fn(self.scale), self.dtype)


@dataclass
class QTensor:
    data: torch.Tensor      # uint8; for 4-bit two values per byte (low nibble = even index)
    scale: torch.Tensor     # fp16, one per group
    zero: torch.Tensor      # fp16 (group minimum)
    bits: int
    axis: int
    group: int
    orig_len: int
    dtype: torch.dtype

    def nbytes(self) -> int:
        return (self.data.numel() + 2 * self.scale.numel() + 2 * self.zero.numel())

    def decode(self) -> torch.Tensor:
        return dequantize_groups(self)

    def apply(self, fn: Callable) -> "QTensor":
        return QTensor(fn(self.data), fn(self.scale), fn(self.zero), self.bits, self.axis,
                       self.group, self.orig_len, self.dtype)


# ---------------------------------------------------------------- group quantisation
def quantize_groups(x: torch.Tensor, bits: int, axis: int, group: int) -> QTensor:
    """Asymmetric min/max quantisation with statistics over groups of `group` elements along `axis`."""
    assert bits in (4, 8), "bits must be 4 or 8"
    dtype = x.dtype
    x = x.float().movedim(axis, -1)
    n = x.shape[-1]
    group = max(1, min(group, n))
    if bits == 4 and group % 2:
        group += 1                                   # nibble packing needs an even group
    pad = (-n) % group
    if pad:                                          # replicate the last element: min/max unchanged
        x = torch.cat([x, x[..., -1:].expand(*x.shape[:-1], pad)], dim=-1)
    g = x.reshape(*x.shape[:-1], -1, group)
    mn, mx = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
    levels = float(2 ** bits - 1)
    scale = ((mx - mn) / levels).clamp_min(1e-4).half()   # fp16-safe floor
    mn = mn.half()
    q = torch.round((g - mn.float()) / scale.float()).clamp_(0, levels).to(torch.uint8)
    if bits == 4:
        q = q.reshape(*q.shape[:-1], group // 2, 2)
        q = q[..., 0] | (q[..., 1] << 4)
    return QTensor(q.contiguous(), scale, mn, bits, axis, group, n, dtype)


def dequantize_groups(qt: QTensor) -> torch.Tensor:
    q = qt.data
    if qt.bits == 4:
        q = torch.stack([q & 0x0F, q >> 4], dim=-1).reshape(*q.shape[:-1], -1)
    x = q.float() * qt.scale.float() + qt.zero.float()
    x = x.reshape(*x.shape[:-2], -1)[..., :qt.orig_len]
    return x.movedim(-1, qt.axis).to(qt.dtype)


def _encode_fp8(x: torch.Tensor) -> FP8:
    if _FP8 is None:
        raise RuntimeError("this torch build has no float8_e4m3fn; use int8")
    scale = (x.float().abs().amax() / _FP8_MAX).clamp_min(1e-8)
    return FP8((x.float() / scale).to(_FP8), scale.reshape(()), x.dtype)


def _encode(x: torch.Tensor, precision: str, axis: int, group: int):
    if precision == "fp16":
        return Raw(x.contiguous())
    if precision == "fp8":
        return _encode_fp8(x)
    if precision == "int8":
        return quantize_groups(x, 8, axis, group)
    if precision == "int4":
        return quantize_groups(x, 4, axis, group)
    raise ValueError(precision)


# ---------------------------------------------------------------- layer / block codecs
@dataclass
class EncodedLayer:
    precision: str
    k: Optional[object]
    v: Optional[object]
    sink_k: Optional[torch.Tensor] = None
    sink_v: Optional[torch.Tensor] = None

    def nbytes(self) -> int:
        n = 0
        for p in (self.k, self.v):
            n += p.nbytes() if p is not None else 0
        for s in (self.sink_k, self.sink_v):
            n += s.numel() * s.element_size() if s is not None else 0
        return n

    def decode(self):
        ks, vs = [], []
        if self.sink_k is not None:
            ks.append(self.sink_k)
            vs.append(self.sink_v)
        if self.k is not None:
            ks.append(self.k.decode())
            vs.append(self.v.decode())
        return torch.cat(ks, dim=1), torch.cat(vs, dim=1)

    def apply(self, fn: Callable) -> "EncodedLayer":
        return EncodedLayer(self.precision,
                            self.k.apply(fn) if self.k is not None else None,
                            self.v.apply(fn) if self.v is not None else None,
                            fn(self.sink_k) if self.sink_k is not None else None,
                            fn(self.sink_v) if self.sink_v is not None else None)


def encode_layer(k: torch.Tensor, v: torch.Tensor, precision: str, key_group: int = 64,
                 value_group: int = 64, sink_tokens: int = 0) -> EncodedLayer:
    """k, v: [H, T, D]. `sink_tokens` leading tokens are kept in fp16 (pass 0 for non-first blocks)."""
    sink_k = sink_v = None
    if sink_tokens > 0 and precision != "fp16":
        s = min(sink_tokens, k.shape[1])
        sink_k, sink_v = k[:, :s].clone(), v[:, :s].clone()
        k, v = k[:, s:], v[:, s:]
        if k.shape[1] == 0:
            return EncodedLayer(precision, None, None, sink_k, sink_v)
    # keys: per-channel -> statistics over tokens (axis=-2); values: per-token -> over channels (axis=-1)
    return EncodedLayer(precision, _encode(k, precision, -2, key_group),
                        _encode(v, precision, -1, value_group), sink_k, sink_v)


@dataclass
class EncodedBlock:
    layers: List[EncodedLayer]
    n_tokens: int

    def nbytes(self) -> int:
        return sum(l.nbytes() for l in self.layers)

    def decode(self):
        ks, vs = zip(*(l.decode() for l in self.layers))
        return list(ks), list(vs)

    def apply(self, fn: Callable) -> "EncodedBlock":
        return EncodedBlock([l.apply(fn) for l in self.layers], self.n_tokens)

    def to(self, device) -> "EncodedBlock":
        return self.apply(lambda t: t.to(device))


def encode_block(ks: Sequence[torch.Tensor], vs: Sequence[torch.Tensor],
                 layer_precisions: Sequence[str], key_group: int = 64, value_group: int = 64,
                 sink_tokens: int = 0) -> EncodedBlock:
    assert len(ks) == len(vs) == len(layer_precisions)
    layers = [encode_layer(k, v, p, key_group, value_group, sink_tokens)
              for k, v, p in zip(ks, vs, layer_precisions)]
    return EncodedBlock(layers, ks[0].shape[1])


def rel_error(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return float((a - b).norm() / a.norm().clamp_min(1e-12))
