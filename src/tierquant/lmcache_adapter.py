"""Adapter sketch: use the tierquant codec as LMCache's per-tier serialisation step.

STATUS: UNVERIFIED against any specific LMCache release. LMCache's internal APIs (storage backends,
serialiser/deserialiser "serde" modules, KV chunk layout) have changed between versions. Treat this
file as the integration contract: pin an LMCache version, read its storage-backend / serde
interface, and wire `TierQuantSerde` into the backends of your L2 (DRAM) and L3 (NVMe) tiers.

Assumed chunk layout (CHECK IT): [num_layers, 2(K,V), chunk_tokens, num_kv_heads, head_dim].
If your version differs, change `chunk_to_layers` / `layers_to_chunk` only.
"""
from __future__ import annotations

import io
from typing import List, Tuple

import torch

from .policy import PrecisionMap
from .quant import EncodedBlock, encode_block


def chunk_to_layers(chunk: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    L = chunk.shape[0]
    ks = [chunk[l, 0].permute(1, 0, 2) for l in range(L)]   # [T,H,D] -> [H,T,D]
    vs = [chunk[l, 1].permute(1, 0, 2) for l in range(L)]
    return ks, vs


def layers_to_chunk(ks, vs) -> torch.Tensor:
    return torch.stack([torch.stack([k.permute(1, 0, 2), v.permute(1, 0, 2)]) for k, v in zip(ks, vs)])


class TierQuantSerde:
    """serialize(chunk)->bytes / deserialize(bytes)->chunk for one tier of the hierarchy."""

    def __init__(self, pmap: PrecisionMap, tier: str):
        self.pmap, self.tier = pmap, tier

    def serialize(self, chunk: torch.Tensor, start_token: int = 0) -> bytes:
        ks, vs = chunk_to_layers(chunk)
        enc = encode_block(ks, vs, self.pmap.layer_precisions(self.tier, len(ks)),
                           self.pmap.key_group, self.pmap.value_group,
                           sink_tokens=self.pmap.sink_tokens if start_token == 0 else 0)
        buf = io.BytesIO()
        torch.save(enc.to("cpu"), buf)
        return buf.getvalue()

    def deserialize(self, data: bytes, device: str = "cuda") -> torch.Tensor:
        enc: EncodedBlock = torch.load(io.BytesIO(data), weights_only=False)
        ks, vs = enc.to(device).decode()
        return layers_to_chunk(ks, vs)
