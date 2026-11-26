"""Block identity and per-block quality metadata (torch-free)."""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Dict, List, Sequence


def prefix_block_ids(token_ids: Sequence[int], block_tokens: int) -> List[str]:
    """Chained prefix hash per full block (same idea as vLLM / LMCache prefix keys).

    id_i = sha256(id_{i-1} || tokens of block i). Two requests share block i iff they
    share the whole prefix up to and including block i.
    """
    ids: List[str] = []
    prev = b""
    n_full = len(token_ids) // block_tokens
    for i in range(n_full):
        chunk = token_ids[i * block_tokens:(i + 1) * block_tokens]
        h = hashlib.sha256(prev + b"|" + ",".join(map(str, chunk)).encode()).hexdigest()
        ids.append(h)
        prev = h.encode()
    return ids


@dataclass
class BlockMeta:
    block_id: str
    n_tokens: int
    start_token: int = 0
    n_lossy: int = 0                 # number of lossy conversions applied so far
    est_error: float = 0.0           # accumulated error proxy (sum of per-conversion proxies)
    precision: str = "fp16"          # nominal precision of the stored copy
    hits: int = 0
    created: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.hits += 1
        self.last_access = time.time()

    def as_dict(self) -> Dict:
        return dict(self.__dict__)
