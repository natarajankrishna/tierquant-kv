"""Synthetic request traces (torch-free). Deterministic given a seed.

Request.prefix_tokens : tokens that *could* be served from cache.
Request.suffix_tokens : fresh tokens (the new question) that are always computed.
Request.store_tokens  : tokens of the prompt that are cached after the request completes
                        (RAG: only the shared document; multi-turn: whole prompt + response).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass
class Request:
    t: float
    prefix_key: str
    prefix_tokens: int
    suffix_tokens: int
    store_tokens: int


def zipf_rag(n_requests: int = 5000, n_docs: int = 2000, alpha: float = 0.9,
             doc_tokens: Sequence[int] = (4096, 8192, 16384, 32768),
             suffix_tokens: int = 128, qps: float = 4.0, seed: int = 0) -> List[Request]:
    """RAG / long-document QA: requests pick one of n_docs documents with Zipf(alpha) popularity."""
    rng = np.random.default_rng(seed)
    ranks = np.arange(1, n_docs + 1, dtype=np.float64)
    p = ranks ** (-alpha)
    p /= p.sum()
    doc_len = rng.choice(np.asarray(doc_tokens), size=n_docs)
    docs = rng.choice(n_docs, size=n_requests, p=p)
    t = np.cumsum(rng.exponential(1.0 / qps, size=n_requests))
    return [Request(float(t[i]), f"doc{docs[i]}", int(doc_len[docs[i]]), suffix_tokens, int(doc_len[docs[i]]))
            for i in range(n_requests)]


def multi_turn(n_sessions: int = 400, turns: int = 8, first_tokens: int = 2048,
               turn_tokens: int = 512, response_tokens: int = 256, qps: float = 4.0,
               think_time_s: float = 30.0, seed: int = 0) -> List[Request]:
    """Chat sessions: each turn's prefix is the entire previous conversation."""
    rng = np.random.default_rng(seed)
    starts = np.cumsum(rng.exponential(1.0 / (qps / turns), size=n_sessions))
    reqs: List[Request] = []
    for s in range(n_sessions):
        t = float(starts[s])
        ctx = 0
        for k in range(turns):
            new = first_tokens if k == 0 else turn_tokens
            store = ctx + new + response_tokens
            reqs.append(Request(t, f"sess{s}", ctx, new, store))
            ctx = store
            t += float(rng.exponential(think_time_s))
    reqs.sort(key=lambda r: r.t)
    return reqs


def make(name: str, **kw) -> List[Request]:
    if name == "zipf_rag":
        return zipf_rag(**kw)
    if name == "multi_turn":
        return multi_turn(**kw)
    raise ValueError(name)
