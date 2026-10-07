"""Small text utilities shared by the mock backends, keyword search and scorers."""

from __future__ import annotations

import hashlib
import math
import re

import numpy as np

_WORD = re.compile(r"[A-Za-z0-9_]+(?:[-'][A-Za-z0-9_]+)*")
STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were will with "
    "what which who whom how when where why do does did can could should would i you we they he she".split()
)


def tokenize(text: str, *, drop_stopwords: bool = True) -> list[str]:
    toks = [t.lower() for t in _WORD.findall(text or "")]
    return [t for t in toks if t not in STOPWORDS] if drop_stopwords else toks


def sha256(text: str | bytes) -> str:
    return hashlib.sha256(text.encode() if isinstance(text, str) else text).hexdigest()


def hash_embed(text: str, dim: int = 384) -> list[float]:
    """Deterministic bag-of-words + bigram feature hashing embedding, L2-normalised.

    Not a semantic model - it exists so the full pipeline runs offline (tests, demos, air-gapped smoke
    tests). Texts sharing words get high cosine similarity, which is enough to exercise every stage.
    """
    vec = np.zeros(dim, dtype=np.float32)
    toks = tokenize(text)
    feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
    for f in feats:
        h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "little")
        vec[h % dim] += 1.0 if (h >> 63) & 1 == 0 else -1.0
    norm = float(np.linalg.norm(vec))
    if norm == 0:
        vec[0] = 1.0
        norm = 1.0
    return (vec / norm).tolist()


def cosine(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    va, vb = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
    if na == 0 or nb == 0:
        return 0.0
    return float(va @ vb / (na * nb))


def overlap_score(query: str, passage: str) -> float:
    """Lexical relevance in [0, 1] used by the mock reranker and grounding checks."""
    q, p = set(tokenize(query)), set(tokenize(passage))
    if not q or not p:
        return 0.0
    inter = len(q & p)
    return inter / math.sqrt(len(q) * len(p))


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])", (text or "").strip())
    return [p.strip() for p in parts if p.strip()]
