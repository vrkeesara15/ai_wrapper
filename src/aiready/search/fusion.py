"""Rank fusion: merge the dense and keyword result lists into one ranking.

* **Reciprocal Rank Fusion** (default, k=60): ``sum_i w_i / (k + rank_i)``. Uses ranks only, so the very
  different score scales of cosine similarity and BM25 never need calibrating.
* **Weighted score fusion**: min-max normalise each list's scores to [0, 1] and take a weighted sum. Useful
  when one signal should dominate and the evaluation set says scores are informative.

Inputs are lists of ``(id, score)`` best-first; output is ``(id, fused_score)`` best-first.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

Ranked = Sequence[tuple[str, float]]


def rrf(
    lists: Sequence[Ranked], *, k: int = 60, weights: Sequence[float] | None = None
) -> list[tuple[str, float]]:
    weights = list(weights) if weights is not None else [1.0] * len(lists)
    fused: dict[str, float] = defaultdict(float)
    for lst, w in zip(lists, weights):
        for rank, (cid, _) in enumerate(lst, start=1):
            fused[cid] += w / (k + rank)
    return sorted(fused.items(), key=lambda x: (-x[1], x[0]))


def _minmax(lst: Ranked) -> dict[str, float]:
    if not lst:
        return {}
    vals = [s for _, s in lst]
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return {cid: 1.0 for cid, _ in lst}
    return {cid: (s - lo) / (hi - lo) for cid, s in lst}


def weighted(lists: Sequence[Ranked], *, weights: Sequence[float] | None = None) -> list[tuple[str, float]]:
    weights = list(weights) if weights is not None else [1.0 / max(1, len(lists))] * len(lists)
    fused: dict[str, float] = defaultdict(float)
    for lst, w in zip(lists, weights):
        for cid, s in _minmax(lst).items():
            fused[cid] += w * s
    return sorted(fused.items(), key=lambda x: (-x[1], x[0]))


def fuse(
    lists: Sequence[Ranked], *, method: str = "rrf", k: int = 60, weights: Sequence[float] | None = None
) -> list[tuple[str, float]]:
    if method == "rrf":
        return rrf(lists, k=k, weights=weights)
    if method == "weighted":
        return weighted(lists, weights=weights)
    raise ValueError(f"unknown fusion method '{method}' (rrf | weighted)")
