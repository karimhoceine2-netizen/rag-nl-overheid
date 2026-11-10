"""Retrieval metrics.

All metrics are computed from one number per question: the 1-based rank of the first
relevant chunk, or None if no relevant chunk was retrieved.
"""

from collections.abc import Callable

from rag.models import Chunk


def first_relevant_rank(results: list[Chunk], is_relevant: Callable[[Chunk], bool]) -> int | None:
    for rank, chunk in enumerate(results, start=1):
        if is_relevant(chunk):
            return rank
    return None


def hit_at_k(ranks: list[int | None], k: int) -> float:
    """Fraction of questions with a relevant chunk in the top k."""
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def mrr(ranks: list[int | None], cutoff: int = 10) -> float:
    """Mean Reciprocal Rank: rewards putting the right chunk first, not just somewhere."""
    if not ranks:
        return 0.0
    return sum(1 / r for r in ranks if r is not None and r <= cutoff) / len(ranks)
