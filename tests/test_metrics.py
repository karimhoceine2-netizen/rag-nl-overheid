import pytest

from evals.metrics import first_relevant_rank, hit_at_k, mrr
from rag.models import Chunk


def test_first_relevant_rank_is_one_based():
    chunks = [
        Chunk(chunk_id=f"c{i}", doc_id="d", url="u", title="t", text=str(i)) for i in range(3)
    ]
    assert first_relevant_rank(chunks, lambda c: c.text == "1") == 2
    assert first_relevant_rank(chunks, lambda c: False) is None


def test_hit_at_k_and_mrr():
    ranks = [1, None, 3]
    assert hit_at_k(ranks, 1) == pytest.approx(1 / 3)
    assert hit_at_k(ranks, 3) == pytest.approx(2 / 3)
    assert mrr(ranks) == pytest.approx((1 + 0 + 1 / 3) / 3)


def test_mrr_cutoff_and_empty():
    assert mrr([11], cutoff=10) == 0.0
    assert mrr([]) == 0.0
    assert hit_at_k([], 5) == 0.0
