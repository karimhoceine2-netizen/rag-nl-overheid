import pytest

from rag.models import Chunk
from rag.retrievers.hybrid import HybridRetriever, reciprocal_rank_fusion


def c(cid: str) -> Chunk:
    return Chunk(chunk_id=cid, doc_id=cid, url=cid, title="", text=cid)


class FixedRetriever:
    def __init__(self, name: str, ids: list[str]) -> None:
        self.name, self.ids, self.calls = name, ids, []

    def search(self, query: str, k: int) -> list[Chunk]:
        self.calls.append(k)
        return [c(i) for i in self.ids[:k]]


def test_rrf_scores_follow_the_formula():
    fused = reciprocal_rank_fusion([[c("a"), c("b")], [c("b"), c("c")]], k_rrf=60)
    scores = {x.chunk_id: x.score for x in fused}
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["c"] == pytest.approx(1 / 62)
    assert [x.chunk_id for x in fused] == ["b", "a", "c"]  # in both lists -> wins


def test_rrf_tie_goes_to_first_retriever():
    fused = reciprocal_rank_fusion([[c("a")], [c("z")]])
    assert [x.chunk_id for x in fused] == ["a", "z"]


def test_hybrid_asks_each_retriever_for_candidates_and_cuts_to_k():
    left = FixedRetriever("l", ["a", "b", "c"])
    right = FixedRetriever("r", ["c", "d"])
    results = HybridRetriever([left, right], candidates=50).search("q", k=2)
    assert left.calls == [50] and right.calls == [50]
    assert [x.chunk_id for x in results] == ["c", "a"]
