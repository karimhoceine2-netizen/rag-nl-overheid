from rag.models import Chunk
from rag.retrievers.rerank import RerankRetriever


def c(cid: str, text: str) -> Chunk:
    return Chunk(chunk_id=cid, doc_id=cid, url=cid, title="T", text=text)


class FixedRetriever:
    name = "fixed"

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks, self.calls = chunks, []

    def search(self, query: str, k: int) -> list[Chunk]:
        self.calls.append(k)
        return self.chunks[:k]


class LengthScorer:
    """Fake cross-encoder: longer text = more relevant."""

    model_name = "fake"

    def __init__(self) -> None:
        self.seen: list[str] = []

    def score(self, query: str, texts: list[str]) -> list[float]:
        self.seen += texts
        return [float(len(t)) for t in texts]


def test_rerank_reorders_candidates_by_scorer():
    base = FixedRetriever([c("short", "a"), c("long", "aaaa"), c("mid", "aa")])
    results = RerankRetriever(base, scorer=LengthScorer(), candidates=20).search("q", k=2)
    assert [r.chunk_id for r in results] == ["long", "mid"]
    assert results[0].score == float(len("T\naaaa"))  # scored on title + text
    assert base.calls == [20]


def test_rerank_only_scores_top_candidates():
    base = FixedRetriever([c(str(i), "x") for i in range(30)])
    scorer = LengthScorer()
    RerankRetriever(base, scorer=scorer, candidates=5).search("q", k=3)
    assert len(scorer.seen) == 5


def test_rerank_empty_base_returns_empty():
    assert RerankRetriever(FixedRetriever([]), scorer=LengthScorer()).search("q", k=3) == []
