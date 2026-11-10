from rag.models import Chunk
from rag.retrievers import BM25Retriever


def chunk(i: int, title: str, text: str) -> Chunk:
    url = f"https://x.nl/{i}"
    return Chunk(chunk_id=f"d{i}#0", doc_id=f"d{i}", url=url, title=title, text=text)


CHUNKS = [
    chunk(1, "Zorgtoeslag", "U krijgt zorgtoeslag als uw inkomen niet te hoog is."),
    chunk(2, "Huurtoeslag", "Huurtoeslag is een bijdrage in de huur van uw woning."),
    chunk(3, "Studiefinanciering", "Studenten in het hoger onderwijs kunnen lenen bij DUO."),
]


def test_bm25_ranks_matching_chunk_first():
    results = BM25Retriever(CHUNKS).search("Wie krijgt zorgtoeslag?", k=3)
    assert results[0].doc_id == "d1"
    assert results[0].score is not None and results[0].score > 0


def test_bm25_uses_stemming():
    # "student" (singular) in the query should still match "Studenten" (plural) in the text.
    results = BM25Retriever(CHUNKS).search("student lenen", k=1)
    assert results[0].doc_id == "d3"


def test_bm25_respects_k_and_drops_zero_scores():
    results = BM25Retriever(CHUNKS).search("huur woning", k=5)
    assert [c.doc_id for c in results] == ["d2"]


def test_bm25_empty_query_returns_nothing():
    assert BM25Retriever(CHUNKS).search("de het een", k=3) == []


def test_bm25_does_not_mutate_corpus_scores():
    BM25Retriever(CHUNKS).search("zorgtoeslag", k=1)
    assert all(c.score is None for c in CHUNKS)
