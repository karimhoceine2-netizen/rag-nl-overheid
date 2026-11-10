"""Dense retriever tests with a tiny fake encoder, so no model download is needed."""

import numpy as np

from rag.models import Chunk
from rag.retrievers.dense import DenseRetriever

VOCAB = ["borg", "waarborgsom", "huur", "zorg", "premie"]
SYNONYMS = {"waarborgsom": "borg"}  # the fake "understands" one synonym, like a real model


class FakeEncoder:
    model_name = "fake"

    def __init__(self) -> None:
        self.queries: list[str] = []

    def _vec(self, text: str) -> np.ndarray:
        words = [SYNONYMS.get(w, w) for w in text.lower().replace("?", "").split()]
        v = np.array([float(words.count(term)) for term in VOCAB])
        return v / (np.linalg.norm(v) or 1.0)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        self.queries += texts
        return np.stack([self._vec(t) for t in texts])

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts])


def chunk(i: int, text: str) -> Chunk:
    return Chunk(chunk_id=f"d{i}#0", doc_id=f"d{i}", url=f"u{i}", title="", text=text)


CHUNKS = [chunk(1, "zorg premie"), chunk(2, "waarborgsom huur"), chunk(3, "huur")]


def test_dense_matches_synonym_and_ranks_by_cosine():
    results = DenseRetriever(CHUNKS, encoder=FakeEncoder()).search("Hoeveel borg?", k=3)
    assert results[0].doc_id == "d2"
    assert results[0].score is not None
    assert [r.score for r in results] == sorted([r.score for r in results], reverse=True)


def test_dense_respects_k_and_passes_raw_query_to_encoder():
    encoder = FakeEncoder()
    assert len(DenseRetriever(CHUNKS, encoder=encoder).search("huur", k=2)) == 2
    assert encoder.queries == ["huur"]  # prefixes are the encoder's job, not the retriever's
