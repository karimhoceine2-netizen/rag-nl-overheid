"""The precomputed index must give the same results, and be ignored when it is stale."""

import numpy as np
import pytest

from rag.dense_index import load_dense_retriever, load_matrix, save_index
from rag.retrievers.dense import DenseRetriever
from tests.test_dense import CHUNKS, FakeEncoder, chunk


class CountingEncoder(FakeEncoder):
    def __init__(self) -> None:
        super().__init__()
        self.passage_calls = 0

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        self.passage_calls += 1
        return super().encode_passages(texts)


def test_saved_index_is_reused_and_gives_same_ranking(tmp_path, monkeypatch):
    path = tmp_path / "index.npz"
    monkeypatch.setenv("RAG_DENSE_INDEX_PATH", str(path))
    fresh = DenseRetriever(CHUNKS, encoder=FakeEncoder())
    save_index(path, fresh, CHUNKS, "fake@1")

    encoder = CountingEncoder()
    cached = load_dense_retriever(CHUNKS, encoder=encoder, model_id="fake@1")
    assert encoder.passage_calls == 0  # no re-embedding at startup
    assert np.array_equal(cached.matrix, fresh.matrix)
    query = "Hoeveel borg?"
    assert [c.chunk_id for c in cached.search(query, 3)] == [
        c.chunk_id for c in fresh.search(query, 3)
    ]


@pytest.mark.parametrize(
    ("chunks", "model_id"),
    [
        ([*CHUNKS[:2], chunk(3, "huur aangepast")], "fake@1"),  # edited chunk text
        (CHUNKS, "fake@2"),  # other model
    ],
)
def test_stale_index_is_ignored(tmp_path, chunks, model_id):
    path = tmp_path / "index.npz"
    save_index(path, DenseRetriever(CHUNKS, encoder=FakeEncoder()), CHUNKS, "fake@1")
    assert load_matrix(path, chunks, model_id) is None


def test_missing_index_falls_back_to_embedding(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_DENSE_INDEX_PATH", str(tmp_path / "absent.npz"))
    encoder = CountingEncoder()
    load_dense_retriever(CHUNKS, encoder=encoder, model_id="fake@1")
    assert encoder.passage_calls == 1


def test_matrix_with_wrong_row_count_is_rejected():
    with pytest.raises(ValueError):
        DenseRetriever(CHUNKS, encoder=FakeEncoder(), matrix=np.zeros((2, 5)))
