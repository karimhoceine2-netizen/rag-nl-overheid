"""Stage 1: dense retrieval with multilingual sentence embeddings.

Query and chunks are embedded separately (a "bi-encoder"), so chunk embeddings are computed
once at index time and a search is one query embedding plus a matrix-vector product.
Unlike BM25 this can match different words with the same meaning ("borg" ~ "waarborgsom").
"""

from typing import Protocol

import numpy as np

from rag.models import Chunk

DEFAULT_MODEL = "intfloat/multilingual-e5-small"
# CPU on purpose: GPU/MPS kernels can give slightly different floats (and so different tie
# orders) per machine, and CI has no GPU. At ~200 chunks CPU is fast enough.
DEFAULT_DEVICE = "cpu"


class Encoder(Protocol):
    model_name: str

    def encode_queries(self, texts: list[str]) -> np.ndarray: ...

    def encode_passages(self, texts: list[str]) -> np.ndarray: ...


class E5Encoder:
    """Wraps an E5 model from sentence-transformers.

    E5 models were trained with "query: " and "passage: " prefixes; the model card says to
    keep them, because queries and passages are embedded slightly differently.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = DEFAULT_DEVICE) -> None:
        from sentence_transformers import SentenceTransformer  # lazy: heavy import (torch)

        self.model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)

    def _encode(self, texts: list[str]) -> np.ndarray:
        # normalize_embeddings=True -> unit vectors, so dot product == cosine similarity.
        return self._model.encode(
            texts, normalize_embeddings=True, batch_size=32, convert_to_numpy=True
        )

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode([f"query: {t}" for t in texts])

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode([f"passage: {t}" for t in texts])


class DenseRetriever:
    name = "dense"

    def __init__(self, chunks: list[Chunk], encoder: Encoder | None = None) -> None:
        self._chunks = list(chunks)
        self._encoder = encoder or E5Encoder()
        self._matrix = self._encoder.encode_passages([c.search_text for c in self._chunks])

    @property
    def model_name(self) -> str:
        return self._encoder.model_name

    def search(self, query: str, k: int) -> list[Chunk]:
        query_vector = self._encoder.encode_queries([query])[0]
        scores = self._matrix @ query_vector  # cosine similarity with every chunk
        order = np.argsort(-scores, kind="stable")[:k]
        return [self._chunks[i].model_copy(update={"score": float(scores[i])}) for i in order]
