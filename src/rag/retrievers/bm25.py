"""Stage 0: BM25 keyword retrieval (the baseline every other stage must beat)."""

import numpy as np
from rank_bm25 import BM25Okapi

from rag.models import Chunk
from rag.text import tokenize


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        # k1 = term-frequency saturation, b = length normalisation; 1.5/0.75 are the
        # textbook defaults and we keep them fixed so the baseline isn't tuned on the eval set.
        self._chunks = list(chunks)
        self._bm25 = BM25Okapi([tokenize(c.search_text) for c in self._chunks], k1=k1, b=b)

    def search(self, query: str, k: int) -> list[Chunk]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        order = np.argsort(-scores, kind="stable")[:k]  # stable = deterministic tie order
        return [
            self._chunks[i].model_copy(update={"score": float(scores[i])})
            for i in order
            if scores[i] > 0  # zero score = no query term in common; not a real result
        ]
