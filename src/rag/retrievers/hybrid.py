"""Stage 2: hybrid retrieval — fuse BM25 and dense rankings with Reciprocal Rank Fusion (RRF).

RRF score(chunk) = sum over retrievers of 1 / (k_rrf + rank_in_that_retriever).

Why RRF instead of adding scores: BM25 scores are unbounded and cosine similarities lie in
[-1, 1], so adding them needs a normalisation and a weight to tune. RRF only looks at ranks,
needs no tuning, and is hard to beat in practice. k_rrf = 60 comes from the original paper
(Cormack, Clarke & Büttcher, 2009); a larger k_rrf flattens the difference between ranks.
"""

from rag.models import Chunk
from rag.retrievers.base import Retriever


def reciprocal_rank_fusion(rankings: list[list[Chunk]], k_rrf: int = 60) -> list[Chunk]:
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1 / (k_rrf + rank)
            chunks.setdefault(chunk.chunk_id, chunk)
    # sorted() is stable: ties keep first-seen order (i.e. the first retriever wins ties).
    ordered = sorted(scores, key=lambda chunk_id: -scores[chunk_id])
    return [chunks[cid].model_copy(update={"score": scores[cid]}) for cid in ordered]


class HybridRetriever:
    name = "hybrid"

    def __init__(self, retrievers: list[Retriever], candidates: int = 50, k_rrf: int = 60) -> None:
        # Each retriever contributes its top `candidates`; a chunk that is missing from one
        # list simply gets no contribution from that retriever.
        self._retrievers = retrievers
        self.candidates = candidates
        self.k_rrf = k_rrf

    def search(self, query: str, k: int) -> list[Chunk]:
        rankings = [r.search(query, self.candidates) for r in self._retrievers]
        return reciprocal_rank_fusion(rankings, self.k_rrf)[:k]
