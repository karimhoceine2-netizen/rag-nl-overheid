"""Stage 3: rerank the top candidates of another retriever with a cross-encoder.

A bi-encoder (stage 1) embeds query and chunk separately. A cross-encoder reads the query and
the chunk *together* in one model pass, so it can weigh details like years, amounts and
negations, but it costs one model call per (query, chunk) pair. Hence the usual pattern: a
cheap retriever for recall, then the cross-encoder only on its top N candidates.

Limitation: the model reads at most 512 tokens. Our chunks are at most 350 words, which is
usually under that, but the end of a long chunk can be cut off.
"""

from typing import Protocol

from rag.models import Chunk
from rag.retrievers.base import Retriever

# Multilingual MiniLM trained on mMARCO (MS MARCO machine-translated into 14 languages,
# including Dutch). Small enough (~118M parameters) to run on CPU in CI.
DEFAULT_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
DEFAULT_REVISION = "1427fd652930e4ba29e8149678df786c240d8825"


class Scorer(Protocol):
    model_name: str

    def score(self, query: str, texts: list[str]) -> list[float]: ...


class CrossEncoderScorer:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu") -> None:
        from sentence_transformers import CrossEncoder  # lazy: heavy import (torch)

        self.model_name = model_name
        self._model = CrossEncoder(
            model_name,
            device=device,
            revision=DEFAULT_REVISION if model_name == DEFAULT_MODEL else None,
        )

    def score(self, query: str, texts: list[str]) -> list[float]:
        pairs = [(query, text) for text in texts]
        return [float(s) for s in self._model.predict(pairs, batch_size=32)]


class RerankRetriever:
    name = "rerank"

    def __init__(self, base: Retriever, scorer: Scorer | None = None, candidates: int = 20) -> None:
        self._base = base
        self._scorer = scorer or CrossEncoderScorer()
        self.candidates = candidates

    @property
    def model_name(self) -> str:
        return self._scorer.model_name

    def search(self, query: str, k: int) -> list[Chunk]:
        candidates = self._base.search(query, self.candidates)
        if not candidates:
            return []
        scores = self._scorer.score(query, [c.search_text for c in candidates])
        order = sorted(range(len(candidates)), key=lambda i: -scores[i])  # stable on ties
        return [candidates[i].model_copy(update={"score": scores[i]}) for i in order[:k]]
