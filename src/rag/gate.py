"""Answerability gate: predicts from retrieval scores whether the corpus can answer a question.

If it says "no", the API can abstain without paying for a language-model call.
The trained model is a logistic regression stored as plain JSON (weights, scaling and
threshold), so loading it needs no scikit-learn and no pickle (which can run code on load).
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

from rag.models import Chunk

DEFAULT_GATE_PATH = Path("models/answerability_gate.json")

# Order matters: the stored weights follow this order.
FEATURES = ["dense_top1", "dense_gap", "bm25_top1", "bm25_gap"]


def _top_and_gap(hits: list[Chunk]) -> tuple[float, float]:
    scores = [h.score or 0.0 for h in hits]
    top1 = scores[0] if scores else 0.0
    top2 = scores[1] if len(scores) > 1 else 0.0
    return top1, top1 - top2


def extract_features(dense_hits: list[Chunk], bm25_hits: list[Chunk]) -> dict[str, float]:
    """Turn the top results of both retrievers into the gate's input features.

    A clear winner (high top score, big gap to number two) suggests the corpus has a page
    about exactly this question; flat, low scores suggest it does not.
    """
    dense_top1, dense_gap = _top_and_gap(dense_hits)
    bm25_top1, bm25_gap = _top_and_gap(bm25_hits)
    return {
        "dense_top1": dense_top1,
        "dense_gap": dense_gap,
        "bm25_top1": bm25_top1,
        "bm25_gap": bm25_gap,
    }


@dataclass(frozen=True)
class AnswerabilityGate:
    features: list[str]
    mean: list[float]
    scale: list[float]
    coef: list[float]
    intercept: float
    threshold: float  # minimum probability of "answerable" to let a question through

    def probability(self, values: dict[str, float]) -> float:
        """Probability that the corpus can answer the question."""
        z = self.intercept
        for name, mean, scale, weight in zip(
            self.features, self.mean, self.scale, self.coef, strict=True
        ):
            z += weight * (values[name] - mean) / scale
        return 1.0 / (1.0 + math.exp(-z))

    def is_answerable(self, values: dict[str, float]) -> bool:
        return self.probability(values) >= self.threshold

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.__dict__, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path = DEFAULT_GATE_PATH) -> "AnswerabilityGate":
        data = json.loads(path.read_text(encoding="utf-8"))
        gate = cls(**data)
        if gate.features != FEATURES:
            raise ValueError(f"{path}: features {gate.features} do not match {FEATURES}")
        return gate
