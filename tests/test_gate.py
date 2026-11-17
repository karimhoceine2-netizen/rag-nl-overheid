from pathlib import Path

import numpy as np
import pytest

from evals.answerability import DenseThreshold, Logistic, Majority, scores
from rag.gate import FEATURES, AnswerabilityGate, extract_features
from rag.models import Chunk


def _hit(score: float) -> Chunk:
    return Chunk(chunk_id="d#0", doc_id="d", url="u", title="t", text="x", score=score)


def _toy_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, len(FEATURES)))
    y = (X[:, 0] + 0.5 * rng.normal(size=40) > 0).astype(int)
    return X, y


def test_extract_features_top_and_gap() -> None:
    values = extract_features([_hit(0.9), _hit(0.7)], [])
    assert values["dense_top1"] == pytest.approx(0.9)
    assert values["dense_gap"] == pytest.approx(0.2)
    # BM25 can return nothing when no query word occurs in the corpus.
    assert values["bm25_top1"] == 0.0 and values["bm25_gap"] == 0.0


def test_gate_matches_scikit_learn() -> None:
    X, y = _toy_data()
    model = Logistic().fit(X, y)
    gate = model.to_gate()
    expected = model.model.predict_proba(model.scaler.transform(X))[:, 1]
    actual = [gate.probability(dict(zip(FEATURES, row, strict=True))) for row in X]
    assert np.allclose(actual, expected)


def test_gate_save_and_load(tmp_path: Path) -> None:
    X, y = _toy_data()
    gate = Logistic().fit(X, y).to_gate()
    path = tmp_path / "gate.json"
    gate.save(path)
    assert AnswerabilityGate.load(path) == gate


def test_load_rejects_other_features(tmp_path: Path) -> None:
    X, y = _toy_data()
    gate = Logistic().fit(X, y).to_gate()
    path = tmp_path / "gate.json"
    AnswerabilityGate(**{**gate.__dict__, "features": list(reversed(FEATURES))}).save(path)
    with pytest.raises(ValueError, match="do not match"):
        AnswerabilityGate.load(path)


def test_baselines() -> None:
    X = np.zeros((6, len(FEATURES)))
    X[:, 0] = [0.1, 0.2, 0.3, 0.8, 0.9, 0.95]
    y = np.array([0, 0, 0, 1, 1, 1])
    assert list(Majority().fit(X, np.array([1, 1, 0, 1, 0, 1])).predict(X)) == [1] * 6
    threshold = DenseThreshold().fit(X, y)
    assert list(threshold.predict(X)) == list(y)


def test_scores_count_unanswerable_as_positive() -> None:
    y = np.array([0, 0, 1, 1])
    pred = np.array([0, 1, 0, 1])  # catches 1 of 2 unanswerable, blocks 1 of 2 answerable
    s = scores(y, pred)
    assert s["recall_unanswerable"] == 0.5
    assert s["precision_unanswerable"] == 0.5
    assert s["answerable_wrongly_blocked"] == 0.5
