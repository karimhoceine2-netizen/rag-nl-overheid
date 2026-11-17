"""Train and evaluate the answerability gate (Task 2).

Run: uv run python -m evals.answerability

Steps, all with fixed seeds:
1. Load every reviewed question (evals/questions.jsonl + evals/gate_questions.jsonl).
2. Compute retrieval features once (dense + BM25 scores; free, no language model).
3. Split off a stratified test set that is only used once, at the end.
4. Compare on the training part with repeated stratified 5-fold cross-validation:
   majority class, a single threshold on the top dense score, and logistic regression.
5. Fit on the whole training part, evaluate once on the test set, save the model.

Writes evals/results/answerability.md / .json and models/answerability_gate.json.
"""

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from evals.questions import Question, load_question_set
from rag.corpus import load_chunks
from rag.gate import DEFAULT_GATE_PATH, FEATURES, AnswerabilityGate, extract_features
from rag.retrievers import BM25Retriever, DenseRetriever

RESULTS_DIR = Path("evals/results")
SEED = 42
TEST_SIZE = 0.25
N_SPLITS, N_REPEATS = 5, 5
N_BOOTSTRAP = 2000
TOP_K = 5


def load_dataset() -> list[Question]:
    questions, _ = load_question_set("v2")
    return [q for q in questions if q.status == "reviewed"]


def compute_features(questions: list[Question]) -> np.ndarray:
    chunks = load_chunks()
    dense, bm25 = DenseRetriever(chunks), BM25Retriever(chunks)
    rows = []
    for q in questions:
        values = extract_features(dense.search(q.question, TOP_K), bm25.search(q.question, TOP_K))
        rows.append([values[name] for name in FEATURES])
    return np.array(rows)


# --- the three models; each has fit(X, y) and predict(X) -> 1 = answerable --------------


class Majority:
    def fit(self, X: np.ndarray, y: np.ndarray) -> "Majority":
        self.label = int(round(y.mean()))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.label)


class DenseThreshold:
    """Answerable if the top dense score is at least t; t is chosen on the training data."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DenseThreshold":
        scores = X[:, FEATURES.index("dense_top1")]
        candidates = np.unique(scores)
        # Pick the cut-off with the best balanced accuracy, so the small class counts equally.
        self.t = max(candidates, key=lambda t: balanced_accuracy(y, (scores >= t).astype(int)))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (X[:, FEATURES.index("dense_top1")] >= self.t).astype(int)


class Logistic:
    def fit(self, X: np.ndarray, y: np.ndarray) -> "Logistic":
        self.scaler = StandardScaler().fit(X)
        # class_weight="balanced": unanswerable questions are the minority but matter as much.
        self.model = LogisticRegression(class_weight="balanced", random_state=SEED)
        self.model.fit(self.scaler.transform(X), y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self.scaler.transform(X))

    def to_gate(self) -> AnswerabilityGate:
        return AnswerabilityGate(
            features=FEATURES,
            mean=self.scaler.mean_.tolist(),
            scale=self.scaler.scale_.tolist(),
            coef=self.model.coef_[0].tolist(),
            intercept=float(self.model.intercept_[0]),
            threshold=0.5,
        )


MODELS = {"majority": Majority, "dense_threshold": DenseThreshold, "logistic": Logistic}


# --- metrics; the "positive" class is UNANSWERABLE, the thing the gate must catch --------


def balanced_accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    recalls = [np.mean(pred[y == c] == c) for c in (0, 1) if np.any(y == c)]
    return float(np.mean(recalls))


def scores(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    # pos_label=0: precision/recall/F1 for catching unanswerable questions.
    p, r, f, _ = precision_recall_fscore_support(
        y, pred, pos_label=0, average="binary", zero_division=0
    )
    blocked = np.mean(pred[y == 1] == 0) if np.any(y == 1) else 0.0
    return {
        "precision_unanswerable": float(p),
        "recall_unanswerable": float(r),
        "f1_unanswerable": float(f),
        "balanced_accuracy": balanced_accuracy(y, pred),
        "answerable_wrongly_blocked": float(blocked),
    }


def cross_validate(X: np.ndarray, y: np.ndarray) -> dict[str, dict[str, list[float]]]:
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    results: dict[str, dict[str, list[float]]] = {name: {} for name in MODELS}
    for train, val in cv.split(X, y):
        for name, cls in MODELS.items():
            pred = cls().fit(X[train], y[train]).predict(X[val])
            for metric, value in scores(y[val], pred).items():
                results[name].setdefault(metric, []).append(value)
    return results


def bootstrap_ci(y: np.ndarray, pred: np.ndarray, metric: str) -> tuple[float, float]:
    """95% interval of a test metric by resampling the test questions."""
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) == 2:  # both classes needed for the metrics to mean anything
            values.append(scores(y[idx], pred[idx])[metric])
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def git_commit() -> str:
    out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return out.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gate-out", type=Path, default=DEFAULT_GATE_PATH)
    args = parser.parse_args()

    questions = load_dataset()
    y = np.array([int(q.answerable) for q in questions])
    X = compute_features(questions)
    idx = np.arange(len(questions))
    train, test = train_test_split(idx, test_size=TEST_SIZE, stratify=y, random_state=SEED)

    cv = cross_validate(X[train], y[train])
    cv_summary = {
        name: {m: (float(np.mean(v)), float(np.std(v))) for m, v in metrics.items()}
        for name, metrics in cv.items()
    }

    test_results = {}
    fitted = {}
    for name, cls in MODELS.items():
        fitted[name] = cls().fit(X[train], y[train])
        pred = fitted[name].predict(X[test])
        test_results[name] = {
            "scores": scores(y[test], pred),
            "ci_f1_unanswerable": bootstrap_ci(y[test], pred, "f1_unanswerable"),
            # rows = true (unanswerable, answerable), columns = predicted, same order
            "confusion": confusion_matrix(y[test], pred, labels=[0, 1]).tolist(),
            "errors": [
                {
                    "id": questions[i].id,
                    "question": questions[i].question,
                    "answerable": bool(y[i]),
                    **{f: round(float(X[i, j]), 3) for j, f in enumerate(FEATURES)},
                }
                for i, p in zip(test, pred, strict=True)
                if p != y[i]
            ],
        }

    gate = fitted["logistic"].to_gate()
    gate.save(args.gate_out)

    meta = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "command": "uv run python -m evals.answerability",
        "n_questions": len(questions),
        "n_answerable": int(y.sum()),
        "n_unanswerable": int(len(y) - y.sum()),
        "n_train": len(train),
        "n_test": len(test),
        "seed": SEED,
        "features": FEATURES,
        "dense_threshold_on_train": float(fitted["dense_threshold"].t),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta, "cv": cv_summary, "test": test_results, "gate": gate.__dict__}
    (RESULTS_DIR / "answerability.json").write_text(json.dumps(payload, indent=2) + "\n")
    (RESULTS_DIR / "answerability.md").write_text(render_markdown(meta, cv_summary, test_results))
    print(render_markdown(meta, cv_summary, test_results))


def render_markdown(meta: dict, cv: dict, test: dict) -> str:
    lines = [
        "# Answerability gate",
        "",
        f"{meta['n_questions']} reviewed questions ({meta['n_answerable']} answerable, "
        f"{meta['n_unanswerable']} unanswerable); train {meta['n_train']}, "
        f"test {meta['n_test']} (stratified, seed {meta['seed']}). "
        "Positive class = unanswerable.",
        "",
        f"## Cross-validation on the training part ({N_REPEATS}x{N_SPLITS}-fold, mean ± sd)",
        "",
        "| model | precision | recall | F1 | balanced acc. | answerable blocked |",
        "|---|---|---|---|---|---|",
    ]
    keys = [
        "precision_unanswerable",
        "recall_unanswerable",
        "f1_unanswerable",
        "balanced_accuracy",
        "answerable_wrongly_blocked",
    ]
    for name, m in cv.items():
        cells = " | ".join(f"{m[k][0]:.2f} ± {m[k][1]:.2f}" for k in keys)
        lines.append(f"| {name} | {cells} |")
    lines += [
        "",
        "## Test set (used once)",
        "",
        "| model | precision | recall | F1 (95% CI) | balanced acc. | answerable blocked |"
        " confusion [[TU, FA], [FU, TA]] |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, r in test.items():
        s, (lo, hi) = r["scores"], r["ci_f1_unanswerable"]
        lines.append(
            f"| {name} | {s['precision_unanswerable']:.2f} | {s['recall_unanswerable']:.2f} | "
            f"{s['f1_unanswerable']:.2f} ({lo:.2f}-{hi:.2f}) | {s['balanced_accuracy']:.2f} | "
            f"{s['answerable_wrongly_blocked']:.2f} | {r['confusion']} |"
        )
    lines += [
        "",
        "Confusion rows = true class (unanswerable, answerable), columns = predicted. "
        "TU = unanswerable caught, FA = unanswerable let through, "
        "FU = answerable wrongly blocked, TA = answerable let through.",
        "",
        "## Test errors of the logistic model",
        "",
        "| id | true label | question | " + " | ".join(FEATURES) + " |",
        "|---|---|---|" + "---|" * len(FEATURES),
    ]
    for e in test["logistic"]["errors"]:
        label = "answerable" if e["answerable"] else "unanswerable"
        feats = " | ".join(str(e[f]) for f in FEATURES)
        lines.append(f"| {e['id']} | {label} | {e['question']} | {feats} |")
    lines += [
        "",
        f"Dense threshold chosen on train: {meta['dense_threshold_on_train']:.3f}.",
        f"Generated {meta['generated_at']} at commit `{meta['git_commit']}`.",
        f"Reproduce: `{meta['command']}`",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
