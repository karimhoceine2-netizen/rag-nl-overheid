"""Print every evaluation result as one table, and check stored answer results in CI.

Run: uv run python -m evals.report            (print the tables; no API calls)
     uv run python -m evals.report --check    (CI: fail on stale or regressed answer results)

Reads evals/results/retrieval.json and evals/results/answer_eval_<stage>.json. The paid answer
runs are not repeated in CI; --check instead verifies that the stored results still belong
to the current question set, re-computes citation correctness from the stored answers, and
compares the scores with evals/answer_thresholds.json.
"""

import argparse
import json
import sys
from pathlib import Path

from evals.answer_eval import citation_correct
from evals.questions import DEFAULT_SET, load_question_set
from rag.corpus import load_chunks
from rag.generate import Answer

RESULTS_DIR = Path("evals/results")
ANSWER_STAGES = ("dense", "rerank")
ANSWER_THRESHOLDS = Path("evals/answer_thresholds.json")


def load_answer_results() -> dict[str, dict]:
    return {
        stage: json.loads((RESULTS_DIR / f"answer_eval_{stage}.json").read_text())
        for stage in ANSWER_STAGES
    }


def pct(rate: dict) -> str:
    return f"{rate['count']}/{rate['total']} ({rate['rate']:.0%})"


def render(retrieval: dict, answers: dict[str, dict]) -> str:
    meta = retrieval["meta"]
    lines = [
        f"Question set {meta['question_set']} ({meta['question_set_sha256']}), "
        f"{meta['n_questions_scored']} answerable questions for retrieval.",
        "",
        "Retrieval | hit@1 | hit@5 | mrr@10 | ms/query",
        "---|---|---|---|---",
    ]
    for stage, m in retrieval["stages"].items():
        lines.append(
            f"{stage} | {m['hit@1']:.3f} | {m['hit@5']:.3f} | {m['mrr@10']:.3f} | "
            f"{m['latency_ms_mean']:.0f}"
        )
    lines += [
        "",
        "Answers | grounded (judge) | citation correct | correct abstention | coverage | "
        "invalid | s/answer | cost USD (gen + judge)",
        "---|---|---|---|---|---|---|---",
    ]
    for stage, r in answers.items():
        m = r["metrics"]
        lines.append(
            f"{stage} | {pct(m['groundedness'])} | {pct(m['citation_correctness'])} | "
            f"{pct(m['correct_abstention'])} | {pct(m['answerable_coverage'])} | "
            f"{m['invalid_answers']} | {m['latency_ms_mean'] / 1000:.1f} | "
            f"{m['generation_cost_usd']:.2f} + {m['judge_cost_usd']:.2f}"
        )
    gen = {r["meta"]["generation_model"] for r in answers.values()}
    judge = {r["meta"]["judge_model"] for r in answers.values()}
    lines += [
        "",
        f"Generator {', '.join(sorted(gen))}; judge {', '.join(sorted(judge))}. "
        "Judge verdicts are spot-checked by hand (evals/results/manual_review_v2.md).",
    ]
    return "\n".join(lines)


def check(answers: dict[str, dict]) -> list[str]:
    """Problems that should fail CI. Free: no API calls."""
    questions, info = load_question_set(DEFAULT_SET)
    by_id = {q.id: q for q in questions}
    chunks_by_id = {c.chunk_id: c for c in load_chunks()}
    thresholds = json.loads(ANSWER_THRESHOLDS.read_text())
    problems = []
    for stage, result in answers.items():
        meta, metrics = result["meta"], result["metrics"]
        if meta.get("question_set_sha256") != info["question_set_sha256"]:
            problems.append(f"{stage}: results are for another question set; re-run the eval")
            continue
        # Re-compute citation correctness from the stored answers: catches labels or chunks
        # that changed after the paid run.
        recomputed = [
            citation_correct(Answer.model_validate(row["response"]), by_id[row["id"]], chunks_by_id)
            for row in result["details"]
        ]
        correct = sum(r is True for r in recomputed)
        if correct != metrics["citation_correctness"]["count"]:
            problems.append(f"{stage}: citation correctness no longer matches the stored answers")
        for name, minimum in thresholds[stage].items():
            if metrics[name]["rate"] < minimum:
                problems.append(f"{stage} {name}: {metrics[name]['rate']:.3f} < {minimum}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    retrieval = json.loads((RESULTS_DIR / "retrieval.json").read_text())
    answers = load_answer_results()
    print(render(retrieval, answers))
    if args.check:
        problems = check(answers)
        if problems:
            print("\nAnswer checks failed:", *problems, sep="\n  ", file=sys.stderr)
            return 1
        print("\nStored answer results match the question set and meet their thresholds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
