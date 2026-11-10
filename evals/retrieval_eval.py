"""Retrieval evaluation: hit@k and MRR for every retrieval stage, in one table.

Run: uv run python -m evals.retrieval_eval                 (all stages)
     uv run python -m evals.retrieval_eval --stages bm25
     uv run python -m evals.retrieval_eval --reviewed-only  (skip draft questions)

Writes evals/results/retrieval.json (machine-readable, used by CI and the README),
evals/results/retrieval.md (the table) and evals/results/retrieval_details.jsonl
(per-question ranks, for error analysis).
"""

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from evals.metrics import first_relevant_rank, hit_at_k, mrr
from evals.questions import QUESTIONS_PATH, Question, check_labels, is_relevant, load_questions
from rag.corpus import load_chunks, load_documents
from rag.models import Chunk
from rag.retrievers import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    RerankRetriever,
    Retriever,
)

K_VALUES = (1, 3, 5, 10)
TOP_K = max(K_VALUES)
RESULTS_DIR = Path("evals/results")

# Order = order of the rows in the results table. Each stage builds on the previous ones:
# hybrid = RRF(bm25, dense), rerank = cross-encoder over the hybrid top 20.
STAGE_NAMES = ("bm25", "dense", "hybrid", "rerank")


def build_retrievers(names: list[str], chunks: list[Chunk]) -> dict[str, Retriever]:
    """Build the requested stages, sharing sub-retrievers (the dense index is built once)."""
    built: dict[str, Retriever] = {}

    def get(name: str) -> Retriever:
        if name not in built:
            if name == "bm25":
                built[name] = BM25Retriever(chunks)
            elif name == "dense":
                built[name] = DenseRetriever(chunks)
            elif name == "hybrid":
                built[name] = HybridRetriever([get("bm25"), get("dense")])
            elif name == "rerank":
                built[name] = RerankRetriever(get("hybrid"))
        return built[name]

    return {name: get(name) for name in names}


def model_names(retrievers: dict[str, Retriever]) -> dict[str, str]:
    return {name: r.model_name for name, r in retrievers.items() if hasattr(r, "model_name")}


def git_commit() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def evaluate_stage(
    retriever: Retriever, questions: list[Question]
) -> tuple[dict[str, float], list[dict]]:
    ranks: list[int | None] = []
    latencies_ms: list[float] = []
    details: list[dict] = []
    # One untimed warm-up query: the first call of a torch model is much slower (lazy init)
    # and would otherwise inflate the mean latency of the neural stages.
    retriever.search("warm-up", TOP_K)
    for q in questions:
        start = time.perf_counter()
        results = retriever.search(q.question, TOP_K)
        latencies_ms.append((time.perf_counter() - start) * 1000)
        rank = first_relevant_rank(results, lambda c, q=q: is_relevant(c, q))
        ranks.append(rank)
        details.append(
            {
                "stage": retriever.name,
                "id": q.id,
                "rank": rank,
                "top3": [c.chunk_id for c in results[:3]],
            }
        )

    metrics = {f"hit@{k}": hit_at_k(ranks, k) for k in K_VALUES}
    metrics["mrr@10"] = mrr(ranks, cutoff=10)
    metrics["latency_ms_mean"] = statistics.fmean(latencies_ms)
    metrics["latency_ms_p95"] = (
        statistics.quantiles(latencies_ms, n=20)[18] if len(latencies_ms) >= 2 else latencies_ms[0]
    )
    return metrics, details


def to_markdown(results: dict, details: list[dict], questions: list[Question]) -> str:
    cols = [f"hit@{k}" for k in K_VALUES] + ["mrr@10", "latency_ms_mean", "latency_ms_p95"]
    lines = [
        "| stage | " + " | ".join(cols) + " |",
        "|---" * (len(cols) + 1) + "|",
    ]
    for stage, m in results["stages"].items():
        cells = [f"{m[c]:.1f}" if c.startswith("latency") else f"{m[c]:.3f}" for c in cols]
        lines.append(f"| {stage} | " + " | ".join(cells) + " |")
    meta = results["meta"]
    lines += [
        "",
        f"n = {meta['n_questions_scored']} answerable questions "
        f"({meta['n_reviewed']} reviewed, {meta['n_draft']} draft); "
        f"{meta['n_docs']} documents, {meta['n_chunks']} chunks.",
        "Latency = milliseconds per query on CPU, including the stages a stage builds on "
        f"(hybrid includes bm25 + dense). Machine: {meta['platform']}.",
        f"Models: {meta['models'] or 'none'}.",
        f"Generated {meta['generated_at']} at commit `{meta['git_commit']}`.",
        f"Reproduce: `{meta['command']}`",
    ]

    # Per-question ranks: the most useful view for error analysis ("-" = not in top 10).
    stages = list(results["stages"])
    rank = {(d["id"], d["stage"]): d["rank"] for d in details}
    lines += [
        "",
        "### Rank of the first relevant chunk per question",
        "",
        "| id | " + " | ".join(stages) + " | notes |",
        "|---" * (len(stages) + 2) + "|",
    ]
    for q in questions:
        cells = [str(rank[(q.id, s)] or "-") for s in stages]
        lines.append(f"| {q.id} | " + " | ".join(cells) + f" | {q.notes} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", default="all", help="comma-separated, or 'all'")
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    parser.add_argument("--reviewed-only", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    stage_names = list(STAGE_NAMES) if args.stages == "all" else args.stages.split(",")
    unknown = [s for s in stage_names if s not in STAGE_NAMES]
    if unknown:
        parser.error(f"unknown stage(s) {unknown}; choose from {list(STAGE_NAMES)}")

    chunks = load_chunks()
    all_questions = load_questions(args.questions)
    problems = check_labels(all_questions, chunks)
    if problems:
        print("Label problems; fix evals/questions.jsonl first:", *problems, sep="\n  ")
        return 1

    # Unanswerable questions have no expected source, so they don't count for retrieval.
    scored = [q for q in all_questions if q.answerable]
    if args.reviewed_only:
        scored = [q for q in scored if q.status == "reviewed"]
    if not scored:
        print("No questions to score.")
        return 1

    retrievers = build_retrievers(stage_names, chunks)
    results: dict = {"stages": {}, "meta": {}}
    all_details: list[dict] = []
    for name, retriever in retrievers.items():
        metrics, details = evaluate_stage(retriever, scored)
        results["stages"][name] = metrics
        all_details += details

    results["meta"] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "command": "uv run python -m evals.retrieval_eval " + " ".join(sys.argv[1:]),
        "n_docs": len(load_documents()),
        "n_chunks": len(chunks),
        "n_questions_scored": len(scored),
        "n_unanswerable_excluded": sum(1 for q in all_questions if not q.answerable),
        "n_reviewed": sum(1 for q in scored if q.status == "reviewed"),
        "n_draft": sum(1 for q in scored if q.status == "draft"),
        "models": model_names(retrievers),
        "platform": f"{platform.machine()} {platform.system()} {platform.release()}, "
        f"Python {platform.python_version()}",
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "retrieval.json").write_text(json.dumps(results, indent=2) + "\n")
    table = to_markdown(results, all_details, scored)
    (args.out_dir / "retrieval.md").write_text(table)
    detail_lines = [json.dumps(d, ensure_ascii=False) for d in all_details]
    (args.out_dir / "retrieval_details.jsonl").write_text("\n".join(detail_lines) + "\n")
    print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
