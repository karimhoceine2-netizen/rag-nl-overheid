# CLAUDE.md — conventions for this repo

Dutch-language RAG over Dutch government information, built as a portfolio project.
**The evaluation is the product**; the chat API is secondary.

## Working rules
- Work in phases and STOP after each one for review:
  1. skeleton + ingest + BM25 + first eval
  2. dense + hybrid + reranker + results table
  3. generation with citations + "ik weet het niet" + FastAPI + logging
  4. CI + Docker + injection tests + README
- Small commits, conventional-commit messages (`feat(retrieval): ...`, `test: ...`, `docs: ...`).
- Write tests alongside the code. `uv run ruff check . && uv run pytest` must pass before every commit.
- Keep the code readable for a first-year AI student: short functions, type hints, a one-line
  comment explaining *why* wherever a choice is non-obvious.
- Do not push to any remote unless Karim asks.

## Never invent numbers
- Every figure in the README must come from a real eval run, committed under `evals/results/`,
  with the exact command that reproduces it.
- CI thresholds (`evals/thresholds.json`) are set from a real run, just below the measured value.
- LLM-judge scores need manual spot checks; always say so where they are reported.

## Interfaces (keep them stable — a Rust index and a Go gateway will plug in later)
- Retriever: `search(query: str, k: int) -> list[Chunk]` (see `src/rag/retrievers/base.py`).
- LLM: everything goes through `LLMClient` (`src/rag/llm/base.py`); providers are swappable.
  Tests and CI use the fake client, never a real API.

## Evaluation data
- `evals/questions.jsonl`: `{id, question, expected_url, evidence, answerable, status, notes}`.
- A retrieved chunk is a hit if it comes from `expected_url` AND contains the `evidence` quote
  (robust to re-chunking).
- Questions written by Claude are `status: "draft"`; only Karim marks them `"reviewed"`.

## Data
- Only add documents whose reuse terms are checked and recorded in `DATA.md`.
- Every document keeps its `url`, `title`, `retrieved_at` and `license`.

## Secrets
- No secrets in the repo. Use `.env` (git-ignored); document variables in `.env.example`.

## Commands
- `uv run python -m rag.ingest` — fetch + clean source pages
- `uv run python -m evals.retrieval_eval` — retrieval metrics table
- `uv run pytest` / `uv run ruff check .` / `uv run ruff format .`
