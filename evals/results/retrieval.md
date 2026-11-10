| stage | hit@1 | hit@3 | hit@5 | hit@10 | mrr@10 | latency_ms_mean |
|---|---|---|---|---|---|---|
| bm25 | 0.529 | 0.765 | 0.824 | 0.941 | 0.660 | 0.3 |

n = 17 answerable questions (0 reviewed, 17 draft); 130 documents, 210 chunks.
Generated 2026-11-10T23:58:21 at commit `3a0c80a`.
Reproduce: `uv run python -m evals.retrieval_eval --stages bm25`
