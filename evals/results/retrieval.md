| stage | hit@1 | hit@3 | hit@5 | hit@10 | mrr@10 | latency_ms_mean | latency_ms_p95 |
|---|---|---|---|---|---|---|---|
| bm25 | 0.529 | 0.765 | 0.824 | 0.941 | 0.660 | 0.5 | 0.8 |
| dense | 0.647 | 0.824 | 0.941 | 1.000 | 0.760 | 10.4 | 12.3 |
| hybrid | 0.471 | 0.824 | 0.941 | 1.000 | 0.676 | 10.8 | 11.5 |
| rerank | 0.588 | 0.882 | 0.941 | 1.000 | 0.735 | 1148.4 | 1195.4 |

n = 17 answerable questions (0 reviewed, 17 draft); 130 documents, 210 chunks.
Latency = milliseconds per query on CPU, including the stages a stage builds on (hybrid includes bm25 + dense). Machine: arm64 Darwin 25.6.0, Python 3.12.16.
Models: {'dense': 'intfloat/multilingual-e5-small', 'rerank': 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'}.
Generated 2026-11-11T00:07:01 at commit `05876a1-dirty`.
Reproduce: `uv run python -m evals.retrieval_eval`

### Rank of the first relevant chunk per question

| id | bm25 | dense | hybrid | rerank | notes |
|---|---|---|---|---|---|
| q001 | 1 | 1 | 1 | 1 | lexical |
| q002 | 3 | 5 | 3 | 5 | lexical |
| q003 | 4 | 2 | 2 | 3 | paraphrase |
| q004 | 2 | 1 | 2 | 2 | lexical |
| q005 | 2 | 2 | 2 | 2 | lexical |
| q006 | 1 | 1 | 1 | 1 | paraphrase: 'ov' vs 'reisproduct' |
| q007 | 3 | 1 | 2 | 1 | paraphrase: 'borg/huisbaas' vs 'waarborgsom/verhuurder' |
| q008 | 1 | 1 | 1 | 1 | lexical; distractor pages for 2024 and 2025; answer is in a table |
| q009 | 1 | 1 | 1 | 1 | lexical |
| q010 | 7 | 4 | 4 | 3 | paraphrase: 'opzegtermijn' not in question |
| q011 | 6 | 3 | 2 | 8 | lexical; distractor pages for 2024 and 2025 |
| q012 | 1 | 1 | 1 | 1 | lexical |
| q013 | - | 1 | 6 | 1 | paraphrase: needs age reasoning |
| q014 | 1 | 1 | 1 | 1 | lexical |
| q015 | 1 | 7 | 4 | 2 | paraphrase: 'vergeet' vs 'niet verzekerd' |
| q016 | 1 | 1 | 1 | 1 | lexical |
| q017 | 1 | 1 | 1 | 1 | lexical |
