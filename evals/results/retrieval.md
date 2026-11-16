| stage | hit@1 | hit@3 | hit@5 | hit@10 | mrr@10 | latency_ms_mean | latency_ms_p95 |
|---|---|---|---|---|---|---|---|
| bm25 | 0.651 | 0.837 | 0.930 | 0.930 | 0.756 | 0.6 | 1.3 |
| dense | 0.651 | 0.930 | 0.977 | 0.977 | 0.790 | 22.4 | 32.2 |
| hybrid | 0.651 | 0.907 | 0.907 | 0.953 | 0.779 | 24.0 | 36.2 |
| rerank | 0.744 | 0.907 | 0.930 | 0.977 | 0.834 | 1645.5 | 2234.7 |

n = 43 answerable questions (43 reviewed, 0 draft); 130 documents, 210 chunks.
Latency = milliseconds per query on CPU, including the stages a stage builds on (hybrid includes bm25 + dense). Machine: arm64 Darwin 25.6.0, Python 3.12.16.
Models: {'dense': 'intfloat/multilingual-e5-small', 'rerank': 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'}.
Generated 2026-11-16T17:56:46 at commit `0d73ee6`.
Reproduce: `uv run python -m evals.retrieval_eval --reviewed-only --out-dir logs/retrieval_final`

### Rank of the first relevant chunk per question

| id | bm25 | dense | hybrid | rerank | notes |
|---|---|---|---|---|---|
| q001 | 3 | 2 | 2 | 1 | lexical |
| q002 | 3 | 5 | 3 | 5 | lexical |
| q003 | 4 | 2 | 2 | 3 | paraphrase |
| q004 | 2 | 1 | 2 | 2 | lexical |
| q005 | 2 | 2 | 2 | 2 | lexical |
| q006 | 1 | 1 | 1 | 1 | duur reisproduct bij een vierjarige hbo-opleiding |
| q007 | - | 4 | 7 | 2 | paraphrase: 'borg/huisbaas' vs 'waarborgsom/verhuurder' |
| q008 | 1 | 1 | 1 | 1 | lexical; distractor pages for 2024 and 2025; answer is in a table |
| q009 | 1 | 1 | 1 | 1 | lexical |
| q010 | 1 | 3 | 1 | 1 | gebruikelijke opzegtermijn bij vertrek op initiatief van de huurder |
| q011 | 4 | 2 | 2 | 7 | tabel; jaar, datum, woningtype, huurprijs en inkomensschijf |
| q012 | 1 | 1 | 1 | 1 | lexical |
| q013 | - | 1 | 6 | 1 | paraphrase: needs age reasoning |
| q014 | 1 | 1 | 1 | 1 | lexical |
| q015 | 2 | 1 | 1 | 1 | termijn na de CAK-brief voor een verzekeringsplichtige |
| q016 | 1 | 1 | 1 | 1 | lexical |
| q017 | 1 | 1 | 1 | 1 | lexical |
| q018 | 1 | 2 | 1 | 1 | leeftijdsgrens |
| q019 | 1 | 3 | 2 | 6 | aanvraagadvies, geen uiterste termijn |
| q020 | 1 | 2 | 1 | 2 | leeftijdsgrens; ook elders in het corpus beschreven |
| q021 | 1 | 1 | 1 | 1 | voorwaarde en termijn |
| q022 | 1 | 1 | 1 | 2 | parafrase van peiljaar bij inkomensdaling |
| q023 | 1 | 1 | 1 | 1 | parafrase |
| q024 | 1 | 1 | 1 | 1 | parafrase, onderscheid met omgekeerde overstap |
| q025 | 1 | 1 | 1 | 1 | aanvraagtermijn |
| q026 | 1 | 1 | 1 | 1 | wijzigingstermijn, andere bedoeling dan aanvraag |
| q027 | 1 | 1 | 1 | 1 | jaar, afkorting en tabel |
| q028 | 1 | 1 | 1 | 1 | tabel, inkomensschijf en kolom |
| q029 | 1 | 1 | 1 | 1 | parafrase met ontkenning |
| q030 | 1 | 1 | 1 | 1 | specifieke uitzondering |
| q031 | 3 | 2 | 2 | 1 | onderscheid met basisverzekering |
| q032 | - | 1 | - | 1 | voorwaarde en parafrase |
| q033 | 2 | 1 | 2 | 1 | termijn voor afsluiten, andere handeling dan q009 |
| q034 | 1 | 1 | 1 | 1 | parafrase en organisatie |
| q035 | 1 | 1 | 1 | 1 | parafrase met ontkenning |
| q036 | 1 | 1 | 1 | 1 | parafrase |
| q037 | 4 | 3 | 3 | 2 | jaar en woningtype |
| q038 | 1 | 1 | 1 | 1 | jaar en woningtype |
| q039 | 2 | 2 | 2 | 1 | onderscheid 14 dagen zonder en 30 dagen met verrekening |
| q040 | 1 | 1 | 1 | 1 | parafrase en voorwaarde |
| q041 | 1 | 2 | 1 | 1 | voorwaarde |
| q042 | 4 | - | - | - | parafrase en voorbeeld |
| q043 | 1 | 1 | 1 | 1 | voorbeeld valt onder de regel |
