| stage | hit@1 | hit@3 | hit@5 | hit@10 | mrr@10 | latency_ms_mean | latency_ms_p95 |
|---|---|---|---|---|---|---|---|
| bm25 | 0.743 | 0.892 | 0.959 | 0.959 | 0.825 | 0.5 | 0.7 |
| dense | 0.770 | 0.959 | 0.986 | 0.986 | 0.864 | 10.4 | 12.1 |
| hybrid | 0.770 | 0.946 | 0.946 | 0.973 | 0.858 | 10.9 | 12.7 |
| rerank | 0.838 | 0.946 | 0.959 | 0.986 | 0.897 | 1128.0 | 1160.4 |

n = 74 answerable questions (74 reviewed, 0 draft); 130 documents, 210 chunks.
Latency = milliseconds per query on CPU, including the stages a stage builds on (hybrid includes bm25 + dense). Machine: arm64 Darwin 25.6.0, Python 3.12.16.
Models: {'dense': 'intfloat/multilingual-e5-small', 'rerank': 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'}.
Generated 2026-11-17T11:09:43 at commit `27d95bf`.
Reproduce: `uv run python -m evals.retrieval_eval`

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
| ga001 | 1 | 1 | 1 | 1 | lexical |
| ga002 | 1 | 1 | 1 | 1 | paraphrase |
| ga003 | 1 | 1 | 1 | 1 | lexical; ook in andere jaarpagina's |
| ga004 | 4 | 1 | 2 | 1 | paraphrase |
| ga005 | 1 | 1 | 1 | 1 | datumredenering |
| ga006 | 3 | 1 | 2 | 1 | lexical |
| ga007 | 1 | 1 | 1 | 1 | uitzondering op 'huisarts telt niet mee' |
| ga008 | 1 | 1 | 1 | 1 | lexical |
| ga009 | 1 | 1 | 1 | 1 | lexical |
| ga010 | 1 | 1 | 1 | 1 | lexical |
| ga011 | 1 | 1 | 1 | 1 | paraphrase: Spanje = Europa |
| ga012 | 1 | 2 | 1 | 1 | lexical |
| ga013 | 1 | 1 | 1 | 1 | lexical |
| ga014 | 1 | 1 | 1 | 1 | paraphrase |
| ga015 | 1 | 1 | 1 | 1 | lexical |
| ga016 | 1 | 1 | 1 | 1 | paraphrase: gevangenis vs detentie |
| ga017 | 1 | 1 | 1 | 1 | redenering: 12 jaar = minderjarig |
| ga018 | 1 | 1 | 1 | 1 | lexical |
| ga019 | 1 | 1 | 1 | 1 | paraphrase |
| ga020 | 2 | 1 | 1 | 1 | voorwaarde |
| ga021 | 1 | 1 | 1 | 2 | paraphrase: huis te klein vs gezin te groot |
| ga022 | 1 | 1 | 1 | 1 | paraphrase |
| ga023 | 1 | 1 | 1 | 1 | paraphrase |
| ga024 | 1 | 1 | 1 | 1 | redenering: 150 <= 186 |
| ga025 | 1 | 1 | 1 | 1 | lexical |
| ga026 | 1 | 1 | 1 | 1 | lexical |
| ga027 | 1 | 1 | 1 | 1 | lexical |
| ga028 | 1 | 2 | 1 | 1 | lexical |
| ga029 | 1 | 1 | 1 | 1 | uitzondering |
| ga030 | 2 | 1 | 1 | 1 | lexical |
| ga031 | 1 | 1 | 1 | 1 | lexical |
