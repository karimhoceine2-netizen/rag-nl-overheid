# Nederlandse overheidsvragen met RAG

Een portfolioproject dat Nederlandse vragen beantwoordt met tekstfragmenten uit Rijksoverheid.nl.
De evaluatie vergelijkt vier manieren om relevante tekst op te halen. De API gebruikt standaard
dense retrieval en geeft per bewering bronverwijzingen terug. Bij onvoldoende informatie geeft
het systeem aan dat het antwoord onbekend is.

## Retrievalresultaten

| Methode | Hit@1 | Hit@3 | Hit@5 | MRR@10 | Gemiddeld ms per vraag |
|---|---:|---:|---:|---:|---:|
| BM25 | 0,529 | 0,765 | 0,824 | 0,660 | 0,5 |
| Dense | 0,647 | 0,824 | 0,941 | 0,760 | 10,4 |
| Hybrid (RRF) | 0,471 | 0,824 | 0,941 | 0,676 | 10,9 |
| Hybrid + reranker | 0,588 | 0,882 | 0,941 | 0,735 | 1131,3 |

Deze run gebruikte 17 beantwoordbare conceptvragen, 130 documenten en 210 fragmenten.
De latencies zijn gemeten op een Apple Silicon CPU. De [volledige run](evals/results/retrieval.md)
bevat de reproduceeropdracht, het commit-ID, de modellen en de rang per vraag.

Hit@k meet het aandeel vragen waarvoor de top k een relevant fragment bevat. MRR@10 beloont
een hogere positie van het eerste relevante fragment. Een fragment telt als relevant als
zowel de bron-URL als het gelabelde citaat overeenkomen.

Dense had op deze set de hoogste Hit@1 en MRR@10. De reranker had de hoogste Hit@3 en
vergde meer rekentijd per vraag. De vraagset is klein en de labels wachten op review.

```bash
uv run --locked python -m evals.retrieval_eval --out-dir logs/retrieval
uv run --locked python -m evals.retrieval_eval \
  --thresholds evals/thresholds.json --out-dir logs/ci-retrieval
```

GitHub Actions draait bij iedere push en pull request de tests, de retrievalvergelijking en
een Docker-starttest. De retrievalcontrole gebruikt voor iedere methode de ondergrenzen voor
Hit@5 en MRR@10 uit [thresholds.json](evals/thresholds.json). Deze liggen iets onder de
gemeten waarden in de tabel.

## Lokaal starten

Installeer [uv](https://docs.astral.sh/uv/getting-started/installation/) en voer dit uit vanuit
de projectmap. Bij de eerste start downloadt het systeem het embeddingmodel. Vul
`ANTHROPIC_API_KEY` in je lokale `.env` in.

```bash
uv sync --locked
cp -n .env.example .env
uv run --locked --env-file .env uvicorn rag.api:app
```

Open [de interactieve API-documentatie](http://127.0.0.1:8000/docs) en kies
**POST /ask → Try it out**, of gebruik curl in een tweede terminal.

```bash
curl -fsS http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"Valt een bezoek aan de huisarts onder mijn eigen risico?"}'
```

Het veld `answer` bevat het antwoord. Bij `citations` staan de bijbehorende bronlinks.
De querylog bevat fragment-ID's, latency, tokengebruik en een kostenraming. Het standaardpad
is `logs/queries.jsonl`.

### Docker

Vul eerst `.env` in zoals hierboven. Als Uvicorn nog lokaal draait, stop het proces met
**Ctrl+C** voordat je dezelfde poort gebruikt.

```bash
docker build -t rag-nl-overheid .
docker run --rm --env-file .env -p 127.0.0.1:8000:8000 \
  -v rag-nl-logs:/app/logs rag-nl-overheid
```

## Antwoorden beoordelen

De handmatige evaluatie meet of de geciteerde fragmenten alle beweringen ondersteunen,
hoe vaak het systeem terecht afziet van een antwoord en welk aandeel van de beantwoordbare
vragen een antwoord krijgt. Bekijk het rapport per vraag en controleer jury-oordelen met
steekproeven in de bronnen.

```bash
uv run --locked --env-file .env python -m evals.answer_eval
```

Deze opdracht gebruikt de betaalde Anthropic API en schrijft naar `logs/answer_eval.json`.
Voeg na de review van [de vragen](evals/questions.jsonl) `--reviewed-only` toe om alleen
goedgekeurde vragen te meten.

## Prompt-injectietests

Synthetische documenten proberen het antwoord te sturen via opdrachten in de broninhoud.
De offline tests controleren dat die tekst in de brongegevens blijft en dat de generator
uitvoer met verzonnen bron-ID's, links of een ongeldig formaat afwijst. De handmatige
injectie-evaluatie gebruikt de betaalde Anthropic API om het antwoord en de geciteerde bron
te controleren.

```bash
uv run --locked pytest -q
uv run --locked --env-file .env python -m evals.injection_eval
```

## Data

Dit is een onafhankelijk studentenproject op basis van een opgeslagen momentopname.
De Rijksoverheid staat los van het project. De [dataherkomst en hergebruikvoorwaarden](DATA.md)
beschrijven de gebruikte pagina's, de selectie op CC0 en hoe je de data opnieuw verzamelt.
