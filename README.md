# Nederlandse overheidsvragen met RAG

Een portfolioproject dat Nederlandse vragen beantwoordt met tekstfragmenten uit Rijksoverheid.nl.
De evaluatie vergelijkt vier manieren om relevante tekst op te halen. De API gebruikt standaard
dense retrieval en geeft per bewering bronverwijzingen terug. Bij onvoldoende informatie geeft
het systeem aan dat het antwoord onbekend is.

## Retrievalresultaten

| Methode | Hit@1 | Hit@3 | Hit@5 | MRR@10 | Gemiddeld ms per vraag |
|---|---:|---:|---:|---:|---:|
| BM25 | 0,651 | 0,837 | 0,930 | 0,756 | 0,6 |
| Dense | 0,651 | 0,930 | 0,977 | 0,790 | 22,4 |
| Hybrid (RRF) | 0,651 | 0,907 | 0,907 | 0,779 | 24,0 |
| Hybrid + reranker | 0,744 | 0,907 | 0,930 | 0,834 | 1645,5 |

Deze run gebruikte 43 beantwoordbare vragen die de projecteigenaar had goedgekeurd,
130 documenten en 210 fragmenten. De latencies zijn gemeten op een Apple Silicon CPU.
De [volledige run](evals/results/retrieval.md) bevat de reproduceeropdracht, het commit-ID,
de modellen en de rang per vraag.

Hit@k meet het aandeel vragen waarvoor de top k een relevant fragment bevat. MRR@10 beloont
een hogere positie van het eerste relevante fragment. Een fragment telt als relevant als
zowel de bron-URL als het gelabelde citaat overeenkomen.

De reranker behaalde de hoogste Hit@1 en MRR@10. Dense heeft de hoogste Hit@3 en Hit@5
en blijft de standaard voor de API vanwege de lagere latency dan de reranker. De scores
meten of het specifiek gelabelde fragment is opgehaald. Een andere pagina kan dezelfde
informatie bevatten. De resultaten gelden voor deze vraagset en deze momentopname.

```bash
uv run --locked python -m evals.retrieval_eval --reviewed-only --out-dir logs/retrieval
uv run --locked python -m evals.retrieval_eval \
  --reviewed-only --thresholds evals/thresholds.json --out-dir logs/ci-retrieval
```

GitHub Actions draait bij iedere push en pull request de tests, de retrievalvergelijking en
een Docker-starttest. De retrievalcontrole gebruikt voor iedere methode de ondergrenzen voor
Hit@5 en MRR@10 uit [thresholds.json](evals/thresholds.json). CI gebruikt de goedgekeurde
vraagset en bewaart het rapport als workflow-artifact.

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

| Metriek | Uitkomst |
|---|---:|
| Door de LLM-jury als onderbouwd beoordeeld | 39 van 42 |
| Terecht afgezien van een antwoord | 7 van 8 |
| Beantwoordbare vragen met een antwoord | 41 van 43 |
| Ongeldige antwoorden | 1 |
| Ongeldige juryoordelen | 0 |

Dense retrieval leverde de bronfragmenten. De LLM-jury beoordeelde de onderbouwing van de
gegeven antwoorden aan de hand van hun citaten. Dekking meet het aandeel beantwoordbare
vragen waarop een antwoord volgde. Open het [rapport per vraag](evals/results/answer_eval.json)
voor de antwoordtekst, bronverwijzingen en juryoordelen. De [handmatige controle](evals/results/manual_review.md)
legt voorbeelden en fouten naast de opgeslagen bronfragmenten. Juryoordelen vragen om
handmatige steekproeven.

```bash
uv run --env-file .env python -m evals.answer_eval --reviewed-only --out logs/answer_eval_final.json
```

Deze opdracht gebruikt de betaalde Anthropic API.

## Prompt-injectietests

Synthetische documenten proberen het antwoord te sturen via opdrachten in de broninhoud.
De offline tests controleren dat die tekst in de brongegevens blijft en dat de generator
uitvoer met verzonnen bron-ID's, links of een ongeldig formaat afwijst. De handmatige
injectie-evaluatie gebruikt de betaalde Anthropic API om het antwoord en de geciteerde bron
te controleren. De [live injectie-evaluatie](evals/results/injection_eval.md) slaagde voor
alle vier de synthetische gevallen.

```bash
uv run --locked pytest -q
uv run --locked --env-file .env python -m evals.injection_eval
```

## Data

Dit is een onafhankelijk studentenproject op basis van een opgeslagen momentopname.
De Rijksoverheid staat los van het project. De [dataherkomst en hergebruikvoorwaarden](DATA.md)
beschrijven de gebruikte pagina's, de selectie op CC0 en hoe je de data opnieuw verzamelt.
