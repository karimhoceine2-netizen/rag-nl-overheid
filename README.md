# Nederlandse overheidsvragen met RAG

Een portfolioproject dat Nederlandse vragen beantwoordt met tekstfragmenten uit Rijksoverheid.nl.
De evaluatie vergelijkt vier manieren om relevante tekst op te halen. De API gebruikt standaard
dense retrieval en geeft per bewering bronverwijzingen terug. Bij onvoldoende informatie geeft
het systeem aan dat het antwoord onbekend is.

## Evaluatie in één tabel

```bash
uv run python -m evals.report
```

Deze opdracht print alle resultaten zonder API-kosten. Alles is gemeten op vragenset **v2**
(112 nagekeken vragen: 74 beantwoordbaar, 38 onbeantwoordbaar), 130 documenten en
210 fragmenten. De vragensets hebben een versie: elk resultaat bewaart de versie en een hash
van de vragenbestanden ([evals/questions.py](evals/questions.py)).

## Retrievalresultaten

| Methode | Hit@1 | Hit@3 | Hit@5 | MRR@10 | Gemiddeld ms per vraag |
|---|---:|---:|---:|---:|---:|
| BM25 | 0,743 | 0,892 | 0,959 | 0,825 | 0,5 |
| Dense | 0,770 | 0,959 | 0,986 | 0,864 | 10,4 |
| Hybrid (RRF) | 0,770 | 0,946 | 0,946 | 0,858 | 10,9 |
| Hybrid + reranker | 0,838 | 0,946 | 0,959 | 0,897 | 1128,0 |

Hit@k meet het aandeel vragen waarvoor de top k een relevant fragment bevat. MRR@10 beloont
een hogere positie van het eerste relevante fragment. Een fragment telt als relevant als
zowel de bron-URL als het gelabelde citaat overeenkomen. De latencies zijn gemeten op een
Apple Silicon CPU. De [volledige run](evals/results/retrieval.md) bevat de reproduceeropdracht,
het commit-ID, de modellen en de rang per vraag.

De reranker haalt de hoogste Hit@1 en MRR@10, dense de hoogste Hit@3 en Hit@5. Dense blijft
de standaard voor de API: de reranker kost ruim een seconde per vraag. Op vragenset v1
(43 beantwoordbare vragen) was de volgorde hetzelfde, met lagere scores (dense Hit@5 0,977,
reranker MRR@10 0,834). De 61 nieuwe vragen van v2 zijn geschreven met kennis van het corpus
en zijn daardoor waarschijnlijk makkelijker dan echte gebruikersvragen.

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

`POST /search` geeft alleen de opgehaalde fragmenten terug, zonder taalmodel. Dat endpoint
kost niets en werkt ook zonder API-key.

```bash
curl -fsS http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"question":"Valt een bezoek aan de huisarts onder mijn eigen risico?","k":3}'
```

### Docker

Met [Docker](https://docs.docker.com/get-docker/) start één opdracht de API vanuit een verse
clone. Als Uvicorn nog lokaal draait, stop het proces met **Ctrl+C** voordat je dezelfde poort
gebruikt.

```bash
docker compose up --build
```

De image bevat het embeddingmodel al, zodat een koude start niets downloadt. Zonder `.env`
werken `/health` en `/search`, en geeft `/ask` een 503 met uitleg. Voor antwoorden kopieer je
`.env.example` naar `.env` en vul je `ANTHROPIC_API_KEY` in. De container draait als
niet-rootgebruiker en heeft een healthcheck op `/health`. CI start de container op dezelfde
manier en meldt de imagegrootte en het piekgeheugen in de samenvatting van de workflow.
De embeddings van het corpus worden bij het bouwen berekend en meegeleverd, zodat de container
ze bij het opstarten niet opnieuw hoeft te berekenen.
In [run 43966524373](https://github.com/karimhoceine2-netizen/rag-nl-overheid/actions/runs/43966524373)
was de image 1903 MiB en het piekgeheugen 781 MiB na het opstarten en één zoekvraag.
De Cloud Run-dienst krijgt 2 GiB geheugen, zodat er ruim marge is.

### Cloud Run

De API draait op [Google Cloud Run](https://rag-nl-overheid-440512847068.europe-west4.run.app/docs).
Probeer `POST /search` (gratis) of `POST /ask` (maximaal 50 vragen per dag) via
**Try it out**. Zonder verkeer schaalt de dienst naar nul. Het eerste verzoek na een rustige
periode wacht daarom op een koude start: op 17 november 2026 duurde dat 45 seconden, daarna
duurde een zoekvraag 0,1 seconde. Krijg je toch een 503, probeer het dan zo opnieuw.

[docs/DEPLOY.md](docs/DEPLOY.md) beschrijft de eenmalige setup, het deployscript, de
kostenbegrenzing en het opruimen. De deploy zelf is één opdracht.

```bash
GCP_PROJECT=jouw-project-id ./scripts/deploy_cloud_run.sh
```

## Antwoorden beoordelen

Twee configuraties op dezelfde 112 vragen, met Claude Haiku als generator en Claude Sonnet
als jury:

| Metriek | Dense (top 5) | Hybrid + reranker (top 5) |
|---|---:|---:|
| Door de LLM-jury als onderbouwd beoordeeld | 72 van 74 | 72 van 75 |
| Citatie correct (een geciteerde bron bevat het gelabelde citaat) | 71 van 72 | 69 van 69 |
| Terecht geweigerd bij onbeantwoordbare vragen | 35 van 38 | 31 van 38 |
| Beantwoordbare vragen met een antwoord | 72 van 74 | 69 van 74 |
| Ongeldige antwoorden | 3 | 4 |
| Seconden per antwoord | 2,3 | 3,5 |
| Kosten generator + jury (USD) | 0,07 + 0,50 | 0,08 + 0,54 |

Dense weigert vaker terecht, beantwoordt meer vragen en is sneller; de reranker scoort iets
beter op citaties. Dense blijft daarom de standaard. Waarom de reranker vaker antwoordt op
onbeantwoordbare vragen is niet gemeten.

**Grenzen van de jury.** Een LLM-jury kan te streng of te mild zijn, en meet alleen of
beweringen in de geciteerde bron staan, niet of de vraag beantwoord is. Daarom is er ook
citatie-correctheid: die is gratis te berekenen en vond een antwoord dat de jury onderbouwd
vond maar de vraag niet beantwoordde. In de [handmatige controle](evals/results/manual_review_v2.md)
klopten 10 van de 11 gecontroleerde juryoordelen; het ene verschil was een te strenge jury.
De weigerscore is een ondergrens: alle 8 beantwoorde onbeantwoordbare vragen zeiden eerst dat
de bronnen het gevraagde niet noemen, maar kregen de status `answered`.

```bash
uv run --env-file .env python -m evals.answer_eval --stage dense --out evals/results/answer_eval_dense.json
uv run --env-file .env python -m evals.answer_eval --stage rerank --out evals/results/answer_eval_rerank.json
```

Deze opdrachten gebruiken de betaalde Anthropic API (samen ongeveer $1,20 voor deze run).

**CI.** GitHub Actions draait bij iedere push de tests, de retrievalvergelijking met
ondergrenzen uit [thresholds.json](evals/thresholds.json) en een starttest met Docker Compose.
De betaalde antwoordevaluatie draait niet in CI. In plaats daarvan controleert
`python -m evals.report --check` dat de opgeslagen antwoordresultaten nog bij de huidige
vragenset horen, berekent het de citatie-correctheid opnieuw en toetst het de scores aan
[answer_thresholds.json](evals/answer_thresholds.json). Wie vragen wijzigt zonder de evaluatie
opnieuw te draaien, laat CI falen.

## Antwoordbaarheid voorspellen

Een klein model voorspelt uit de retrievalscores of het corpus een vraag kan beantwoorden.
Zegt het nee, dan zou de API kunnen weigeren zonder een betaalde taalmodelaanroep. De
features zijn de hoogste score en het verschil tussen de eerste twee resultaten, voor zowel
dense retrieval als BM25. De dataset telt 112 nagekeken vragen: 74 beantwoordbaar en 38 niet.
Daarvan zijn 28 vragen apart gehouden als testset, die één keer is gebruikt.

| Model (testset, n = 28) | Precisie | Recall | F1 (95%-BI) | Beantwoordbaar onterecht geblokkeerd |
|---|---:|---:|---:|---:|
| Meest voorkomende klasse | 0,00 | 0,00 | 0,00 | 0 van 18 |
| Drempel op dense-score | 0,67 | 0,80 | 0,73 (0,47-0,91) | 4 van 18 |
| Logistische regressie | 0,73 | 0,80 | 0,76 (0,50-0,94) | 3 van 18 |

Precisie, recall en F1 gaan over het herkennen van onbeantwoordbare vragen. In 5x5-voudige
kruisvalidatie op de trainingsvragen scoren de drempel en de logistische regressie allebei een
F1 van 0,64. **Het model verslaat de eenvoudige drempel dus niet aantoonbaar**: het verschil op
de testset is één vraag en de intervallen overlappen bijna volledig. De meeste fouten zijn vragen
waarbij retrieval de juiste pagina vindt, maar het gevraagde feit ontbreekt, zoals een
percentage voor 2027 op een pagina over 2026. Retrievalscores zien dat verschil niet.

De gate staat daarom niet aan in de API. Hij blokkeert op de testset 3 van de 18
beantwoordbare vragen, terwijl het taalmodel met dense retrieval al 35 van de 38 onbeantwoordbare
vragen terecht weigert. Het getrainde model staat als JSON in `models/answerability_gate.json` en laadt via
`AnswerabilityGate` in `src/rag/gate.py`. Alle cijfers staan in
[evals/results/answerability.md](evals/results/answerability.md). Hoe de labels tot stand
kwamen, staat in [DATA.md](DATA.md).

```bash
uv run python -m evals.answerability
```

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
