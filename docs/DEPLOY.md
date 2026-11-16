# Deployen naar Google Cloud Run

De API draait als één container op Cloud Run in `europe-west4` (Eemshaven). Zonder verkeer
schaalt de dienst naar nul instanties. De eerste vraag na een rustige periode wacht daarom op
een koude start.

## Kosten en misbruik begrenzen

`/ask` roept de betaalde Anthropic API aan en de URL is openbaar. De rem bestaat uit vier lagen.
De eerste laag is de belangrijkste:

1. **Aparte Anthropic-key met een uitgavelimiet.** Maak in de Anthropic Console een aparte
   workspace voor deze demo, zet de maandlimiet laag (bijvoorbeeld $5) en maak daar een key aan.
   Gebruik nooit een key van ander werk. Alleen deze limiet stopt de uitgaven echt.
2. **Dagplafond in de app.** `RAG_DAILY_ASK_LIMIT=50`: na 50 vragen per UTC-dag geeft `/ask`
   een 429. De teller staat in het geheugen en begint opnieuw na een herstart, dus hij is een
   eerste rem en geen harde grens.
3. **Eén instantie.** `--max-instances 1` begrenst de rekenkosten en houdt de teller globaal.
4. **Budgetmelding in Google Cloud.** Een melding waarschuwt alleen en stopt niets.

`/health` en `/search` roepen geen taalmodel aan en zijn gratis.

## Eenmalige setup

Installeer de [gcloud CLI](https://cloud.google.com/sdk/docs/install) (op macOS:
`brew install --cask google-cloud-sdk`) en log in.

```bash
gcloud auth login
```

Maak in de [Google Cloud Console](https://console.cloud.google.com/) een project aan,
koppel een factureringsaccount en stel onder **Billing → Budgets & alerts** een budget met
melding in. Zet daarna het project-ID in je shell.

```bash
export GCP_PROJECT=jouw-project-id
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  artifactregistry.googleapis.com secretmanager.googleapis.com --project "$GCP_PROJECT"
```

Bewaar de Anthropic-key in Secret Manager. De opdracht leest de key van je toetsenbord: plak
hem, druk op Enter en dan op **Ctrl+D**. Zo komt de key niet in je shellgeschiedenis.

```bash
gcloud secrets create anthropic-api-key --data-file=- --project "$GCP_PROJECT"
```

Geef het standaard serviceaccount van Cloud Run leesrecht op dat ene secret.

```bash
PROJECT_NUMBER=$(gcloud projects describe "$GCP_PROJECT" --format='value(projectNumber)')
gcloud secrets add-iam-policy-binding anthropic-api-key --project "$GCP_PROJECT" \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role=roles/secretmanager.secretAccessor
```

## Deployen

Cloud Build bouwt de `Dockerfile` in de cloud, dus lokaal Docker is hiervoor niet nodig.

```bash
GCP_PROJECT=jouw-project-id ./scripts/deploy_cloud_run.sh
```

Het script toont aan het eind de URL van de dienst. Controleer de dienst daarna.

```bash
URL=$(gcloud run services describe rag-nl-overheid --region europe-west4 \
  --project "$GCP_PROJECT" --format='value(status.url)')
curl -fsS "$URL/health"
curl -fsS "$URL/search" -H 'Content-Type: application/json' \
  -d '{"question":"Valt een bezoek aan de huisarts onder mijn eigen risico?","k":3}'
```

## Opruimen

Verwijder de dienst en de opgeslagen images als de demo niet meer nodig is.

```bash
gcloud run services delete rag-nl-overheid --region europe-west4 --project "$GCP_PROJECT"
gcloud artifacts repositories delete cloud-run-source-deploy \
  --location europe-west4 --project "$GCP_PROJECT"
gcloud secrets delete anthropic-api-key --project "$GCP_PROJECT"
```

Zet daarna ook de Anthropic-key van deze demo uit in de Anthropic Console.
