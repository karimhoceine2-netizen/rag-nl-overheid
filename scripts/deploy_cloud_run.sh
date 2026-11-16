#!/usr/bin/env bash
# Build the Dockerfile with Cloud Build and deploy it to Cloud Run.
# One-time setup (project, billing, secret, permissions) is in docs/DEPLOY.md.
set -euo pipefail

: "${GCP_PROJECT:?Set GCP_PROJECT to your Google Cloud project ID}"
REGION="${GCP_REGION:-europe-west4}"  # Eemshaven, the Netherlands
SERVICE="${SERVICE:-rag-nl-overheid}"

# min-instances 0: scale to zero, so idle time costs nothing (first request waits).
# max-instances 1: caps cost and keeps the in-memory daily /ask limit global.
gcloud run deploy "$SERVICE" \
  --project "$GCP_PROJECT" \
  --region "$REGION" \
  --source . \
  --memory 2Gi \
  --cpu 1 \
  --cpu-boost \
  --min-instances 0 \
  --max-instances 1 \
  --concurrency 4 \
  --timeout 60 \
  --allow-unauthenticated \
  --set-secrets ANTHROPIC_API_KEY=anthropic-api-key:latest \
  --set-env-vars RAG_DAILY_ASK_LIMIT=50
