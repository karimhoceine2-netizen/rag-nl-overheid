# Interview notes

Plain-language notes per upgrade: what I built, why, the trade-offs, and questions I expect.

## Task 1: Docker and cloud deployment

### What I built
- **Multi-stage Dockerfile.** The builder stage installs dependencies with `uv` and downloads
  the pinned embedding model (`multilingual-e5-small`). The runtime stage copies only the
  virtualenv, the model and the corpus. It runs as a non-root user, works offline
  (`HF_HUB_OFFLINE=1`) and has a `HEALTHCHECK` on `/health`.
- **docker-compose.yml.** `docker compose up --build` starts the API from a clean clone. The
  `.env` file is optional: without a key, `/health` and `/search` work and `/ask` returns 503.
- **Free `/search` endpoint.** Retrieval only, no language model, so it costs nothing.
- **Daily cap on `/ask`.** `RAG_DAILY_ASK_LIMIT` returns 429 after N paid calls per UTC day.
- **CI.** GitHub Actions starts the container with Compose (no `.env`), tests `/health`,
  `/search` and the 503 on `/ask`, and writes the image size and peak memory to the job summary.
- **Cloud Run deploy.** `scripts/deploy_cloud_run.sh` builds with Cloud Build and deploys with
  scale-to-zero and at most one instance. `docs/DEPLOY.md` lists every step, including cleanup.

### Why
- Baking the model into the image avoids a download on every cold start and makes the
  container independent of Hugging Face being reachable.
- Starting without a key means anyone can run and test the container, including CI, without
  secrets.
- Cloud Run scales to zero, so an idle portfolio demo costs close to nothing, and it runs any
  container, so the same image runs locally, in CI and in the cloud.

### Trade-offs
- **No database in Compose.** The index is 210 chunks in memory. A database would add a
  moving part without solving a problem. With many thousands of documents I would move the
  vectors to something like pgvector.
- **Cold start.** Scale-to-zero means the first request after idle time waits for the
  container to start and the model to load. Three changes made this work: the corpus
  embeddings are computed at build time and shipped in the image (local startup went from
  about 16 s to 5.5 s, CI peak memory from 1226 to 781 MiB); the model loads in a background
  thread, so the port opens at once and the first request waits instead of being rejected
  with a 429 by Cloud Run; and the service gets 2 CPUs without throttling, so loading is not
  slowed down after the port opens. A measured cold start went from 49.4 s to 45.0 s; a warm
  search takes about 0.1 s. Keeping one instance warm would remove the wait but costs money
  every month, so I accepted it for a demo.
- **The daily cap lives in memory.** It resets when the container restarts. That is fine as a
  first brake because the real limit is the spending cap on a separate Anthropic key. A
  durable counter would need shared storage such as Firestore or Redis.
- **The query log is not durable on Cloud Run.** It is written inside the container, so it
  disappears when the instance scales to zero. For a demo that is fine; for real use I would
  log to stdout so Cloud Logging keeps it.
- **No build cache mounts.** BuildKit cache mounts speed up local rebuilds, but I left them out
  so the same Dockerfile builds unchanged in Cloud Build.
- **Manual deploy, not deploy-on-push.** Deploying from GitHub Actions needs keyless auth
  (Workload Identity Federation). It is a sensible next step but not needed to prove the
  container runs in the cloud.

### Likely interview questions
1. **Why a multi-stage build?** The build tools (uv, caches) stay in the first stage. The final
   image only has what the app needs to run, which makes it smaller and gives an attacker less
   to work with.
2. **Why run as a non-root user?** If someone breaks into the app process, they do not get root
   inside the container. It is a cheap default that limits damage.
3. **How do you keep the API key safe?** It is never in the image or the repo. Locally it is in
   a git-ignored `.env`. On Cloud Run it is in Secret Manager and injected as an environment
   variable at runtime. `.dockerignore` and `.gcloudignore` both exclude `.env*`.
4. **What happens if someone spams the public URL?** `/search` costs nothing. `/ask` stops after
   the daily cap, only one instance can run, and the Anthropic key has its own spending limit,
   which is the hard stop.
5. **How do you know the container works before deploying?** CI builds the image and starts it
   with Compose on every push, then calls the endpoints. Compose waits on the Docker healthcheck,
   so a container that starts but never becomes healthy fails the build.
