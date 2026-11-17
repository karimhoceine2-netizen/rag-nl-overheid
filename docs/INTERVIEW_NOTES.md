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

## Task 2: an answerability gate (ML fundamentals)

### What I built
- **A labelled dataset.** 112 reviewed questions (74 answerable, 38 unanswerable). Answerable
  questions carry a source URL and a verbatim quote that a script checks; the label source is
  documented in `DATA.md`.
- **Features from retrieval.** Top score and the gap between the first two results, for dense
  retrieval and BM25. They are already computed for every question, so they cost nothing.
- **Baselines first.** Majority class, then a single threshold on the top dense score, then
  logistic regression with standardised features and balanced class weights.
- **Evaluation.** A stratified test set of 28 questions, touched once. 5x5 repeated stratified
  cross-validation on the other 84. Precision, recall and F1 for the unanswerable class, a
  confusion matrix, bootstrap 95% intervals and a table of every test error. Fixed seed 42.
- **Model behind an interface.** The logistic regression is saved as JSON (weights, scaling,
  threshold) and loaded by `AnswerabilityGate`; a test checks it gives the same probabilities as
  scikit-learn.

### Result
Logistic regression does not clearly beat the threshold. Both reach F1 0.64 in
cross-validation; on the test set it is 0.76 against 0.73, a difference of one question with
almost fully overlapping intervals. I did not switch the gate on: it blocks 3 of 18 answerable
test questions, while the language model already refuses 7 of 8 unanswerable questions.

### Why
- The gate targets a real cost: every refused question is an LLM call not paid for.
- Logistic regression fits 84 training rows; anything bigger would overfit, and its weights are
  easy to read and to ship as JSON without pickle.
- Positive class = unanswerable, because catching those is the job; "answerable wrongly
  blocked" is reported separately because that error hurts users most.

### Trade-offs
- **Small data.** 28 test questions give wide intervals, so I report them instead of a single
  number, and use repeated cross-validation for model comparison.
- **Labels written by Claude, reviewed by me.** Cheaper than writing 61 questions by hand, but
  cleaner than real user questions, so real-world performance is probably lower.
- **Features cannot see missing facts.** Most errors are questions where retrieval finds the
  right page but the specific year or amount is missing. A better gate would need to read the
  text, for example a cross-encoder or an LLM check, which brings back cost and latency.

### Likely interview questions
1. **Why start with a baseline?** Without it, "F1 0.76" means nothing. Here the baseline showed
   the model adds almost nothing, which is the most important finding.
2. **Why a separate test set when you already do cross-validation?** Cross-validation compares
   models on the training part; the test set gives one untouched estimate at the end. If I tune
   on the test set, its score is optimistic.
3. **Why bootstrap intervals?** With 28 test questions one question changes F1 a lot. The
   interval shows how much the score could move with a different sample.
4. **Why not ship it, if F1 is 0.76?** Because of the cost of errors. Blocking a real question is
   worse than paying a few cents for an LLM call, and the LLM already refuses most unanswerable
   questions. Accuracy alone does not decide whether a model is useful.
5. **How would you improve it?** More and more realistic questions, from real logs. And features
   that look at content, such as whether the reranker finds the asked-for year or amount in the
   top chunk.
