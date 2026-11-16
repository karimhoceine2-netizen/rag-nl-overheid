# Two stages: the builder has uv and downloads everything; the runtime image only gets the
# finished virtualenv, the embedding model and the corpus. Same base image in both stages,
# so the venv's Python symlinks stay valid.

FROM python:3.12.16-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /bin/uv
ENV UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1 \
    HF_HOME=/opt/hf

WORKDIR /app
# torch needs the OpenMP runtime, also here because we load the model once below.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first, so editing src/ does not re-download torch.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project
COPY src ./src
RUN uv sync --locked --no-dev --no-editable

# Bake the pinned embedding model into the image: a cold start then reads it from disk
# instead of downloading it from Hugging Face.
RUN /app/.venv/bin/python -c "from rag.retrievers.dense import E5Encoder; E5Encoder()"


FROM python:3.12.16-slim-bookworm AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --no-log-init rag

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /opt/hf /opt/hf
COPY data/docs.jsonl ./data/docs.jsonl
RUN install -d -o rag -g rag /app/logs

ENV PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

USER rag
EXPOSE 8000

# No curl in the slim image, so the healthcheck uses Python's standard library.
# The start period covers loading the model and embedding the corpus.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen(f\"http://127.0.0.1:{os.environ['PORT']}/health\", timeout=4)"]

# Cloud Run tells the container which port to use through $PORT.
CMD ["sh", "-c", "exec uvicorn rag.api:app --host 0.0.0.0 --port ${PORT}"]
