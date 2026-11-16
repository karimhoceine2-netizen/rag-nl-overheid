"""FastAPI entry point for asking Dutch questions about the corpus."""

import logging
import os
import threading
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Annotated

import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, StringConstraints

from rag.generate import Answer
from rag.limits import DailyLimit
from rag.llm.anthropic import AnthropicClient
from rag.llm.base import LLMError
from rag.query import QueryService
from rag.retrievers.base import Retriever

logger = logging.getLogger(__name__)

Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]

# How long a request may wait for the model to finish loading. Below Cloud Run's 60 s timeout.
STARTUP_WAIT_S = 50.0


class AskRequest(BaseModel):
    question: Question


class SearchRequest(BaseModel):
    question: Question
    k: int = Field(default=5, ge=1, le=10)


class SearchHit(BaseModel):
    chunk_id: str
    url: str
    title: str
    text: str
    score: float | None


def default_retriever() -> Retriever:
    # Imported here so the web server can start before torch is loaded.
    from rag.corpus import load_chunks
    from rag.dense_index import load_dense_retriever

    return load_dense_retriever(load_chunks())


class Startup:
    """Loads the retriever in a background thread so the port opens right away.

    Cloud Run only forwards requests once the port is open. With the model loading first,
    the first request after scale-to-zero could be rejected (429). Now it waits here instead.
    """

    def __init__(self, load: Callable[[], None]) -> None:
        self.ready = threading.Event()
        self.failed = False
        self._load = load

    def start(self) -> None:
        threading.Thread(target=self._run, name="load-retriever", daemon=True).start()

    def _run(self) -> None:
        try:
            self._load()
        except Exception:
            logger.exception("Loading the retriever failed")
            self.failed = True
        finally:
            self.ready.set()

    def wait(self, timeout: float) -> None:
        if not self.ready.wait(timeout) or self.failed:
            raise HTTPException(503, "Het model wordt nog geladen. Probeer het zo opnieuw.")


def create_app(
    service: QueryService | None = None,
    *,
    retriever_factory: Callable[[], Retriever] = default_retriever,
    daily_limit: DailyLimit | None = None,
    startup_wait_s: float = STARTUP_WAIT_S,
) -> FastAPI:
    """Build the app. Tests inject a service or retriever factory; production builds its own."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.query_service = None
        key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if service is None and not key:
            # Start anyway so `docker compose up` works from a clean clone: /health and
            # /search need no key, and /ask explains what is missing.
            logger.warning("ANTHROPIC_API_KEY is not set; /ask is disabled")

        with httpx.Client(timeout=60) as http:

            def load() -> None:
                if service is not None:
                    app.state.retriever = service.retriever
                    app.state.query_service = service
                    return
                app.state.retriever = retriever_factory()
                if key:
                    client = AnthropicClient.from_env(http)
                    app.state.query_service = QueryService(app.state.retriever, client)

            app.state.startup = Startup(load)
            app.state.startup.start()
            yield

    app = FastAPI(title="Nederlandse overheidsvragen", lifespan=lifespan)
    limit = daily_limit or DailyLimit.from_env()

    @app.get("/health")
    def health(request: Request) -> dict[str, str]:
        startup: Startup = request.app.state.startup
        if startup.failed:
            raise HTTPException(500, "Het model kon niet worden geladen.")
        return {"status": "ok" if startup.ready.is_set() else "loading"}

    @app.post("/search", response_model=list[SearchHit])
    def search(body: SearchRequest, request: Request) -> list[SearchHit]:
        request.app.state.startup.wait(startup_wait_s)
        # Retrieval only: no language model call, so it is free and always available.
        chunks = request.app.state.retriever.search(body.question, body.k)
        return [SearchHit(**chunk.model_dump()) for chunk in chunks]

    @app.post("/ask", response_model=Answer)
    def ask(body: AskRequest, request: Request) -> Answer:
        request.app.state.startup.wait(startup_wait_s)
        query_service: QueryService | None = request.app.state.query_service
        if query_service is None:
            raise HTTPException(503, "Antwoorden staat uit: ANTHROPIC_API_KEY ontbreekt.")
        if not limit.try_acquire():
            raise HTTPException(
                429,
                "De dagelijkse demolimiet is bereikt. Probeer morgen opnieuw of gebruik /search.",
            )
        try:
            return query_service.ask(body.question).answer
        except LLMError as exc:
            raise HTTPException(502, "De taalmodeldienst is tijdelijk niet beschikbaar.") from exc

    return app


app = create_app()
