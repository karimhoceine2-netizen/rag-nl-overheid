"""FastAPI entry point for asking Dutch questions about the corpus."""

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, StringConstraints

from rag.corpus import load_chunks
from rag.generate import Answer
from rag.limits import DailyLimit
from rag.llm.anthropic import AnthropicClient
from rag.llm.base import LLMError
from rag.query import QueryService
from rag.retrievers.base import Retriever
from rag.retrievers.dense import DenseRetriever

logger = logging.getLogger(__name__)

Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]


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


def create_app(
    service: QueryService | None = None,
    *,
    retriever: Retriever | None = None,
    daily_limit: DailyLimit | None = None,
) -> FastAPI:
    """Build the app. Tests inject a service or retriever; production builds its own."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if service is not None:
            app.state.retriever = service.retriever
            app.state.query_service = service
            yield
            return

        app.state.retriever = retriever or DenseRetriever(load_chunks())
        if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
            # Start anyway so `docker compose up` works from a clean clone: /health and
            # /search need no key, and /ask explains what is missing.
            logger.warning("ANTHROPIC_API_KEY is not set; /ask is disabled")
            app.state.query_service = None
            yield
            return

        with httpx.Client(timeout=60) as http:
            client = AnthropicClient.from_env(http)
            app.state.query_service = QueryService(app.state.retriever, client)
            yield

    app = FastAPI(title="Nederlandse overheidsvragen", lifespan=lifespan)
    limit = daily_limit or DailyLimit.from_env()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/search", response_model=list[SearchHit])
    def search(body: SearchRequest, request: Request) -> list[SearchHit]:
        # Retrieval only: no language model call, so it is free and always available.
        chunks = request.app.state.retriever.search(body.question, body.k)
        return [SearchHit(**chunk.model_dump()) for chunk in chunks]

    @app.post("/ask", response_model=Answer)
    def ask(body: AskRequest, request: Request) -> Answer:
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
