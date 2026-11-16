"""FastAPI entry point for asking Dutch questions about the corpus."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, StringConstraints

from rag.corpus import load_chunks
from rag.generate import Answer
from rag.llm.anthropic import AnthropicClient
from rag.llm.base import LLMError
from rag.query import QueryService
from rag.retrievers.dense import DenseRetriever


class AskRequest(BaseModel):
    question: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)
    ]


def create_app(service: QueryService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if service is not None:
            app.state.query_service = service
            yield
            return

        with httpx.Client(timeout=60) as http:
            client = AnthropicClient.from_env(http)
            retriever = DenseRetriever(load_chunks())
            app.state.query_service = QueryService(retriever, client)
            yield

    app = FastAPI(title="Nederlandse overheidsvragen", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/ask", response_model=Answer)
    def ask(body: AskRequest, request: Request) -> Answer:
        query_service: QueryService = request.app.state.query_service
        try:
            return query_service.ask(body.question).answer
        except LLMError as exc:
            raise HTTPException(502, "De taalmodeldienst is tijdelijk niet beschikbaar.") from exc

    return app


app = create_app()
