"""Shared retrieval and generation flow, with one JSONL record per query."""

import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock

from rag.generate import DEFAULT_TOP_K, Answer, GeneratedAnswer, generate_answer
from rag.llm.base import LLMClient, LLMError, Usage
from rag.models import Chunk
from rag.retrievers.base import Retriever

DEFAULT_LOG_PATH = Path("logs/queries.jsonl")


@dataclass(frozen=True)
class QueryResult:
    answer: Answer
    chunks: list[Chunk]
    usage: Usage | None = None  # None when no language model was called


class QueryService:
    def __init__(
        self,
        retriever: Retriever,
        client: LLMClient,
        log_path: Path | None = None,
        *,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        self.retriever = retriever
        self.top_k = top_k
        self._client = client
        self._log_path = log_path or Path(os.environ.get("RAG_LOG_PATH") or DEFAULT_LOG_PATH)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_lock = Lock()

    def ask(self, question: str) -> QueryResult:
        started = time.perf_counter()
        chunks: list[Chunk] = []
        generated: GeneratedAnswer | None = None
        status = "error"
        try:
            chunks = self.retriever.search(question, self.top_k)
            generated = generate_answer(self._client, question, chunks)
            status = generated.answer.status
            completion = generated.completion
            return QueryResult(generated.answer, chunks, completion.usage if completion else None)
        except LLMError:
            status = "provider_error"
            raise
        finally:
            self._record(started, status, chunks, generated)

    def _record(
        self, started: float, status: str, chunks: list[Chunk], generated: GeneratedAnswer | None
    ) -> None:
        completion = generated.completion if generated else None
        if completion:
            usage = asdict(completion.usage)
        else:
            # A failed call can still have been billed, so its usage stays unknown.
            no_call = status == "insufficient_context"
            usage = {
                "input_tokens": 0 if no_call else None,
                "output_tokens": 0 if no_call else None,
                "estimated_cost_usd": 0.0 if no_call else None,
            }
        # Store IDs and metrics so the log does not capture personal details in a question.
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "status": status,
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "retriever": self.retriever.name,
            "retrieved_chunk_ids": [chunk.chunk_id for chunk in chunks],
            "model": completion.model if completion else None,
            **usage,
        }
        with self._log_lock, self._log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
