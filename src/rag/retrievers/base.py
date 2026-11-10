"""The retriever interface. Every retrieval stage (and later a Rust index) implements it."""

from typing import Protocol

from rag.models import Chunk


class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int) -> list[Chunk]:
        """Return at most `k` chunks, best first, each with `score` set."""
        ...
