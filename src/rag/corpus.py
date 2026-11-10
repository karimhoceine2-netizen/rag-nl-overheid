"""Load the ingested corpus (data/docs.jsonl) and chunk it."""

from pathlib import Path

from rag.chunking import MAX_WORDS, chunk_document
from rag.models import Chunk, Document

DOCS_PATH = Path("data/docs.jsonl")


def load_documents(path: Path = DOCS_PATH) -> list[Document]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `uv run python -m rag.ingest` first")
    lines = path.read_text(encoding="utf-8").splitlines()
    return [Document.model_validate_json(line) for line in lines if line]


def load_chunks(path: Path = DOCS_PATH, max_words: int = MAX_WORDS) -> list[Chunk]:
    # Chunking is deterministic and fast, so we re-chunk on load instead of storing chunks.
    # That keeps chunk size a single parameter we can vary in experiments.
    return [chunk for doc in load_documents(path) for chunk in chunk_document(doc, max_words)]
