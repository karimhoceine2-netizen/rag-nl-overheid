"""Save and load precomputed chunk embeddings for the dense retriever.

Embedding all chunks is the slowest step when the API starts (about 10 s on a laptop), so the
Docker build computes them once. At startup the API uses the file only if it was made from
exactly the same chunks and model; otherwise it embeds the chunks again.

Build it with: uv run python -m rag.dense_index
"""

import logging
import os
from pathlib import Path

import numpy as np

from rag.corpus import load_chunks
from rag.models import Chunk
from rag.retrievers.dense import DEFAULT_MODEL, DEFAULT_REVISION, DenseRetriever, Encoder

logger = logging.getLogger(__name__)

DEFAULT_INDEX_PATH = Path("data/dense_index.npz")
MODEL_ID = f"{DEFAULT_MODEL}@{DEFAULT_REVISION}"


def index_path() -> Path:
    return Path(os.environ.get("RAG_DENSE_INDEX_PATH") or DEFAULT_INDEX_PATH)


def save_index(path: Path, retriever: DenseRetriever, chunks: list[Chunk], model_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        matrix=retriever.matrix,
        chunk_ids=np.array([c.chunk_id for c in chunks]),
        texts=np.array([c.search_text for c in chunks]),
        model_id=np.array(model_id),
    )


def load_matrix(path: Path, chunks: list[Chunk], model_id: str) -> np.ndarray | None:
    """Return the saved matrix, or None if it is missing or does not match these chunks."""
    if not path.exists():
        return None
    with np.load(path) as data:
        same_model = str(data["model_id"]) == model_id
        # Compare the embedded text, not just IDs, so an edited chunk is never served stale.
        same_chunks = data["texts"].tolist() == [c.search_text for c in chunks]
        if same_model and same_chunks:
            return data["matrix"]
    logger.warning("Ignoring %s: it was built from other chunks or another model", path)
    return None


def load_dense_retriever(
    chunks: list[Chunk], encoder: Encoder | None = None, model_id: str = MODEL_ID
) -> DenseRetriever:
    matrix = load_matrix(index_path(), chunks, model_id)
    return DenseRetriever(chunks, encoder=encoder, matrix=matrix)


def main() -> None:
    chunks = load_chunks()
    path = index_path()
    save_index(path, DenseRetriever(chunks), chunks, MODEL_ID)
    print(f"Saved embeddings for {len(chunks)} chunks to {path}")


if __name__ == "__main__":
    main()
