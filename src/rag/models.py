"""Core data types shared by ingestion, retrieval and evaluation."""

from pydantic import BaseModel


class Document(BaseModel):
    doc_id: str  # stable, human-readable id: the last segment of the URL path
    url: str
    title: str
    text: str  # cleaned text; headings are kept as "## ..." lines for the chunker
    retrieved_at: str  # ISO date (YYYY-MM-DD) of the HTTP fetch
    modified: str | None = None  # page's own DCTERMS.modified date, if present
    license: str  # page's DCTERMS.rights meta tag, e.g. "CC0 1.0 Universal"


class Chunk(BaseModel):
    chunk_id: str  # f"{doc_id}#{index}"
    doc_id: str
    url: str
    title: str
    text: str
    score: float | None = None  # filled in by a retriever

    @property
    def search_text(self) -> str:
        # Chunks from the middle of a page often don't repeat the topic ("Voorwaarden ...").
        # Prefixing the page title gives every chunk that context at index time.
        return f"{self.title}\n{self.text}"
