"""Split documents into retrieval chunks.

Strategy:
1. Split on headings: on these pages a section usually answers one sub-question.
2. Split sections longer than `max_words` into paragraph groups, with one paragraph of
   overlap so a fact sitting at a boundary still appears whole in at least one chunk.
3. Merge neighbouring small pieces up to `max_words`, so we don't index 20-word chunks
   that have lost all context.
"""

from rag.models import Chunk, Document

MAX_WORDS = 350


def word_count(text: str) -> int:
    return len(text.split())


def split_sections(text: str) -> list[tuple[str | None, list[str]]]:
    """Return (heading, paragraphs) pairs. Headings without any paragraphs are dropped."""
    sections: list[tuple[str | None, list[str]]] = []
    heading: str | None = None
    paragraphs: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            if paragraphs:
                sections.append((heading, paragraphs))
            heading, paragraphs = line[3:], []
        elif line.strip():
            paragraphs.append(line)
    if paragraphs:
        sections.append((heading, paragraphs))
    return sections


def pack(items: list[str], max_words: int, overlap: int = 0) -> list[list[str]]:
    """Greedily group consecutive items so each group stays within `max_words`.

    A single item longer than `max_words` becomes its own group (we never cut mid-paragraph).
    """
    groups: list[list[str]] = []
    current: list[str] = []
    size = 0
    for item in items:
        n = word_count(item)
        if current and size + n > max_words:
            groups.append(current)
            current = current[-overlap:] if overlap else []
            size = sum(word_count(x) for x in current)
        current.append(item)
        size += n
    if current:
        groups.append(current)
    return groups


def chunk_document(doc: Document, max_words: int = MAX_WORDS) -> list[Chunk]:
    pieces: list[str] = []
    for heading, paragraphs in split_sections(doc.text):
        for group in pack(paragraphs, max_words, overlap=1):
            body = "\n".join(group)
            pieces.append(f"{heading}\n{body}" if heading else body)

    return [
        Chunk(
            chunk_id=f"{doc.doc_id}#{i}",
            doc_id=doc.doc_id,
            url=doc.url,
            title=doc.title,
            text="\n\n".join(group),
        )
        for i, group in enumerate(pack(pieces, max_words))
    ]
