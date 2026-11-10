from rag.chunking import chunk_document, pack, split_sections, word_count
from rag.models import Document


def make_doc(text: str) -> Document:
    return Document(
        doc_id="voorbeeld",
        url="https://example.org/voorbeeld",
        title="Voorbeeld",
        text=text,
        retrieved_at="2026-11-10",
        license="CC0 1.0 Universal",
    )


def test_split_sections_drops_heading_without_paragraphs():
    text = "Intro.\n## Lege kop\n## Kop\nAlinea een.\nAlinea twee."
    assert split_sections(text) == [(None, ["Intro."]), ("Kop", ["Alinea een.", "Alinea twee."])]


def test_pack_respects_max_words_and_overlap():
    items = ["a b c", "d e f", "g h i"]
    assert pack(items, max_words=6) == [["a b c", "d e f"], ["g h i"]]
    assert pack(items, max_words=6, overlap=1) == [["a b c", "d e f"], ["d e f", "g h i"]]


def test_pack_keeps_oversized_item_whole():
    assert pack(["a b c d e"], max_words=2) == [["a b c d e"]]


def test_chunk_document_ids_and_no_text_lost():
    paragraphs = [f"zin {i} " + "woord " * 40 for i in range(12)]
    text = "## Kop A\n" + "\n".join(paragraphs[:6]) + "\n## Kop B\n" + "\n".join(paragraphs[6:])
    chunks = chunk_document(make_doc(text), max_words=120)

    assert [c.chunk_id for c in chunks] == [f"voorbeeld#{i}" for i in range(len(chunks))]
    assert all(word_count(c.text) <= 120 + 45 for c in chunks)  # a heading line may be added
    joined = "\n".join(c.text for c in chunks)
    for p in paragraphs:
        assert p in joined


def test_small_sections_are_merged():
    text = "## Een\nKort.\n## Twee\nOok kort."
    chunks = chunk_document(make_doc(text))
    assert len(chunks) == 1
    assert "Een\nKort." in chunks[0].text and "Twee\nOok kort." in chunks[0].text
