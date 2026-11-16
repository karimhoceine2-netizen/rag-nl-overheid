import json

import pytest

from rag.generate import UNKNOWN_ANSWER, generate_answer
from rag.llm.base import Completion, Usage
from rag.models import Chunk


class FakeClient:
    def __init__(self, text: str, complete: bool = True) -> None:
        self.completion = Completion(text, "fake", Usage(10, 5, 0.0), complete)
        self.calls: list[dict] = []

    def complete(self, **kwargs) -> Completion:
        self.calls.append(kwargs)
        return self.completion


def make_chunk(chunk_id: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="zorg",
        url="https://www.rijksoverheid.nl/zorg",
        title="Zorgverzekering",
        text="Een bezoek aan de huisarts valt niet onder het verplichte eigen risico.",
    )


def test_each_claim_gets_citations_from_the_retriever():
    first, second = make_chunk("zorg#0"), make_chunk("zorg#1")
    client = FakeClient(
        json.dumps(
            {
                "claims": [
                    {
                        "text": "De huisarts valt buiten het eigen risico.",
                        "chunk_ids": [second.chunk_id],
                    },
                    {
                        "text": "Dat geldt voor het verplichte eigen risico.",
                        "chunk_ids": [first.chunk_id, second.chunk_id],
                    },
                ]
            }
        )
    )
    question = 'Valt de "huisarts" onder mijn eigen risico?'
    result = generate_answer(client, question, [first, second])

    assert result.answer.status == "answered"
    assert result.answer.answer.endswith("[2] [1]")
    assert [source.chunk_id for source in result.answer.citations] == ["zorg#1", "zorg#0"]
    assert result.answer.citations[0].url == second.url
    payload = json.loads(client.calls[0]["prompt"])
    assert payload["question"] == question
    assert payload["sources"][0]["text"] == first.text


@pytest.mark.parametrize(
    ("text", "complete", "status"),
    [
        ('{"claims":[]}', True, "insufficient_context"),
        ("geen json", True, "invalid_output"),
        ('{"claims":[]}', False, "invalid_output"),
        ('{"claims":[{"text":"Ja.","chunk_ids":[]}]}', True, "invalid_output"),
        ('{"claims":[{"text":"Ja.","chunk_ids":["verzonnen"]}]}', True, "invalid_output"),
        ('{"claims":[{"text":"  ","chunk_ids":["zorg#0"]}]}', True, "invalid_output"),
        ('{"claims":[{"text":"Ja. [99]","chunk_ids":["zorg#0"]}]}', True, "invalid_output"),
    ],
)
def test_unusable_answers_fall_back(text, complete, status):
    result = generate_answer(FakeClient(text, complete), "Welke zorg?", [make_chunk("zorg#0")])
    answer = result.answer

    assert answer.status == status
    assert answer.answer == UNKNOWN_ANSWER
    assert answer.citations == []


def test_empty_retrieval_skips_provider():
    client = FakeClient("")
    result = generate_answer(client, "Een vraag?", [])

    assert result.answer.status == "insufficient_context"
    assert not client.calls
