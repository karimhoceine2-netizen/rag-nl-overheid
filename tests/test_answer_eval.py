import json

from evals.answer_eval import evaluate_answers
from evals.questions import Question
from rag.llm.base import Completion, Usage
from rag.models import Chunk
from rag.query import QueryService


class FakeRetriever:
    name = "fake"

    def search(self, query: str, k: int) -> list[Chunk]:
        return [
            Chunk(
                chunk_id="a#0",
                doc_id="a",
                url="https://example.test/a",
                title="Zorg",
                text="Huisartsenzorg valt buiten het eigen risico.",
            )
        ]


class FakeGenerator:
    def complete(self, *, prompt: str, **kwargs) -> Completion:
        question = json.loads(prompt)["question"]
        text = {
            "huisarts": '{"claims":[{"text":"Geen eigen risico.","chunk_ids":["a#0"]}]}',
            "parkeren": '{"claims":[]}',
            "rente": '{"claims":[{"text":"Drie procent.","chunk_ids":["unknown"]}]}',
        }[question]
        return Completion(text, "fake", Usage(0, 0))


class FakeJudge:
    def complete(self, *, prompt: str, **kwargs) -> Completion:
        payload = json.loads(prompt)
        assert payload["question"] == "huisarts"
        assert payload["sources"][0]["number"] == 1
        return Completion(
            '{"grounded":true,"rationale":"De bron steunt het antwoord."}',
            "fake-judge",
            Usage(0, 0),
        )


def test_invalid_output_does_not_count_as_correct_abstention(tmp_path):
    questions = [
        Question(
            id="q001",
            question="huisarts",
            status="draft",
            answerable=True,
            expected_url="https://example.test/a",
            evidence="Huisartsenzorg",
        ),
        Question(id="u001", question="parkeren", status="draft", answerable=False),
        Question(id="u002", question="rente", status="draft", answerable=False),
    ]
    service = QueryService(FakeRetriever(), FakeGenerator(), tmp_path / "queries.jsonl")

    metrics, details = evaluate_answers(service, FakeJudge(), questions)

    assert metrics["groundedness"] == {"count": 1, "total": 1, "rate": 1.0}
    assert metrics["correct_abstention"] == {"count": 1, "total": 2, "rate": 0.5}
    assert metrics["invalid_answers"] == 1
    assert details[0]["first_relevant_rank"] == 1
    assert details[0]["judgment"]["verdict"]["grounded"] is True
    assert details[1]["judgment"] is None
