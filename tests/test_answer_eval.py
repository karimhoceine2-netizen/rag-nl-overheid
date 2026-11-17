import json

from evals.answer_eval import citation_correct, evaluate_answers
from evals.questions import Question
from rag.generate import Answer
from rag.llm.base import Completion, Usage
from rag.models import Chunk
from rag.query import QueryService

CHUNK = Chunk(
    chunk_id="a#0",
    doc_id="a",
    url="https://example.test/a",
    title="Zorg",
    text="Huisartsenzorg valt buiten het eigen risico.",
)


class FakeRetriever:
    name = "fake"

    def search(self, query: str, k: int) -> list[Chunk]:
        return [CHUNK]


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

    metrics, details = evaluate_answers(service, FakeJudge(), questions, [CHUNK])

    assert metrics["groundedness"] == {"count": 1, "total": 1, "rate": 1.0}
    assert metrics["correct_abstention"] == {"count": 1, "total": 2, "rate": 0.5}
    assert metrics["invalid_answers"] == 1
    assert details[0]["first_relevant_rank"] == 1
    assert details[0]["judgment"]["verdict"]["grounded"] is True
    assert details[1]["judgment"] is None
    assert metrics["citation_correctness"] == {"count": 1, "total": 1, "rate": 1.0}
    assert details[0]["citation_correct"] is True
    assert details[1]["citation_correct"] is None  # abstentions cite nothing


def test_citation_of_wrong_page_is_not_correct():
    question = Question(
        id="q",
        question="huisarts",
        status="draft",
        answerable=True,
        expected_url="https://example.test/other",
        evidence="Huisartsenzorg",
    )
    answer = Answer.model_validate(
        {
            "status": "answered",
            "answer": "x [1]",
            "citations": [{"number": 1, "chunk_id": "a#0", "title": "Zorg", "url": CHUNK.url}],
        }
    )
    assert citation_correct(answer, question, {"a#0": CHUNK}) is False
