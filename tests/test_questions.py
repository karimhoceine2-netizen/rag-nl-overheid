import pytest
from pydantic import ValidationError

from evals.questions import Question, check_labels, is_relevant, load_questions
from rag.models import Chunk

URL = "https://x.nl/zorgtoeslag"
CHUNK = Chunk(
    chunk_id="z#0", doc_id="z", url=URL, title="Zorgtoeslag", text="U bent 18 jaar\nof ouder."
)


def q(**kwargs) -> Question:
    base = {"id": "q1", "question": "?", "answerable": True, "status": "draft"}
    return Question(**(base | kwargs))


def test_answerable_question_needs_source():
    with pytest.raises(ValidationError):
        q(expected_url=URL)  # evidence missing


def test_unanswerable_question_must_not_have_source():
    with pytest.raises(ValidationError):
        q(answerable=False, expected_url=URL, evidence="x")
    assert q(answerable=False).expected_url is None


def test_is_relevant_needs_url_and_evidence():
    assert is_relevant(CHUNK, q(expected_url=URL, evidence="U bent 18 jaar of ouder"))
    assert not is_relevant(CHUNK, q(expected_url=URL, evidence="21 jaar"))
    assert not is_relevant(CHUNK, q(expected_url="https://x.nl/other", evidence="18 jaar"))


def test_check_labels_flags_unfindable_evidence():
    good = q(id="good", expected_url=URL, evidence="18 jaar")
    bad = q(id="bad", expected_url=URL, evidence="typo jaar")
    problems = check_labels([good, bad], [CHUNK])
    assert len(problems) == 1 and problems[0].startswith("bad:")


def test_questions_file_is_valid():
    questions = load_questions()
    assert len(questions) >= 20
    assert any(not x.answerable for x in questions)
