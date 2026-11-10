"""Evaluation questions: schema, loading and the relevance rule."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, model_validator

from rag.models import Chunk
from rag.text import normalize_for_match

QUESTIONS_PATH = Path("evals/questions.jsonl")


class Question(BaseModel):
    id: str
    question: str
    expected_url: str | None = None
    evidence: str | None = None  # short verbatim quote from the expected page
    answerable: bool
    status: Literal["draft", "reviewed"]
    notes: str = ""

    @model_validator(mode="after")
    def _check_labels(self) -> "Question":
        if self.answerable and not (self.expected_url and self.evidence):
            raise ValueError(f"{self.id}: answerable questions need expected_url and evidence")
        if not self.answerable and (self.expected_url or self.evidence):
            raise ValueError(f"{self.id}: unanswerable questions must not have a source")
        return self


def load_questions(path: Path = QUESTIONS_PATH) -> list[Question]:
    lines = path.read_text(encoding="utf-8").splitlines()
    questions = [Question.model_validate_json(line) for line in lines if line.strip()]
    ids = [q.id for q in questions]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"duplicate question ids: {sorted(duplicates)}")
    return questions


def is_relevant(chunk: Chunk, question: Question) -> bool:
    """A chunk is relevant if it comes from the expected page AND contains the evidence quote.

    Labelling by (url, quote) instead of chunk id means the labels stay valid when we change
    the chunking strategy.
    """
    if question.evidence is None or chunk.url != question.expected_url:
        return False
    return normalize_for_match(question.evidence) in normalize_for_match(chunk.text)


def check_labels(questions: list[Question], chunks: list[Chunk]) -> list[str]:
    """Return a problem description for every answerable question no chunk can satisfy.

    Without this check a typo in an evidence quote would silently count as a retrieval miss.
    """
    problems = []
    for q in questions:
        if q.answerable and not any(is_relevant(c, q) for c in chunks):
            problems.append(
                f"{q.id}: evidence not found in any chunk of {q.expected_url} "
                "(typo, page changed, or quote spans a chunk boundary)"
            )
    return problems
