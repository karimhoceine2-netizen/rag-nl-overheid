"""Evaluation questions: schema, loading and the relevance rule."""

import hashlib
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


GATE_QUESTIONS_PATH = Path("evals/gate_questions.jsonl")

# Versioned question sets. Results record the version and a hash of the files, so a number
# can always be traced back to the exact questions it was measured on. Never edit a released
# version in place: add questions in a new version instead.
QUESTION_SETS: dict[str, list[Path]] = {
    "v1": [QUESTIONS_PATH],  # 51 questions, 2026-11-16
    "v2": [QUESTIONS_PATH, GATE_QUESTIONS_PATH],  # + 61 questions, 2026-11-17
}
DEFAULT_SET = "v2"


def load_question_set(name: str = DEFAULT_SET) -> tuple[list[Question], dict[str, str]]:
    """Load a versioned question set and describe it (version + sha256 of its files)."""
    paths = QUESTION_SETS[name]
    digest = hashlib.sha256()
    questions: list[Question] = []
    for path in paths:
        digest.update(path.read_bytes())
        questions += load_questions(path)
    ids = [q.id for q in questions]
    if len(ids) != len(set(ids)):
        raise ValueError(f"question set {name} has duplicate ids")
    return questions, {"question_set": name, "question_set_sha256": digest.hexdigest()[:12]}


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
