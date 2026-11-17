from pathlib import Path

from evals.questions import Question, load_questions
from evals.review_questions import review, save


def _drafts(path: Path) -> None:
    save(
        path,
        [
            Question(id="a", question="Vraag a?", answerable=False, status="draft"),
            Question(id="b", question="Vraag b?", answerable=False, status="draft"),
            Question(id="c", question="Vraag c?", answerable=False, status="draft"),
            Question(id="d", question="Vraag d?", answerable=False, status="reviewed"),
        ],
    )


def test_approve_reject_skip(tmp_path: Path) -> None:
    path = tmp_path / "q.jsonl"
    _drafts(path)
    answers = iter(["j", "x", "n", "s"])  # "x" is invalid and asked again
    counts = review(path, ask=lambda _: next(answers))

    assert counts == {"approved": 1, "rejected": 1, "skipped": 1}
    status = {q.id: q.status for q in load_questions(path)}
    assert status == {"a": "reviewed", "c": "draft", "d": "reviewed"}
    rejected = load_questions(tmp_path / "q.rejected.jsonl")
    assert [q.id for q in rejected] == ["b"]


def test_quit_keeps_earlier_answers(tmp_path: Path) -> None:
    path = tmp_path / "q.jsonl"
    _drafts(path)
    answers = iter(["j", "q"])
    review(path, ask=lambda _: next(answers))

    status = {q.id: q.status for q in load_questions(path)}
    assert status == {"a": "reviewed", "b": "draft", "c": "draft", "d": "reviewed"}
