import copy
import json

import pytest

import evals.report as report
from evals.questions import load_question_set


def _result(stage_answers: list[dict], sha: str) -> dict:
    rate = {"count": 0, "total": 0, "rate": 1.0}
    return {
        "meta": {"question_set_sha256": sha, "generation_model": "g", "judge_model": "j"},
        "metrics": {
            "groundedness": rate,
            "citation_correctness": {"count": 0, "total": 0, "rate": 1.0},
            "correct_abstention": rate,
            "answerable_coverage": rate,
            "invalid_answers": 0,
            "latency_ms_mean": 1000.0,
            "generation_cost_usd": 0.0,
            "judge_cost_usd": 0.0,
        },
        "details": stage_answers,
    }


@pytest.fixture
def thresholds(tmp_path, monkeypatch):
    path = tmp_path / "t.json"
    path.write_text(json.dumps({s: {"groundedness": 0.5} for s in report.ANSWER_STAGES}))
    monkeypatch.setattr(report, "ANSWER_THRESHOLDS", path)


def test_check_passes_for_current_set(thresholds) -> None:
    _, info = load_question_set()
    answers = {s: _result([], info["question_set_sha256"]) for s in report.ANSWER_STAGES}
    assert report.check(answers) == []


def test_check_flags_stale_results(thresholds) -> None:
    answers = {s: _result([], "oldhash") for s in report.ANSWER_STAGES}
    problems = report.check(answers)
    assert len(problems) == 2 and "another question set" in problems[0]


def test_check_flags_scores_below_threshold(thresholds) -> None:
    _, info = load_question_set()
    good = _result([], info["question_set_sha256"])
    bad = copy.deepcopy(good)
    bad["metrics"]["groundedness"] = {"count": 1, "total": 4, "rate": 0.25}
    problems = report.check({"dense": good, "rerank": bad})
    assert problems == ["rerank groundedness: 0.250 < 0.5"]


def test_render_mentions_every_stage() -> None:
    retrieval = {
        "meta": {"question_set": "v2", "question_set_sha256": "abc", "n_questions_scored": 1},
        "stages": {"dense": {"hit@1": 1.0, "hit@5": 1.0, "mrr@10": 1.0, "latency_ms_mean": 10.0}},
    }
    text = report.render(retrieval, {s: _result([], "abc") for s in report.ANSWER_STAGES})
    assert "dense" in text and "rerank" in text and "v2" in text
