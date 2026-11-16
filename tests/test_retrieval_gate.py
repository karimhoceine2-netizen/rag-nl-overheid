import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(("mrr_min", "exit_code"), [(0.63, 0), (1.0, 1)])
def test_retrieval_cli_writes_report_and_enforces_minimum(tmp_path, mrr_min, exit_code):
    floors = tmp_path / "thresholds.json"
    floors.write_text(json.dumps({"bm25": {"hit@5": 0.79, "mrr@10": mrr_min}}), encoding="utf-8")
    report = tmp_path / "report"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "evals.retrieval_eval",
            "--stages",
            "bm25",
            "--thresholds",
            str(floors),
            "--out-dir",
            str(report),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == exit_code, completed.stderr
    assert (report / "retrieval.json").exists()
    if exit_code:
        assert "bm25 mrr@10" in completed.stderr
