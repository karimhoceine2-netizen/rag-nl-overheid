from datetime import date

import pytest

from rag.limits import DailyLimit


def test_limit_blocks_after_cap_and_resets_next_day():
    day = [date(2026, 11, 16)]
    limit = DailyLimit(2, today=lambda: day[0])
    assert [limit.try_acquire() for _ in range(3)] == [True, True, False]

    day[0] = date(2026, 11, 17)
    assert limit.try_acquire() is True


def test_zero_means_no_cap():
    limit = DailyLimit(0)
    assert all(limit.try_acquire() for _ in range(100))


def test_limit_reads_env(monkeypatch):
    monkeypatch.setenv("RAG_DAILY_ASK_LIMIT", "3")
    assert DailyLimit.from_env().limit == 3
    monkeypatch.delenv("RAG_DAILY_ASK_LIMIT")
    assert DailyLimit.from_env().limit == 0


def test_negative_limit_is_rejected():
    with pytest.raises(ValueError):
        DailyLimit(-1)
