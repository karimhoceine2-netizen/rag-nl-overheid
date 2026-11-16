"""A small daily request cap for the paid /ask endpoint.

The counter lives in memory, so it resets when the process restarts (for example after a
Cloud Run scale-to-zero). That is acceptable because it is only the first brake: the hard
limit is the spending cap on the Anthropic API key itself.
"""

import os
from collections.abc import Callable
from datetime import UTC, date, datetime
from threading import Lock


def _utc_today() -> date:
    return datetime.now(UTC).date()


class DailyLimit:
    """Allow at most `limit` calls per UTC day. A limit of 0 means no cap (local use)."""

    def __init__(self, limit: int, today: Callable[[], date] = _utc_today) -> None:
        if limit < 0:
            raise ValueError("limit must be 0 (no cap) or positive")
        self.limit = limit
        self._today = today
        self._day = today()
        self._count = 0
        self._lock = Lock()  # FastAPI runs sync endpoints in a thread pool

    def try_acquire(self) -> bool:
        if self.limit == 0:
            return True
        with self._lock:
            today = self._today()
            if today != self._day:
                self._day, self._count = today, 0
            if self._count >= self.limit:
                return False
            self._count += 1
            return True

    @classmethod
    def from_env(cls) -> "DailyLimit":
        return cls(int(os.environ.get("RAG_DAILY_ASK_LIMIT", "0") or 0))
