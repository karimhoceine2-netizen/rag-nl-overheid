import json
import threading

from fastapi.testclient import TestClient

from rag.api import create_app
from rag.corpus import load_chunks
from rag.limits import DailyLimit
from rag.llm.base import Completion, LLMError, Usage
from rag.query import QueryService
from rag.retrievers.bm25 import BM25Retriever


class FakeClient:
    def complete(self, *, prompt: str, **kwargs) -> Completion:
        payload = json.loads(prompt)
        if payload["question"] == "eigen risico providerfout":
            raise LLMError("provider diagnostic")
        claims = []
        if payload["question"] == "eigen risico huisarts":
            chunk = next(
                source for source in payload["sources"] if "huisartsenpost" in source["text"]
            )
            claims.append(
                {
                    "text": "De huisarts valt buiten het eigen risico.",
                    "chunk_ids": [chunk["chunk_id"]],
                }
            )
        return Completion(json.dumps({"claims": claims}), "fake", Usage(20, 8, 0.0))


def test_api_logs_answer_abstention_and_provider_error(tmp_path):
    log_path = tmp_path / "queries.jsonl"
    service = QueryService(BM25Retriever(load_chunks()), FakeClient(), log_path)
    with TestClient(create_app(service)) as client:
        assert client.get("/health").json() == {"status": "ok"}
        response = client.post("/ask", json={"question": "  eigen risico huisarts "})
        assert response.status_code == 200
        assert response.json()["status"] == "answered"
        assert response.json()["citations"][0]["url"].startswith("https://www.rijksoverheid.nl/")

        unknown = client.post("/ask", json={"question": "Dekt de zorgverzekering iets op Mars?"})
        assert unknown.json()["status"] == "insufficient_context"
        assert unknown.json()["citations"] == []

        error = client.post("/ask", json={"question": "eigen risico providerfout"})
        assert error.status_code == 502
        assert "provider diagnostic" not in error.text
        assert client.post("/ask", json={"question": "  "}).status_code == 422

    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert [record["status"] for record in records] == [
        "answered",
        "insufficient_context",
        "provider_error",
    ]
    assert response.json()["citations"][0]["chunk_id"] in records[0]["retrieved_chunk_ids"]
    assert records[0]["input_tokens"] == 20
    assert records[0]["output_tokens"] == 8
    assert records[0]["latency_ms"] >= 0
    assert records[0]["estimated_cost_usd"] == 0.0
    assert records[1]["input_tokens"] == 20
    assert records[2]["input_tokens"] is None
    assert all("question" not in record for record in records)


def test_search_is_free_and_ask_reports_missing_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    app = create_app(retriever_factory=lambda: BM25Retriever(load_chunks()))
    with TestClient(app) as client:
        hits = client.post("/search", json={"question": "eigen risico huisarts", "k": 3})
        assert hits.status_code == 200
        assert len(hits.json()) == 3
        assert hits.json()[0]["url"].startswith("https://www.rijksoverheid.nl/")
        assert client.post("/search", json={"question": "x", "k": 50}).status_code == 422

        missing = client.post("/ask", json={"question": "eigen risico huisarts"})
        assert missing.status_code == 503
        assert "ANTHROPIC_API_KEY" in missing.json()["detail"]


def test_ask_returns_429_after_daily_limit(tmp_path):
    service = QueryService(BM25Retriever(load_chunks()), FakeClient(), tmp_path / "q.jsonl")
    with TestClient(create_app(service, daily_limit=DailyLimit(1))) as client:
        assert client.post("/ask", json={"question": "eigen risico huisarts"}).status_code == 200
        blocked = client.post("/ask", json={"question": "eigen risico huisarts"})
        assert blocked.status_code == 429
        # Only the allowed call reached the model and the log.
        assert len((tmp_path / "q.jsonl").read_text().splitlines()) == 1


def test_port_opens_while_retriever_loads_and_requests_wait(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    release = threading.Event()

    def slow_retriever():
        release.wait(5)
        return BM25Retriever(load_chunks())

    app = create_app(retriever_factory=slow_retriever, startup_wait_s=0.05)
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "loading"}
        waiting = client.post("/search", json={"question": "eigen risico"})
        assert waiting.status_code == 503

        release.set()
        app.state.startup.ready.wait(5)
        assert client.get("/health").json() == {"status": "ok"}
        assert client.post("/search", json={"question": "eigen risico"}).status_code == 200


def test_failed_load_makes_health_fail(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    def broken():
        raise RuntimeError("model missing")

    app = create_app(retriever_factory=broken)
    with TestClient(app) as client:
        app.state.startup.ready.wait(5)
        assert client.get("/health").status_code == 500
        assert client.post("/search", json={"question": "eigen risico"}).status_code == 503
