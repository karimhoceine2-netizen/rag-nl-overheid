import json

from fastapi.testclient import TestClient

from rag.api import create_app
from rag.corpus import load_chunks
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
