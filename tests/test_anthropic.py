import json

import httpx
import pytest

from rag.llm.anthropic import GENERATION_MODEL, AnthropicClient, estimate_cost
from rag.llm.base import LLMError


def test_anthropic_adapter_sends_schema_and_parses_usage():
    schema = {"type": "object", "properties": {}, "additionalProperties": False}

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://api.anthropic.com/v1/messages"
        assert request.headers["Authorization"] == "Bearer fake-key"
        body = json.loads(request.content)
        assert body["model"] == GENERATION_MODEL
        assert body["messages"] == [{"role": "user", "content": "vraag"}]
        assert body["output_config"]["format"]["schema"] == schema
        return httpx.Response(
            200,
            json={
                "model": GENERATION_MODEL,
                "content": [{"type": "thinking"}, {"type": "text", "text": "{}"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1000, "output_tokens": 100},
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as http:
        client = AnthropicClient(http, model=GENERATION_MODEL, api_key="fake-key")
        result = client.complete(system="prompt", prompt="vraag", schema=schema, max_tokens=100)

    assert result.complete
    assert result.text == "{}"
    assert result.usage.input_tokens == 1000
    assert result.usage.estimated_cost_usd == pytest.approx(0.00015)
    assert estimate_cost(GENERATION_MODEL, 100_001, 100) == pytest.approx(0.0502505)
    assert estimate_cost("unknown", 1000, 100) is None


def test_provider_errors_do_not_include_response_bodies():
    transport = httpx.MockTransport(lambda request: httpx.Response(429, text="private detail"))
    with httpx.Client(transport=transport) as http:
        client = AnthropicClient(http, model=GENERATION_MODEL, api_key="fake-key")
        with pytest.raises(LLMError, match=r"^Anthropic returned HTTP 429$"):
            client.complete(system="", prompt="", schema={}, max_tokens=1)
