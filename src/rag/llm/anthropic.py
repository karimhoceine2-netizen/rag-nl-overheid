"""Anthropic Messages API adapter using the project's existing HTTP client."""

import os
from typing import Any

import httpx
from pydantic import BaseModel, Field, ValidationError

from rag.llm.base import Completion, LLMError, Usage

GENERATION_MODEL = "claude-haiku-5-5"
JUDGE_MODEL = "claude-sonnet-5-5"


class _TokenCounts(BaseModel):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class _ContentBlock(BaseModel):
    type: str
    text: str = ""


class _Message(BaseModel):
    model: str
    content: list[_ContentBlock]
    stop_reason: str
    usage: _TokenCounts


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    # Standard direct API prices, no prompt caching or batch pricing.
    # https://platform.claude.com/docs/en/about-claude/pricing
    if model == GENERATION_MODEL:
        input_rate, output_rate = (0.10, 0.50) if input_tokens <= 100_000 else (0.50, 2.50)
    elif model == JUDGE_MODEL:
        input_rate, output_rate = (2.0, 10.0)
    else:
        return None  # A custom model needs its own rates before we can estimate its cost.
    return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000


class AnthropicClient:
    def __init__(self, http: httpx.Client, *, model: str, api_key: str) -> None:
        self._http = http
        self.model = model
        self._api_key = api_key

    @classmethod
    def from_env(cls, http: httpx.Client, *, judge: bool = False) -> "AnthropicClient":
        key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not key:
            raise ValueError("Set ANTHROPIC_API_KEY or run with uv run --env-file .env")
        setting = "RAG_JUDGE_MODEL" if judge else "RAG_GENERATION_MODEL"
        default_model = JUDGE_MODEL if judge else GENERATION_MODEL
        return cls(http, model=os.environ.get(setting, "").strip() or default_model, api_key=key)

    def complete(
        self, *, system: str, prompt: str, schema: dict[str, Any], max_tokens: int
    ) -> Completion:
        try:
            response = self._http.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "anthropic-version": "2026-10-21",
                },
                json={
                    "model": self.model,
                    "max_tokens": max_tokens,
                    "system": system,
                    "messages": [{"role": "user", "content": prompt}],
                    "output_config": {"format": {"type": "json_schema", "schema": schema}},
                },
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise LLMError(f"Anthropic returned HTTP {exc.response.status_code}") from exc
        except httpx.RequestError as exc:
            raise LLMError("Could not reach Anthropic") from exc

        try:
            message = _Message.model_validate_json(response.content)
        except ValidationError as exc:
            raise LLMError("Anthropic returned an invalid response envelope") from exc

        input_tokens = message.usage.input_tokens
        output_tokens = message.usage.output_tokens
        return Completion(
            text="".join(block.text for block in message.content if block.type == "text"),
            model=message.model,
            usage=Usage(
                input_tokens,
                output_tokens,
                estimate_cost(message.model, input_tokens, output_tokens),
            ),
            complete=message.stop_reason == "end_turn",
        )
