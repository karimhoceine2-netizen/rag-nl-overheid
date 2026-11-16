"""The provider-independent interface shared by generation and the judge."""

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float | None = None


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    usage: Usage
    complete: bool = True


class LLMError(RuntimeError):
    """The provider could not return a usable response envelope."""


class LLMClient(Protocol):
    def complete(
        self, *, system: str, prompt: str, schema: dict[str, Any], max_tokens: int
    ) -> Completion: ...
