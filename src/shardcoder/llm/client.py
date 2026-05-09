"""Abstract local LLM client interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class ChatMessage:
    role: str  # "system", "user", or "assistant"
    content: str


@dataclass
class ChatResult:
    text: str
    finish_reason: str | None = None
    raw: dict | None = None


class LLMError(RuntimeError):
    """Raised when the local LLM server is unavailable or misbehaves."""


@runtime_checkable
class LLMClient(Protocol):
    """Minimal interface every backend must implement."""

    model: str

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> ChatResult:
        ...
