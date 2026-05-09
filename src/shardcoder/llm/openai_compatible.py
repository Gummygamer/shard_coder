"""OpenAI-compatible local LLM client.

Works with any server that exposes the OpenAI ``/chat/completions`` API:
Ollama, llama.cpp ``server``, LM Studio, vLLM, and others.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from ..config import LLMConfig
from .client import ChatMessage, ChatResult, LLMError


class OpenAICompatibleClient:
    """Tiny synchronous client for the OpenAI chat-completions endpoint."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.model = config.model
        self._base_url = config.base_url.rstrip("/")
        self._timeout = config.timeout_seconds
        self._headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if config.api_key:
            self._headers["Authorization"] = f"Bearer {config.api_key}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> ChatResult:
        """Send a chat-completion request and return the assistant text."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": m.role, "content": m.content} for m in messages
            ],
            "temperature": (
                temperature if temperature is not None else self.config.temperature
            ),
            "max_tokens": max_tokens or self.config.max_output_tokens,
            "stream": False,
        }
        if stop:
            payload["stop"] = stop

        url = f"{self._base_url}/chat/completions"
        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(url, headers=self._headers, json=payload)
        except httpx.ConnectError as exc:
            raise LLMError(
                f"Could not reach local model server at {self._base_url}. "
                "Is the server running? Original error: "
                f"{exc.__class__.__name__}: {exc}"
            ) from exc
        except httpx.ReadTimeout as exc:
            raise LLMError(
                f"Local model server timed out after {self._timeout}s."
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"HTTP error talking to local model server: {exc}") from exc

        if response.status_code >= 400:
            raise LLMError(
                f"Local model server returned HTTP {response.status_code}: "
                f"{_truncate(response.text, 500)}"
            )

        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise LLMError(
                f"Local model server returned non-JSON response: "
                f"{_truncate(response.text, 500)}"
            ) from exc

        try:
            choice = data["choices"][0]
            message = choice["message"]
            text = message.get("content", "") or ""
            finish_reason = choice.get("finish_reason")
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(
                f"Local model server response was missing expected fields: {data}"
            ) from exc

        return ChatResult(text=text, finish_reason=finish_reason, raw=data)


def _truncate(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n] + "..."
