"""OpenAI-compatible local LLM client.

Works with any server that exposes the OpenAI ``/chat/completions`` API:
Ollama, llama.cpp ``server``, LM Studio, vLLM, and others.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import httpx

from ..config import LLMConfig
from .client import ChatMessage, ChatResult, LLMError


@dataclass
class LLMConnectionInfo:
    base_url: str
    models: list[str]
    selected_model: str


class OpenAICompatibleClient:
    """Tiny synchronous client for the OpenAI chat-completions endpoint."""

    def __init__(
        self,
        config: LLMConfig,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.config = config
        self.model = config.model.strip() or "auto"
        self._base_url = config.base_url.rstrip("/")
        self._timeout = config.timeout_seconds
        self._timeout_retries = config.timeout_retries
        self._timeout_retry_backoff_seconds = config.timeout_retry_backoff_seconds
        self._transport = transport
        self._resolved_model: str | None = None
        self._headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if config.api_key:
            self._headers["Authorization"] = f"Bearer {config.api_key}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_models(self) -> list[str]:
        """Return model ids advertised by the local OpenAI-compatible server."""
        data = self._request("GET", f"{self._base_url}/models")
        models = data.get("data")
        if not isinstance(models, list):
            raise LLMError(
                f"Local model server response was missing model data: {data}"
            )

        ids: list[str] = []
        for item in models:
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                ids.append(item["id"])
        return ids

    def check_connection(self) -> LLMConnectionInfo:
        """Probe the local server and resolve the model ShardCoder will use."""
        models = self.list_models()
        selected = self._resolve_model(models=models)
        return LLMConnectionInfo(
            base_url=self._base_url,
            models=models,
            selected_model=selected,
        )

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> ChatResult:
        """Send a chat-completion request and return the assistant text."""
        model = self._resolve_model()
        payload: dict[str, Any] = {
            "model": model,
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
        data = self._request("POST", url, payload=payload)

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

    def _client_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"timeout": self._timeout}
        if self._transport is not None:
            kwargs["transport"] = self._transport
        return kwargs

    def _request(
        self,
        method: str,
        url: str,
        *,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        for attempt in range(self._timeout_retries + 1):
            try:
                with httpx.Client(**self._client_kwargs()) as client:
                    response = client.request(
                        method,
                        url,
                        headers=self._headers,
                        json=payload,
                    )
                break
            except httpx.ConnectError as exc:
                raise LLMError(
                    f"Could not reach local model server at {self._base_url}. "
                    "Is the server running? Original error: "
                    f"{exc.__class__.__name__}: {exc}"
                ) from exc
            except httpx.ReadTimeout as exc:
                if attempt < self._timeout_retries:
                    if self._timeout_retry_backoff_seconds:
                        time.sleep(self._timeout_retry_backoff_seconds)
                    continue
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
            return response.json()
        except json.JSONDecodeError as exc:
            raise LLMError(
                f"Local model server returned non-JSON response: "
                f"{_truncate(response.text, 500)}"
            ) from exc

    def _resolve_model(self, *, models: list[str] | None = None) -> str:
        if self.model != "auto":
            if models is not None and self.model not in models:
                raise LLMError(
                    f"Configured model {self.model!r} is not loaded at "
                    f"{self._base_url}. Available models: "
                    f"{', '.join(models) or '(none)'}"
                )
            return self.model
        if self._resolved_model is not None:
            return self._resolved_model

        models = models if models is not None else self.list_models()
        if not models:
            raise LLMError(
                f"No models are loaded at {self._base_url}. "
                "Load a model in LM Studio, then start the local server."
            )
        self._resolved_model = models[0]
        return self._resolved_model


def _truncate(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n] + "..."
