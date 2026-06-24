"""OpenAI-compatible client behaviour."""

from __future__ import annotations

import json

import httpx
import pytest

from shardcoder.config import LLMConfig
from shardcoder.llm.client import ChatMessage, LLMError
from shardcoder.llm.openai_compatible import OpenAICompatibleClient


def test_auto_model_uses_first_model_from_lm_studio_models_endpoint() -> None:
    seen_payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {"id": "loaded-model-a", "object": "model"},
                        {"id": "loaded-model-b", "object": "model"},
                    ],
                },
            )
        if request.url.path == "/v1/chat/completions":
            payload = json.loads(request.content.decode())
            seen_payloads.append(payload)
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ]
                },
            )
        return httpx.Response(404, text="not found")

    client = OpenAICompatibleClient(
        LLMConfig(base_url="http://localhost:1234/v1", model="auto"),
        transport=httpx.MockTransport(handler),
    )

    info = client.check_connection()
    result = client.chat([ChatMessage(role="user", content="hello")])

    assert info.models == ["loaded-model-a", "loaded-model-b"]
    assert info.selected_model == "loaded-model-a"
    assert result.text == "ok"
    assert seen_payloads[0]["model"] == "loaded-model-a"


def test_explicit_model_is_sent_even_when_server_lists_another_model_first() -> None:
    seen_payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {"id": "gemma4-uncensored", "object": "model"},
                        {"id": "google/gemma-4-26b-a4b-qat", "object": "model"},
                    ],
                },
            )
        if request.url.path == "/v1/chat/completions":
            payload = json.loads(request.content.decode())
            seen_payloads.append(payload)
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ]
                },
            )
        return httpx.Response(404, text="not found")

    client = OpenAICompatibleClient(
        LLMConfig(
            base_url="http://localhost:1234/v1",
            model="google/gemma-4-26b-a4b-qat",
        ),
        transport=httpx.MockTransport(handler),
    )

    info = client.check_connection()
    result = client.chat([ChatMessage(role="user", content="hello")])

    assert info.models == ["gemma4-uncensored", "google/gemma-4-26b-a4b-qat"]
    assert info.selected_model == "google/gemma-4-26b-a4b-qat"
    assert result.text == "ok"
    assert seen_payloads[0]["model"] == "google/gemma-4-26b-a4b-qat"


def test_explicit_model_check_fails_when_configured_model_is_not_loaded() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"id": "gemma4-uncensored", "object": "model"}],
            },
        )
    )
    client = OpenAICompatibleClient(
        LLMConfig(
            base_url="http://localhost:1234/v1",
            model="google/gemma-4-26b-a4b-qat",
        ),
        transport=transport,
    )

    with pytest.raises(LLMError, match="Configured model .* is not loaded"):
        client.check_connection()


def test_auto_model_reports_clear_error_when_no_model_is_loaded() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={"object": "list", "data": []},
        )
    )
    client = OpenAICompatibleClient(
        LLMConfig(base_url="http://localhost:1234/v1", model="auto"),
        transport=transport,
    )

    with pytest.raises(LLMError, match="No models are loaded"):
        client.check_connection()
