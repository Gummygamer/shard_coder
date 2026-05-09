"""Pluggable web search backend.

ShardCoder does not ship with a real search backend. The default
:class:`DisabledBackend` returns a clear error so that the agent fails
safely when ``--web`` is used without configuration. Subclass
:class:`WebBackend` to plug in your own search/fetch implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


class WebSearchUnavailable(RuntimeError):
    """Raised when no real web backend is configured."""


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    score: float = 0.0
    is_official_docs: bool = False


@dataclass
class FetchedPage:
    url: str
    title: str
    text: str
    is_official_docs: bool = False
    extra: dict[str, str] = field(default_factory=dict)


@runtime_checkable
class WebBackend(Protocol):
    """Interface implemented by every web backend."""

    name: str

    def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]: ...

    def fetch(self, url: str) -> FetchedPage: ...


class DisabledBackend:
    """Safe default backend that refuses every call with a clear message."""

    name = "disabled"

    def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
        raise WebSearchUnavailable(
            "External documentation retrieval is enabled, but no web backend is "
            "configured. Set [web].backend in shardcoder.toml to a backend you "
            "have implemented."
        )

    def fetch(self, url: str) -> FetchedPage:
        raise WebSearchUnavailable(
            "External documentation retrieval is enabled, but no web backend is "
            "configured. Set [web].backend in shardcoder.toml to a backend you "
            "have implemented."
        )


_BACKENDS: dict[str, WebBackend] = {"disabled": DisabledBackend()}


def register_backend(name: str, backend: WebBackend) -> None:
    """Register a custom backend so ``[web].backend = "<name>"`` resolves it."""
    _BACKENDS[name] = backend


def get_backend(name: str) -> WebBackend:
    if name not in _BACKENDS:
        # Unknown backend names are treated as disabled to fail closed.
        return DisabledBackend()
    return _BACKENDS[name]
