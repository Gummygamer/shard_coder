"""High-level external documentation retrieval pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import WebConfig
from ..llm.client import ChatMessage, LLMClient, LLMError
from ..prompting.schemas import parse_web_queries
from ..prompting.templates import external_query_prompt
from .context_normalizer import (
    CompactNote,
    normalize_page,
    notes_under_budget,
)
from .web_search import (
    SearchResult,
    WebBackend,
    WebSearchUnavailable,
    get_backend,
)


@dataclass
class ExternalContextResult:
    notes: list[CompactNote] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    error: str = ""
    backend_name: str = ""
    used: bool = False


def generate_queries(
    task: str,
    reason: str,
    *,
    llm: LLMClient | None = None,
    max_queries: int = 3,
) -> list[str]:
    """Ask the model for precise queries; fall back to a deterministic one."""
    fallback = [f"{task} official documentation"][:max_queries]
    if llm is None:
        return fallback
    try:
        result = llm.chat(
            [
                ChatMessage(
                    role="system",
                    content=(
                        "You write precise web search queries. "
                        "Always respond with valid JSON only."
                    ),
                ),
                ChatMessage(role="user", content=external_query_prompt(task, reason)),
            ],
            temperature=0.0,
            max_tokens=200,
        )
    except LLMError:
        return fallback
    parsed = parse_web_queries(result.text)
    if not parsed or not parsed.queries:
        return fallback
    return parsed.queries[:max_queries]


def retrieve_external_context(
    task: str,
    *,
    reason: str,
    config: WebConfig,
    max_web_context_tokens: int,
    llm: LLMClient | None = None,
    backend: WebBackend | None = None,
    approve_fetch=lambda url: True,
) -> ExternalContextResult:
    """Run the full external retrieval pipeline."""
    if not config.enabled:
        return ExternalContextResult(
            error="web search disabled in config", backend_name=config.backend
        )

    backend = backend or get_backend(config.backend)
    queries = generate_queries(task, reason, llm=llm, max_queries=config.max_queries)

    candidates: list[SearchResult] = []
    try:
        for query in queries:
            for result in backend.search(query, max_results=config.max_results_per_query):
                candidates.append(result)
    except WebSearchUnavailable as exc:
        return ExternalContextResult(
            error=str(exc), backend_name=getattr(backend, "name", "unknown")
        )

    if not candidates:
        return ExternalContextResult(
            error="no relevant web results found",
            backend_name=getattr(backend, "name", "unknown"),
            used=True,
        )

    # Prefer official docs first, otherwise the order returned by the backend.
    if config.prefer_official_docs:
        candidates.sort(key=lambda r: (0 if r.is_official_docs else 1, -r.score))

    fetched_pages = []
    for result in candidates[: config.max_fetched_pages]:
        if config.require_user_approval and not approve_fetch(result.url):
            continue
        try:
            page = backend.fetch(result.url)
        except WebSearchUnavailable as exc:
            return ExternalContextResult(
                error=str(exc), backend_name=getattr(backend, "name", "unknown")
            )
        page.is_official_docs = result.is_official_docs or page.is_official_docs
        fetched_pages.append((page, result))

    notes: list[CompactNote] = []
    for page, result in fetched_pages:
        note = normalize_page(page, relevance=result.snippet[:120], llm=llm)
        notes.append(note)

    bounded = notes_under_budget(notes, config, max_web_context_tokens)
    return ExternalContextResult(
        notes=bounded,
        sources=[n.source_url for n in bounded],
        backend_name=getattr(backend, "name", "unknown"),
        used=True,
    )
