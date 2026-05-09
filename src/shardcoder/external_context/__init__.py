"""Optional external documentation retrieval and compression."""

from .context_normalizer import (
    CompactNote,
    enforce_web_budget,
    normalize_page,
    notes_under_budget,
    rank_notes,
)
from .docs_retriever import (
    ExternalContextResult,
    generate_queries,
    retrieve_external_context,
)
from .web_search import (
    DisabledBackend,
    FetchedPage,
    SearchResult,
    WebBackend,
    WebSearchUnavailable,
    get_backend,
    register_backend,
)

__all__ = [
    "CompactNote",
    "DisabledBackend",
    "ExternalContextResult",
    "FetchedPage",
    "SearchResult",
    "WebBackend",
    "WebSearchUnavailable",
    "enforce_web_budget",
    "generate_queries",
    "get_backend",
    "normalize_page",
    "notes_under_budget",
    "rank_notes",
    "register_backend",
    "retrieve_external_context",
]
