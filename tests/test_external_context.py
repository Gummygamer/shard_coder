"""External documentation pipeline tests."""

from __future__ import annotations

from datetime import date

import pytest

from shardcoder.config import WebConfig
from shardcoder.external_context.context_normalizer import (
    CompactNote,
    enforce_web_budget,
    normalize_page,
    notes_under_budget,
    rank_notes,
)
from shardcoder.external_context.docs_retriever import retrieve_external_context
from shardcoder.external_context.web_search import (
    DisabledBackend,
    FetchedPage,
    SearchResult,
    WebSearchUnavailable,
)
from shardcoder.tokens.budget import estimate_tokens


def test_disabled_backend_returns_safe_error_when_used() -> None:
    backend = DisabledBackend()
    with pytest.raises(WebSearchUnavailable):
        backend.search("anything")
    with pytest.raises(WebSearchUnavailable):
        backend.fetch("https://example.com")


def test_retrieve_external_context_disabled_returns_clear_error() -> None:
    config = WebConfig(enabled=False)
    result = retrieve_external_context(
        task="anything",
        reason="example",
        config=config,
        max_web_context_tokens=200,
    )
    assert not result.notes
    assert "disabled" in result.error.lower()


def test_retrieve_external_context_with_disabled_backend_fails_safely() -> None:
    config = WebConfig(enabled=True, backend="disabled", require_user_approval=False)
    result = retrieve_external_context(
        task="task",
        reason="reason",
        config=config,
        max_web_context_tokens=200,
    )
    assert not result.notes
    assert "no web backend" in result.error.lower()


def test_normalized_page_includes_required_fields() -> None:
    page = FetchedPage(
        url="https://docs.example.com/foo",
        title="Foo Docs",
        text=(
            "Foo is a helper module. It supports lifespan. "
            "Use the lifespan parameter for startup logic. "
            "It is the modern approach. Older handlers may still work."
        ),
        is_official_docs=True,
    )
    note = normalize_page(page, relevance="dependency behaviour", llm=None)
    assert note.source_title == "Foo Docs"
    assert note.source_url == page.url
    assert note.date_accessed == date.today().isoformat()
    assert note.relevance == "dependency behaviour"
    assert note.facts, "facts list should not be empty"


def test_official_docs_rank_above_generic_when_configured() -> None:
    today = date.today().isoformat()
    notes = [
        CompactNote(
            source_title="Random Blog",
            source_url="https://blog.random.com/x",
            date_accessed=today,
            relevance="r",
            is_official_docs=False,
        ),
        CompactNote(
            source_title="Official Docs",
            source_url="https://docs.example.com/y",
            date_accessed=today,
            relevance="r",
            is_official_docs=True,
        ),
    ]
    ranked = rank_notes(notes, prefer_official=True)
    assert ranked[0].source_title == "Official Docs"


def test_web_notes_are_capped_by_budget() -> None:
    today = date.today().isoformat()
    big_facts = ["This is a long fact sentence about behaviour." for _ in range(8)]
    notes = [
        CompactNote(
            source_title=f"doc-{i}",
            source_url=f"https://docs.example.com/{i}",
            date_accessed=today,
            relevance="dependency behaviour",
            facts=big_facts,
            is_official_docs=True,
            rank=i,
        )
        for i in range(5)
    ]
    bounded = enforce_web_budget(notes, max_tokens=80)
    rendered = "\n".join(n.to_rendered().render() for n in bounded)
    assert estimate_tokens(rendered) <= 80
    # Some notes were dropped to fit the budget.
    assert len(bounded) < len(notes)


def test_raw_pages_are_never_passed_through() -> None:
    """Notes only contain the structured fields, not full page text."""
    page = FetchedPage(
        url="https://example.com",
        title="Example",
        text="<html><body>" + ("X" * 5000) + "</body></html>",
    )
    note = normalize_page(page, relevance="x", llm=None)
    rendered = note.to_rendered().render()
    assert "<html>" not in rendered
    assert len(rendered) < 1500


def test_notes_under_budget_combines_ranking_and_capping() -> None:
    today = date.today().isoformat()
    notes = [
        CompactNote(
            source_title="Blog",
            source_url="https://blog.example.com/x",
            date_accessed=today,
            relevance="r",
            facts=["fact one."],
            is_official_docs=False,
            rank=0,
        ),
        CompactNote(
            source_title="Docs",
            source_url="https://docs.example.com/y",
            date_accessed=today,
            relevance="r",
            facts=["fact two."],
            is_official_docs=True,
            rank=1,
        ),
    ]
    bounded = notes_under_budget(notes, WebConfig(prefer_official_docs=True), 200)
    assert bounded[0].source_title == "Docs"


def test_search_result_dataclass_default_score() -> None:
    sr = SearchResult(title="t", url="u", snippet="s")
    assert sr.score == 0.0
    assert sr.is_official_docs is False
