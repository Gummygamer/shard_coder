"""Compress fetched web pages into compact, factual notes.

Raw HTML is never inserted into the model context. The normaliser asks
the local model to extract a small JSON summary, and falls back to a
deterministic excerpt when the model is unavailable or returns garbage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Iterable

from ..config import WebConfig
from ..llm.client import ChatMessage, LLMClient, LLMError
from ..prompting.schemas import WebNote
from ..prompting.templates import external_summary_prompt
from ..retrieval.context_pack import WebNoteRendered
from ..tokens.budget import cap_text, estimate_tokens
from .web_search import FetchedPage


_OFFICIAL_DOMAIN_HINTS = (
    "docs.",
    "developer.",
    "fastapi.tiangolo.com",
    "python.org",
    "nodejs.org",
    "go.dev",
    "rust-lang.org",
    "kubernetes.io",
    "rfc-editor.org",
    "w3.org",
)


@dataclass
class CompactNote:
    source_title: str
    source_url: str
    date_accessed: str
    relevance: str
    facts: list[str] = field(default_factory=list)
    is_official_docs: bool = False
    rank: int = 0

    def to_rendered(self) -> WebNoteRendered:
        return WebNoteRendered(
            source_title=self.source_title,
            source_url=self.source_url,
            date_accessed=self.date_accessed,
            relevance=self.relevance,
            facts=self.facts,
            rank=self.rank,
        )


def _looks_official(url: str) -> bool:
    lowered = url.lower()
    return any(h in lowered for h in _OFFICIAL_DOMAIN_HINTS)


def _deterministic_facts(text: str, max_facts: int = 5) -> list[str]:
    if not text:
        return []
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
    out: list[str] = []
    for sentence in sentences:
        if len(sentence) < 30 or len(sentence) > 240:
            continue
        out.append(sentence + ".")
        if len(out) >= max_facts:
            break
    return out


def normalize_page(
    page: FetchedPage,
    *,
    relevance: str,
    llm: LLMClient | None = None,
    today: str | None = None,
    max_input_tokens: int = 1500,
) -> CompactNote:
    """Compress *page* into a :class:`CompactNote`."""
    today = today or date.today().isoformat()
    is_official = page.is_official_docs or _looks_official(page.url)

    capped = cap_text(page.text, max_input_tokens)

    if llm is not None:
        try:
            result = llm.chat(
                [
                    ChatMessage(
                        role="system",
                        content=(
                            "You compress web pages into compact factual notes. "
                            "Always respond with valid JSON only."
                        ),
                    ),
                    ChatMessage(
                        role="user",
                        content=external_summary_prompt(page.title, page.url, capped),
                    ),
                ],
                temperature=0.0,
                max_tokens=400,
            )
            from ..prompting.schemas import extract_json

            raw = extract_json(result.text)
            if raw:
                try:
                    parsed = WebNote.model_validate(
                        {
                            "source_title": raw.get("source_title", page.title),
                            "source_url": raw.get("source_url", page.url),
                            "date_accessed": today,
                            "relevance": raw.get("relevance", relevance),
                            "facts": [str(f) for f in raw.get("facts", [])][:5],
                        }
                    )
                    return CompactNote(
                        source_title=parsed.source_title,
                        source_url=parsed.source_url,
                        date_accessed=parsed.date_accessed,
                        relevance=parsed.relevance,
                        facts=parsed.facts,
                        is_official_docs=is_official,
                    )
                except Exception:
                    pass
        except LLMError:
            pass

    return CompactNote(
        source_title=page.title or page.url,
        source_url=page.url,
        date_accessed=today,
        relevance=relevance,
        facts=_deterministic_facts(capped),
        is_official_docs=is_official,
    )


def rank_notes(
    notes: Iterable[CompactNote], *, prefer_official: bool = True
) -> list[CompactNote]:
    """Sort notes by relevance heuristics."""
    notes = list(notes)
    for i, n in enumerate(notes):
        n.rank = i
    if prefer_official:
        notes.sort(key=lambda n: (0 if n.is_official_docs else 1, n.rank))
    for i, n in enumerate(notes):
        n.rank = i
    return notes


def enforce_web_budget(
    notes: Iterable[CompactNote], max_tokens: int
) -> list[CompactNote]:
    """Drop the lowest-ranked notes until the total fits in *max_tokens*."""
    kept: list[CompactNote] = []
    used = 0
    for note in sorted(notes, key=lambda n: n.rank):
        cost = estimate_tokens(note.to_rendered().render())
        if used + cost > max_tokens:
            continue
        kept.append(note)
        used += cost
    return kept


def notes_under_budget(
    notes: Iterable[CompactNote], config: WebConfig, max_tokens: int
) -> list[CompactNote]:
    ranked = rank_notes(notes, prefer_official=config.prefer_official_docs)
    return enforce_web_budget(ranked, max_tokens)
