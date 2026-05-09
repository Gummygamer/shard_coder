"""Hybrid retrieval over the SQLite metadata store.

Retrieval combines several deterministic heuristics:

* path tokens that match the query
* symbol names that match the query
* keyword matches against summary text
* import / dependency graph (files importing matching modules)
* explicit "likely_files" hints from the planner
* previously-touched files (memory)
* test failure paths

It returns ranked :class:`Candidate` objects which the context-pack
assembler then turns into snippets within a strict token budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from ..summarization.store import StoredSummary, SummaryStore
from .ranking import Candidate, keyword_score, path_score, tokenize


@dataclass
class RetrievalRequest:
    task: str
    queries: list[str] = field(default_factory=list)
    likely_files: list[str] = field(default_factory=list)
    touched_files: list[str] = field(default_factory=list)
    failing_files: list[str] = field(default_factory=list)
    max_candidates: int = 20


def _summary_text(summary: StoredSummary) -> str:
    s = summary.summary
    parts = [
        s.get("file_purpose", ""),
        " ".join(s.get("dependencies", []) or []),
        " ".join(s.get("likely_edit_points", []) or []),
        " ".join(
            sym.get("name", "") + " " + sym.get("signature", "")
            for sym in s.get("symbols", []) or []
        ),
    ]
    return " ".join(p for p in parts if p)


def retrieve(
    store: SummaryStore,
    request: RetrievalRequest,
) -> list[Candidate]:
    """Return ranked candidate files for the request."""
    summaries = {s.path: s for s in store.all_summaries()}
    if not summaries:
        # Fall back to all known files from the indexer.
        summaries = {}
        for row in store.all_files():
            summaries[row["path"]] = StoredSummary(
                path=row["path"],
                sha256=row["sha256"],
                summary={
                    "path": row["path"],
                    "language": row["language"],
                    "file_purpose": "",
                    "symbols": [],
                    "dependencies": store.imports_for(row["path"]),
                    "side_effects": [],
                    "likely_edit_points": [],
                    "hash": row["sha256"],
                },
            )

    query_text = " ".join([request.task, *request.queries])
    query_tokens = tokenize(query_text)

    candidates: dict[str, Candidate] = {}

    def _bump(path: str, score: float, reason: str, summary_text: str) -> None:
        existing = candidates.get(path)
        if existing is None:
            candidates[path] = Candidate(
                path=path, score=score, reason=reason, summary_text=summary_text
            )
        else:
            existing.score += score
            if reason and reason not in existing.reason:
                existing.reason = f"{existing.reason}; {reason}"

    # Always-include: explicit hints, recent memory, failing files.
    boosts: list[tuple[Iterable[str], float, str]] = [
        (request.likely_files, 1.5, "planner-likely"),
        (request.failing_files, 1.8, "validation-failure"),
        (request.touched_files, 0.6, "recent-memory"),
    ]
    for paths, boost, reason in boosts:
        for path in paths:
            if path in summaries:
                _bump(path, boost, reason, _summary_text(summaries[path]))

    # Symbol-name matches.
    for token in {t for t in query_tokens if len(t) >= 3}:
        for row in store.search_symbols(token, limit=50):
            path = row["path"]
            if path in summaries:
                _bump(path, 0.6, f"symbol:{row['name']}", _summary_text(summaries[path]))

    # Import-graph matches.
    for token in {t for t in query_tokens if len(t) >= 3}:
        for path in store.files_importing(token):
            if path in summaries:
                _bump(path, 0.3, f"imports:{token}", _summary_text(summaries[path]))

    # Path + keyword matches across summaries.
    for path, summary in summaries.items():
        text = _summary_text(summary)
        ks = keyword_score(query_tokens, text)
        ps = path_score(query_tokens, path)
        score = ks * 1.0 + ps * 0.8
        if score > 0:
            _bump(path, score, "keyword", text)

    ranked = sorted(candidates.values(), key=lambda c: c.score, reverse=True)
    return ranked[: request.max_candidates]


def queries_from_task(task: str, hint_tokens: Sequence[str] = ()) -> list[str]:
    """Heuristically generate retrieval queries from the user task."""
    base = list(dict.fromkeys(tokenize(task)))
    base = [t for t in base if len(t) >= 3]
    queries = base[:6]
    for hint in hint_tokens:
        if hint not in queries:
            queries.append(hint)
    return queries
