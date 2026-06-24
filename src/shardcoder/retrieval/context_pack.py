"""Assemble a budget-aware context pack for the local model."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from ..config import ContextConfig
from ..tokens.budget import BudgetManager, cap_text, estimate_tokens
from .ranking import Candidate


@dataclass
class Snippet:
    path: str
    start_line: int
    end_line: int
    text: str
    rank: int = 0

    def render(self) -> str:
        header = f"--- path: {self.path} lines {self.start_line}-{self.end_line} ---"
        return f"{header}\n{self.text}"


@dataclass
class WebNoteRendered:
    source_title: str
    source_url: str
    date_accessed: str
    relevance: str
    facts: list[str]
    rank: int = 0

    def render(self) -> str:
        bullet_facts = "\n    - " + "\n    - ".join(self.facts) if self.facts else ""
        return (
            f"- source_title: {self.source_title}\n"
            f"  source_url: {self.source_url}\n"
            f"  date_accessed: {self.date_accessed}\n"
            f"  relevance: {self.relevance}\n"
            f"  facts:{bullet_facts}"
        )


@dataclass
class ContextPack:
    task: str
    constraints: list[str] = field(default_factory=list)
    summaries: list[Candidate] = field(default_factory=list)
    snippets: list[Snippet] = field(default_factory=list)
    web_notes: list[WebNoteRendered] = field(default_factory=list)
    memory: str = ""
    validation: str = ""
    used_tokens: int = 0
    max_tokens: int = 0
    pruned_sections: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self) -> str:
        sections: list[str] = []
        sections.append(f"TASK:\n{self.task}")
        if self.constraints:
            constraints_text = "\n".join(f"- {c}" for c in self.constraints)
            sections.append(f"CONSTRAINTS:\n{constraints_text}")
        if self.summaries:
            lines = ["RELEVANT SUMMARIES:"]
            for c in self.summaries:
                lines.append(f"- path: {c.path}")
                if c.summary_text:
                    lines.append(f"  purpose: {cap_text(c.summary_text, 120)}")
                lines.append(f"  reason: {c.reason} (score={c.score:.2f})")
            sections.append("\n".join(lines))
        if self.snippets:
            lines = ["SNIPPETS:"]
            for snippet in self.snippets:
                lines.append(snippet.render())
            sections.append("\n".join(lines))
        if self.web_notes:
            lines = ["WEB NOTES:"]
            for note in self.web_notes:
                lines.append(note.render())
            sections.append("\n".join(lines))
        if self.memory:
            sections.append(f"RECENT MEMORY:\n{self.memory}")
        if self.validation:
            sections.append(f"VALIDATION:\n{self.validation}")
        return "\n\n".join(sections)


def _read_snippet(repo_root: Path, path: str, start: int, end: int) -> Snippet | None:
    try:
        full = (repo_root / path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    lines = full.splitlines()
    if not lines:
        return None
    start = max(1, start)
    end = min(len(lines), end)
    if end < start:
        return None
    excerpt = "\n".join(lines[start - 1 : end])
    return Snippet(path=path, start_line=start, end_line=end, text=excerpt)


def _select_snippet_for_candidate(
    repo_root: Path,
    candidate: Candidate,
    max_lines: int,
) -> Snippet | None:
    snippet = _read_snippet(repo_root, candidate.path, 1, max_lines)
    return snippet


def build_context_pack(
    *,
    task: str,
    candidates: Iterable[Candidate],
    repo_root: Path,
    config: ContextConfig,
    constraints: list[str] | None = None,
    web_notes: list[WebNoteRendered] | None = None,
    memory: str = "",
    validation: str = "",
    max_snippet_lines: int = 80,
    max_tokens: int | None = None,
) -> ContextPack:
    """Build a context pack constrained to the configured budgets.

    The token budgets are applied per section first (summaries, snippets,
    web notes, memory, validation), then the *full* pack is run through a
    :class:`BudgetManager` to enforce a global cap.
    """
    candidates = list(candidates)

    # Build summaries (top-N by score, capped to the summaries token budget).
    summaries_kept: list[Candidate] = []
    summaries_used = 0
    for c in candidates:
        line = f"- path: {c.path}\n  purpose: {cap_text(c.summary_text, 120)}\n  reason: {c.reason}"
        cost = estimate_tokens(line)
        if summaries_used + cost > config.max_summary_tokens:
            break
        summaries_kept.append(c)
        summaries_used += cost

    # Read snippets for the highest-ranked summaries within snippet budget.
    snippets_kept: list[Snippet] = []
    snippets_used = 0
    for rank, c in enumerate(summaries_kept):
        snippet = _select_snippet_for_candidate(repo_root, c, max_snippet_lines)
        if snippet is None:
            continue
        snippet.rank = rank
        cost = estimate_tokens(snippet.render())
        if snippets_used + cost > config.max_snippet_tokens:
            break
        snippets_kept.append(snippet)
        snippets_used += cost

    # Compress web notes within the web budget.
    web_notes = web_notes or []
    web_kept: list[WebNoteRendered] = []
    web_used = 0
    for rank, note in enumerate(sorted(web_notes, key=lambda n: n.rank)):
        cost = estimate_tokens(note.render())
        if web_used + cost > config.max_web_context_tokens:
            break
        web_kept.append(note)
        web_used += cost

    # Truncate memory + validation to their per-section budgets.
    memory_text = cap_text(memory, config.max_memory_tokens) if memory else ""
    validation_text = (
        cap_text(validation, config.max_validation_tokens) if validation else ""
    )

    pack = ContextPack(
        task=task,
        constraints=constraints or [],
        summaries=summaries_kept,
        snippets=snippets_kept,
        web_notes=web_kept,
        memory=memory_text,
        validation=validation_text,
    )

    # Apply a global cap using BudgetManager so that critical sections (task,
    # constraints) stay reserved.
    configured_budget = (
        config.max_summary_tokens
        + config.max_snippet_tokens
        + config.max_web_context_tokens
        + config.max_memory_tokens
        + config.max_validation_tokens
        + 256  # task + constraints headroom
    )
    budget = BudgetManager(max_tokens=max_tokens or configured_budget)
    budget.add("task", f"TASK:\n{task}", reserved=True)
    if pack.constraints:
        budget.add(
            "instructions",
            "CONSTRAINTS:\n" + "\n".join(f"- {c}" for c in pack.constraints),
            reserved=True,
        )
    if pack.summaries:
        budget.add(
            "summaries",
            "\n".join(
                [
                    "RELEVANT SUMMARIES:",
                    *[
                        f"- path: {c.path}\n  purpose: {cap_text(c.summary_text, 120)}\n  reason: {c.reason}"
                        for c in pack.summaries
                    ],
                ]
            ),
        )
    if pack.snippets:
        budget.add(
            "snippets",
            "SNIPPETS:\n" + "\n".join(s.render() for s in pack.snippets),
        )
    if pack.web_notes:
        budget.add(
            "external_notes",
            "WEB NOTES:\n" + "\n".join(n.render() for n in pack.web_notes),
        )
    if pack.memory:
        budget.add("memory", f"RECENT MEMORY:\n{pack.memory}")
    if pack.validation:
        budget.add("validation", f"VALIDATION:\n{pack.validation}")

    report = budget.assemble()
    pack.used_tokens = report.used_tokens
    pack.max_tokens = report.max_tokens
    pack.pruned_sections = report.pruned

    if "snippets" in report.pruned:
        pack.snippets = []
    if "summaries" in report.pruned:
        pack.summaries = []
    if "external_notes" in report.pruned:
        pack.web_notes = []
    if "memory" in report.pruned:
        pack.memory = ""
    if "validation" in report.pruned:
        pack.validation = ""

    return pack
