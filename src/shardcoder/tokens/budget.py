"""Token budget management for compact context construction.

Token counts are estimated heuristically (about four characters per token);
this is sufficient for budgeting since model context windows are not the
bottleneck — prompt noise is. The :class:`BudgetManager` keeps a section
table of the form::

    {
        "instructions": (priority, text, reserved),
        "task":          (priority, text, reserved),
        ...
    }

and exposes :meth:`assemble`, which prunes lower-priority sections first
until the total token estimate fits inside ``max_tokens``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


# Approximate "4 chars per token" heuristic — fine for budgeting.
_CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Rough heuristic that does not require a tokenizer."""
    if not text:
        return 0
    return max(1, (len(text) + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN)


# Lower priority numbers are pruned first.
DEFAULT_PRIORITIES: dict[str, int] = {
    "memory": 10,
    "external_notes_extra": 20,
    "snippets_extra": 30,
    "summaries_verbose": 40,
    "validation_old": 50,
    "comments": 55,
    "imports_unrelated": 58,
    "external_notes": 60,
    "snippets": 70,
    "summaries": 80,
    "validation": 85,
    "task": 95,
    "instructions": 100,
    "schema": 100,
}


@dataclass
class BudgetSection:
    name: str
    text: str
    priority: int
    reserved: bool = False
    tokens: int = field(init=False)

    def __post_init__(self) -> None:
        self.tokens = estimate_tokens(self.text)

    def update_text(self, text: str) -> None:
        self.text = text
        self.tokens = estimate_tokens(text)


@dataclass
class BudgetReport:
    used_tokens: int
    max_tokens: int
    sections: list[BudgetSection]
    pruned: list[str]

    @property
    def fits(self) -> bool:
        return self.used_tokens <= self.max_tokens


class BudgetManager:
    """Hold sections of context text and assemble them within a budget."""

    def __init__(self, max_tokens: int) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.max_tokens = max_tokens
        self._sections: dict[str, BudgetSection] = {}

    # ------------------------------------------------------------------
    # Section management
    # ------------------------------------------------------------------

    def add(
        self,
        name: str,
        text: str,
        priority: int | None = None,
        reserved: bool = False,
    ) -> BudgetSection:
        """Add or replace a section. Reserved sections are never pruned."""
        if priority is None:
            priority = DEFAULT_PRIORITIES.get(name, 50)
        section = BudgetSection(
            name=name, text=text, priority=priority, reserved=reserved
        )
        self._sections[name] = section
        return section

    def update(self, name: str, text: str) -> None:
        if name not in self._sections:
            raise KeyError(name)
        self._sections[name].update_text(text)

    def remove(self, name: str) -> None:
        self._sections.pop(name, None)

    def total_tokens(self) -> int:
        return sum(section.tokens for section in self._sections.values())

    def sections(self) -> Iterable[BudgetSection]:
        return self._sections.values()

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    def assemble(self) -> BudgetReport:
        """Return a :class:`BudgetReport` after pruning to fit the budget.

        Pruning order:
            1. Drop non-reserved, lowest-priority sections entirely until
               the budget fits.
            2. If reserved sections alone overflow the budget, return the
               report unchanged with ``fits=False`` so the caller can react.
        """
        kept = list(self._sections.values())
        pruned: list[str] = []
        kept.sort(key=lambda s: (-s.priority, s.name))

        total = sum(section.tokens for section in kept)
        if total <= self.max_tokens:
            return BudgetReport(total, self.max_tokens, kept, pruned)

        # Identify pruning candidates (low priority -> high priority).
        prunable = sorted(
            [s for s in kept if not s.reserved],
            key=lambda s: (s.priority, s.name),
        )

        for section in prunable:
            if total <= self.max_tokens:
                break
            kept.remove(section)
            pruned.append(section.name)
            total -= section.tokens

        kept.sort(key=lambda s: (-s.priority, s.name))
        return BudgetReport(total, self.max_tokens, kept, pruned)

    def render(self, separator: str = "\n\n") -> str:
        """Return the assembled, in-budget text."""
        report = self.assemble()
        return separator.join(s.text for s in report.sections if s.text)


def cap_text(text: str, max_tokens: int) -> str:
    """Truncate *text* to roughly *max_tokens* tokens, preserving the head."""
    if max_tokens <= 0:
        return ""
    max_chars = max_tokens * _CHARS_PER_TOKEN
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 12)] + "\n... [truncated]"
