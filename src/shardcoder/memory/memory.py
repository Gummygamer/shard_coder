"""Sliding-window agent memory.

Active memory is kept small and feeds into the model context only as a
compact rendered string. Older entries are summarised down to a single
line each, so the memory section never balloons.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable

from ..tokens.budget import cap_text


@dataclass
class MemoryEntry:
    user_request: str
    subtask: str
    files_touched: list[str] = field(default_factory=list)
    patch_summary: str = ""
    validation: str = ""
    remaining_issues: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    def render_short(self) -> str:
        files = ", ".join(self.files_touched) or "(none)"
        return (
            f"[{self.timestamp}] {self.subtask} | files: {files} | "
            f"validation: {self.validation or 'n/a'}"
        )

    def render_long(self) -> str:
        files = ", ".join(self.files_touched) or "(none)"
        out = [
            f"[{self.timestamp}] subtask: {self.subtask}",
            f"  files: {files}",
        ]
        if self.patch_summary:
            out.append(f"  patch: {cap_text(self.patch_summary, 80)}")
        if self.validation:
            out.append(f"  validation: {cap_text(self.validation, 80)}")
        if self.remaining_issues:
            out.append(f"  remaining: {cap_text(self.remaining_issues, 80)}")
        return "\n".join(out)


class SlidingMemory:
    """Hold a fixed-length window of recent steps plus a long-tail summary."""

    def __init__(self, window_size: int = 5) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.window_size = window_size
        self._entries: deque[MemoryEntry] = deque(maxlen=window_size)
        self._archive_summary: str = ""

    def add(self, entry: MemoryEntry) -> None:
        if len(self._entries) == self.window_size:
            evicted = self._entries[0]
            self._archive_summary = self._fold(self._archive_summary, evicted)
        self._entries.append(entry)

    @staticmethod
    def _fold(prev_summary: str, evicted: MemoryEntry) -> str:
        line = evicted.render_short()
        if not prev_summary:
            return line
        combined = f"{prev_summary}\n{line}"
        return cap_text(combined, 200)

    def recent(self) -> Iterable[MemoryEntry]:
        return list(self._entries)

    def archive(self) -> str:
        return self._archive_summary

    def render(self, max_tokens: int) -> str:
        if not self._entries and not self._archive_summary:
            return ""
        parts: list[str] = []
        if self._archive_summary:
            parts.append("Archive:\n" + self._archive_summary)
        if self._entries:
            recent_text = "\n".join(e.render_long() for e in self._entries)
            parts.append("Recent:\n" + recent_text)
        return cap_text("\n\n".join(parts), max_tokens)
