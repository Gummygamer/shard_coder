"""Heuristic ranking helpers for retrieval candidates."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]+|[A-Za-z]+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def keyword_score(query_tokens: list[str], target_text: str) -> float:
    if not query_tokens:
        return 0.0
    target_tokens = set(tokenize(target_text))
    if not target_tokens:
        return 0.0
    hits = sum(1 for t in query_tokens if t in target_tokens)
    return hits / len(query_tokens)


def path_score(query_tokens: list[str], path: str) -> float:
    if not query_tokens:
        return 0.0
    pieces = re.split(r"[/\\._-]", path.lower())
    pieces = [p for p in pieces if p]
    if not pieces:
        return 0.0
    matches = sum(1 for t in query_tokens if any(t == p or t in p for p in pieces))
    return matches / len(query_tokens)


@dataclass
class Candidate:
    path: str
    score: float
    reason: str
    summary_text: str = ""

    def label(self) -> str:
        return f"{Path(self.path).name} ({self.path})"
