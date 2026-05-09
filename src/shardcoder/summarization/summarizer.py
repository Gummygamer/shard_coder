"""Compact summarisation of source files for retrieval."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..indexing.scanner import FileRecord, detect_language
from ..indexing.symbols import SymbolReport, extract_symbols
from ..llm.client import ChatMessage, LLMClient, LLMError
from ..prompting.templates import file_summary_prompt
from ..tokens.budget import cap_text


@dataclass
class FileSummary:
    path: str
    language: str
    file_purpose: str
    symbols: list[dict[str, Any]]
    dependencies: list[str]
    side_effects: list[str]
    likely_edit_points: list[str]
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "language": self.language,
            "file_purpose": self.file_purpose,
            "symbols": self.symbols,
            "dependencies": self.dependencies,
            "side_effects": self.side_effects,
            "likely_edit_points": self.likely_edit_points,
            "hash": self.sha256,
        }


def deterministic_summary(record: FileRecord, report: SymbolReport) -> FileSummary:
    """Build a summary from indexer facts only — never hallucinated."""
    likely = [
        s.name
        for s in report.symbols
        if s.kind in {"function", "method", "class"}
    ][:10]

    purpose = _heuristic_purpose(record, report)
    return FileSummary(
        path=record.path,
        language=record.language,
        file_purpose=purpose,
        symbols=[
            {
                "name": s.name,
                "kind": s.kind,
                "lines": [s.start_line, s.end_line],
                "signature": s.signature,
            }
            for s in report.symbols
        ],
        dependencies=list(dict.fromkeys(report.imports)),
        side_effects=[],
        likely_edit_points=likely,
        sha256=record.sha256,
    )


def _heuristic_purpose(record: FileRecord, report: SymbolReport) -> str:
    name = Path(record.path).stem
    if report.symbols:
        kinds = {s.kind for s in report.symbols}
        head = sorted(kinds)
        return f"{name}: contains {', '.join(head)} ({len(report.symbols)} symbols)"
    return f"{name}: {record.language} file"


def summarize_file(
    record: FileRecord,
    *,
    llm: LLMClient | None = None,
    max_input_tokens: int = 1500,
) -> FileSummary:
    """Summarise a file. If *llm* is None, use the deterministic fallback."""
    source = record.text()
    report = extract_symbols(source, record.language)

    base = deterministic_summary(record, report)
    if llm is None:
        return base

    capped_source = cap_text(source, max_input_tokens)
    prompt = file_summary_prompt(record.path, record.language, capped_source)

    try:
        result = llm.chat(
            [
                ChatMessage(
                    role="system",
                    content=(
                        "You are a code summariser for a small-context coding agent. "
                        "Always respond with valid JSON only."
                    ),
                ),
                ChatMessage(role="user", content=prompt),
            ],
            temperature=0.0,
            max_tokens=600,
        )
    except LLMError:
        return base

    parsed = _safe_json_parse(result.text)
    if not parsed:
        return base

    return FileSummary(
        path=record.path,
        language=record.language,
        file_purpose=str(parsed.get("file_purpose", base.file_purpose))[:300],
        symbols=base.symbols if not parsed.get("symbols") else parsed["symbols"],
        dependencies=_listify(parsed.get("dependencies"), fallback=base.dependencies),
        side_effects=_listify(parsed.get("side_effects"), fallback=base.side_effects),
        likely_edit_points=_listify(
            parsed.get("likely_edit_points"), fallback=base.likely_edit_points
        ),
        sha256=record.sha256,
    )


def _listify(value: Any, fallback: list[str]) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value][:25]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return fallback


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.S)


def _safe_json_parse(text: str) -> dict[str, Any] | None:
    """Be tolerant of small models wrapping JSON in prose or fences."""
    if not text:
        return None
    match = _JSON_BLOCK_RE.search(text)
    if not match:
        return None
    snippet = match.group(0)
    try:
        data = json.loads(snippet)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None
