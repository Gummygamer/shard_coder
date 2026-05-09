"""Pydantic schemas used to validate small-model outputs."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field, ValidationError


class Subtask(BaseModel):
    id: str
    goal: str
    reason: str = ""
    search_queries: list[str] = Field(default_factory=list)
    likely_files: list[str] = Field(default_factory=list)
    edit_scope: str = "small"
    validation: str = ""
    needs_external_docs: bool = False
    external_doc_reason: str = ""


class Plan(BaseModel):
    task: str
    assumptions: list[str] = Field(default_factory=list)
    subtasks: list[Subtask] = Field(default_factory=list)
    stop_conditions: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)


class WebQueryList(BaseModel):
    queries: list[str] = Field(default_factory=list)


class WebNote(BaseModel):
    source_title: str
    source_url: str
    date_accessed: str
    relevance: str = ""
    facts: list[str] = Field(default_factory=list)


class FailureAnalysis(BaseModel):
    failing_files: list[str] = Field(default_factory=list)
    failing_lines: list[dict[str, Any]] = Field(default_factory=list)
    root_cause: str = ""
    is_dependency_or_version_issue: bool = False


_JSON_RE = re.compile(r"\{.*\}", re.S)


def extract_json(text: str) -> dict[str, Any] | None:
    """Pull the first JSON object out of small-model output."""
    if not text:
        return None
    # Strip code fences if present.
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    match = _JSON_RE.search(text)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return data


def parse_plan(text: str) -> Plan | None:
    raw = extract_json(text)
    if raw is None:
        return None
    try:
        return Plan.model_validate(raw)
    except ValidationError:
        return None


def parse_web_queries(text: str) -> WebQueryList | None:
    raw = extract_json(text)
    if raw is None:
        return None
    try:
        return WebQueryList.model_validate(raw)
    except ValidationError:
        return None


def parse_failure_analysis(text: str) -> FailureAnalysis | None:
    raw = extract_json(text)
    if raw is None:
        return None
    try:
        return FailureAnalysis.model_validate(raw)
    except ValidationError:
        return None
