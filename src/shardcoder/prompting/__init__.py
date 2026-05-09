"""Structured prompts for small-context LLMs."""

from .schemas import (
    FailureAnalysis,
    Plan,
    Subtask,
    WebNote,
    WebQueryList,
    extract_json,
    parse_failure_analysis,
    parse_plan,
    parse_web_queries,
)
from .templates import (
    SYSTEM_PATCH,
    external_query_prompt,
    external_summary_prompt,
    failure_analysis_prompt,
    file_summary_prompt,
    final_summary_prompt,
    patch_prompt,
    repair_prompt,
    retrieval_query_prompt,
    task_decomposition_prompt,
)

__all__ = [
    "FailureAnalysis",
    "Plan",
    "SYSTEM_PATCH",
    "Subtask",
    "WebNote",
    "WebQueryList",
    "external_query_prompt",
    "external_summary_prompt",
    "extract_json",
    "failure_analysis_prompt",
    "file_summary_prompt",
    "final_summary_prompt",
    "parse_failure_analysis",
    "parse_plan",
    "parse_web_queries",
    "patch_prompt",
    "repair_prompt",
    "retrieval_query_prompt",
    "task_decomposition_prompt",
]
