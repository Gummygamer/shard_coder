"""Prompt templates tuned for small local models.

Design rules:
1. Be short.
2. Use numbered instructions.
3. Use explicit output schemas.
4. Avoid broad open-ended requests.
5. Include only the relevant context.
6. Repeat the current sub-task at the end.
7. Tell the model what *not* to do.
8. Provide validation criteria.
9. Prefer "the smallest correct change."
10. Prefer ``NO_PATCH: <reason>`` over hallucination.
"""

from __future__ import annotations


SYSTEM_PATCH = (
    "You are a careful, concise coding assistant for a small local model. "
    "You must follow the rules exactly and respond in the requested format only."
)


def file_summary_prompt(path: str, language: str, code: str) -> str:
    return f"""Summarise this code for a small-context coding agent.

Rules:
1. Be factual.
2. Be compact.
3. Mention symbols, dependencies, side effects, and likely edit points.
4. Do not include unnecessary prose.
5. Output valid JSON only. No prose. No code fences.

File: {path}
Language: {language}

Code:
{code}

JSON schema:
{{
  "path": "{path}",
  "language": "{language}",
  "file_purpose": "<one sentence>",
  "symbols": [
    {{"name": "...", "kind": "function|class|method|const", "lines": [start, end], "signature": "..."}}
  ],
  "dependencies": ["..."],
  "side_effects": ["..."],
  "likely_edit_points": ["..."]
}}
"""


def task_decomposition_prompt(task: str, repo_summary: str) -> str:
    return f"""Decompose the user task into small atomic subtasks.

Rules:
1. Prefer 1 to 5 subtasks.
2. Every subtask must be small and testable.
3. Output valid JSON only. No prose. No code fences.
4. If the task is trivial, return a single subtask.
5. Set "needs_external_docs" to true only if the task depends on
   current external API behaviour.

Repository summary:
{repo_summary}

User task:
{task}

JSON schema:
{{
  "task": "...",
  "assumptions": ["..."],
  "subtasks": [
    {{
      "id": "T1",
      "goal": "...",
      "reason": "...",
      "search_queries": ["..."],
      "likely_files": ["..."],
      "edit_scope": "small|medium|large",
      "validation": "...",
      "needs_external_docs": false,
      "external_doc_reason": ""
    }}
  ],
  "stop_conditions": ["..."],
  "risks": ["..."]
}}

Repeat the user task: {task}
"""


def retrieval_query_prompt(task: str, summaries: str) -> str:
    return f"""Suggest retrieval queries for the task.

Rules:
1. Output JSON only.
2. At most 5 queries.
3. Queries should be short keywords or symbol names.
4. Prefer concrete identifiers from the repository summaries.

Repository summaries:
{summaries}

Task:
{task}

JSON schema:
{{"queries": ["..."]}}
"""


def external_query_prompt(task: str, reason: str) -> str:
    return f"""Generate at most 3 precise web search queries.

Rules:
1. Output JSON only.
2. Queries must target official documentation when possible.
3. Each query must be self-contained.
4. No prose.

Reason external docs are needed:
{reason}

Task:
{task}

JSON schema:
{{"queries": ["..."]}}
"""


def external_summary_prompt(source_title: str, source_url: str, snippet: str) -> str:
    return f"""Compress the following web snippet into compact factual notes
for a small-context coding agent.

Rules:
1. Output valid JSON only.
2. Each fact must be a short sentence.
3. Drop opinion, marketing copy, and example boilerplate.
4. At most 5 facts.

Source title: {source_title}
Source URL: {source_url}

Snippet:
{snippet}

JSON schema:
{{
  "source_title": "{source_title}",
  "source_url": "{source_url}",
  "relevance": "<one sentence>",
  "facts": ["..."]
}}
"""


def patch_prompt(subtask: str, context_pack: str, validation_command: str) -> str:
    return f"""You are editing or creating code.

Goal:
{subtask}

Rules:
1. Make the smallest correct change.
2. Do not rewrite unrelated code.
3. Do not invent external APIs not shown in the context.
4. For edits to existing files, use only the provided snippets/context.
   For new file creation (files that do not yet exist), write complete working code.
5. Return NO_PATCH: <reason> only if you cannot identify which existing code to change.
   Do NOT return NO_PATCH for tasks that require creating new files.
6. Output only a unified diff or NO_PATCH.
7. Do not include prose before or after the diff.
8. Diff headers must use "--- a/<path>" and "+++ b/<path>".
   New files use "--- /dev/null" and "+++ b/<path>".

Relevant context:
{context_pack}

Validation command:
{validation_command}

Repeat goal:
{subtask}

Output:
Unified diff only, or NO_PATCH: <reason>.
"""


def repair_prompt(
    subtask: str,
    context_pack: str,
    failure_summary: str,
    previous_patch: str,
) -> str:
    return f"""Your previous patch failed. Propose a focused repair patch.

Rules:
1. Output a unified diff only, or NO_PATCH: <reason>.
2. The diff must address the failure shown below.
3. Do not modify unrelated files.
4. Prefer the smallest correct change.
5. If the failure is unclear from the context, return NO_PATCH.

Sub-task:
{subtask}

Previous patch:
{previous_patch}

Failure summary:
{failure_summary}

Context:
{context_pack}

Repeat sub-task: {subtask}

Output:
Unified diff only, or NO_PATCH: <reason>.
"""


def failure_analysis_prompt(failure_log: str) -> str:
    return f"""Summarise this validation failure for a small-context coding agent.

Rules:
1. Output JSON only.
2. Identify failing files and lines if possible.
3. Identify the apparent root cause in <= 2 sentences.

Failure log (truncated):
{failure_log}

JSON schema:
{{
  "failing_files": ["..."],
  "failing_lines": [{{"path": "...", "line": 0}}],
  "root_cause": "...",
  "is_dependency_or_version_issue": false
}}
"""


def final_summary_prompt(
    task: str,
    files_changed: list[str],
    validation_status: str,
    used_web: bool,
) -> str:
    files = "\n".join(f"- {p}" for p in files_changed) or "(none)"
    return f"""Write a 2-3 sentence final report.

Task: {task}

Files changed:
{files}

Validation status: {validation_status}
Used external docs: {used_web}

Rules:
1. Output plain text.
2. No marketing language.
3. Mention any remaining work explicitly.
"""
