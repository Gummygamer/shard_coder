"""High-level orchestration for ShardCoder.

The agent ties everything together:

* config + LLM client
* indexing and summarisation
* retrieval and context-pack assembly
* planning + multi-subtask execution
* patch generation, validation, application
* validation command + iterative repair
* memory updates and final report

``Agent.run_edit`` walks every subtask of the plan in order, folding
each outcome into :class:`SlidingMemory` so later subtasks can see the
patches the earlier ones produced. The walk stops early when a subtask
returns ``NO_PATCH`` or two subtasks in a row hard-fail validation.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .config import ShardCoderConfig, load_config
from .editing.patcher import (
    PatchApplyResult,
    PatchValidationResult,
    apply_validated_patch,
    validate_model_output,
)
from .execution.runner import CommandResult, run_command, summarize_result
from .external_context.docs_retriever import (
    ExternalContextResult,
    retrieve_external_context,
)
from .indexing.scanner import scan_repository
from .indexing.symbols import extract_symbols
from .llm.client import ChatMessage, LLMClient, LLMError
from .llm.openai_compatible import OpenAICompatibleClient
from .memory.memory import MemoryEntry, SlidingMemory
from .planning.planner import make_plan
from .prompting.schemas import Subtask, parse_failure_analysis
from .prompting.templates import (
    SYSTEM_PATCH,
    failure_analysis_prompt,
    patch_prompt,
    repair_prompt,
)
from .retrieval.context_pack import ContextPack, build_context_pack
from .retrieval.retriever import RetrievalRequest, queries_from_task, retrieve
from .safety.commands import assess_command
from .summarization.store import SummaryStore
from .summarization.summarizer import FileSummary, summarize_file


DEFAULT_DB_DIR = ".shardcoder"
DEFAULT_DB_NAME = "shardcoder.db"


@dataclass
class SubtaskOutcome:
    subtask: Subtask
    context_pack: ContextPack | None = None
    patch_text: str = ""
    validation: PatchValidationResult | None = None
    apply_result: PatchApplyResult | None = None
    command_result: CommandResult | None = None
    iterations: int = 0
    no_patch_reason: str = ""
    used_external_docs: bool = False
    external_sources: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class AgentReport:
    task: str
    plan_used_fallback: bool
    outcomes: list[SubtaskOutcome] = field(default_factory=list)
    files_changed: list[str] = field(default_factory=list)
    used_external_docs: bool = False
    external_sources: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    final_validation: CommandResult | None = None
    stop_reason: str = ""


class Agent:
    """Orchestrates an end-to-end task."""

    def __init__(
        self,
        repo_root: str | Path,
        config: ShardCoderConfig | None = None,
        llm: LLMClient | None = None,
        memory: SlidingMemory | None = None,
        db_path: str | Path | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.config = config or load_config()
        self.llm = llm  # may be None for CLI commands that do not need it
        self.memory = memory or SlidingMemory()
        if db_path is None:
            db_path = self.repo_root / DEFAULT_DB_DIR / DEFAULT_DB_NAME
        self.store = SummaryStore(db_path)

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

    @classmethod
    def with_default_llm(cls, repo_root: str | Path, config: ShardCoderConfig | None = None) -> "Agent":
        config = config or load_config()
        llm = OpenAICompatibleClient(config.llm)
        return cls(repo_root, config=config, llm=llm)

    # ------------------------------------------------------------------
    # Indexing + summarisation
    # ------------------------------------------------------------------

    def index(self, *, only_changed: bool = True) -> tuple[int, int]:
        """Walk the repo and refresh metadata. Returns (added_or_updated, skipped)."""
        added, skipped = 0, 0
        for record in scan_repository(self.repo_root, self.config.repository):
            existing = self.store.get_file_hash(record.path)
            if only_changed and existing == record.sha256:
                skipped += 1
                continue
            self.store.upsert_file(
                path=record.path,
                language=record.language,
                size=record.size,
                sha256=record.sha256,
            )
            try:
                source = record.text()
            except OSError:
                continue
            report = extract_symbols(source, record.language)
            self.store.replace_imports(record.path, report.imports)
            self.store.replace_symbols(
                record.path,
                [
                    {
                        "name": s.name,
                        "kind": s.kind,
                        "start_line": s.start_line,
                        "end_line": s.end_line,
                        "signature": s.signature,
                    }
                    for s in report.symbols
                ],
            )
            added += 1
        return added, skipped

    def summarize(self, *, use_llm: bool = True) -> tuple[int, int]:
        """Summarise files, skipping unchanged content. Returns (updated, skipped)."""
        updated, skipped = 0, 0
        for record in scan_repository(self.repo_root, self.config.repository):
            existing_summary = self.store.get_summary(record.path)
            if existing_summary and existing_summary.sha256 == record.sha256:
                skipped += 1
                continue
            llm = self.llm if use_llm else None
            try:
                summary: FileSummary = summarize_file(record, llm=llm)
            except OSError:
                continue
            self.store.upsert_summary(record.path, record.sha256, summary.to_dict())
            self.store.upsert_file(
                path=record.path,
                language=record.language,
                size=record.size,
                sha256=record.sha256,
            )
            updated += 1
        return updated, skipped

    # ------------------------------------------------------------------
    # Planning + first subtask execution
    # ------------------------------------------------------------------

    def plan(self, task: str):
        repo_summary = self._tiny_repo_summary()
        return make_plan(task, repo_summary, llm=self.llm)

    def _tiny_repo_summary(self, max_files: int = 30) -> str:
        rows = self.store.all_files()[:max_files]
        if not rows:
            return "(repository not yet indexed)"
        parts = []
        for row in rows:
            parts.append(f"- {row['path']} ({row['language']})")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Public end-to-end run
    # ------------------------------------------------------------------

    def run_edit(
        self,
        task: str,
        *,
        web: bool = False,
        allow_dirty: bool = False,
        approve_fetch: Callable[[str], bool] = lambda _url: True,
        max_subtasks: int | None = None,
    ) -> AgentReport:
        """Execute every subtask of a plan in order, with early-stop guards.

        ``max_subtasks`` caps how many of the planner's subtasks are
        attempted; pass ``None`` for "all of them".
        """
        warnings: list[str] = []

        if not allow_dirty and self._git_is_dirty():
            warnings.append(
                "git working tree is dirty; rerun with --allow-dirty to override"
            )
            return AgentReport(task=task, plan_used_fallback=False, warnings=warnings)

        # Always refresh the index before editing.
        self.index(only_changed=True)

        planner_result = self.plan(task)
        plan = planner_result.plan
        if not plan.subtasks:
            warnings.append("plan contained no subtasks")
            return AgentReport(
                task=task,
                plan_used_fallback=planner_result.fallback_used,
                warnings=warnings,
            )

        report = AgentReport(
            task=task,
            plan_used_fallback=planner_result.fallback_used,
        )

        cap = len(plan.subtasks) if max_subtasks is None else max(max_subtasks, 0)
        consecutive_failures = 0
        for subtask in plan.subtasks[:cap]:
            outcome = self.run_subtask(
                subtask,
                web=web,
                approve_fetch=approve_fetch,
            )
            report.outcomes.append(outcome)
            if outcome.apply_result:
                for path in outcome.apply_result.changed_files:
                    if path not in report.files_changed:
                        report.files_changed.append(path)
            if outcome.used_external_docs:
                report.used_external_docs = True
            for src in outcome.external_sources:
                if src not in report.external_sources:
                    report.external_sources.append(src)
            if outcome.command_result is not None:
                report.final_validation = outcome.command_result

            self._update_memory(task, outcome)

            if outcome.no_patch_reason:
                report.stop_reason = (
                    f"NO_PATCH on subtask {subtask.id}: {outcome.no_patch_reason}"
                )
                warnings.append(report.stop_reason)
                break

            if self._subtask_hard_failed(outcome):
                consecutive_failures += 1
                if consecutive_failures >= 2:
                    report.stop_reason = (
                        f"two consecutive validation failures "
                        f"(last: subtask {subtask.id})"
                    )
                    warnings.append(report.stop_reason)
                    break
            else:
                consecutive_failures = 0

        if max_subtasks is not None and len(plan.subtasks) > cap and not report.stop_reason:
            warnings.append(
                f"--max-subtasks={max_subtasks} reached; "
                f"{len(plan.subtasks) - cap} subtask(s) skipped"
            )

        report.warnings.extend(warnings)
        return report

    def build_context_for(self, task: str) -> ContextPack:
        """Public helper: produce a context pack for an arbitrary question.

        Used by ``shardcoder ask`` and any caller that wants a context
        view without running the full edit loop. The task is wrapped in a
        single-subtask plan so retrieval gets the same shape it does
        during ``run_edit``.
        """
        planner_result = self.plan(task)
        first = (
            planner_result.plan.subtasks[0]
            if planner_result.plan.subtasks
            else Subtask(id="T1", goal=task, reason="ask helper", edit_scope="small")
        )
        return self._build_context_pack(
            subtask=first,
            failing_files=[],
            validation_text="",
            external=None,
        )

    @staticmethod
    def _subtask_hard_failed(outcome: SubtaskOutcome) -> bool:
        """True when the subtask exhausted its iterations without success."""
        if outcome.no_patch_reason:
            return False
        if outcome.validation is None or not outcome.validation.ok:
            return True
        if outcome.apply_result is not None and not outcome.apply_result.ok:
            return True
        if outcome.command_result is not None and not outcome.command_result.ok:
            return True
        return False

    # ------------------------------------------------------------------
    # Subtask execution (one iteration + repair loop)
    # ------------------------------------------------------------------

    def run_subtask(
        self,
        subtask: Subtask,
        *,
        web: bool = False,
        approve_fetch: Callable[[str], bool] = lambda _url: True,
    ) -> SubtaskOutcome:
        outcome = SubtaskOutcome(subtask=subtask)

        external_result: ExternalContextResult | None = None
        if web and (subtask.needs_external_docs or self.config.web.enabled):
            external_result = retrieve_external_context(
                subtask.goal,
                reason=subtask.external_doc_reason or "task-supplied --web",
                config=self.config.web,
                max_web_context_tokens=self.config.context.max_web_context_tokens,
                llm=self.llm,
                approve_fetch=approve_fetch,
            )
            if external_result.error:
                outcome.notes.append(f"external docs: {external_result.error}")
            outcome.used_external_docs = external_result.used and bool(
                external_result.notes
            )
            outcome.external_sources = external_result.sources

        context_pack = self._build_context_pack(
            subtask=subtask,
            failing_files=[],
            validation_text="",
            external=external_result,
        )
        outcome.context_pack = context_pack

        validation_command = (
            subtask.validation or self.config.validation.test_command or ""
        )

        for iteration in range(1, self.config.agent.max_iterations + 1):
            outcome.iterations = iteration
            patch_text = self._ask_for_patch(subtask, context_pack, validation_command)
            outcome.patch_text = patch_text

            validation_result = validate_model_output(
                patch_text,
                self.repo_root,
                repo_config=self.config.repository,
                expected_files=subtask.likely_files or None,
            )
            outcome.validation = validation_result

            if validation_result.is_no_patch:
                outcome.no_patch_reason = validation_result.reason
                break

            if not validation_result.ok:
                outcome.notes.append(
                    f"iteration {iteration}: patch rejected — {'; '.join(validation_result.errors)}"
                )
                # Feed the failure back as 'validation' for the next iteration.
                context_pack = self._build_context_pack(
                    subtask=subtask,
                    failing_files=[],
                    validation_text="\n".join(validation_result.errors),
                    external=external_result,
                )
                continue

            apply_result = apply_validated_patch(
                validation_result,
                self.repo_root,
                dry_run=self.config.agent.dry_run or not self.config.agent.auto_apply,
            )
            outcome.apply_result = apply_result
            if not apply_result.ok:
                outcome.notes.append(
                    f"iteration {iteration}: apply failed — {'; '.join(apply_result.errors)}"
                )
                break

            if (
                self.config.agent.auto_run_tests
                and validation_command
                and not self.config.agent.dry_run
            ):
                cmd_result = self._run_validation(validation_command)
                outcome.command_result = cmd_result
                if cmd_result.ok:
                    break
                # Build a focused failure context and retry.
                failure_summary = summarize_result(
                    cmd_result, max_tokens=self.config.context.max_validation_tokens
                )
                analysis = self._analyse_failure(failure_summary)
                failing_files = analysis.failing_files if analysis else []
                context_pack = self._build_context_pack(
                    subtask=subtask,
                    failing_files=failing_files or cmd_result.failing_paths,
                    validation_text=failure_summary,
                    external=external_result,
                )
                # Ask for a repair patch on the next iteration.
                subtask = Subtask(
                    id=subtask.id,
                    goal=subtask.goal,
                    reason=subtask.reason,
                    search_queries=subtask.search_queries,
                    likely_files=(failing_files or subtask.likely_files),
                    edit_scope=subtask.edit_scope,
                    validation=subtask.validation,
                    needs_external_docs=subtask.needs_external_docs,
                    external_doc_reason=subtask.external_doc_reason,
                )
                outcome.notes.append(
                    f"iteration {iteration}: validation failed, attempting repair"
                )
                continue

            # No validation command — accept the patch.
            break

        return outcome

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_context_pack(
        self,
        *,
        subtask: Subtask,
        failing_files: list[str],
        validation_text: str,
        external: ExternalContextResult | None,
    ) -> ContextPack:
        request = RetrievalRequest(
            task=subtask.goal,
            queries=subtask.search_queries
            or queries_from_task(subtask.goal, hint_tokens=subtask.likely_files),
            likely_files=subtask.likely_files,
            failing_files=failing_files,
            touched_files=self._recent_touched_files(),
        )
        candidates = retrieve(self.store, request)
        memory_text = self.memory.render(self.config.context.max_memory_tokens)
        web_notes = (
            [n.to_rendered() for n in external.notes] if external and external.notes else []
        )

        constraints = [
            "Make the smallest correct change.",
            "Do not modify unrelated files.",
            "For edits to existing files, use only the provided snippets and notes.",
            "For new file creation, write complete working implementations.",
            "Return NO_PATCH only if you cannot identify existing code to edit; never for new file creation.",
        ]
        if subtask.edit_scope == "small":
            constraints.append("This sub-task is scoped as small — keep the diff tight.")

        return build_context_pack(
            task=subtask.goal,
            candidates=candidates,
            repo_root=self.repo_root,
            config=self.config.context,
            constraints=constraints,
            web_notes=web_notes,
            memory=memory_text,
            validation=validation_text,
        )

    def _ask_for_patch(
        self, subtask: Subtask, pack: ContextPack, validation_command: str
    ) -> str:
        if self.llm is None:
            return "NO_PATCH: no LLM client configured"
        try:
            result = self.llm.chat(
                [
                    ChatMessage(role="system", content=SYSTEM_PATCH),
                    ChatMessage(
                        role="user",
                        content=patch_prompt(
                            subtask.goal, pack.render(), validation_command
                        ),
                    ),
                ],
                temperature=0.0,
                max_tokens=self.config.llm.max_output_tokens,
            )
            return result.text
        except LLMError as exc:
            return f"NO_PATCH: local model error — {exc}"

    def _ask_for_repair(
        self,
        subtask: Subtask,
        pack: ContextPack,
        previous_patch: str,
        failure_summary: str,
    ) -> str:
        if self.llm is None:
            return "NO_PATCH: no LLM client configured"
        try:
            result = self.llm.chat(
                [
                    ChatMessage(role="system", content=SYSTEM_PATCH),
                    ChatMessage(
                        role="user",
                        content=repair_prompt(
                            subtask.goal, pack.render(), failure_summary, previous_patch
                        ),
                    ),
                ],
                temperature=0.0,
                max_tokens=self.config.llm.max_output_tokens,
            )
            return result.text
        except LLMError as exc:
            return f"NO_PATCH: local model error — {exc}"

    def _analyse_failure(self, failure_summary: str):
        if self.llm is None:
            return None
        try:
            result = self.llm.chat(
                [
                    ChatMessage(
                        role="system",
                        content=(
                            "You analyse failing test output. "
                            "Always respond with valid JSON only."
                        ),
                    ),
                    ChatMessage(
                        role="user",
                        content=failure_analysis_prompt(failure_summary),
                    ),
                ],
                temperature=0.0,
                max_tokens=300,
            )
        except LLMError:
            return None
        return parse_failure_analysis(result.text)

    def _run_validation(self, command: str) -> CommandResult:
        report = assess_command(command)
        if not report.is_safe:
            return CommandResult(
                command=command,
                exit_code=126,
                stdout="",
                stderr=f"safety: refused to run command — {report.explain()}",
                elapsed=0.0,
            )
        return run_command(
            command,
            cwd=self.repo_root,
            timeout=self.config.llm.timeout_seconds * 2 or 300,
        )

    def _recent_touched_files(self) -> list[str]:
        out: list[str] = []
        for entry in self.memory.recent():
            for path in entry.files_touched:
                if path not in out:
                    out.append(path)
        return out

    def _update_memory(self, task: str, outcome: SubtaskOutcome) -> None:
        self.memory.add(
            MemoryEntry(
                user_request=task,
                subtask=outcome.subtask.goal,
                files_touched=(
                    outcome.apply_result.changed_files if outcome.apply_result else []
                ),
                patch_summary=("NO_PATCH: " + outcome.no_patch_reason)
                if outcome.no_patch_reason
                else outcome.patch_text[:200],
                validation=(
                    outcome.command_result.command + (" OK" if outcome.command_result.ok else " FAIL")
                )
                if outcome.command_result
                else "",
                remaining_issues="; ".join(outcome.notes)[:200],
            )
        )

    def _git_is_dirty(self) -> bool:
        if shutil.which("git") is None:
            return False
        try:
            completed = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        if completed.returncode != 0:
            return False
        return bool(completed.stdout.strip())
