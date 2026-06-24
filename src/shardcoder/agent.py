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
    continuation_prompt,
    failure_analysis_prompt,
    patch_prompt,
    repair_prompt,
)
from .retrieval.context_pack import ContextPack, build_context_pack
from .retrieval.retriever import RetrievalRequest, queries_from_task, retrieve
from .safety.commands import assess_command
from .summarization.store import SummaryStore
from .summarization.summarizer import FileSummary, summarize_file
from .tokens.budget import cap_text


DEFAULT_DB_DIR = ".shardcoder"
DEFAULT_DB_NAME = "shardcoder.db"

# Maximum number of continuation subtasks chained off a single original
# subtask before we give up. Stops runaway loops when the model keeps
# producing length-truncated output.
MAX_CONTINUATIONS_PER_SUBTASK = 8

# Non-context prompt text, chat role overhead, and rough tokenizer drift.
PROMPT_CONTEXT_HEADROOM_TOKENS = 1024
MIN_CONTEXT_PACK_TOKENS = 512
TRUNCATED_OUTPUT_EXCERPT_TOKENS = 180


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
    truncated: bool = False
    continuation_subtask: Subtask | None = None


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
    dry_run: bool = False


class Agent:
    """Orchestrates an end-to-end task."""

    def __init__(
        self,
        repo_root: str | Path,
        config: ShardCoderConfig | None = None,
        llm: LLMClient | None = None,
        memory: SlidingMemory | None = None,
        db_path: str | Path | None = None,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.config = config or load_config()
        self.llm = llm  # may be None for CLI commands that do not need it
        self.memory = memory or SlidingMemory()
        if db_path is None:
            db_path = self.repo_root / DEFAULT_DB_DIR / DEFAULT_DB_NAME
        self.store = SummaryStore(db_path)
        self._log_fn = log

    def _emit(self, msg: str) -> None:
        if self._log_fn is not None:
            self._log_fn(msg)

    def _context_pack_token_budget(self) -> int:
        available = (
            self.config.llm.max_context_tokens
            - self.config.llm.max_output_tokens
            - PROMPT_CONTEXT_HEADROOM_TOKENS
        )
        return max(MIN_CONTEXT_PACK_TOKENS, available)

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

    @classmethod
    def with_default_llm(
        cls,
        repo_root: str | Path,
        config: ShardCoderConfig | None = None,
        log: Callable[[str], None] | None = None,
    ) -> "Agent":
        config = config or load_config()
        llm = OpenAICompatibleClient(config.llm)
        return cls(repo_root, config=config, llm=llm, log=log)

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
        self._emit("Indexing repository...")
        self.index(only_changed=True)

        self._emit("Planning subtasks...")
        planner_result = self.plan(task)
        plan = planner_result.plan
        if not plan.subtasks:
            warnings.append("plan contained no subtasks")
            return AgentReport(
                task=task,
                plan_used_fallback=planner_result.fallback_used,
                warnings=warnings,
            )

        src = "fallback" if planner_result.fallback_used else "model"
        self._emit(f"Plan ({src}): {len(plan.subtasks)} subtask(s)")
        for st in plan.subtasks:
            self._emit(f"  {st.id}: {st.goal}")

        is_dry_run = self.config.agent.dry_run or not self.config.agent.auto_apply
        report = AgentReport(
            task=task,
            plan_used_fallback=planner_result.fallback_used,
            dry_run=is_dry_run,
        )

        cap = len(plan.subtasks) if max_subtasks is None else max(max_subtasks, 0)
        consecutive_failures = 0
        # Mutable queue so continuation subtasks (queued after a length-
        # truncated output) can be inserted ahead of the remaining plan.
        queue: list[Subtask] = list(plan.subtasks[:cap])
        continuations_per_root: dict[str, int] = {}
        while queue:
            subtask = queue.pop(0)
            self._emit(f"\n[{subtask.id}] Starting: {subtask.goal}")
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

            is_continuation = "-cont" in subtask.id
            root_id = subtask.id.split("-cont", 1)[0]

            if outcome.continuation_subtask is not None:
                count = continuations_per_root.get(root_id, 0)
                if count < MAX_CONTINUATIONS_PER_SUBTASK:
                    continuations_per_root[root_id] = count + 1
                    queue.insert(0, outcome.continuation_subtask)
                else:
                    msg = (
                        f"continuation limit ({MAX_CONTINUATIONS_PER_SUBTASK}) "
                        f"reached for {root_id}; the file may still be incomplete"
                    )
                    warnings.append(msg)
                    self._emit(msg)

            if outcome.no_patch_reason:
                if is_continuation and "complete" in outcome.no_patch_reason.lower():
                    # Model says the file is done — treat as success.
                    pass
                elif is_continuation:
                    # NO_PATCH for a non-completion reason; retry the continuation.
                    count = continuations_per_root.get(root_id, 0)
                    if count < MAX_CONTINUATIONS_PER_SUBTASK:
                        retry = self._build_continuation_subtask(subtask, [])
                        continuations_per_root[root_id] = count + 1
                        queue.insert(0, retry)
                        self._emit(
                            f"[{subtask.id}] NO_PATCH on continuation; retrying as {retry.id}"
                        )
                    else:
                        msg = (
                            f"continuation limit ({MAX_CONTINUATIONS_PER_SUBTASK}) "
                            f"reached for {root_id}; the file may still be incomplete"
                        )
                        warnings.append(msg)
                        self._emit(msg)
                else:
                    report.stop_reason = (
                        f"NO_PATCH on subtask {subtask.id}: {outcome.no_patch_reason}"
                    )
                    warnings.append(report.stop_reason)
                    break

            # Track consecutive hard-failures only for original subtasks.
            # Continuation failures re-enqueue a retry rather than aborting.
            if not is_continuation:
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
            elif (
                outcome.continuation_subtask is None
                and not outcome.no_patch_reason
                and self._subtask_hard_failed(outcome)
            ):
                # Apply/validation failed for this continuation; retry it.
                count = continuations_per_root.get(root_id, 0)
                if count < MAX_CONTINUATIONS_PER_SUBTASK:
                    retry = self._build_continuation_subtask(subtask, [])
                    continuations_per_root[root_id] = count + 1
                    queue.insert(0, retry)
                    self._emit(
                        f"[{subtask.id}] continuation hard-failed; retrying as {retry.id}"
                    )
                else:
                    msg = (
                        f"continuation limit ({MAX_CONTINUATIONS_PER_SUBTASK}) "
                        f"reached for {root_id}; the file may still be incomplete"
                    )
                    warnings.append(msg)
                    self._emit(msg)

        if not report.stop_reason and report.outcomes:
            last = report.outcomes[-1]
            if self._subtask_hard_failed(last):
                report.stop_reason = self._failure_stop_reason(last)
                warnings.append(report.stop_reason)

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
        if outcome.apply_result is None:
            return True
        if outcome.apply_result is not None and not outcome.apply_result.ok:
            return True
        if outcome.command_result is not None and not outcome.command_result.ok:
            return True
        return False

    @staticmethod
    def _failure_stop_reason(outcome: SubtaskOutcome) -> str:
        prefix = f"subtask {outcome.subtask.id} failed"
        if outcome.validation and outcome.validation.errors:
            return f"{prefix}: {'; '.join(outcome.validation.errors)}"
        if outcome.apply_result and outcome.apply_result.errors:
            return f"{prefix}: {'; '.join(outcome.apply_result.errors)}"
        if outcome.command_result and not outcome.command_result.ok:
            return (
                f"{prefix}: validation command exited "
                f"{outcome.command_result.exit_code}"
            )
        if outcome.truncated:
            return f"{prefix}: model output was truncated before a safe patch was applied"
        return f"{prefix}: no patch was applied"

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
            self._emit(
                f"[{subtask.id}] Fetching external docs"
                + (f" — {subtask.external_doc_reason}" if subtask.external_doc_reason else "")
            )
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
                self._emit(f"[{subtask.id}] External docs error: {external_result.error}")
            elif external_result.sources:
                self._emit(f"[{subtask.id}] External docs: {', '.join(external_result.sources)}")
            outcome.used_external_docs = external_result.used and bool(
                external_result.notes
            )
            outcome.external_sources = external_result.sources

        queries = subtask.search_queries or queries_from_task(
            subtask.goal, hint_tokens=subtask.likely_files
        )
        self._emit(
            f"[{subtask.id}] Retrieving context — queries: {', '.join(queries[:4])}"
            + (" ..." if len(queries) > 4 else "")
        )
        context_pack = self._build_context_pack(
            subtask=subtask,
            failing_files=[],
            validation_text="",
            external=external_result,
        )
        outcome.context_pack = context_pack
        self._emit(
            f"[{subtask.id}] Context: {len(context_pack.snippets)} snippet(s), "
            f"{len(context_pack.summaries)} summary/ies, {context_pack.used_tokens} tokens"
        )

        validation_command = (
            subtask.validation or self.config.validation.test_command or ""
        )

        is_continuation = "-cont" in subtask.id

        for iteration in range(1, self.config.agent.max_iterations + 1):
            outcome.iterations = iteration
            self._emit(f"[{subtask.id}] Iteration {iteration}: calling model for patch...")
            if is_continuation:
                patch_text, finish_reason = self._ask_for_continuation(
                    subtask, context_pack
                )
            else:
                patch_text, finish_reason = self._ask_for_patch(
                    subtask, context_pack, validation_command
                )
            outcome.patch_text = patch_text
            truncated = finish_reason == "length"

            if patch_text.lstrip().startswith("NO_PATCH"):
                self._emit(f"[{subtask.id}] Iteration {iteration}: model returned NO_PATCH")
            else:
                nlines = patch_text.count("\n") + 1
                trunc_note = " (truncated at max_output_tokens)" if truncated else ""
                self._emit(
                    f"[{subtask.id}] Iteration {iteration}: "
                    f"model returned {nlines}-line diff{trunc_note}"
                )

            self._emit(f"[{subtask.id}] Iteration {iteration}: validating patch...")
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
                self._emit(
                    f"[{subtask.id}] Iteration {iteration}: patch rejected — "
                    + "; ".join(validation_result.errors)
                )
                if truncated:
                    outcome.truncated = True
                    msg = (
                        "model hit max_output_tokens before producing a valid "
                        "unified diff; no files were written from the partial response"
                    )
                    self._emit(f"[{subtask.id}] Iteration {iteration}: {msg}")
                    outcome.notes.append(f"iteration {iteration}: {msg}")
                    if patch_text.strip():
                        excerpt = cap_text(
                            patch_text.strip(), TRUNCATED_OUTPUT_EXCERPT_TOKENS
                        )
                        outcome.notes.append(
                            f"iteration {iteration}: truncated output excerpt:\n{excerpt}"
                        )
                outcome.notes.append(
                    f"iteration {iteration}: patch rejected — {'; '.join(validation_result.errors)}"
                )
                # Feed the failure back as 'validation' for the next iteration.
                retry_guidance = "\n".join(validation_result.errors)
                if truncated:
                    retry_guidance = (
                        retry_guidance
                        + "\nThe previous response hit max_output_tokens before a "
                        "valid diff was produced. Reply with a smaller unified diff "
                        "only, or split the work into the first complete file-sized "
                        "patch."
                    )
                context_pack = self._build_context_pack(
                    subtask=subtask,
                    failing_files=[],
                    validation_text=retry_guidance,
                    external=external_result,
                )
                continue

            # If the model hit max_output_tokens, only apply when the partial
            # diff is safe: a new-file write, or an edit that only ADDS lines.
            # Anything that removes or rewrites existing lines could land in a
            # corrupted state mid-edit, so we reject and let the loop retry.
            safe_partial = self._is_safe_partial_diff(validation_result.diffs)
            if truncated and not safe_partial:
                msg = (
                    f"output truncated mid-edit at max_output_tokens="
                    f"{self.config.llm.max_output_tokens}; refusing partial "
                    "edit (diff contains removals or rewrites)"
                )
                self._emit(f"[{subtask.id}] Iteration {iteration}: {msg}")
                outcome.notes.append(f"iteration {iteration}: {msg}")
                context_pack = self._build_context_pack(
                    subtask=subtask,
                    failing_files=[],
                    validation_text=msg + "\nReply with a smaller diff that fits.",
                    external=external_result,
                )
                continue

            target_files = [fd.target_path for fd in validation_result.diffs]
            self._emit(
                f"[{subtask.id}] Iteration {iteration}: applying patch"
                + (f" → {', '.join(target_files)}" if target_files else "")
            )
            apply_result = apply_validated_patch(
                validation_result,
                self.repo_root,
                dry_run=self.config.agent.dry_run or not self.config.agent.auto_apply,
            )
            outcome.apply_result = apply_result
            if not apply_result.ok:
                self._emit(
                    f"[{subtask.id}] Iteration {iteration}: apply failed — "
                    + "; ".join(apply_result.errors)
                )
                outcome.notes.append(
                    f"iteration {iteration}: apply failed — {'; '.join(apply_result.errors)}"
                )
                break

            if apply_result.changed_files:
                self._emit(
                    f"[{subtask.id}] Iteration {iteration}: wrote "
                    + ", ".join(apply_result.changed_files)
                )
                # Refresh the index so retrieval can see newly-written /
                # extended files in any continuation subtask.
                if not self.config.agent.dry_run:
                    self.index(only_changed=True)

            if truncated:
                outcome.truncated = True
                outcome.continuation_subtask = self._build_continuation_subtask(
                    subtask, validation_result.diffs
                )
                self._emit(
                    f"[{subtask.id}] Iteration {iteration}: output truncated; "
                    f"queued continuation {outcome.continuation_subtask.id}"
                )
                break

            if (
                self.config.agent.auto_run_tests
                and validation_command
                and not self.config.agent.dry_run
            ):
                self._emit(
                    f"[{subtask.id}] Iteration {iteration}: running validation: {validation_command}"
                )
                cmd_result = self._run_validation(validation_command)
                outcome.command_result = cmd_result
                if cmd_result.ok:
                    self._emit(f"[{subtask.id}] Iteration {iteration}: validation passed")
                    break
                self._emit(f"[{subtask.id}] Iteration {iteration}: validation failed, preparing repair...")
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

    @staticmethod
    def _is_safe_partial_diff(diffs: list) -> bool:
        """Decide whether a length-truncated diff is safe to apply.

        New-file writes are always safe (we're just creating a partial file
        the continuation will extend). Edits are safe only if every hunk is
        pure-addition: no ``-`` lines that would remove or rewrite existing
        content. Everything else risks corrupting a file mid-edit.
        """
        for fd in diffs:
            if fd.is_new_file:
                continue
            if fd.is_deleted_file:
                return False
            for hunk in fd.hunks:
                for line in hunk.lines:
                    if line.startswith("-") and not line.startswith("---"):
                        return False
        return True

    def _build_continuation_subtask(
        self, parent: Subtask, diffs: list
    ) -> Subtask:
        """Build a follow-up subtask that asks the model to extend the
        partially-written file produced by *parent*.

        ``diffs`` is the list of :class:`FileDiff` from the truncated patch.
        We pin ``likely_files`` to the new files so retrieval surfaces the
        partial content; the goal text carries the continuation framing the
        model needs.
        """
        new_paths = (
            [fd.target_path for fd in diffs if fd.is_new_file and not fd.is_deleted_file]
            or [fd.target_path for fd in diffs]
            or list(parent.likely_files)
        )

        # Keep a continuation depth counter on the id so we can detect runaway
        # chains and avoid colliding ids in the report. T1 -> T1-cont1 -> T1-cont2 ...
        if "-cont" in parent.id:
            base, _, depth = parent.id.rpartition("-cont")
            try:
                next_depth = int(depth) + 1
            except ValueError:
                next_depth = 1
        else:
            base, next_depth = parent.id, 1

        goal = (
            f"Continue writing {', '.join(new_paths)}: the previous output "
            f"was cut off by max_output_tokens. Append the remaining code so "
            f"the file fulfils the original goal. Original goal: {parent.goal}"
        )

        return Subtask(
            id=f"{base}-cont{next_depth}",
            goal=goal,
            reason="continuation after truncated output",
            search_queries=parent.search_queries,
            likely_files=new_paths,
            edit_scope=parent.edit_scope,
            validation=parent.validation,
            needs_external_docs=False,
            external_doc_reason="",
        )

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
            max_tokens=self._context_pack_token_budget(),
        )

    def _ask_for_patch(
        self, subtask: Subtask, pack: ContextPack, validation_command: str
    ) -> tuple[str, str | None]:
        if self.llm is None:
            return "NO_PATCH: no LLM client configured", None
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
            return result.text, result.finish_reason
        except LLMError as exc:
            return f"NO_PATCH: local model error — {exc}", None

    def _ask_for_continuation(
        self, subtask: Subtask, pack: ContextPack
    ) -> tuple[str, str | None]:
        """Ask the model to extend a partially-written file.

        ``subtask.likely_files[0]`` is the file we're continuing. We read its
        current line count + last few lines and feed them into a dedicated
        continuation prompt that demands an append-only edit-style diff.
        """
        if self.llm is None:
            return "NO_PATCH: no LLM client configured", None

        file_path = subtask.likely_files[0] if subtask.likely_files else ""
        full_path = self.repo_root / file_path if file_path else None
        try:
            current_text = (
                full_path.read_text(encoding="utf-8", errors="replace")
                if full_path and full_path.exists()
                else ""
            )
        except OSError:
            current_text = ""
        current_lines = current_text.splitlines()
        line_count = len(current_lines)
        tail = "\n".join(current_lines[-10:]) if current_lines else "(file empty)"

        # Strip the wrapping "Original goal: ..." that the continuation
        # subtask carries, to surface only the original user-visible goal.
        if "Original goal:" in subtask.goal:
            original_goal = subtask.goal.split("Original goal:", 1)[1].strip()
        else:
            original_goal = subtask.goal

        try:
            result = self.llm.chat(
                [
                    ChatMessage(role="system", content=SYSTEM_PATCH),
                    ChatMessage(
                        role="user",
                        content=continuation_prompt(
                            original_goal=original_goal,
                            file_path=file_path,
                            current_line_count=line_count,
                            current_tail=tail,
                            context_pack=pack.render(),
                        ),
                    ),
                ],
                temperature=0.0,
                max_tokens=self.config.llm.max_output_tokens,
            )
            return result.text, result.finish_reason
        except LLMError as exc:
            return f"NO_PATCH: local model error — {exc}", None

    def _ask_for_repair(
        self,
        subtask: Subtask,
        pack: ContextPack,
        previous_patch: str,
        failure_summary: str,
    ) -> tuple[str, str | None]:
        if self.llm is None:
            return "NO_PATCH: no LLM client configured", None
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
            return result.text, result.finish_reason
        except LLMError as exc:
            return f"NO_PATCH: local model error — {exc}", None

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
