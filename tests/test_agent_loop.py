"""Multi-subtask plan execution (M1).

These tests pin the loop in ``Agent.run_edit``: every subtask in the plan
runs in order, memory is folded between them, and the early-stop guards
fire on ``NO_PATCH`` and on two consecutive validation failures.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shardcoder.agent import Agent
from shardcoder.config import AgentConfig, ShardCoderConfig
from shardcoder.llm.client import ChatResult


def _plan(*subtask_specs: tuple[str, str, str]) -> dict:
    """Build a JSON plan dict with the supplied (id, goal, file) subtasks."""
    return {
        "task": "stub",
        "assumptions": [],
        "subtasks": [
            {
                "id": sid,
                "goal": goal,
                "reason": "",
                "search_queries": [],
                "likely_files": [target],
                "edit_scope": "small",
                "validation": "",
                "needs_external_docs": False,
                "external_doc_reason": "",
            }
            for sid, goal, target in subtask_specs
        ],
        "stop_conditions": [],
        "risks": [],
    }


def _diff(path: str, before: str, after: str) -> str:
    return (
        f"--- a/{path}\n"
        f"+++ b/{path}\n"
        f"@@ -1,1 +1,1 @@\n"
        f"-{before}\n"
        f"+{after}\n"
    )


_PATCH_PROMPT_MARKERS = (
    "You are editing or creating code.",
    "Your previous patch failed",
    "Your previous output was cut off",
)


class StubLLM:
    """Minimal LLM stub that can return a plan and a queue of patches.

    Patches can be either a string (text only, finish_reason=stop) or a
    ``(text, finish_reason)`` tuple if a test wants to simulate truncation.
    """

    model = "stub"

    def __init__(self, *, plan: dict, patches: list):
        self._plan = plan
        self._patches = list(patches)
        self.plan_calls = 0
        self.patch_calls = 0

    def chat(self, messages, *, temperature=None, max_tokens=None, stop=None):
        user = messages[-1].content
        if "Decompose the user task" in user:
            self.plan_calls += 1
            return ChatResult(text=json.dumps(self._plan))
        if any(marker in user for marker in _PATCH_PROMPT_MARKERS):
            self.patch_calls += 1
            entry = self._patches[self.patch_calls - 1]
            if isinstance(entry, tuple):
                text, finish_reason = entry
            else:
                text, finish_reason = entry, "stop"
            return ChatResult(text=text, finish_reason=finish_reason)
        # Failure analysis or anything else — return harmless empty JSON.
        return ChatResult(text="{}")


def _build_repo(tmp_path: Path) -> Path:
    (tmp_path / "alpha.py").write_text("a = 1\n")
    (tmp_path / "beta.py").write_text("b = 1\n")
    (tmp_path / "gamma.py").write_text("c = 1\n")
    return tmp_path


def _agent(tmp_path: Path, llm: StubLLM, *, max_iterations: int = 1) -> Agent:
    config = ShardCoderConfig()
    config.agent = AgentConfig(
        max_iterations=max_iterations,
        dry_run=True,
        auto_apply=False,
        auto_run_tests=False,
    )
    return Agent(repo_root=tmp_path, config=config, llm=llm)


def test_two_subtask_plan_runs_both_in_dry_run(tmp_path: Path) -> None:
    repo = _build_repo(tmp_path)
    plan = _plan(
        ("T1", "rename a in alpha.py", "alpha.py"),
        ("T2", "rename b in beta.py", "beta.py"),
    )
    llm = StubLLM(
        plan=plan,
        patches=[
            _diff("alpha.py", "a = 1", "a = 2"),
            _diff("beta.py", "b = 1", "b = 2"),
        ],
    )
    agent = _agent(repo, llm)

    report = agent.run_edit("two-step task", allow_dirty=True)

    assert llm.plan_calls == 1
    assert llm.patch_calls == 2
    assert len(report.outcomes) == 2
    assert [o.subtask.id for o in report.outcomes] == ["T1", "T2"]
    assert report.files_changed == ["alpha.py", "beta.py"]
    assert report.stop_reason == ""
    # Dry-run leaves disk untouched.
    assert (repo / "alpha.py").read_text() == "a = 1\n"
    assert (repo / "beta.py").read_text() == "b = 1\n"
    # Both outcomes were folded into memory.
    assert len(list(agent.memory.recent())) == 2


def test_no_patch_stops_loop_immediately(tmp_path: Path) -> None:
    repo = _build_repo(tmp_path)
    plan = _plan(
        ("T1", "edit alpha", "alpha.py"),
        ("T2", "edit beta", "beta.py"),
    )
    llm = StubLLM(
        plan=plan,
        patches=[
            "NO_PATCH: not enough context",
            _diff("beta.py", "b = 1", "b = 2"),
        ],
    )
    agent = _agent(repo, llm)

    report = agent.run_edit("two-step task", allow_dirty=True)

    assert llm.patch_calls == 1
    assert len(report.outcomes) == 1
    assert report.outcomes[0].no_patch_reason
    assert "NO_PATCH on subtask T1" in report.stop_reason


def test_two_consecutive_validation_failures_stop_loop(tmp_path: Path) -> None:
    repo = _build_repo(tmp_path)
    plan = _plan(
        ("T1", "edit alpha", "alpha.py"),
        ("T2", "edit beta", "beta.py"),
        ("T3", "edit gamma", "gamma.py"),
    )
    bad = "this is not a diff"
    llm = StubLLM(
        plan=plan,
        patches=[bad, bad, _diff("gamma.py", "c = 1", "c = 2")],
    )
    agent = _agent(repo, llm)

    report = agent.run_edit("three-step task", allow_dirty=True)

    # Stops after the second consecutive failure; T3 is never attempted.
    assert llm.patch_calls == 2
    assert len(report.outcomes) == 2
    assert "two consecutive validation failures" in report.stop_reason


def test_max_subtasks_caps_execution(tmp_path: Path) -> None:
    repo = _build_repo(tmp_path)
    plan = _plan(
        ("T1", "edit alpha", "alpha.py"),
        ("T2", "edit beta", "beta.py"),
        ("T3", "edit gamma", "gamma.py"),
    )
    llm = StubLLM(
        plan=plan,
        patches=[
            _diff("alpha.py", "a = 1", "a = 2"),
            _diff("beta.py", "b = 1", "b = 2"),
            _diff("gamma.py", "c = 1", "c = 2"),
        ],
    )
    agent = _agent(repo, llm)

    report = agent.run_edit("three-step task", allow_dirty=True, max_subtasks=2)

    assert llm.patch_calls == 2
    assert len(report.outcomes) == 2
    assert any("max-subtasks=2" in w for w in report.warnings)
    assert report.stop_reason == ""


def test_truncated_new_file_queues_continuation(tmp_path: Path) -> None:
    """A length-truncated new-file patch should land partial content,
    queue a continuation subtask, and the continuation should append the
    rest of the file before the loop ends."""
    repo = _build_repo(tmp_path)
    plan = _plan(("T1", "create new module foo.py", "foo.py"))

    partial_new_file = (
        "--- /dev/null\n"
        "+++ b/foo.py\n"
        "@@ -0,0 +1,2 @@\n"
        "+def foo():\n"
        "+    return 1\n"
    )
    # Continuation appends two more lines starting at line 3 of foo.py.
    continuation_diff = (
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
        "@@ -2,1 +2,3 @@\n"
        "     return 1\n"
        "+\n"
        "+def bar():\n"
        "+    return 2\n"
    )
    llm = StubLLM(
        plan=plan,
        patches=[
            (partial_new_file, "length"),
            continuation_diff,  # finish_reason defaults to "stop"
        ],
    )
    # auto_apply=True so the partial file actually lands on disk and the
    # continuation can read it back from the repo.
    config = ShardCoderConfig()
    config.agent = AgentConfig(
        max_iterations=1,
        dry_run=False,
        auto_apply=True,
        auto_run_tests=False,
    )
    agent = Agent(repo_root=repo, config=config, llm=llm)

    report = agent.run_edit("create foo.py", allow_dirty=True)

    # Two LLM patch calls: original + one continuation.
    assert llm.patch_calls == 2
    assert len(report.outcomes) == 2
    assert report.outcomes[0].truncated is True
    assert report.outcomes[0].continuation_subtask is not None
    assert report.outcomes[1].subtask.id == "T1-cont1"
    assert "foo.py" in report.files_changed
    # Final file has both halves stitched together.
    final = (repo / "foo.py").read_text()
    assert "def foo()" in final
    assert "def bar()" in final
    assert report.stop_reason == ""


def test_truncated_edit_with_removals_is_refused(tmp_path: Path) -> None:
    """A length-truncated edit-style diff that removes lines should NOT
    be applied — the loop must retry rather than corrupt the file."""
    repo = _build_repo(tmp_path)
    plan = _plan(("T1", "edit alpha", "alpha.py"))

    # Truncated diff that removes a line — unsafe to apply mid-edit.
    bad_partial = (
        "--- a/alpha.py\n"
        "+++ b/alpha.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-a = 1\n"
        "+a = 999\n"
    )
    llm = StubLLM(plan=plan, patches=[(bad_partial, "length")])
    agent = _agent(repo, llm)

    report = agent.run_edit("edit alpha", allow_dirty=True)

    # The loop ran out of iterations without applying.
    assert llm.patch_calls == 1
    assert report.outcomes[0].apply_result is None
    # File untouched on disk.
    assert (repo / "alpha.py").read_text() == "a = 1\n"
    # No continuation queued for unsafe partial.
    assert report.outcomes[0].continuation_subtask is None


def test_truncated_invalid_output_records_feedback(tmp_path: Path) -> None:
    """If the model hits max_tokens before a diff header, keep useful feedback."""
    repo = _build_repo(tmp_path)
    plan = _plan(("T1", "create new module foo.py", "foo.py"))
    truncated_prose = (
        "I will create foo.py with the requested helpers.\n"
        "First, here is the implementation approach before the diff starts"
    )
    llm = StubLLM(plan=plan, patches=[(truncated_prose, "length")])
    agent = _agent(repo, llm)

    report = agent.run_edit("create foo.py", allow_dirty=True)

    assert llm.patch_calls == 1
    outcome = report.outcomes[0]
    assert outcome.truncated is True
    assert outcome.apply_result is None
    assert outcome.continuation_subtask is None
    assert any("max_output_tokens" in note for note in outcome.notes)
    assert any("truncated output excerpt" in note for note in outcome.notes)
    assert "model output does not look like a unified diff" in report.stop_reason
    assert not (repo / "foo.py").exists()


def test_recovery_after_single_failure_does_not_stop(tmp_path: Path) -> None:
    """One bad subtask between two good ones should not trigger early stop."""
    repo = _build_repo(tmp_path)
    plan = _plan(
        ("T1", "edit alpha", "alpha.py"),
        ("T2", "edit beta", "beta.py"),
        ("T3", "edit gamma", "gamma.py"),
    )
    llm = StubLLM(
        plan=plan,
        patches=[
            _diff("alpha.py", "a = 1", "a = 2"),
            "this is not a diff",
            _diff("gamma.py", "c = 1", "c = 2"),
        ],
    )
    agent = _agent(repo, llm)

    report = agent.run_edit("three-step task", allow_dirty=True)

    assert llm.patch_calls == 3
    assert len(report.outcomes) == 3
    assert report.stop_reason == ""
