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


class StubLLM:
    """Minimal LLM stub that can return a plan and a queue of patches."""

    model = "stub"

    def __init__(self, *, plan: dict, patches: list[str]):
        self._plan = plan
        self._patches = list(patches)
        self.plan_calls = 0
        self.patch_calls = 0

    def chat(self, messages, *, temperature=None, max_tokens=None, stop=None):
        user = messages[-1].content
        if "Decompose the user task" in user:
            self.plan_calls += 1
            return ChatResult(text=json.dumps(self._plan))
        if "You are editing code" in user or "Your previous patch failed" in user:
            self.patch_calls += 1
            patch = self._patches[self.patch_calls - 1]
            return ChatResult(text=patch)
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
