"""Planner fallback behaviour."""

from __future__ import annotations

from shardcoder.llm.client import ChatResult
from shardcoder.planning.planner import fallback_plan, make_plan


class InvalidPlanLLM:
    model = "stub"

    def chat(self, messages, *, temperature=None, max_tokens=None, stop=None):
        return ChatResult(text="not json")


def test_pacman_fallback_plan_is_split_into_browser_game_shards() -> None:
    plan = fallback_plan("Create a Pac-man clone", "(repository not yet indexed)")

    assert [subtask.id for subtask in plan.subtasks] == ["T1", "T2", "T3", "T4", "T5"]
    assert plan.subtasks[0].likely_files == ["index.html"]
    assert "only the HTML" in plan.subtasks[0].goal
    assert plan.subtasks[1].likely_files == ["game.js"]
    assert any("ghost" in subtask.goal.lower() for subtask in plan.subtasks)
    assert any("pellet" in subtask.goal.lower() for subtask in plan.subtasks)
    assert any("responsive" in subtask.goal.lower() for subtask in plan.subtasks)


def test_broad_fallback_prefers_existing_browser_entry_points() -> None:
    repo_summary = "\n".join(
        [
            "- public/index.html (html)",
            "- src/main.ts (typescript)",
            "- src/styles.css (css)",
        ]
    )

    plan = fallback_plan("Build an arcade game", repo_summary)

    assert [subtask.id for subtask in plan.subtasks] == ["T1", "T2", "T3", "T4"]
    assert plan.subtasks[0].likely_files == ["public/index.html"]
    assert "src/styles.css" in plan.subtasks[0].goal
    assert "src/main.ts" in plan.subtasks[0].goal
    assert plan.subtasks[1].likely_files == ["src/main.ts"]
    assert plan.subtasks[3].likely_files == ["src/styles.css", "public/index.html"]


def test_invalid_model_plan_uses_heuristic_pacman_fallback() -> None:
    result = make_plan(
        "Create a Pac-man clone",
        "(repository not yet indexed)",
        llm=InvalidPlanLLM(),
    )

    assert result.fallback_used is True
    assert result.fallback_reason == "model produced no valid plan"
    assert len(result.plan.subtasks) == 5
    assert result.plan.subtasks[0].likely_files == ["index.html"]


def test_narrow_fallback_plan_stays_single_subtask() -> None:
    plan = fallback_plan("Fix the failing login test")

    assert len(plan.subtasks) == 1
    assert plan.subtasks[0].goal == "Fix the failing login test"


def test_non_game_clone_uses_generic_broad_fallback() -> None:
    plan = fallback_plan("Create a Twitter clone")

    assert [subtask.id for subtask in plan.subtasks] == ["T1", "T2", "T3", "T4"]
    assert plan.subtasks[0].likely_files == []
    assert "game shell" not in plan.subtasks[0].goal.lower()
