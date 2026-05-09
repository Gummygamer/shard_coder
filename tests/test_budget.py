"""Token-budget pruning behaviour."""

from __future__ import annotations

from shardcoder.tokens.budget import BudgetManager, cap_text, estimate_tokens


def test_estimate_tokens_is_approximate() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("a") == 1
    assert estimate_tokens("a" * 4) == 1
    assert estimate_tokens("a" * 8) == 2


def test_lower_priority_sections_are_pruned_first() -> None:
    bm = BudgetManager(max_tokens=20)
    bm.add("instructions", "I" * 16, reserved=True)  # 4 tokens
    bm.add("task", "T" * 16, reserved=True)  # 4 tokens
    bm.add("memory", "M" * 80)  # 20 tokens, lowest priority
    bm.add("snippets", "S" * 24)  # 6 tokens
    bm.add("validation", "V" * 16)  # 4 tokens

    report = bm.assemble()
    kept_names = {s.name for s in report.sections}
    assert "instructions" in kept_names
    assert "task" in kept_names
    # memory is the lowest-priority section and must be pruned first
    assert "memory" in report.pruned
    assert report.used_tokens <= 20


def test_critical_task_instructions_remain_when_overflowing() -> None:
    bm = BudgetManager(max_tokens=12)
    bm.add("instructions", "I" * 32, reserved=True)  # 8 tokens
    bm.add("task", "T" * 16, reserved=True)  # 4 tokens
    bm.add("memory", "M" * 100)  # 25 tokens

    report = bm.assemble()
    kept_names = {s.name for s in report.sections}
    assert kept_names == {"instructions", "task"}


def test_output_schema_budget_is_preserved() -> None:
    bm = BudgetManager(max_tokens=30)
    bm.add("schema", "S" * 40, reserved=True)  # 10 tokens reserved
    bm.add("memory", "M" * 80)
    bm.add("snippets", "X" * 80)

    report = bm.assemble()
    kept_names = {s.name for s in report.sections}
    assert "schema" in kept_names
    assert report.used_tokens <= 30


def test_external_notes_are_capped_independently() -> None:
    # Simulate the per-section cap behaviour the agent relies on.
    note = "x" * 1000
    capped = cap_text(note, max_tokens=10)
    assert estimate_tokens(capped) <= 11  # tolerate the truncation suffix
    assert capped.endswith("[truncated]")
