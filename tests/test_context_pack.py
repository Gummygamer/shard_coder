"""Context-pack assembly and budget enforcement."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from shardcoder.config import ContextConfig
from shardcoder.external_context.context_normalizer import CompactNote
from shardcoder.retrieval.context_pack import build_context_pack
from shardcoder.retrieval.ranking import Candidate
from shardcoder.tokens.budget import estimate_tokens


@pytest.fixture()
def repo_with_files(tmp_path: Path) -> Path:
    # ~20 short lines per file; each snippet costs roughly the same.
    (tmp_path / "high.py").write_text("\n".join(f"hi_{i}" for i in range(1, 21)))
    (tmp_path / "low.py").write_text("\n".join(f"lo_{i}" for i in range(1, 21)))
    return tmp_path


def test_context_pack_includes_paths_and_line_numbers(repo_with_files: Path) -> None:
    candidates = [
        Candidate(
            path="high.py",
            score=10.0,
            reason="planner-likely",
            summary_text="high priority module",
        )
    ]
    pack = build_context_pack(
        task="add greeting",
        candidates=candidates,
        repo_root=repo_with_files,
        config=ContextConfig(),
        max_snippet_lines=20,
    )
    assert pack.snippets, "expected at least one snippet"
    snippet = pack.snippets[0]
    assert snippet.path == "high.py"
    assert snippet.start_line == 1
    assert snippet.end_line >= 1
    rendered = pack.render()
    assert "high.py" in rendered
    assert "lines 1-" in rendered


def test_lower_ranked_snippets_are_pruned_first(repo_with_files: Path) -> None:
    candidates = [
        Candidate(path="high.py", score=10.0, reason="high", summary_text="high"),
        Candidate(path="low.py", score=0.5, reason="low", summary_text="low"),
    ]
    # Constrain the snippet budget so only one snippet fits.
    config = ContextConfig(
        max_snippet_tokens=50,
        max_summary_tokens=200,
        max_memory_tokens=0,
        max_validation_tokens=0,
        max_web_context_tokens=0,
    )
    pack = build_context_pack(
        task="task",
        candidates=candidates,
        repo_root=repo_with_files,
        config=config,
        max_snippet_lines=80,
    )
    snippet_paths = [s.path for s in pack.snippets]
    assert snippet_paths == ["high.py"]
    assert pack.used_tokens <= pack.max_tokens


def test_context_pack_stays_under_global_budget(repo_with_files: Path) -> None:
    candidates = [
        Candidate(
            path="high.py",
            score=10.0,
            reason="planner-likely",
            summary_text="x" * 500,
        )
    ]
    config = ContextConfig(
        max_summary_tokens=20,
        max_snippet_tokens=20,
        max_memory_tokens=0,
        max_validation_tokens=0,
        max_web_context_tokens=0,
    )
    pack = build_context_pack(
        task="x",
        candidates=candidates,
        repo_root=repo_with_files,
        config=config,
        max_snippet_lines=80,
    )
    assert pack.used_tokens <= pack.max_tokens


def test_compact_web_notes_fit_within_budget(repo_with_files: Path) -> None:
    today = date.today().isoformat()
    notes = [
        CompactNote(
            source_title=f"doc-{i}",
            source_url=f"https://docs.example.com/{i}",
            date_accessed=today,
            relevance="dependency behaviour",
            facts=["fact one.", "fact two."],
            is_official_docs=True,
            rank=i,
        ).to_rendered()
        for i in range(5)
    ]
    config = ContextConfig(max_web_context_tokens=80)
    pack = build_context_pack(
        task="x",
        candidates=[],
        repo_root=repo_with_files,
        config=config,
        web_notes=notes,
    )
    web_text = "\n".join(n.render() for n in pack.web_notes)
    assert estimate_tokens(web_text) <= config.max_web_context_tokens
