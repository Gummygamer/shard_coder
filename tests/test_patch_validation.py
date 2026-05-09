"""Patch parser, validator, and applier tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from shardcoder.config import RepositoryConfig
from shardcoder.editing.patcher import (
    apply_validated_patch,
    validate_model_output,
)


VALID_DIFF = """\
--- a/src/hello.py
+++ b/src/hello.py
@@ -1,3 +1,3 @@
 def greet():
-    return 'hi'
+    return 'hello'
 # tail
"""


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    (src / "hello.py").write_text("def greet():\n    return 'hi'\n# tail\n")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "lib.js").write_text("module.exports = 1;\n")
    return tmp_path


def test_valid_diff_is_accepted_and_applied(repo: Path) -> None:
    result = validate_model_output(VALID_DIFF, repo, RepositoryConfig())
    assert result.ok, result.errors
    assert not result.is_no_patch
    apply_result = apply_validated_patch(result, repo)
    assert apply_result.ok, apply_result.errors
    assert apply_result.changed_files == ["src/hello.py"]
    assert (repo / "src/hello.py").read_text() == "def greet():\n    return 'hello'\n# tail\n"


def test_diff_with_prose_is_rejected(repo: Path) -> None:
    polluted = "Here is the diff you asked for:\n" + VALID_DIFF
    result = validate_model_output(polluted, repo, RepositoryConfig())
    assert not result.ok
    assert any("prose" in err for err in result.errors)


def test_no_patch_response_is_recognised(repo: Path) -> None:
    result = validate_model_output("NO_PATCH: insufficient context", repo, RepositoryConfig())
    assert result.is_no_patch
    assert "insufficient" in result.reason


def test_diff_targeting_ignored_path_is_rejected(repo: Path) -> None:
    bad = (
        "--- a/node_modules/lib.js\n"
        "+++ b/node_modules/lib.js\n"
        "@@ -1,1 +1,1 @@\n"
        "-module.exports = 1;\n"
        "+module.exports = 2;\n"
    )
    result = validate_model_output(bad, repo, RepositoryConfig())
    assert not result.ok
    assert any("ignored path" in err for err in result.errors)


def test_diff_targeting_binary_extension_is_rejected(repo: Path) -> None:
    (repo / "logo.png").write_bytes(b"x")
    bad = (
        "--- a/logo.png\n"
        "+++ b/logo.png\n"
        "@@ -1,1 +1,1 @@\n"
        "-x\n"
        "+y\n"
    )
    result = validate_model_output(bad, repo, RepositoryConfig())
    assert not result.ok
    assert any("binary" in err for err in result.errors)


def test_invalid_diff_is_rejected(repo: Path) -> None:
    bad = "+++ this is not a valid diff\nrandom text"
    result = validate_model_output(bad, repo, RepositoryConfig())
    assert not result.ok


def test_unrelated_files_warn_when_expected_files_known(repo: Path) -> None:
    result = validate_model_output(
        VALID_DIFF, repo, RepositoryConfig(), expected_files=["other/file.py"]
    )
    assert result.ok
    assert any("not in the planner's expected files" in w for w in result.warnings)


def test_dry_run_does_not_modify_files(repo: Path) -> None:
    result = validate_model_output(VALID_DIFF, repo, RepositoryConfig())
    apply_result = apply_validated_patch(result, repo, dry_run=True)
    assert apply_result.ok
    # File on disk is unchanged.
    assert (repo / "src/hello.py").read_text() == "def greet():\n    return 'hi'\n# tail\n"


def test_diff_creating_new_file_works(repo: Path) -> None:
    diff = (
        "--- /dev/null\n"
        "+++ b/src/new.py\n"
        "@@ -0,0 +1,2 @@\n"
        "+# new\n"
        "+x = 1\n"
    )
    result = validate_model_output(diff, repo, RepositoryConfig())
    assert result.ok, result.errors
    apply_result = apply_validated_patch(result, repo)
    assert apply_result.ok
    assert (repo / "src/new.py").read_text().startswith("# new\n")


def test_diff_deleting_file_unlinks_it(repo: Path) -> None:
    target = repo / "src" / "hello.py"
    assert target.exists()
    diff = (
        "--- a/src/hello.py\n"
        "+++ /dev/null\n"
        "@@ -1,3 +0,0 @@\n"
        "-def greet():\n"
        "-    return 'hi'\n"
        "-# tail\n"
    )
    result = validate_model_output(diff, repo, RepositoryConfig())
    assert result.ok, result.errors
    apply_result = apply_validated_patch(result, repo)
    assert apply_result.ok, apply_result.errors
    assert apply_result.changed_files == ["src/hello.py"]
    assert not target.exists()


def test_diff_with_huge_deletion_is_blocked(repo: Path) -> None:
    big_file = repo / "src" / "big.py"
    lines = "\n".join(f"x = {i}" for i in range(1, 21))
    big_file.write_text(lines + "\n")
    diff_lines = ["--- a/src/big.py", "+++ b/src/big.py", "@@ -1,20 +1,1 @@"]
    diff_lines.extend(f"-x = {i}" for i in range(1, 21))
    diff_lines.append("+x = 1")
    diff = "\n".join(diff_lines) + "\n"
    result = validate_model_output(
        diff, repo, RepositoryConfig(), max_deleted_lines_per_file=10
    )
    assert not result.ok
    assert any("deletes" in e for e in result.errors)
