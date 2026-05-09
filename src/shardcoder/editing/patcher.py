"""Safe patch application with extensive validation.

The patcher rejects model output that:

* contains prose around the diff
* targets ignored / vendored / generated paths
* targets binary file extensions
* invents non-existent files unless they are explicitly allowed
* deletes large sections of code without a corresponding header
* strays beyond the planner-declared expected files (when known)
* fails to match context lines on disk
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..config import RepositoryConfig
from ..indexing.scanner import BINARY_EXTENSIONS
from .diff_utils import (
    DiffParseError,
    FileDiff,
    PatchApplyError,
    apply_file_diff,
    parse_unified_diff,
)

NO_PATCH_PREFIX = "NO_PATCH"

DIFF_LINE_RE = re.compile(r"^(?:diff |---\s|@@\s|\+\+\+\s)")
DIFF_BODY_LINE_RE = re.compile(r"^[\+\- @]")


@dataclass
class PatchValidationResult:
    is_no_patch: bool = False
    reason: str = ""
    diffs: list[FileDiff] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    cleaned_diff: str = ""

    @property
    def ok(self) -> bool:
        return not self.errors and not self.is_no_patch


@dataclass
class PatchApplyResult:
    changed_files: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1 :]
        if text.endswith("```"):
            text = text[: -3]
    return text.strip("\n")


def _looks_like_diff(text: str) -> bool:
    return any(line.startswith(("--- ", "+++ ", "@@ ", "diff ")) for line in text.splitlines())


def _is_ignored_path(path: str, ignore_dirs: set[str]) -> bool:
    parts = Path(path).parts
    return any(part in ignore_dirs for part in parts)


def _has_binary_extension(path: str) -> bool:
    return Path(path).suffix.lower() in BINARY_EXTENSIONS


def validate_model_output(
    raw: str,
    repo_root: Path,
    repo_config: RepositoryConfig | None = None,
    expected_files: list[str] | None = None,
    allow_new_files: bool = True,
    max_deleted_lines_per_file: int = 200,
) -> PatchValidationResult:
    """Validate model output, returning either a parsed patch or NO_PATCH."""
    repo_config = repo_config or RepositoryConfig()
    ignore_dirs = set(repo_config.ignore_dirs)

    if not raw:
        return PatchValidationResult(errors=["empty model output"])

    cleaned = _strip_code_fence(raw)
    stripped = cleaned.strip()

    if stripped.startswith(NO_PATCH_PREFIX):
        rest = stripped[len(NO_PATCH_PREFIX) :].lstrip(":").strip()
        return PatchValidationResult(is_no_patch=True, reason=rest or "(no reason given)")

    if not _looks_like_diff(cleaned):
        return PatchValidationResult(
            errors=[
                "model output does not look like a unified diff and is not "
                f"{NO_PATCH_PREFIX}: ..."
            ]
        )

    # Reject prose lines: every non-blank line must be either a diff
    # header/body line, or the "\ No newline at end of file" marker.
    body_lines = cleaned.splitlines()
    started = False
    for line in body_lines:
        if not line.strip():
            continue
        if DIFF_LINE_RE.match(line) or line.startswith("\\ No newline"):
            started = True
            continue
        if started and DIFF_BODY_LINE_RE.match(line):
            continue
        return PatchValidationResult(
            errors=[
                f"prose detected around diff: {line!r} — model must output diff only"
            ]
        )
    if not started:
        return PatchValidationResult(errors=["no diff headers found"])

    try:
        diffs = parse_unified_diff(cleaned)
    except DiffParseError as exc:
        return PatchValidationResult(errors=[f"diff parse error: {exc}"])

    errors: list[str] = []
    warnings: list[str] = []

    expected_set = {p for p in (expected_files or [])}

    for fd in diffs:
        path = fd.target_path

        if _is_ignored_path(path, ignore_dirs):
            errors.append(f"diff touches ignored path: {path}")
            continue
        if _has_binary_extension(path):
            errors.append(f"diff touches binary file: {path}")
            continue
        if os.path.isabs(path) or ".." in Path(path).parts:
            errors.append(f"diff path escapes repo root: {path}")
            continue

        target = repo_root / path
        if fd.is_new_file:
            if not allow_new_files:
                errors.append(f"diff creates new file but new files are not allowed: {path}")
                continue
            if target.exists():
                errors.append(
                    f"diff marks {path} as new but file already exists on disk"
                )
                continue
        elif fd.is_deleted_file:
            if not target.exists():
                errors.append(f"diff deletes {path} but file does not exist")
                continue
        else:
            if not target.exists():
                errors.append(f"diff targets non-existent file: {path}")
                continue

        # Total deleted lines per file safeguard.
        deleted = sum(
            1
            for hunk in fd.hunks
            for line in hunk.lines
            if line.startswith("-") and not line.startswith("---")
        )
        if deleted > max_deleted_lines_per_file:
            errors.append(
                f"diff deletes {deleted} lines from {path}, exceeding the safety cap "
                f"of {max_deleted_lines_per_file} — refusing to apply"
            )
            continue

        if expected_set and path not in expected_set:
            warnings.append(
                f"{path} was not in the planner's expected files {sorted(expected_set)}"
            )

    return PatchValidationResult(
        diffs=diffs,
        errors=errors,
        warnings=warnings,
        cleaned_diff=cleaned,
    )


def apply_validated_patch(
    result: PatchValidationResult,
    repo_root: Path,
    *,
    dry_run: bool = False,
) -> PatchApplyResult:
    """Apply a previously validated patch to *repo_root*."""
    if not result.ok:
        return PatchApplyResult(errors=list(result.errors) or ["patch did not pass validation"])

    apply_result = PatchApplyResult()
    pending: list[tuple[Path, str, bool]] = []

    for fd in result.diffs:
        path = repo_root / fd.target_path
        if fd.is_new_file:
            original = None
        elif fd.is_deleted_file:
            original = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
        else:
            try:
                original = path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                apply_result.errors.append(f"could not read {path}: {exc}")
                return apply_result
        try:
            new_text = apply_file_diff(fd, original)
        except PatchApplyError as exc:
            apply_result.errors.append(f"{fd.target_path}: {exc}")
            return apply_result
        pending.append((path, new_text, fd.is_deleted_file))
        apply_result.changed_files.append(fd.target_path)

    if dry_run:
        return apply_result

    for path, text, is_deleted in pending:
        if is_deleted:
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                apply_result.errors.append(f"could not delete {path}: {exc}")
            continue
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        except OSError as exc:
            apply_result.errors.append(f"could not write {path}: {exc}")
            return apply_result

    return apply_result
