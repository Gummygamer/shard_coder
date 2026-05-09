"""Pure-Python unified-diff parser and applier.

We deliberately do not call out to the system ``patch`` binary so that
ShardCoder works on every platform. The parser is forgiving but rejects
diffs that do not specify ``--- a/<path>`` / ``+++ b/<path>`` headers, or
whose hunks fail to apply cleanly to the on-disk file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


HUNK_HEADER_RE = re.compile(
    r"^@@\s*-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s*@@"
)
FILE_HEADER_OLD_RE = re.compile(r"^---\s+(.+?)\s*$")
FILE_HEADER_NEW_RE = re.compile(r"^\+\+\+\s+(.+?)\s*$")


@dataclass
class Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[str] = field(default_factory=list)


@dataclass
class FileDiff:
    old_path: str
    new_path: str
    hunks: list[Hunk] = field(default_factory=list)
    is_new_file: bool = False
    is_deleted_file: bool = False

    @property
    def target_path(self) -> str:
        return self.new_path if not self.is_deleted_file else self.old_path


class DiffParseError(ValueError):
    """Raised when a unified diff cannot be parsed."""


class PatchApplyError(RuntimeError):
    """Raised when a hunk cannot be cleanly applied."""


def _strip_prefix(path: str) -> str:
    if path == "/dev/null":
        return path
    for prefix in ("a/", "b/"):
        if path.startswith(prefix):
            return path[len(prefix) :]
    return path


def parse_unified_diff(diff_text: str) -> list[FileDiff]:
    """Parse *diff_text* and return one :class:`FileDiff` per file."""
    if not diff_text or not diff_text.strip():
        raise DiffParseError("empty diff")

    lines = diff_text.splitlines()
    files: list[FileDiff] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        if line.startswith("diff "):
            i += 1
            continue
        if line.startswith("Index "):
            i += 1
            continue
        if line.startswith("---"):
            old_match = FILE_HEADER_OLD_RE.match(line)
            if not old_match:
                raise DiffParseError(f"malformed --- header: {line!r}")
            if i + 1 >= n:
                raise DiffParseError("--- header without +++ counterpart")
            new_line = lines[i + 1]
            new_match = FILE_HEADER_NEW_RE.match(new_line)
            if not new_match:
                raise DiffParseError(f"expected +++ header, got {new_line!r}")
            old_path = _strip_prefix(old_match.group(1))
            new_path = _strip_prefix(new_match.group(1))
            file_diff = FileDiff(
                old_path=old_path,
                new_path=new_path,
                is_new_file=(old_path == "/dev/null"),
                is_deleted_file=(new_path == "/dev/null"),
            )
            i += 2
            while i < n and lines[i].startswith("@@"):
                hunk_header = HUNK_HEADER_RE.match(lines[i])
                if not hunk_header:
                    raise DiffParseError(f"malformed hunk header: {lines[i]!r}")
                old_start = int(hunk_header.group(1))
                old_count = int(hunk_header.group(2) or 1)
                new_start = int(hunk_header.group(3))
                new_count = int(hunk_header.group(4) or 1)
                i += 1
                hunk = Hunk(old_start, old_count, new_start, new_count)
                while i < n and not lines[i].startswith("@@") and not lines[i].startswith("--- "):
                    if lines[i].startswith("\\ No newline"):
                        i += 1
                        continue
                    if lines[i] == "" or lines[i][0] in {" ", "+", "-"}:
                        hunk.lines.append(lines[i])
                        i += 1
                    else:
                        break
                file_diff.hunks.append(hunk)
            files.append(file_diff)
        else:
            i += 1

    if not files:
        raise DiffParseError("no file headers found in diff")
    return files


def apply_file_diff(file_diff: FileDiff, original_text: str | None) -> str:
    """Apply a single :class:`FileDiff` to *original_text* and return the new text.

    Pass ``original_text=None`` for new files.
    """
    if file_diff.is_new_file:
        if original_text not in (None, ""):
            raise PatchApplyError(
                f"diff marks {file_diff.new_path} as new, but file is not empty"
            )
        new_lines: list[str] = []
        for hunk in file_diff.hunks:
            for line in hunk.lines:
                if line.startswith("+"):
                    new_lines.append(line[1:])
        return "\n".join(new_lines) + ("\n" if new_lines else "")

    if file_diff.is_deleted_file:
        return ""

    if original_text is None:
        raise PatchApplyError(f"file not found for diff: {file_diff.old_path}")

    original_lines = original_text.splitlines()
    out: list[str] = []
    cursor = 0  # zero-based index into original_lines

    for hunk in file_diff.hunks:
        target_index = hunk.old_start - 1
        if target_index < cursor:
            raise PatchApplyError("hunks are out of order")
        out.extend(original_lines[cursor:target_index])
        cursor = target_index

        for line in hunk.lines:
            if not line:
                # Blank line inside diff: treat as context blank.
                if cursor >= len(original_lines) or original_lines[cursor] != "":
                    raise PatchApplyError(
                        f"context mismatch in {file_diff.old_path} at line {cursor + 1}: expected blank line"
                    )
                out.append("")
                cursor += 1
                continue
            tag, content = line[0], line[1:]
            if tag == " ":
                if cursor >= len(original_lines) or original_lines[cursor] != content:
                    raise PatchApplyError(
                        f"context mismatch in {file_diff.old_path} at line {cursor + 1}: "
                        f"expected {content!r}, got "
                        f"{original_lines[cursor] if cursor < len(original_lines) else '<EOF>'!r}"
                    )
                out.append(content)
                cursor += 1
            elif tag == "-":
                if cursor >= len(original_lines) or original_lines[cursor] != content:
                    raise PatchApplyError(
                        f"removal mismatch in {file_diff.old_path} at line {cursor + 1}: "
                        f"expected {content!r}, got "
                        f"{original_lines[cursor] if cursor < len(original_lines) else '<EOF>'!r}"
                    )
                cursor += 1
            elif tag == "+":
                out.append(content)
            else:
                raise PatchApplyError(f"unexpected line in hunk: {line!r}")

    out.extend(original_lines[cursor:])
    trailing_nl = original_text.endswith("\n") or original_text == ""
    text = "\n".join(out)
    if trailing_nl and not text.endswith("\n"):
        text += "\n"
    return text


def apply_unified_diff(diff_text: str, file_reader, file_writer) -> list[str]:
    """Parse and apply a unified diff via the supplied I/O callables.

    *file_reader* maps path -> str | None (None for missing).
    *file_writer* takes (path, text) and writes the file.

    Returns the list of changed paths.
    """
    diffs = parse_unified_diff(diff_text)
    changed: list[str] = []
    for fd in diffs:
        path = fd.target_path
        original = file_reader(fd.old_path if not fd.is_new_file else fd.new_path)
        new_text = apply_file_diff(fd, original)
        file_writer(path, new_text)
        changed.append(path)
    return changed
