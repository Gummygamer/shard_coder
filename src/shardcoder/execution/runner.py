"""Run validation commands and summarise their output safely."""

from __future__ import annotations

import re
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..tokens.budget import cap_text


@dataclass
class CommandResult:
    command: str
    exit_code: int
    stdout: str
    stderr: str
    elapsed: float
    timed_out: bool = False
    failing_paths: list[str] = field(default_factory=list)
    failing_lines: list[tuple[str, int]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


_PATH_LINE_RE = re.compile(
    r"""
    (?P<path>[\w./\\\-]+\.(?:py|js|ts|tsx|jsx|go|rs|java|kt|rb|php|cpp|c|cs|swift))
    [:\(](?P<line>\d+)
    """,
    re.X,
)


def _detect_failing_locations(text: str) -> tuple[list[str], list[tuple[str, int]]]:
    paths: list[str] = []
    locations: list[tuple[str, int]] = []
    for match in _PATH_LINE_RE.finditer(text):
        path = match.group("path")
        line = int(match.group("line"))
        if path not in paths:
            paths.append(path)
        locations.append((path, line))
    return paths[:10], locations[:25]


def run_command(
    command: str,
    *,
    cwd: str | Path = ".",
    timeout: int = 300,
    env: dict[str, str] | None = None,
) -> CommandResult:
    """Run *command* as a child process and capture its output."""
    if not command:
        return CommandResult(command="", exit_code=0, stdout="", stderr="", elapsed=0.0)

    start = time.monotonic()
    try:
        completed = subprocess.run(
            shlex.split(command),
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            check=False,
        )
    except FileNotFoundError as exc:
        return CommandResult(
            command=command,
            exit_code=127,
            stdout="",
            stderr=f"command not found: {exc}",
            elapsed=time.monotonic() - start,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - start
        return CommandResult(
            command=command,
            exit_code=124,
            stdout=(exc.stdout or "") if isinstance(exc.stdout, str) else "",
            stderr=(exc.stderr or "") if isinstance(exc.stderr, str) else "",
            elapsed=elapsed,
            timed_out=True,
        )

    combined = f"{completed.stdout}\n{completed.stderr}"
    paths, locations = _detect_failing_locations(combined)
    return CommandResult(
        command=command,
        exit_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        elapsed=time.monotonic() - start,
        failing_paths=paths,
        failing_lines=locations,
    )


def summarize_result(result: CommandResult, max_tokens: int = 1000) -> str:
    """Produce a compact text summary suitable for the model context."""
    if not result.command:
        return "(no validation command configured)"
    head = (
        f"$ {result.command}\n"
        f"exit_code={result.exit_code} elapsed={result.elapsed:.2f}s"
        + (" TIMED_OUT" if result.timed_out else "")
    )
    interesting = result.stderr or result.stdout
    excerpt = cap_text(interesting.strip(), max_tokens - 200) if interesting else ""
    paths = (
        ("\nfailing_paths: " + ", ".join(result.failing_paths))
        if result.failing_paths
        else ""
    )
    return f"{head}{paths}\n---\n{excerpt}".strip()
