"""Detect and gate dangerous shell commands."""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass


# Patterns that we always refuse to run unless ``allow_dangerous`` is true.
DANGEROUS_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?:^|\s)rm\s+-r?f"), "uses rm -rf"),
    (re.compile(r"(?:^|\s)sudo(?:\s|$)"), "uses sudo"),
    (re.compile(r"(?:^|\s)chmod\s+-R"), "recursive chmod"),
    (re.compile(r"(?:^|\s)chown\s+-R"), "recursive chown"),
    (re.compile(r"curl\b.*\|\s*sh"), "pipes curl into a shell"),
    (re.compile(r"wget\b.*\|\s*sh"), "pipes wget into a shell"),
    (re.compile(r"git\s+reset\s+--hard"), "git reset --hard"),
    (re.compile(r"git\s+clean\s+-fd"), "git clean -fd"),
    (re.compile(r"git\s+push\s+--force"), "git push --force"),
    (re.compile(r"npm\s+publish"), "package publish command"),
    (re.compile(r"yarn\s+publish"), "package publish command"),
    (re.compile(r"pip\s+upload"), "package publish command"),
    (re.compile(r"twine\s+upload"), "package publish command"),
    (re.compile(r"cargo\s+publish"), "package publish command"),
    (re.compile(r"docker\s+push"), "docker push"),
    (re.compile(r"kubectl\s+apply"), "kubectl apply"),
    (re.compile(r"terraform\s+apply"), "terraform apply"),
    (re.compile(r"aws\s+deploy"), "aws deploy"),
    (re.compile(r"gcloud\s+app\s+deploy"), "gcloud app deploy"),
    (re.compile(r"heroku\s+releases:create"), "heroku release"),
]


@dataclass
class CommandSafetyReport:
    command: str
    is_safe: bool
    reasons: list[str]

    def explain(self) -> str:
        if self.is_safe:
            return "command appears safe"
        return "blocked: " + "; ".join(self.reasons)


def assess_command(command: str) -> CommandSafetyReport:
    """Inspect *command* for dangerous patterns."""
    reasons: list[str] = []
    text = command or ""
    for pattern, label in DANGEROUS_PATTERNS:
        if pattern.search(text):
            reasons.append(label)
    return CommandSafetyReport(
        command=text,
        is_safe=not reasons,
        reasons=reasons,
    )


def split_for_display(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return [command]
