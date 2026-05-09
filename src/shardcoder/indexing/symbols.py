"""Lightweight, regex-based symbol extraction.

Tree-sitter would be more accurate, but regex extraction works without
native dependencies and is good enough to feed the retriever and
summariser. Each extractor is intentionally permissive: when in doubt we
prefer to over-include a symbol rather than miss it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Symbol:
    name: str
    kind: str  # "function" | "class" | "method" | "const" | "interface" | ...
    start_line: int
    end_line: int
    signature: str = ""


@dataclass
class SymbolReport:
    language: str
    imports: list[str] = field(default_factory=list)
    symbols: list[Symbol] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------

_PY_IMPORT_RE = re.compile(r"^\s*(?:from\s+([\w\.]+)\s+import|import\s+([\w\.]+))", re.M)
_PY_DEF_RE = re.compile(r"^(?P<indent>[ \t]*)def\s+(?P<name>[A-Za-z_]\w*)\s*\(", re.M)
_PY_CLASS_RE = re.compile(r"^(?P<indent>[ \t]*)class\s+(?P<name>[A-Za-z_]\w*)\s*[:\(]", re.M)


def _python_block_end(lines: list[str], start_idx: int, indent: str) -> int:
    """Find the end line (1-indexed) of a Python block starting at start_idx."""
    n = len(lines)
    i = start_idx + 1
    while i < n:
        line = lines[i]
        stripped = line.strip()
        if stripped:
            current_indent = re.match(r"[ \t]*", line).group(0)
            if len(current_indent) <= len(indent) and not line.startswith(indent + (" " if indent else "")):
                if not stripped.startswith(("#",)):
                    return i  # 1-indexed end is i (the line before this one was last)
        i += 1
    return n


def extract_python(source: str) -> SymbolReport:
    imports: list[str] = []
    for match in _PY_IMPORT_RE.finditer(source):
        mod = match.group(1) or match.group(2)
        if mod:
            imports.append(mod)

    lines = source.splitlines()
    symbols: list[Symbol] = []
    for match in _PY_CLASS_RE.finditer(source):
        line_no = source[: match.start()].count("\n")
        indent = match.group("indent")
        end = _python_block_end(lines, line_no, indent)
        signature_line = lines[line_no].strip().rstrip(":")
        symbols.append(
            Symbol(
                name=match.group("name"),
                kind="class",
                start_line=line_no + 1,
                end_line=end,
                signature=signature_line,
            )
        )

    for match in _PY_DEF_RE.finditer(source):
        line_no = source[: match.start()].count("\n")
        indent = match.group("indent")
        end = _python_block_end(lines, line_no, indent)
        signature_line = lines[line_no].strip().rstrip(":")
        kind = "method" if indent else "function"
        symbols.append(
            Symbol(
                name=match.group("name"),
                kind=kind,
                start_line=line_no + 1,
                end_line=end,
                signature=signature_line,
            )
        )

    symbols.sort(key=lambda s: s.start_line)
    return SymbolReport("python", imports, symbols)


# ---------------------------------------------------------------------------
# JavaScript / TypeScript
# ---------------------------------------------------------------------------

_JS_IMPORT_RE = re.compile(
    r"""(?:
        import\s+(?:[\w*\s{},]+)\s+from\s+['"]([^'"]+)['"]   # import x from "y"
      | import\s+['"]([^'"]+)['"]                              # import "y"
      | require\(\s*['"]([^'"]+)['"]\s*\)                      # require("y")
    )
    """,
    re.X | re.M,
)

_JS_FUNC_RE = re.compile(
    r"""^(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\(""",
    re.M,
)

_JS_ARROW_RE = re.compile(
    r"""^(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>""",
    re.M,
)

_JS_CLASS_RE = re.compile(
    r"""^(?:export\s+)?(?:abstract\s+)?class\s+([A-Za-z_$][\w$]*)""",
    re.M,
)

_TS_INTERFACE_RE = re.compile(
    r"""^(?:export\s+)?interface\s+([A-Za-z_$][\w$]*)""",
    re.M,
)

_TS_TYPE_RE = re.compile(
    r"""^(?:export\s+)?type\s+([A-Za-z_$][\w$]*)\s*=""",
    re.M,
)


def _block_end_by_braces(lines: list[str], start_idx: int) -> int:
    depth = 0
    started = False
    for i in range(start_idx, len(lines)):
        for ch in lines[i]:
            if ch == "{":
                depth += 1
                started = True
            elif ch == "}":
                depth -= 1
                if started and depth == 0:
                    return i + 1
    return len(lines)


def extract_js_like(source: str, language: str) -> SymbolReport:
    imports: list[str] = []
    for match in _JS_IMPORT_RE.finditer(source):
        mod = match.group(1) or match.group(2) or match.group(3)
        if mod:
            imports.append(mod)

    lines = source.splitlines()
    symbols: list[Symbol] = []

    def _add(match: re.Match[str], kind: str) -> None:
        line_no = source[: match.start()].count("\n")
        end = _block_end_by_braces(lines, line_no)
        signature = lines[line_no].strip()
        symbols.append(
            Symbol(
                name=match.group(1),
                kind=kind,
                start_line=line_no + 1,
                end_line=end,
                signature=signature,
            )
        )

    for match in _JS_CLASS_RE.finditer(source):
        _add(match, "class")
    for match in _JS_FUNC_RE.finditer(source):
        _add(match, "function")
    for match in _JS_ARROW_RE.finditer(source):
        _add(match, "function")
    if language == "typescript":
        for match in _TS_INTERFACE_RE.finditer(source):
            line_no = source[: match.start()].count("\n")
            end = _block_end_by_braces(lines, line_no)
            symbols.append(
                Symbol(
                    name=match.group(1),
                    kind="interface",
                    start_line=line_no + 1,
                    end_line=end,
                    signature=lines[line_no].strip(),
                )
            )
        for match in _TS_TYPE_RE.finditer(source):
            line_no = source[: match.start()].count("\n")
            symbols.append(
                Symbol(
                    name=match.group(1),
                    kind="type",
                    start_line=line_no + 1,
                    end_line=line_no + 1,
                    signature=lines[line_no].strip(),
                )
            )

    symbols.sort(key=lambda s: s.start_line)
    return SymbolReport(language, imports, symbols)


# ---------------------------------------------------------------------------
# Generic fallback
# ---------------------------------------------------------------------------


def extract_generic(source: str, language: str) -> SymbolReport:
    """Heuristic catch-all extractor for unsupported languages."""
    imports: list[str] = []
    return SymbolReport(language, imports, [])


def extract_symbols(source: str, language: str) -> SymbolReport:
    """Dispatch to the language-specific extractor."""
    if language == "python":
        return extract_python(source)
    if language in {"javascript", "typescript"}:
        return extract_js_like(source, language)
    return extract_generic(source, language)
