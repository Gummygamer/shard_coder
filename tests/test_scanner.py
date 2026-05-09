"""Repository scanner and symbol extractor tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from shardcoder.config import RepositoryConfig
from shardcoder.indexing.scanner import scan_repository
from shardcoder.indexing.symbols import extract_symbols


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text(
        "import os\n\n"
        "class Greeter:\n"
        "    def hello(self):\n"
        "        return os.getenv('GREETING', 'hi')\n\n"
        "def add(a, b):\n"
        "    return a + b\n"
    )
    (tmp_path / "src" / "ui.tsx").write_text(
        'import React from "react";\n'
        'export function Greeting({name}: {name: string}) {\n'
        '    return <span>{name}</span>;\n'
        '}\n'
        'export class Sidebar {}\n'
        'export interface Props { name: string }\n'
    )
    # Ignored directory contents:
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "ignored.js").write_text(
        "module.exports = function () { return 1 }\n"
    )
    # Binary file (should be skipped).
    (tmp_path / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 256)
    # Unsupported text file (should still be scanned).
    (tmp_path / "README.md").write_text("# hi\n")
    return tmp_path


def test_ignored_directories_are_skipped(repo: Path) -> None:
    paths = {fr.path for fr in scan_repository(repo, RepositoryConfig())}
    assert "src/main.py" in paths
    assert "src/ui.tsx" in paths
    assert "README.md" in paths
    assert all("node_modules" not in p for p in paths)


def test_binary_files_are_skipped(repo: Path) -> None:
    paths = {fr.path for fr in scan_repository(repo, RepositoryConfig())}
    assert "logo.png" not in paths


def test_python_symbols_are_extracted(repo: Path) -> None:
    source = (repo / "src" / "main.py").read_text()
    report = extract_symbols(source, "python")
    names = {s.name for s in report.symbols}
    assert {"Greeter", "hello", "add"}.issubset(names)
    assert "os" in report.imports


def test_typescript_symbols_are_extracted(repo: Path) -> None:
    source = (repo / "src" / "ui.tsx").read_text()
    report = extract_symbols(source, "typescript")
    names = {s.name for s in report.symbols}
    kinds = {s.kind for s in report.symbols}
    assert "Greeting" in names
    assert "Sidebar" in names
    assert "Props" in names
    assert "interface" in kinds
    assert "react" in report.imports


def test_max_file_bytes_is_enforced(tmp_path: Path) -> None:
    big = tmp_path / "big.py"
    big.write_text("x = 1\n" * 1000)
    config = RepositoryConfig(max_file_bytes=10)
    paths = [fr.path for fr in scan_repository(tmp_path, config)]
    assert "big.py" not in paths
