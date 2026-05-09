"""Repository scanner.

Walks the working tree, skipping ignored directories and binary files,
records file metadata + a content hash, and emits :class:`FileRecord`
objects that downstream summarisation/retrieval consume.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from ..config import RepositoryConfig

# Map of file extension -> language slug. Extensions are lowercased.
LANGUAGE_BY_EXTENSION: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".cs": "csharp",
    ".scala": "scala",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".sql": "sql",
    ".md": "markdown",
    ".rst": "rst",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".html": "html",
    ".css": "css",
    ".lua": "lua",
}

# A small allow-list of binary-looking extensions we never index even if
# the byte sniff says otherwise.
BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico", ".svgz",
    ".pdf", ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    ".so", ".dylib", ".dll", ".class", ".jar", ".war", ".ear",
    ".o", ".obj", ".a", ".lib", ".pyc", ".pyo", ".whl", ".wasm",
    ".woff", ".woff2", ".ttf", ".otf", ".eot",
    ".mp3", ".mp4", ".mov", ".wav", ".flac", ".ogg", ".webm",
    ".bin", ".dat", ".db", ".sqlite", ".pickle", ".pkl",
}


@dataclass
class FileRecord:
    """A single file discovered by the scanner."""

    path: str
    abs_path: str
    size: int
    language: str
    sha256: str

    def text(self) -> str:
        return Path(self.abs_path).read_text(encoding="utf-8", errors="replace")


def _looks_binary(sample: bytes) -> bool:
    if not sample:
        return False
    if b"\x00" in sample:
        return True
    text_chars = bytes(range(32, 127)) + b"\n\r\t\f\b"
    nontext = sum(1 for b in sample if b not in text_chars)
    return nontext / max(1, len(sample)) > 0.30


def detect_language(path: str | Path) -> str:
    return LANGUAGE_BY_EXTENSION.get(Path(path).suffix.lower(), "text")


def is_ignored(path: Path, root: Path, ignore_dirs: Iterable[str]) -> bool:
    rel = path.resolve().relative_to(root.resolve())
    parts = set(rel.parts)
    return any(d in parts for d in ignore_dirs)


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def scan_repository(
    root: str | Path,
    config: RepositoryConfig | None = None,
) -> Iterator[FileRecord]:
    """Yield :class:`FileRecord` for each text file in the repo."""
    config = config or RepositoryConfig()
    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise NotADirectoryError(f"{root_path} is not a directory")

    ignore_dirs = set(config.ignore_dirs)
    max_bytes = config.max_file_bytes

    for current_root, dirnames, filenames in os.walk(root_path):
        # Mutate dirnames in-place to skip ignored directories.
        dirnames[:] = [d for d in dirnames if d not in ignore_dirs]

        for name in filenames:
            full = Path(current_root) / name
            ext = full.suffix.lower()
            if ext in BINARY_EXTENSIONS:
                continue

            try:
                size = full.stat().st_size
            except OSError:
                continue

            if size > max_bytes:
                continue

            try:
                with full.open("rb") as fh:
                    sample = fh.read(min(size, 8192))
            except OSError:
                continue

            if _looks_binary(sample):
                continue

            try:
                full_bytes = full.read_bytes()
            except OSError:
                continue

            rel = full.resolve().relative_to(root_path).as_posix()
            yield FileRecord(
                path=rel,
                abs_path=str(full.resolve()),
                size=size,
                language=detect_language(full),
                sha256=hash_bytes(full_bytes),
            )
