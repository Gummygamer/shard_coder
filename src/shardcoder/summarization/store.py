"""SQLite-backed store for file metadata, symbols and summaries."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    path        TEXT PRIMARY KEY,
    language    TEXT NOT NULL,
    size        INTEGER NOT NULL,
    sha256      TEXT NOT NULL,
    indexed_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS symbols (
    path        TEXT NOT NULL,
    name        TEXT NOT NULL,
    kind        TEXT NOT NULL,
    start_line  INTEGER NOT NULL,
    end_line    INTEGER NOT NULL,
    signature   TEXT,
    PRIMARY KEY (path, name, start_line),
    FOREIGN KEY (path) REFERENCES files(path) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS imports (
    path        TEXT NOT NULL,
    module      TEXT NOT NULL,
    PRIMARY KEY (path, module),
    FOREIGN KEY (path) REFERENCES files(path) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS summaries (
    path        TEXT PRIMARY KEY,
    sha256      TEXT NOT NULL,
    summary_json TEXT NOT NULL,
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);
CREATE INDEX IF NOT EXISTS idx_imports_module ON imports(module);
"""


@dataclass
class StoredSummary:
    path: str
    sha256: str
    summary: dict[str, Any]


class SummaryStore:
    """Thin SQLite wrapper for ShardCoder metadata."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Files
    # ------------------------------------------------------------------

    def upsert_file(
        self,
        path: str,
        language: str,
        size: int,
        sha256: str,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO files(path, language, size, sha256)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                language=excluded.language,
                size=excluded.size,
                sha256=excluded.sha256,
                indexed_at=datetime('now')
            """,
            (path, language, size, sha256),
        )
        self._conn.commit()

    def get_file_hash(self, path: str) -> str | None:
        cur = self._conn.execute("SELECT sha256 FROM files WHERE path=?", (path,))
        row = cur.fetchone()
        return row["sha256"] if row else None

    def all_files(self) -> list[sqlite3.Row]:
        return list(self._conn.execute("SELECT * FROM files ORDER BY path"))

    def known_paths(self) -> set[str]:
        return {row["path"] for row in self._conn.execute("SELECT path FROM files")}

    def delete_file(self, path: str) -> None:
        self._conn.execute("DELETE FROM files WHERE path=?", (path,))
        self._conn.commit()

    # ------------------------------------------------------------------
    # Symbols + imports
    # ------------------------------------------------------------------

    def replace_symbols(
        self,
        path: str,
        symbols: list[dict[str, Any]],
    ) -> None:
        self._conn.execute("DELETE FROM symbols WHERE path=?", (path,))
        if symbols:
            self._conn.executemany(
                """
                INSERT OR REPLACE INTO symbols
                    (path, name, kind, start_line, end_line, signature)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        path,
                        s["name"],
                        s["kind"],
                        s["start_line"],
                        s["end_line"],
                        s.get("signature", ""),
                    )
                    for s in symbols
                ],
            )
        self._conn.commit()

    def replace_imports(self, path: str, imports: list[str]) -> None:
        self._conn.execute("DELETE FROM imports WHERE path=?", (path,))
        if imports:
            self._conn.executemany(
                "INSERT OR IGNORE INTO imports(path, module) VALUES (?, ?)",
                [(path, mod) for mod in imports],
            )
        self._conn.commit()

    def search_symbols(self, query: str, limit: int = 25) -> list[sqlite3.Row]:
        like = f"%{query}%"
        cur = self._conn.execute(
            "SELECT * FROM symbols WHERE name LIKE ? LIMIT ?",
            (like, limit),
        )
        return list(cur)

    def files_importing(self, module: str) -> list[str]:
        cur = self._conn.execute(
            "SELECT path FROM imports WHERE module=? OR module LIKE ?",
            (module, f"{module}.%"),
        )
        return [row["path"] for row in cur]

    def imports_for(self, path: str) -> list[str]:
        cur = self._conn.execute(
            "SELECT module FROM imports WHERE path=? ORDER BY module", (path,)
        )
        return [row["module"] for row in cur]

    def symbols_for(self, path: str) -> list[sqlite3.Row]:
        cur = self._conn.execute(
            "SELECT * FROM symbols WHERE path=? ORDER BY start_line", (path,)
        )
        return list(cur)

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------

    def upsert_summary(self, path: str, sha256: str, summary: dict[str, Any]) -> None:
        self._conn.execute(
            """
            INSERT INTO summaries(path, sha256, summary_json)
            VALUES (?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                sha256=excluded.sha256,
                summary_json=excluded.summary_json,
                updated_at=datetime('now')
            """,
            (path, sha256, json.dumps(summary, ensure_ascii=False)),
        )
        self._conn.commit()

    def get_summary(self, path: str) -> StoredSummary | None:
        cur = self._conn.execute("SELECT * FROM summaries WHERE path=?", (path,))
        row = cur.fetchone()
        if not row:
            return None
        return StoredSummary(
            path=row["path"],
            sha256=row["sha256"],
            summary=json.loads(row["summary_json"]),
        )

    def all_summaries(self) -> list[StoredSummary]:
        out: list[StoredSummary] = []
        for row in self._conn.execute("SELECT * FROM summaries ORDER BY path"):
            out.append(
                StoredSummary(
                    path=row["path"],
                    sha256=row["sha256"],
                    summary=json.loads(row["summary_json"]),
                )
            )
        return out

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def close(self) -> None:
        self._conn.close()
