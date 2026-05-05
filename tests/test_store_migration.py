"""Migration safety: an old-schema DB upgrades cleanly via init_db()."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from memory import store


OLD_SCHEMA = """
CREATE TABLE memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    title TEXT NOT NULL,
    summary TEXT NOT NULL,
    content TEXT NOT NULL,
    tags_json TEXT NOT NULL DEFAULT '[]',
    source_session_id TEXT NOT NULL DEFAULT '',
    source_cwd TEXT NOT NULL DEFAULT '',
    source_branch TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE TABLE embeddings (
    memory_id INTEGER PRIMARY KEY,
    vector BLOB NOT NULL,
    dim INTEGER NOT NULL,
    model TEXT NOT NULL
);
CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    last_offset INTEGER NOT NULL DEFAULT 0,
    ingested_at TEXT NOT NULL
);
"""


def test_old_schema_db_upgrades_in_place(tmp_path):
    db = Path(store._db_path())
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db)
    try:
        conn.executescript(OLD_SCHEMA)
        conn.execute(
            "INSERT INTO memories (kind, title, summary, content, created_at) "
            "VALUES ('preference', 'old', 'old summary', 'old content', "
            "'2024-01-01T00:00:00+00:00')"
        )
        conn.commit()
    finally:
        conn.close()

    # init_db should add the new columns without losing the row.
    store.init_db()

    conn = sqlite3.connect(db)
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(memories)").fetchall()}
        for needed in ("confidence", "last_used_at", "use_count",
                       "pinned", "promoted_to_skill"):
            assert needed in cols, f"missing {needed}"

        # Maintenance_runs table created.
        tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        assert "maintenance_runs" in tables

        # The original row is intact, with safe defaults for new columns.
        row = conn.execute(
            "SELECT title, confidence, use_count, pinned, promoted_to_skill, "
            "last_used_at FROM memories WHERE title = 'old'"
        ).fetchone()
        assert row[0] == "old"
        assert row[1] == 5      # default confidence
        assert row[2] == 0      # default use_count
        assert row[3] == 0      # default pinned
        assert row[4] is None   # default promoted_to_skill
        assert row[5] is None   # default last_used_at
    finally:
        conn.close()
