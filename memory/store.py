"""SQLite-backed memory store with numpy cosine top-k.

At personal-memory scale (a few thousand items) a brute-force dot product over
an in-RAM (N, dim) matrix is faster than any ANN index and has no extra deps.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import load_config


def get_logger(name: str) -> logging.Logger:
    """Module-level logger writing to AGENTIC_BRAIN_LOG. Safe to call repeatedly."""
    log = logging.getLogger(name)
    if not log.handlers:
        path = Path(os.environ.get("AGENTIC_BRAIN_LOG", "/tmp/agentic-brain.log"))
        h = logging.FileHandler(path)
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(h)
        log.setLevel(logging.INFO)
    return log


@dataclass
class Memory:
    id: int
    kind: str
    title: str
    summary: str
    content: str
    tags: list[str]
    source_session_id: str
    source_cwd: str
    source_branch: str
    created_at: str
    score: float = 0.0
    confidence: int = 5
    last_used_at: str | None = None
    use_count: int = 0
    pinned: bool = False
    promoted_to_skill: str | None = None


SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    title TEXT NOT NULL,
    summary TEXT NOT NULL,
    content TEXT NOT NULL,
    tags_json TEXT NOT NULL DEFAULT '[]',
    source_session_id TEXT NOT NULL DEFAULT '',
    source_cwd TEXT NOT NULL DEFAULT '',
    source_branch TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    confidence INTEGER NOT NULL DEFAULT 5,
    last_used_at TEXT,
    use_count INTEGER NOT NULL DEFAULT 0,
    pinned INTEGER NOT NULL DEFAULT 0,
    promoted_to_skill TEXT
);
CREATE TABLE IF NOT EXISTS embeddings (
    memory_id INTEGER PRIMARY KEY,
    vector BLOB NOT NULL,
    dim INTEGER NOT NULL,
    model TEXT NOT NULL,
    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    last_offset INTEGER NOT NULL DEFAULT 0,
    ingested_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS maintenance_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ran_at TEXT NOT NULL,
    sessions_at_run INTEGER NOT NULL,
    purged INTEGER NOT NULL DEFAULT 0,
    promoted INTEGER NOT NULL DEFAULT 0
);
"""

# Indexes created after migration so they can reference newly-added columns.
_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind);
CREATE INDEX IF NOT EXISTS idx_memories_cwd ON memories(source_cwd);
CREATE INDEX IF NOT EXISTS idx_memories_pinned ON memories(pinned);
"""

# Columns added after v0.1; safe to ALTER on existing DBs.
_MIGRATIONS = [
    ("ALTER TABLE memories ADD COLUMN confidence INTEGER NOT NULL DEFAULT 5", "confidence"),
    ("ALTER TABLE memories ADD COLUMN last_used_at TEXT", "last_used_at"),
    ("ALTER TABLE memories ADD COLUMN use_count INTEGER NOT NULL DEFAULT 0", "use_count"),
    ("ALTER TABLE memories ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0", "pinned"),
    ("ALTER TABLE memories ADD COLUMN promoted_to_skill TEXT", "promoted_to_skill"),
]


def _migrate(conn: sqlite3.Connection) -> None:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(memories)").fetchall()}
    for sql, col in _MIGRATIONS:
        if col not in cols:
            conn.execute(sql)


def _db_path() -> Path:
    return Path(os.path.expanduser(load_config().db_path))


def _connect() -> sqlite3.Connection:
    p = _db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p, timeout=10.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> Path:
    conn = _connect()
    try:
        conn.executescript(SCHEMA)
        _migrate(conn)
        conn.executescript(_INDEXES)
        conn.commit()
    finally:
        conn.close()
    return _db_path()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


_FULL_COLS = (
    "id, kind, title, summary, content, tags_json, "
    "source_session_id, source_cwd, source_branch, created_at, "
    "confidence, last_used_at, use_count, pinned, promoted_to_skill"
)


def add_memory(
    kind: str,
    title: str,
    summary: str,
    content: str,
    tags: list[str],
    source_session_id: str,
    source_cwd: str,
    source_branch: str,
    vector: np.ndarray,
    model: str,
    confidence: int = 5,
) -> int:
    conn = _connect()
    try:
        cur = conn.execute(
            """INSERT INTO memories
               (kind, title, summary, content, tags_json,
                source_session_id, source_cwd, source_branch, created_at,
                confidence)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (kind, title, summary, content, json.dumps(tags),
             source_session_id, source_cwd, source_branch, _now(),
             max(1, min(10, int(confidence)))),
        )
        mid = cur.lastrowid
        vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        conn.execute(
            "INSERT INTO embeddings (memory_id, vector, dim, model) VALUES (?, ?, ?, ?)",
            (mid, vec.tobytes(), int(vec.size), model),
        )
        conn.commit()
        return mid
    finally:
        conn.close()


def _row_to_memory(row, score: float = 0.0) -> Memory:
    return Memory(
        id=row[0], kind=row[1], title=row[2], summary=row[3], content=row[4],
        tags=json.loads(row[5] or "[]"),
        source_session_id=row[6], source_cwd=row[7], source_branch=row[8],
        created_at=row[9], score=score,
        confidence=int(row[10]),
        last_used_at=row[11],
        use_count=int(row[12]),
        pinned=bool(row[13]),
        promoted_to_skill=row[14],
    )


def list_memories(kind: str | None = None, cwd_prefix: str | None = None,
                  limit: int = 100) -> list[Memory]:
    conn = _connect()
    try:
        sql = f"SELECT {_FULL_COLS} FROM memories WHERE 1=1"
        args: list = []
        if kind:
            sql += " AND kind = ?"
            args.append(kind)
        if cwd_prefix:
            sql += " AND source_cwd LIKE ?"
            args.append(f"{cwd_prefix}%")
        sql += " ORDER BY datetime(created_at) DESC LIMIT ?"
        args.append(limit)
        rows = conn.execute(sql, args).fetchall()
        return [_row_to_memory(r) for r in rows]
    finally:
        conn.close()


def get_memory(memory_id: int) -> Memory | None:
    conn = _connect()
    try:
        row = conn.execute(
            f"SELECT {_FULL_COLS} FROM memories WHERE id = ?",
            (memory_id,),
        ).fetchone()
        return _row_to_memory(row) if row else None
    finally:
        conn.close()


def bump_usage(ids: list[int]) -> None:
    """Record that these memories were just retrieved (recency + use_count)."""
    if not ids:
        return
    conn = _connect()
    try:
        placeholders = ",".join("?" * len(ids))
        conn.execute(
            f"UPDATE memories SET use_count = use_count + 1, "
            f"last_used_at = ? WHERE id IN ({placeholders})",
            [_now(), *ids],
        )
        conn.commit()
    finally:
        conn.close()


def set_pinned(memory_id: int, pinned: bool) -> bool:
    conn = _connect()
    try:
        cur = conn.execute(
            "UPDATE memories SET pinned = ? WHERE id = ?",
            (1 if pinned else 0, memory_id),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def boost(memory_id: int, amount: int = 2) -> bool:
    """Manually raise a memory's confidence (clamped 1-10)."""
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT confidence FROM memories WHERE id = ?", (memory_id,),
        ).fetchone()
        if not row:
            return False
        new_conf = max(1, min(10, int(row[0]) + amount))
        conn.execute(
            "UPDATE memories SET confidence = ? WHERE id = ?",
            (new_conf, memory_id),
        )
        conn.commit()
        return True
    finally:
        conn.close()


def delete_memory(memory_id: int) -> bool:
    conn = _connect()
    try:
        cur = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        conn.execute("DELETE FROM embeddings WHERE memory_id = ?", (memory_id,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def mark_promoted(ids: list[int], skill_name: str) -> None:
    if not ids:
        return
    conn = _connect()
    try:
        placeholders = ",".join("?" * len(ids))
        conn.execute(
            f"UPDATE memories SET promoted_to_skill = ? WHERE id IN ({placeholders})",
            [skill_name, *ids],
        )
        conn.commit()
    finally:
        conn.close()


def stats() -> dict:
    conn = _connect()
    try:
        total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        by_kind = dict(conn.execute(
            "SELECT kind, COUNT(*) FROM memories GROUP BY kind").fetchall())
        models = dict(conn.execute(
            "SELECT model, COUNT(*) FROM embeddings GROUP BY model").fetchall())
        sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        pinned = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE pinned = 1").fetchone()[0]
        promoted = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE promoted_to_skill IS NOT NULL"
        ).fetchone()[0]
        return {
            "total": total, "by_kind": by_kind, "by_model": models,
            "sessions": sessions, "pinned": pinned, "promoted": promoted,
            "db_path": str(_db_path()),
        }
    finally:
        conn.close()


def session_count() -> int:
    conn = _connect()
    try:
        return conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    finally:
        conn.close()


def last_maintenance_session_count() -> int:
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT sessions_at_run FROM maintenance_runs "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return int(row[0]) if row else 0
    finally:
        conn.close()


def record_maintenance(purged: int, promoted: int) -> None:
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO maintenance_runs (ran_at, sessions_at_run, purged, promoted) "
            "VALUES (?, ?, ?, ?)",
            (_now(), session_count(), purged, promoted),
        )
        conn.commit()
    finally:
        conn.close()


def load_all_for_scoring() -> tuple[list[Memory], np.ndarray | None]:
    """Return every memory plus an aligned embedding matrix (or None)."""
    conn = _connect()
    try:
        rows = conn.execute(
            f"SELECT {_FULL_COLS} FROM memories ORDER BY id"
        ).fetchall()
        memories = [_row_to_memory(r) for r in rows]
        if not memories:
            return [], None
        ids = [m.id for m in memories]
        placeholders = ",".join("?" * len(ids))
        emb_rows = conn.execute(
            f"SELECT memory_id, vector, dim FROM embeddings WHERE memory_id IN ({placeholders})",
            ids,
        ).fetchall()
        by_mid = {r[0]: (np.frombuffer(r[1], dtype=np.float32), r[2]) for r in emb_rows}
        if not by_mid:
            return memories, None
        dim = next(iter(by_mid.values()))[1]
        matrix = np.zeros((len(memories), dim), dtype=np.float32)
        for i, m in enumerate(memories):
            entry = by_mid.get(m.id)
            if entry and entry[1] == dim:
                matrix[i] = entry[0]
        return memories, matrix
    finally:
        conn.close()


def _load_all_vectors(conn: sqlite3.Connection, expected_model: str
                      ) -> tuple[list[int], np.ndarray, list[str]]:
    rows = conn.execute(
        "SELECT memory_id, vector, dim, model FROM embeddings").fetchall()
    if not rows:
        return [], np.zeros((0, 0), dtype=np.float32), []
    ids: list[int] = []
    vecs: list[np.ndarray] = []
    models: list[str] = []
    dim = rows[0][2]
    for mid, blob, d, model in rows:
        if d != dim:
            continue  # mismatched dim — skip; rebuild needed
        ids.append(mid)
        vecs.append(np.frombuffer(blob, dtype=np.float32))
        models.append(model)
    matrix = np.vstack(vecs) if vecs else np.zeros((0, dim), dtype=np.float32)
    return ids, matrix, models


def top_k(query_vec: np.ndarray, k: int = 5, kind: str | None = None,
          cwd_prefix: str | None = None) -> list[Memory]:
    """Cosine top-k. Vectors are L2-normalized at write time, so dot == cosine."""
    if query_vec.size == 0:
        return []
    q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(q))
    if n > 0:
        q = q / n

    conn = _connect()
    try:
        ids, matrix, _ = _load_all_vectors(conn, expected_model="")
        if not ids or matrix.shape[1] != q.shape[0]:
            return []
        scores = matrix @ q
        keep_mask = np.ones(len(ids), dtype=bool)
        if kind or cwd_prefix:
            placeholders = ",".join("?" * len(ids))
            rows = conn.execute(
                f"SELECT id, kind, source_cwd FROM memories WHERE id IN ({placeholders})",
                ids,
            ).fetchall()
            id_to_meta = {r[0]: (r[1], r[2]) for r in rows}
            for i, mid in enumerate(ids):
                m_kind, m_cwd = id_to_meta.get(mid, ("", ""))
                if kind and m_kind != kind:
                    keep_mask[i] = False
                if cwd_prefix and not (m_cwd or "").startswith(cwd_prefix):
                    keep_mask[i] = False

        if not keep_mask.any():
            return []

        idxs = np.where(keep_mask)[0]
        sub_scores = scores[idxs]
        order = np.argsort(-sub_scores)[:k]
        winners = [(int(ids[idxs[o]]), float(sub_scores[o])) for o in order]

        placeholders = ",".join("?" * len(winners))
        rows = conn.execute(
            f"SELECT {_FULL_COLS} FROM memories WHERE id IN ({placeholders})",
            [w[0] for w in winners],
        ).fetchall()
        by_id = {r[0]: r for r in rows}
        return [_row_to_memory(by_id[mid], score=sc) for mid, sc in winners if mid in by_id]
    finally:
        conn.close()


def claude_rerank_top_k(query: str, k: int = 5, kind: str | None = None,
                         cwd_prefix: str | None = None) -> list[Memory]:
    """Ask the Claude CLI to pick the most relevant memories for a query.

    Used when the configured embedder is `claude-rerank`. Loads candidate
    titles+summaries (capped), passes them to `claude -p`, parses returned IDs.
    """
    candidates = list_memories(kind=kind, cwd_prefix=cwd_prefix, limit=200)
    if not candidates:
        return []

    catalog_lines = []
    for m in candidates:
        line = f"[{m.id}] ({m.kind}) {m.title} :: {m.summary}"
        catalog_lines.append(line[:300])
    catalog = "\n".join(catalog_lines)

    prompt = (
        "You are a relevance ranker. Given a query and a numbered catalog of "
        "memory items, return ONLY a JSON array of the top "
        f"{k} most relevant memory IDs (integers), most relevant first. No "
        "prose, no code fence — raw JSON like [3, 17, 2].\n\n"
        f"QUERY: {query}\n\nCATALOG:\n{catalog}"
    )

    import subprocess
    try:
        proc = subprocess.run(
            ["claude", "-p", prompt],
            capture_output=True, text=True, timeout=60,
        )
        out = proc.stdout.strip()
    except Exception:
        return candidates[:k]  # graceful fallback: recency

    # Find first JSON array in output
    import re as _re
    m = _re.search(r"\[[\s\d,]+\]", out)
    if not m:
        return candidates[:k]
    try:
        ids = json.loads(m.group(0))
    except Exception:
        return candidates[:k]

    by_id = {c.id: c for c in candidates}
    picked = [by_id[i] for i in ids if i in by_id]
    return picked[:k] if picked else candidates[:k]


# ---------- session offset tracking (idempotent ingest) ----------

def get_offset(session_id: str) -> int:
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT last_offset FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return int(row[0]) if row else 0
    finally:
        conn.close()


def mark_ingested(session_id: str, last_offset: int) -> None:
    conn = _connect()
    try:
        conn.execute(
            """INSERT INTO sessions (session_id, last_offset, ingested_at)
               VALUES (?, ?, ?)
               ON CONFLICT(session_id) DO UPDATE SET
                 last_offset = excluded.last_offset,
                 ingested_at = excluded.ingested_at""",
            (session_id, last_offset, _now()),
        )
        conn.commit()
    finally:
        conn.close()
