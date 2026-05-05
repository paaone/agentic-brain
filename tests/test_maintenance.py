"""Tests for composite scoring and purge."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from memory import maintenance, store
from memory.embed import HashTfidfEmbedder


def _seed(emb: HashTfidfEmbedder, *, kind="code-pattern",
          title="x", session="s1", cwd="/repo",
          confidence=5) -> int:
    v = emb.encode([f"{title} {kind}"])[0]
    return store.add_memory(kind, title, title, f"content of {title}", [],
                            session, cwd, "main", v, emb.name,
                            confidence=confidence)


def test_score_components_in_unit_range():
    store.init_db()
    emb = HashTfidfEmbedder()
    _seed(emb, title="alpha")
    _seed(emb, title="beta")

    scored = maintenance.score_all()
    assert len(scored) == 2
    for s in scored:
        for k, v in s["components"].items():
            assert 0.0 <= v <= 1.0, (k, v)
        assert 0.0 <= s["score"] <= 1.0


def test_pinned_never_in_purge_candidates():
    store.init_db()
    emb = HashTfidfEmbedder()
    mid = _seed(emb, title="weak", confidence=1)
    store.set_pinned(mid, True)

    # Force the row to be very old so its raw score is low.
    conn = store._connect()
    try:
        old = (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()
        conn.execute("UPDATE memories SET created_at = ? WHERE id = ?",
                     (old, mid))
        conn.commit()
    finally:
        conn.close()

    scored = maintenance.score_all()
    cands = maintenance.candidates_for_purge(scored, threshold=1.1, keep_recent=0)
    assert all(c["id"] != mid for c in cands)


def test_keep_recent_protects_newest():
    store.init_db()
    emb = HashTfidfEmbedder()
    ids = [_seed(emb, title=f"t{i}", confidence=1) for i in range(5)]
    scored = maintenance.score_all()
    cands = maintenance.candidates_for_purge(scored, threshold=1.1, keep_recent=3)
    cand_ids = {c["id"] for c in cands}
    # The 3 newest (highest ids) must be protected.
    for protected in ids[-3:]:
        assert protected not in cand_ids


def test_promoted_memories_are_not_purge_candidates():
    store.init_db()
    emb = HashTfidfEmbedder()
    mid = _seed(emb, title="already-a-skill", confidence=1)
    store.mark_promoted([mid], "some-skill")
    scored = maintenance.score_all()
    cands = maintenance.candidates_for_purge(scored, threshold=1.1, keep_recent=0)
    assert all(c["id"] != mid for c in cands)


def test_purge_dry_run_mutates_nothing():
    store.init_db()
    emb = HashTfidfEmbedder()
    for i in range(5):
        _seed(emb, title=f"x{i}", confidence=1)
    before = store.stats()["total"]
    summary = maintenance.purge(threshold=1.1, keep_recent=0, dry_run=True)
    after = store.stats()["total"]
    assert summary["dry_run"] is True
    assert summary["deleted"] == 0
    assert before == after


def test_use_z_zero_when_no_uses():
    store.init_db()
    emb = HashTfidfEmbedder()
    _seed(emb, title="never-used")
    scored = maintenance.score_all()
    assert scored[0]["components"]["use_z"] == 0.0


def test_bump_usage_changes_score():
    store.init_db()
    emb = HashTfidfEmbedder()
    mid = _seed(emb, title="will-be-used")
    before = maintenance.score_all()[0]["score"]
    store.bump_usage([mid])
    store.bump_usage([mid])
    after = maintenance.score_all()[0]["score"]
    assert after > before
