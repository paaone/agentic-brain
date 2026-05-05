"""Tests for skill clustering, scope detection, and SKILL.md generation."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from memory import skills, store
from memory.embed import HashTfidfEmbedder


@pytest.fixture(autouse=True)
def _no_claude_cli(monkeypatch):
    """Force the stub-template path so tests don't shell out."""
    monkeypatch.setattr(skills, "_claude_generate", lambda slug, members: None)


def _add(emb, *, title, kind="code-pattern", session, cwd, content=None):
    text = content or title
    v = emb.encode([f"{title} {text}"])[0]
    return store.add_memory(kind, title, title, text, [], session, cwd,
                            "main", v, emb.name, confidence=7)


def test_cluster_in_one_repo_writes_project_scope(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    store.init_db()
    emb = HashTfidfEmbedder()
    repo = tmp_path / "repoA"
    repo.mkdir()
    # 3 near-identical memories spanning 2 sessions, all in one repo.
    for i, sess in enumerate(["sA", "sB", "sA"]):
        _add(emb, title="prefer-fstrings", session=sess, cwd=str(repo),
             content="always prefer python f-strings over .format() and %")

    summary = skills.promote(min_cluster=3, min_sessions=2)
    assert summary["promoted"] == 1
    s = summary["skills"][0]
    assert s["scope"] == "project"
    assert Path(s["path"]).is_relative_to(repo / ".claude" / "skills")
    assert Path(s["path"]).exists()
    body = Path(s["path"]).read_text()
    assert body.startswith("---\n"), "missing YAML frontmatter"
    assert "name:" in body and "description:" in body

    # All three memories should now be marked promoted.
    for mid in s["member_ids"]:
        m = store.get_memory(mid)
        assert m is not None and m.promoted_to_skill == s["slug"]


def test_cluster_spanning_repos_writes_global_scope(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    store.init_db()
    # Re-import skills so its module-level GLOBAL_SKILLS_DIR honours the new HOME.
    import importlib
    importlib.reload(skills)
    monkeypatch.setattr(skills, "_claude_generate", lambda slug, members: None)

    emb = HashTfidfEmbedder()
    for i, (sess, cwd) in enumerate([("s1", "/a"), ("s2", "/b"), ("s3", "/c")]):
        _add(emb, title="prefer-fstrings", session=sess, cwd=cwd,
             content="always prefer python f-strings over .format() and %")

    summary = skills.promote(min_cluster=3, min_sessions=2)
    assert summary["promoted"] == 1
    s = summary["skills"][0]
    assert s["scope"] == "global"
    assert ".claude/skills" in s["path"]


def test_too_small_cluster_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    store.init_db()
    emb = HashTfidfEmbedder()
    _add(emb, title="lone", session="s1", cwd="/r")
    _add(emb, title="lone", session="s2", cwd="/r")
    summary = skills.promote(min_cluster=3, min_sessions=2)
    assert summary["promoted"] == 0


def test_dry_run_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    store.init_db()
    emb = HashTfidfEmbedder()
    repo = tmp_path / "repo"
    repo.mkdir()
    for sess in ["sA", "sB", "sC"]:
        _add(emb, title="prefer-fstrings", session=sess, cwd=str(repo),
             content="always prefer python f-strings over .format() and %")

    summary = skills.promote(dry_run=True, min_cluster=3, min_sessions=2)
    assert summary["promoted"] == 1
    assert summary["dry_run"] is True
    skill_path = Path(summary["skills"][0]["path"])
    assert not skill_path.exists()
    # Memories must NOT be marked promoted on a dry run.
    for mid in summary["skills"][0]["member_ids"]:
        m = store.get_memory(mid)
        assert m is not None and m.promoted_to_skill is None


def test_existing_proposed_dir_does_not_overwrite(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    store.init_db()
    emb = HashTfidfEmbedder()
    repo = tmp_path / "repo"
    (repo / ".claude" / "skills" / "proposed-prefer-fstrings").mkdir(parents=True)

    for sess in ["sA", "sB", "sC"]:
        _add(emb, title="prefer-fstrings", session=sess, cwd=str(repo),
             content="always prefer python f-strings over .format() and %")

    summary = skills.promote(min_cluster=3, min_sessions=2)
    assert summary["promoted"] == 1
    assert "proposed-prefer-fstrings-2" in summary["skills"][0]["path"]
