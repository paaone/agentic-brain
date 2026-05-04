"""Test fixtures: redirect the agentic-brain config + DB to a temp dir per test."""
from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolated_brain(tmp_path, monkeypatch):
    """Each test gets its own ~/.agentic-brain (HOME redirected to tmp_path)."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("AGENTIC_BRAIN_LOG", str(tmp_path / "log.txt"))

    # Reload modules so they pick up the new HOME
    from memory import config as cfg_mod
    importlib.reload(cfg_mod)
    from memory import store
    importlib.reload(store)

    cfg = cfg_mod.Config(
        db_path=str(tmp_path / "memory.db"),
        embedder="hash-tfidf",
        model_name="hash-tfidf",
    )
    cfg.save()
    yield tmp_path
