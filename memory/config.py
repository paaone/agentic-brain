"""Config loader/saver for agentic-brain.

Stores user-chosen DB path and embedder selection at ~/.agentic-brain/config.json.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

CONFIG_DIR = Path(os.path.expanduser("~/.agentic-brain"))
CONFIG_PATH = CONFIG_DIR / "config.json"
DEFAULT_DB = CONFIG_DIR / "memory.db"

VALID_EMBEDDERS = ("minilm", "claude-rerank", "hash-tfidf")


@dataclass
class Config:
    db_path: str = str(DEFAULT_DB)
    embedder: str = "minilm"
    model_name: str = "all-MiniLM-L6-v2"

    def save(self) -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls) -> "Config":
        if not CONFIG_PATH.exists():
            return cls()
        data = json.loads(CONFIG_PATH.read_text())
        return cls(**{k: data[k] for k in data if k in cls.__annotations__})


def interactive_init(autonomous: bool = False) -> Config:
    """Prompt user for embedder + DB location. Use defaults if autonomous."""
    cfg = Config()
    if autonomous or not sys.stdin.isatty():
        cfg.save()
        return cfg

    print("agentic-brain init")
    print("------------------")

    print(f"\nDatabase location:")
    print(f"  1) {DEFAULT_DB}  (recommended — cross-project global brain)")
    print(f"  2) ./data/memory.db  (per-repo only)")
    choice = input("Choose [1]: ").strip() or "1"
    if choice == "2":
        cfg.db_path = str(Path.cwd() / "data" / "memory.db")

    print(f"\nEmbedder (semantic search backend):")
    print(f"  1) MiniLM (sentence-transformers, ~80MB local model — recommended)")
    print(f"  2) claude-rerank (no model; uses your Claude CLI auth at query time)")
    choice = input("Choose [1]: ").strip() or "1"
    if choice == "2":
        cfg.embedder = "claude-rerank"
        cfg.model_name = "claude-rerank"

    cfg.save()
    print(f"\nSaved config to {CONFIG_PATH}")
    print(f"DB will be created at: {cfg.db_path}")
    print(f"Embedder: {cfg.embedder}")
    return cfg


def load_config() -> Config:
    return Config.load()
