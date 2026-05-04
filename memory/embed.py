"""Embedders for semantic search.

Three local-only backends:

- MiniLMEmbedder      — sentence-transformers all-MiniLM-L6-v2 (384-dim).
- ClaudeRerankEmbedder — no embeddings; relevance is decided by the Claude CLI
                         at query time. `encode` returns zero vectors so storage
                         stays uniform; `top_k` is overridden in store.py.
- HashTfidfEmbedder   — pure stdlib + numpy fallback (512-dim signed hashing).

All vectors are L2-normalized so cosine similarity reduces to dot product.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    name: str
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray: ...


# ---------- MiniLM (local sentence-transformers) ----------

class MiniLMEmbedder:
    name = "minilm:all-MiniLM-L6-v2"
    dim = 384

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer  # lazy import
        self._model = SentenceTransformer("all-MiniLM-L6-v2")

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return vecs.astype(np.float32)


# ---------- Claude rerank (no local embeddings) ----------

class ClaudeRerankEmbedder:
    """Sentinel embedder. Storage uses zero vectors; ranking happens in store.py
    via a Claude CLI call at query time."""
    name = "claude-rerank"
    dim = 1  # tiny placeholder

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.zeros((len(texts), self.dim), dtype=np.float32)


# ---------- Hash-TFIDF fallback ----------

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{1,}")


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _char_ngrams(text: str, n_min: int = 3, n_max: int = 5) -> list[str]:
    s = re.sub(r"\s+", " ", text.lower())
    grams: list[str] = []
    for n in range(n_min, n_max + 1):
        for i in range(0, max(0, len(s) - n + 1)):
            grams.append(s[i:i + n])
    return grams


def _hash_idx(token: str, dim: int) -> tuple[int, int]:
    h = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    idx = int.from_bytes(h[:4], "little") % dim
    sign = 1 if (h[4] & 1) == 0 else -1
    return idx, sign


class HashTfidfEmbedder:
    name = "hash-tfidf-512"
    dim = 512

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            toks = _tokens(text) + _char_ngrams(text)
            if not toks:
                continue
            counts: dict[int, float] = {}
            signs: dict[int, int] = {}
            for tok in toks:
                idx, sign = _hash_idx(tok, self.dim)
                counts[idx] = counts.get(idx, 0.0) + 1.0
                signs[idx] = sign
            # Sublinear TF
            for idx, c in counts.items():
                out[row, idx] = signs[idx] * (1.0 + math.log(c))
            n = float(np.linalg.norm(out[row]))
            if n > 0:
                out[row] /= n
        return out


# ---------- Factory ----------

def get_embedder(name: str | None = None) -> Embedder:
    """Return an embedder by name. Falls back to HashTfidf if MiniLM import fails."""
    from .config import load_config

    if name is None:
        name = load_config().embedder

    if name == "minilm":
        try:
            return MiniLMEmbedder()
        except Exception as e:  # pragma: no cover - env-dependent
            import sys
            print(f"[agentic-brain] MiniLM unavailable ({e}); falling back to hash-tfidf",
                  file=sys.stderr)
            return HashTfidfEmbedder()
    if name == "claude-rerank":
        return ClaudeRerankEmbedder()
    if name in ("hash-tfidf", "hash-tfidf-512"):
        return HashTfidfEmbedder()
    raise ValueError(f"Unknown embedder: {name}")
