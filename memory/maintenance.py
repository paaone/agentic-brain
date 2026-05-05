"""Memory maintenance: composite scoring, purge, and skill-promotion driver.

Composite score per memory (all components in 0..1):
    score = 0.4*recency_decay + 0.3*use_z + 0.2*confidence_norm + 0.1*non_redundancy

  recency_decay  = exp(-days_since_last_seen / RECENCY_HALFLIFE_DAYS)
                   (uses created_at if never used)
  use_z          = log(1+use_count) / log(1+max_use_count)
  confidence_norm = confidence / 10
  non_redundancy = 1 - max_cosine_to_other (0 if alone)

Purge rules:
  - Pinned memories are never purged.
  - Always keep the `keep_recent` newest memories regardless of score.
  - Anything below `threshold` after that is purged.
"""
from __future__ import annotations

import math
import os
from datetime import datetime, timezone

import numpy as np

from . import store

_log = store.get_logger("agentic-brain.maintenance")


RECENCY_HALFLIFE_DAYS = float(os.environ.get("AGENTIC_BRAIN_HALFLIFE_DAYS", "90"))
DEFAULT_THRESHOLD = float(os.environ.get("AGENTIC_BRAIN_PURGE_THRESHOLD", "0.30"))
DEFAULT_KEEP_RECENT = int(os.environ.get("AGENTIC_BRAIN_KEEP_RECENT", "30"))
# Above this N, an N x N cosine matrix gets expensive (5k -> 100MB float32).
# Skip the non_redundancy term and use a neutral 0.5 instead.
PAIRWISE_MAX_N = int(os.environ.get("AGENTIC_BRAIN_PAIRWISE_MAX_N", "5000"))


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None


def _days_since(ts: str | None, now: datetime) -> float:
    dt = _parse_iso(ts)
    if dt is None:
        return float("inf")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return max(0.0, (now - dt).total_seconds() / 86400.0)


def _non_redundancy(matrix: np.ndarray | None) -> np.ndarray:
    """For each row, return 1 - max cosine similarity to any *other* row."""
    if matrix is None or matrix.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)
    n = matrix.shape[0]
    if n == 1:
        return np.array([1.0], dtype=np.float32)
    if n > PAIRWISE_MAX_N:
        return np.full(n, 0.5, dtype=np.float32)  # neutral; skip O(N^2) cost
    sims = matrix @ matrix.T
    np.fill_diagonal(sims, -np.inf)
    max_other = sims.max(axis=1)
    max_other = np.clip(max_other, -1.0, 1.0)
    return (1.0 - max_other).astype(np.float32)


def score_all(loaded: tuple[list, np.ndarray | None] | None = None) -> list[dict]:
    """Return [{id, score, components, memory}] for every memory.

    Pass `loaded` to reuse a previous (memories, matrix) load and avoid
    a second sqlite + embedding pull.
    """
    memories, matrix = loaded if loaded is not None else store.load_all_for_scoring()
    if not memories:
        return []
    now = datetime.now(timezone.utc)

    use_counts = np.array([m.use_count for m in memories], dtype=np.float32)
    max_use = float(use_counts.max()) if use_counts.size else 0.0
    log_max = math.log(1.0 + max_use) if max_use > 0 else 1.0
    use_z = np.log1p(use_counts) / log_max if log_max > 0 else np.zeros_like(use_counts)

    recency = np.zeros(len(memories), dtype=np.float32)
    for i, m in enumerate(memories):
        ref = m.last_used_at or m.created_at
        d = _days_since(ref, now)
        if d == float("inf"):
            recency[i] = 0.0
        else:
            recency[i] = float(math.exp(-d / RECENCY_HALFLIFE_DAYS))

    confidence = np.array([m.confidence / 10.0 for m in memories], dtype=np.float32)
    non_red = _non_redundancy(matrix)
    if non_red.size != len(memories):
        non_red = np.ones(len(memories), dtype=np.float32) * 0.5  # neutral fallback

    composite = (0.4 * recency + 0.3 * use_z + 0.2 * confidence + 0.1 * non_red)

    out: list[dict] = []
    for i, m in enumerate(memories):
        out.append({
            "id": m.id,
            "memory": m,
            "score": float(composite[i]),
            "components": {
                "recency": float(recency[i]),
                "use_z": float(use_z[i]),
                "confidence": float(confidence[i]),
                "non_redundancy": float(non_red[i]),
            },
        })
    return out


def candidates_for_purge(scored: list[dict],
                         threshold: float = DEFAULT_THRESHOLD,
                         keep_recent: int = DEFAULT_KEEP_RECENT
                         ) -> list[dict]:
    """Apply pin and keep-recent guards, then filter by threshold."""
    if not scored:
        return []
    # newest-first; tie-break on id so creates-in-the-same-second are deterministic
    by_recent = sorted(scored,
                       key=lambda s: (s["memory"].created_at, s["id"]),
                       reverse=True)
    protected_ids = {s["id"] for s in by_recent[:keep_recent]}
    return [s for s in scored
            if s["score"] < threshold
            and not s["memory"].pinned
            and s["id"] not in protected_ids
            and s["memory"].promoted_to_skill is None]


def purge(threshold: float = DEFAULT_THRESHOLD,
          keep_recent: int = DEFAULT_KEEP_RECENT,
          dry_run: bool = False,
          loaded: tuple[list, np.ndarray | None] | None = None) -> dict:
    scored = score_all(loaded=loaded)
    targets = candidates_for_purge(scored, threshold=threshold,
                                   keep_recent=keep_recent)
    deleted: list[int] = []
    if not dry_run:
        for t in targets:
            if store.delete_memory(t["id"]):
                deleted.append(t["id"])
    summary = {
        "scored": len(scored),
        "candidates": len(targets),
        "deleted": len(deleted) if not dry_run else 0,
        "deleted_ids": deleted,
        "candidate_ids": [t["id"] for t in targets],
        "threshold": threshold,
        "keep_recent": keep_recent,
        "dry_run": dry_run,
    }
    _log.info("purge: %s", summary)
    return summary


def run_maintenance(dry_run: bool = False) -> dict:
    """Run purge + skill promotion + record the maintenance event.

    Loads the memory+embedding snapshot once and reuses it for both passes.
    Skill promotion runs against the post-purge view of memories so anything
    just-deleted can't seed a cluster.
    """
    from .skills import promote as promote_skills

    snapshot = store.load_all_for_scoring()
    purge_summary = purge(dry_run=dry_run, loaded=snapshot)

    if not dry_run and purge_summary["deleted_ids"]:
        # Re-load after deletes so promotion sees the updated set.
        snapshot = store.load_all_for_scoring()
    skill_summary = promote_skills(dry_run=dry_run, loaded=snapshot)

    if not dry_run:
        store.record_maintenance(
            purged=purge_summary["deleted"],
            promoted=skill_summary["promoted"],
        )
    return {"purge": purge_summary, "skills": skill_summary}
