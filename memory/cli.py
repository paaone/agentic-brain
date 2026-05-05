"""agentic-brain CLI.

Subcommands:
  init          — interactive setup (DB path, embedder)
  reflect       — read transcript, distill, store (used by Stop hook)
  prime         — emit SessionStart additionalContext (used by SessionStart hook)
  query         — semantic search; cross-tool entry point
  list          — list recent memories
  stats         — DB stats
  rebuild       — re-embed all memories with a different embedder
  export-copilot, export-cursor — write tool-specific instruction files
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import config, transcript
from .embed import get_embedder
from .exporters import copilot as copilot_exporter
from .exporters import cursor as cursor_exporter
from .reflector import run_reflection
from . import store
from .store import (
    Memory,
    boost,
    bump_usage,
    claude_rerank_top_k,
    delete_memory,
    init_db,
    list_memories,
    set_pinned,
    stats,
    top_k,
)


def _format_memory_md(m: Memory) -> str:
    tag_str = " ".join(f"#{t}" for t in m.tags) if m.tags else ""
    body = m.content if len(m.content) <= 500 else m.content[:500] + "…"
    head = f"### {m.title}  _({m.kind})_"
    meta = f"_{m.source_branch or m.source_cwd}_  {tag_str}".strip()
    return "\n".join(p for p in [head, meta, body] if p)


def _semantic_search(query: str, k: int, kind: str | None,
                     cwd_prefix: str | None,
                     track: bool = True) -> list[Memory]:
    cfg = config.load_config()
    if cfg.embedder == "claude-rerank":
        results = claude_rerank_top_k(query, k=k, kind=kind, cwd_prefix=cwd_prefix)
    else:
        embedder = get_embedder()
        qv = embedder.encode([query])
        if qv.shape[0] == 0:
            return []
        results = top_k(qv[0], k=k, kind=kind, cwd_prefix=cwd_prefix)
    if track and results:
        bump_usage([m.id for m in results])
    return results


# ---------- subcommand handlers ----------

def cmd_init(args: argparse.Namespace) -> int:
    cfg = config.interactive_init(autonomous=args.yes)
    init_db()
    print(f"DB initialized at {cfg.db_path}")
    return 0


def cmd_reflect(args: argparse.Namespace) -> int:
    n = run_reflection(args.session, args.cwd, transcript_path=args.transcript)
    print(f"Reflected {n} memories from session {args.session}")
    return 0


def cmd_prime(args: argparse.Namespace) -> int:
    """Print a SessionStart hookSpecificOutput JSON with relevant memories.

    Strategy: build the query from the most recent user turn (if the transcript
    already exists) plus cwd/branch. If no memories yet, exit silently.
    """
    init_db()
    query_parts: list[str] = []

    tpath = Path(args.transcript) if args.transcript else \
        transcript.derive_path(args.session, args.cwd)
    if tpath.exists():
        turns = transcript.parse(tpath)
        first = transcript.first_user_text(turns)
        if first:
            query_parts.append(first[:1500])
        if turns and turns[0].git_branch:
            query_parts.append(turns[0].git_branch)
    query_parts.append(args.cwd)
    query = "\n".join(query_parts).strip()
    if not query:
        return 0

    try:
        memories = _semantic_search(query, k=args.k, kind=None, cwd_prefix=None)
    except Exception:
        memories = []
    if not memories:
        return 0

    blocks = [_format_memory_md(m) for m in memories]
    md = ("## Memory recall (from past sessions)\n\n"
          "These items were distilled from your prior work to keep your style "
          "and preferences consistent across sessions.\n\n"
          + "\n\n".join(blocks)
          + "\n\n_— agentic-brain_")
    out = {"hookSpecificOutput": {
        "hookEventName": "SessionStart",
        "additionalContext": md,
    }}
    print(json.dumps(out))
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    init_db()
    memories = _semantic_search(args.query, k=args.k, kind=args.kind,
                                cwd_prefix=args.cwd_prefix)
    if args.json:
        print(json.dumps([{
            "id": m.id, "kind": m.kind, "title": m.title, "summary": m.summary,
            "content": m.content, "tags": m.tags,
            "source_cwd": m.source_cwd, "source_branch": m.source_branch,
            "created_at": m.created_at, "score": m.score,
        } for m in memories], indent=2))
        return 0

    if not memories:
        print("(no matches)")
        return 0
    for m in memories:
        print(f"[{m.id}] {m.score:.3f}  ({m.kind})  {m.title}")
        print(f"    {m.summary}")
        if m.tags:
            print(f"    tags: {', '.join(m.tags)}")
        print()
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    init_db()
    memories = list_memories(kind=args.kind, cwd_prefix=args.cwd_prefix,
                             limit=args.limit)
    if not memories:
        print("(no memories yet)")
        return 0
    for m in memories:
        print(f"[{m.id}] {m.created_at}  ({m.kind})  {m.title}")
        print(f"    {m.summary}")
    return 0


def cmd_stats(_args: argparse.Namespace) -> int:
    init_db()
    print(json.dumps(stats(), indent=2))
    return 0


def cmd_rebuild(args: argparse.Namespace) -> int:
    """Re-embed every memory with the chosen embedder. Useful after switching."""
    init_db()
    cfg = config.load_config()
    cfg.embedder = args.embedder
    cfg.model_name = args.embedder
    cfg.save()

    import sqlite3
    import numpy as np
    from .store import _connect, _now  # type: ignore

    embedder = get_embedder()
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT id, title, summary, content FROM memories"
        ).fetchall()
        if not rows:
            print("No memories to rebuild.")
            return 0
        texts = [f"{r[1]}\n{r[2]}\n{r[3]}" for r in rows]
        vecs = embedder.encode(texts)
        conn.execute("DELETE FROM embeddings")
        for (mid, *_), vec in zip(rows, vecs):
            v = np.asarray(vec, dtype=np.float32).reshape(-1)
            conn.execute(
                "INSERT INTO embeddings (memory_id, vector, dim, model) "
                "VALUES (?, ?, ?, ?)",
                (mid, v.tobytes(), int(v.size), embedder.name),
            )
        conn.commit()
        print(f"Rebuilt {len(rows)} embeddings using {embedder.name}")
        return 0
    finally:
        conn.close()


def cmd_purge(args: argparse.Namespace) -> int:
    init_db()
    from .maintenance import purge
    summary = purge(threshold=args.threshold, keep_recent=args.keep_recent,
                    dry_run=args.dry_run)
    print(json.dumps(summary, indent=2))
    return 0


def cmd_promote_skills(args: argparse.Namespace) -> int:
    init_db()
    from .skills import promote
    summary = promote(dry_run=args.dry_run, min_cluster=args.min_cluster,
                      min_sessions=args.min_sessions)
    print(json.dumps(summary, indent=2))
    return 0


def cmd_maintenance(args: argparse.Namespace) -> int:
    init_db()
    from .maintenance import run_maintenance
    summary = run_maintenance(dry_run=args.dry_run)
    print(json.dumps(summary, indent=2))
    return 0


def cmd_pin(args: argparse.Namespace) -> int:
    init_db()
    ok = set_pinned(args.id, True)
    print(f"pinned {args.id}" if ok else f"no memory with id {args.id}")
    return 0 if ok else 1


def cmd_unpin(args: argparse.Namespace) -> int:
    init_db()
    ok = set_pinned(args.id, False)
    print(f"unpinned {args.id}" if ok else f"no memory with id {args.id}")
    return 0 if ok else 1


def cmd_boost(args: argparse.Namespace) -> int:
    init_db()
    ok = boost(args.id, amount=args.by)
    print(f"boosted {args.id} by {args.by}" if ok
          else f"no memory with id {args.id}")
    return 0 if ok else 1


def cmd_forget(args: argparse.Namespace) -> int:
    init_db()
    ok = delete_memory(args.id)
    print(f"deleted {args.id}" if ok else f"no memory with id {args.id}")
    return 0 if ok else 1


def cmd_score(args: argparse.Namespace) -> int:
    init_db()
    from .maintenance import score_all
    scored = score_all()
    hit = next((s for s in scored if s["id"] == args.id), None)
    if not hit:
        print(f"no memory with id {args.id}")
        return 1
    m = hit["memory"]
    out = {
        "id": m.id, "title": m.title, "kind": m.kind,
        "pinned": m.pinned, "promoted_to_skill": m.promoted_to_skill,
        "use_count": m.use_count, "confidence": m.confidence,
        "last_used_at": m.last_used_at, "created_at": m.created_at,
        "score": hit["score"], "components": hit["components"],
    }
    print(json.dumps(out, indent=2))
    return 0


def cmd_export_copilot(args: argparse.Namespace) -> int:
    init_db()
    out = copilot_exporter.export(out_path=args.out, repo=args.repo)
    print(f"Wrote {out}")
    return 0


def cmd_export_cursor(args: argparse.Namespace) -> int:
    init_db()
    out = cursor_exporter.export(out_path=args.out, repo=args.repo)
    print(f"Wrote {out}")
    return 0


# ---------- argparse plumbing ----------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agentic-brain",
                                description="Local semantic memory.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("init", help="Interactive setup")
    pi.add_argument("--yes", action="store_true",
                    help="Use recommended defaults (autonomous mode)")
    pi.set_defaults(func=cmd_init)

    pr = sub.add_parser("reflect",
                        help="Distill memories from a finished session")
    pr.add_argument("--session", required=True)
    pr.add_argument("--cwd", required=True)
    pr.add_argument("--transcript", help="Override transcript path")
    pr.set_defaults(func=cmd_reflect)

    pp = sub.add_parser("prime",
                        help="Emit SessionStart additionalContext JSON")
    pp.add_argument("--session", required=True)
    pp.add_argument("--cwd", required=True)
    pp.add_argument("--transcript")
    pp.add_argument("--k", type=int, default=5)
    pp.set_defaults(func=cmd_prime)

    pq = sub.add_parser("query", help="Semantic search")
    pq.add_argument("query")
    pq.add_argument("--k", type=int, default=5)
    pq.add_argument("--kind")
    pq.add_argument("--cwd-prefix", dest="cwd_prefix")
    pq.add_argument("--json", action="store_true")
    pq.set_defaults(func=cmd_query)

    pl = sub.add_parser("list", help="List recent memories")
    pl.add_argument("--kind")
    pl.add_argument("--cwd-prefix", dest="cwd_prefix")
    pl.add_argument("--limit", type=int, default=20)
    pl.set_defaults(func=cmd_list)

    ps = sub.add_parser("stats")
    ps.set_defaults(func=cmd_stats)

    pb = sub.add_parser("rebuild",
                        help="Re-embed all memories with a different embedder")
    pb.add_argument("--embedder", required=True,
                    choices=("minilm", "claude-rerank", "hash-tfidf"))
    pb.set_defaults(func=cmd_rebuild)

    pp_purge = sub.add_parser("purge",
                              help="Drop low-scoring memories (run periodically)")
    pp_purge.add_argument("--threshold", type=float, default=0.30,
                          help="Composite-score floor (default 0.30)")
    pp_purge.add_argument("--keep-recent", type=int, default=30, dest="keep_recent",
                          help="Always keep the N newest memories")
    pp_purge.add_argument("--dry-run", action="store_true", dest="dry_run")
    pp_purge.set_defaults(func=cmd_purge)

    pp_skills = sub.add_parser("promote-skills",
                               help="Cluster memories into draft SKILL.md files")
    pp_skills.add_argument("--dry-run", action="store_true", dest="dry_run")
    pp_skills.add_argument("--min-cluster", type=int, default=3, dest="min_cluster")
    pp_skills.add_argument("--min-sessions", type=int, default=2, dest="min_sessions")
    pp_skills.set_defaults(func=cmd_promote_skills)

    pp_maint = sub.add_parser("maintenance",
                              help="Run purge + skill promotion together")
    pp_maint.add_argument("--dry-run", action="store_true", dest="dry_run")
    pp_maint.set_defaults(func=cmd_maintenance)

    pp_pin = sub.add_parser("pin", help="Pin a memory (never purged)")
    pp_pin.add_argument("id", type=int)
    pp_pin.set_defaults(func=cmd_pin)

    pp_unpin = sub.add_parser("unpin", help="Unpin a memory")
    pp_unpin.add_argument("id", type=int)
    pp_unpin.set_defaults(func=cmd_unpin)

    pp_boost = sub.add_parser("boost", help="Raise a memory's confidence")
    pp_boost.add_argument("id", type=int)
    pp_boost.add_argument("--by", type=int, default=2)
    pp_boost.set_defaults(func=cmd_boost)

    pp_forget = sub.add_parser("forget", help="Delete a single memory")
    pp_forget.add_argument("id", type=int)
    pp_forget.set_defaults(func=cmd_forget)

    pp_score = sub.add_parser("score", help="Show composite score for one memory")
    pp_score.add_argument("id", type=int)
    pp_score.set_defaults(func=cmd_score)

    pc = sub.add_parser("export-copilot",
                        help="Write .github/copilot-instructions.md")
    pc.add_argument("--out")
    pc.add_argument("--repo", help="Repo root (defaults to cwd)")
    pc.set_defaults(func=cmd_export_copilot)

    pcu = sub.add_parser("export-cursor", help="Write .cursorrules")
    pcu.add_argument("--out")
    pcu.add_argument("--repo")
    pcu.set_defaults(func=cmd_export_cursor)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
