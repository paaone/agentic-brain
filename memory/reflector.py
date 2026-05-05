"""Reflector: read a finished transcript, distill durable memories via Claude CLI.

Pipeline: parse → redact → summarize → claude -p → validate → embed → store.
All errors logged to /tmp/agentic-brain.log; nothing raises out of run_reflection.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from . import redact, transcript
from .embed import get_embedder
from .store import (
    add_memory,
    get_logger,
    get_offset,
    init_db,
    last_maintenance_session_count,
    mark_ingested,
    session_count,
)

MAINTENANCE_EVERY = int(os.environ.get("AGENTIC_BRAIN_MAINTENANCE_EVERY", "50"))
_log = get_logger("agentic-brain.reflector")


VALID_KINDS = {"writing-style", "code-pattern", "decision", "snippet",
               "preference", "domain-fact"}

PROMPT_TEMPLATE = """You are a memory-distillation agent. Read the transcript and extract 0-8 DURABLE items the user will want remembered next session: writing voice/tone rules, recurring code patterns/conventions, named decisions with rationale, reusable snippets, stated preferences, project-specific facts. SKIP: ephemeral debugging, secrets, tokens, file contents the user pasted, anything tied to one bug. Each `content` must be self-contained (no "see above").

Also assign each memory a `confidence` 1-10:
  10 = lifelong rule the user clearly endorses (e.g. "always X")
   8 = strong recurring preference, mentioned with conviction
   5 = reasonable guess from one decision
   2 = weak signal, might not generalize
   1 = ephemeral, would only matter for this single bug/task

Return ONLY raw JSON, no prose, no code fence:
{{"memories":[{{"kind":"writing-style|code-pattern|decision|snippet|preference|domain-fact","title":"<=80 chars","summary":"<=240 chars","content":"<=1500 chars, self-contained","tags":["<=6 tags"],"confidence":1-10}}]}}

If nothing durable, return {{"memories":[]}}.

TRANSCRIPT:
<<<
{transcript}
>>>
"""


def _claude_available() -> bool:
    return shutil.which("claude") is not None


def _call_claude(prompt: str, timeout: int = 90) -> str | None:
    """Invoke `claude -p` non-interactively. Returns stdout or None on failure."""
    if not _claude_available():
        _log.warning("claude CLI not on PATH; reflection skipped")
        return None
    try:
        proc = subprocess.run(
            ["claude", "-p", prompt],
            capture_output=True, text=True, timeout=timeout,
            env={**os.environ, "CLAUDECODE": "1"},
        )
    except subprocess.TimeoutExpired:
        _log.warning("claude -p timed out after %ss", timeout)
        return None
    except Exception as e:
        _log.exception("claude -p failed: %s", e)
        return None
    if proc.returncode != 0:
        _log.warning("claude -p exit=%s stderr=%s", proc.returncode, proc.stderr[:500])
        return None
    return proc.stdout


_JSON_OBJ_RE = re.compile(r"\{[\s\S]*\}")


def _parse_response(raw: str) -> list[dict]:
    """Tolerate markdown fences or leading prose around the JSON object."""
    if not raw:
        return []
    # Strip code fences if present
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    # Find first {...} block
    m = _JSON_OBJ_RE.search(cleaned)
    if not m:
        return []
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    items = obj.get("memories") if isinstance(obj, dict) else None
    if not isinstance(items, list):
        return []
    return items


def _validate(item: dict) -> dict | None:
    kind = (item.get("kind") or "").strip()
    if kind not in VALID_KINDS:
        return None
    title = (item.get("title") or "").strip()[:80]
    summary = (item.get("summary") or "").strip()[:240]
    content = (item.get("content") or "").strip()[:1500]
    if not (title and summary and content):
        return None
    tags_raw = item.get("tags") or []
    tags = [str(t).strip().lower()[:32] for t in tags_raw if str(t).strip()][:6]
    try:
        confidence = int(item.get("confidence", 5))
    except (TypeError, ValueError):
        confidence = 5
    confidence = max(1, min(10, confidence))
    return {"kind": kind, "title": title, "summary": summary,
            "content": content, "tags": tags, "confidence": confidence}


def extract_memories(transcript_text: str) -> list[dict]:
    if not transcript_text or len(transcript_text) < 200:
        return []
    prompt = PROMPT_TEMPLATE.format(transcript=transcript_text)
    raw = _call_claude(prompt)
    if raw is None:
        return []
    items = _parse_response(raw)
    out: list[dict] = []
    for it in items:
        v = _validate(it)
        if v:
            out.append(v)
    return out[:8]


def run_reflection(session_id: str, cwd: str,
                   transcript_path: str | None = None) -> int:
    """Read the transcript for (session_id, cwd), extract memories, store them.

    Returns the number of memories saved. Idempotent: re-running on the same
    session won't duplicate (offset is tracked in `sessions` table).
    """
    try:
        init_db()
        path = Path(transcript_path) if transcript_path else \
            transcript.derive_path(session_id, cwd)
        if not path.exists():
            _log.info("transcript not found: %s", path)
            return 0

        turns = transcript.parse(path)
        if len(turns) < 3:
            _log.info("transcript too short (%d turns), skipping", len(turns))
            return 0

        # Idempotency: skip if we've already processed this many turns.
        prev = get_offset(session_id)
        if len(turns) <= prev:
            _log.info("session %s already ingested at offset %d", session_id, prev)
            return 0

        # Drop turns whose text references sensitive files entirely
        clean_turns = [t for t in turns if not redact.looks_sensitive(t.text)]
        for t in clean_turns:
            t.text = redact.scrub(t.text)

        text = transcript.summarize(clean_turns, max_chars=60000)
        if len(text) < 500:
            _log.info("transcript content too small after redaction; skipping")
            return 0

        items = extract_memories(text)
        if not items:
            _log.info("no durable memories extracted for session %s", session_id)
            mark_ingested(session_id, len(turns))
            return 0

        embedder = get_embedder()
        texts = [f"{m['title']}\n{m['summary']}\n{m['content']}" for m in items]
        vecs = embedder.encode(texts)

        branch = clean_turns[0].git_branch if clean_turns else ""
        for m, vec in zip(items, vecs):
            add_memory(
                kind=m["kind"], title=m["title"], summary=m["summary"],
                content=m["content"], tags=m["tags"],
                source_session_id=session_id, source_cwd=cwd, source_branch=branch,
                vector=vec, model=embedder.name,
                confidence=m.get("confidence", 5),
            )

        mark_ingested(session_id, len(turns))
        _log.info("saved %d memories from session %s", len(items), session_id)

        # Auto-maintenance: every Nth session, run purge + skill promotion.
        try:
            _maybe_run_maintenance()
        except Exception as e:  # never fail reflection on a maintenance error
            _log.exception("auto-maintenance failed: %s", e)

        return len(items)
    except Exception as e:  # never propagate from a Stop-hook background job
        _log.exception("run_reflection failed: %s", e)
        return 0


def _maybe_run_maintenance() -> None:
    if MAINTENANCE_EVERY <= 0:
        return
    delta = session_count() - last_maintenance_session_count()
    if delta < MAINTENANCE_EVERY:
        return
    _log.info("auto-maintenance triggered (delta=%d)", delta)
    from .maintenance import run_maintenance
    summary = run_maintenance(dry_run=False)
    _log.info("auto-maintenance: %s", summary)
