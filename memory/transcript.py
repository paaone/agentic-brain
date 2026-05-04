"""Parse Claude Code JSONL transcripts.

Transcripts live at ~/.claude/projects/<encoded-cwd>/<session-id>.jsonl.
Each line is a JSON event; we keep user/assistant turns and skip queue-operation
and other internal events.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class Turn:
    role: str             # "user" | "assistant"
    text: str
    timestamp: str
    cwd: str = ""
    git_branch: str = ""


def _encode_cwd(cwd: str) -> str:
    # Claude Code encodes cwd by replacing "/" with "-"; leading "/" becomes leading "-"
    return cwd.replace("/", "-")


def derive_path(session_id: str, cwd: str) -> Path:
    """Compute the JSONL transcript path for a session in a given cwd."""
    base = Path(os.path.expanduser("~/.claude/projects"))
    return base / _encode_cwd(cwd) / f"{session_id}.jsonl"


def _extract_text(message) -> str:
    """Pull plain text out of a transcript message field (str OR list-of-blocks)."""
    if message is None:
        return ""
    if isinstance(message, str):
        return message
    if isinstance(message, dict):
        content = message.get("content")
        return _extract_text(content)
    if isinstance(message, list):
        parts = []
        for block in message:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                t = block.get("type")
                if t == "text":
                    parts.append(block.get("text", ""))
                elif t == "tool_use":
                    name = block.get("name", "tool")
                    parts.append(f"<tool_use:{name}>")
                elif t == "tool_result":
                    body = block.get("content", "")
                    body_text = _extract_text(body) if not isinstance(body, str) else body
                    if len(body_text) > 2000:
                        parts.append(f"<tool_result len={len(body_text)} elided>")
                    else:
                        parts.append(f"<tool_result>{body_text}</tool_result>")
        return "\n".join(p for p in parts if p)
    return str(message)


def parse(jsonl_path: Path | str) -> list[Turn]:
    """Parse a Claude Code transcript JSONL file into a list of Turns."""
    path = Path(jsonl_path)
    turns: list[Turn] = []
    if not path.exists():
        return turns
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                continue
            etype = evt.get("type")
            if etype not in ("user", "assistant"):
                continue
            text = _extract_text(evt.get("message"))
            if not text:
                continue
            turns.append(Turn(
                role=etype,
                text=text,
                timestamp=evt.get("timestamp", ""),
                cwd=evt.get("cwd", ""),
                git_branch=evt.get("gitBranch", ""),
            ))
    return turns


def iter_lines(jsonl_path: Path | str) -> Iterator[dict]:
    """Yield parsed JSON event objects (used by reflector for offset tracking)."""
    path = Path(jsonl_path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def first_user_text(turns: list[Turn]) -> str:
    for t in turns:
        if t.role == "user":
            return t.text
    return ""


def summarize(turns: list[Turn], max_chars: int = 60000) -> str:
    """Render turns into a compact text block for the reflector prompt.

    Strategy: keep the first user prompt verbatim and the last 20 turns; if still
    over max_chars, head/tail truncate the middle with an elision marker.
    """
    if not turns:
        return ""

    rendered = []
    for t in turns:
        prefix = "USER" if t.role == "user" else "ASSISTANT"
        rendered.append(f"[{prefix}] {t.text}")

    full = "\n\n".join(rendered)
    if len(full) <= max_chars:
        return full

    # head/tail truncation
    keep_head = max_chars // 3
    keep_tail = max_chars - keep_head - 60
    head = full[:keep_head]
    tail = full[-keep_tail:]
    elided = len(full) - keep_head - keep_tail
    return f"{head}\n\n[...{elided} chars elided...]\n\n{tail}"
