"""Pre-extraction secret scrubbing.

Defense in depth: the reflector prompt also tells Claude to skip secrets, but we
remove them mechanically first so they never leave this process.
"""
from __future__ import annotations

import re

# Order matters — more specific patterns first.
_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"), "<REDACTED:anthropic-key>"),
    (re.compile(r"sk-[A-Za-z0-9]{20,}"), "<REDACTED:openai-key>"),
    (re.compile(r"\bgh[ps]_[A-Za-z0-9]{20,}\b"), "<REDACTED:github-token>"),
    (re.compile(r"\bxox[abp]-[A-Za-z0-9-]{10,}\b"), "<REDACTED:slack-token>"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "<REDACTED:aws-key-id>"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]+?-----END [A-Z ]*PRIVATE KEY-----"),
     "<REDACTED:private-key>"),
    # JWT (three base64url segments)
    (re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b"),
     "<REDACTED:jwt>"),
    # KEY=value lines that look like .env entries (long opaque values)
    (re.compile(r"(?m)^([A-Z][A-Z0-9_]{2,})=([^\s\"']{16,})$"),
     r"\1=<REDACTED:env-value>"),
]

# Path patterns that suggest a turn is mostly secret material; drop entirely.
_SENSITIVE_PATH = re.compile(
    r"(?:^|/)(?:\.env(?:\.[a-z]+)?|.*credentials.*|.*\.pem|.*\.key|id_rsa|id_ed25519)\b",
    re.IGNORECASE,
)


def looks_sensitive(text: str) -> bool:
    return bool(_SENSITIVE_PATH.search(text))


def scrub(text: str) -> str:
    """Replace likely-secret substrings with redaction markers."""
    if not text:
        return text
    out = text
    for pat, repl in _PATTERNS:
        out = pat.sub(repl, out)
    return out
