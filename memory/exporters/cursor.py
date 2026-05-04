"""Export memories to .cursorrules (Cursor IDE rules file)."""
from __future__ import annotations

from pathlib import Path

from . import copilot as _copilot


def export(out_path: str | None = None, repo: str | None = None) -> Path:
    """Cursor consumes .cursorrules in the same shape as Copilot's instructions
    file, so we reuse the same content generation."""
    repo_root = Path(repo) if repo else Path.cwd()
    target = Path(out_path) if out_path else repo_root / ".cursorrules"
    # Generate to a temp via copilot, then copy to .cursorrules location
    tmp = _copilot.export(out_path=str(target), repo=str(repo_root))
    return tmp
