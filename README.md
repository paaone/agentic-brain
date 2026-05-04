# agentic-brain

A local semantic memory for your dev tools. Every Claude Code session is reflected at the end into a small, durable set of "memories" — your writing voice, recurring code patterns, decisions, preferences. The next session (or your other tools) can pull the most relevant ones back in so you never start from scratch.

Everything lives on your machine. No third‑party APIs.

## What it does

1. **Reflect** — when a Claude Code session ends, a Stop hook spawns a background process that reads the transcript, calls Claude (your existing CLI auth) to extract durable items, embeds them, and saves them to SQLite.
2. **Recall** — when a new session starts, a SessionStart hook injects the top-k relevant memories as `additionalContext`, so Claude opens already imprinted with your style.
3. **Share** — a CLI (`agentic-brain query …`, `agentic-brain export-copilot`) lets GitHub Copilot, Cursor, or anything else use the same memories.

## Install

```bash
cd /home/user/agentic-brain
pip install -e .            # core (numpy)
pip install -e .[minilm]    # add local sentence-transformer embedder
agentic-brain init          # interactive setup
# or:
agentic-brain init --yes    # accept recommended defaults
```

`init` asks two things:

- **Where the DB lives** — default `~/.agentic-brain/memory.db` (cross-project global brain) or `./data/memory.db` (per-repo).
- **Embedder** — `minilm` (recommended; fully local) or `claude-rerank` (no model; uses your Claude CLI auth at query time).

## Hooks (already wired in this repo)

`.claude/settings.json` registers two hooks at the project level:

- `SessionStart` → `.claude/hooks/session-start.sh` runs `agentic-brain prime`
- `Stop` → `.claude/hooks/session-end.sh` spawns a detached `agentic-brain reflect`

Both merge with global hooks (e.g. the existing git‑check Stop hook). Reflection is fire‑and‑forget so it never delays session end.

## CLI

```
agentic-brain init [--yes]
agentic-brain reflect --session SID --cwd CWD [--transcript PATH]
agentic-brain prime   --session SID --cwd CWD [--k 5]
agentic-brain query   "<text>" [--k 5] [--kind ...] [--cwd-prefix /repo] [--json]
agentic-brain list    [--kind ...] [--cwd-prefix ...] [--limit 20]
agentic-brain stats
agentic-brain rebuild --embedder minilm|claude-rerank|hash-tfidf
agentic-brain export-copilot [--out PATH]   # writes .github/copilot-instructions.md
agentic-brain export-cursor  [--out PATH]   # writes .cursorrules
```

## Memory kinds

`writing-style`, `code-pattern`, `decision`, `snippet`, `preference`, `domain-fact`.

## Cross-tool integration

- **Claude Code**: hooks (automatic).
- **GitHub Copilot**: `agentic-brain export-copilot` — emits a `.github/copilot-instructions.md` from your style + pattern memories scoped to the current repo. Run manually or via cron.
- **Cursor**: `agentic-brain export-cursor` — same content, written to `.cursorrules`.
- **Anything else**: `agentic-brain query "..." --json` returns top memories as JSON.

## Privacy

Before transcript text leaves this process, `memory/redact.py` strips common secret shapes (API keys, JWTs, private key blocks, `KEY=value` env lines) and drops turns referencing `.env`, `id_rsa`, etc. The reflector prompt also asks Claude to skip secrets as defense in depth.

## Layout

```
memory/                   # Python package
  config.py               # ~/.agentic-brain/config.json
  transcript.py           # parse Claude Code JSONL
  redact.py               # secret scrubber
  embed.py                # MiniLM | claude-rerank | hash-tfidf (fallback)
  store.py                # SQLite + numpy cosine top-k
  reflector.py            # transcript -> claude -p -> memories
  cli.py                  # entry points
  exporters/{copilot,cursor}.py
.claude/
  settings.json           # registers hooks (project-level)
  hooks/{session-start,session-end}.sh
tests/                    # pytest smoke tests
```

## Troubleshooting

- Reflection logs go to `/tmp/agentic-brain.log`.
- If MiniLM download fails on first use, the system silently falls back to `hash-tfidf` so nothing breaks. Run `agentic-brain rebuild --embedder minilm` once the model is cached.
- DB path: `agentic-brain stats` prints it.
