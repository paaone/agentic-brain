#!/bin/bash
# SessionStart hook: query agentic-brain for relevant memories and inject them
# as additionalContext so Claude starts with the user's style/patterns loaded.

set -uo pipefail

input="$(cat)"
src="$(echo "$input" | jq -r '.source // ""')"

# Don't double-load on auto-compact restarts
if [ "$src" = "compact" ]; then
  exit 0
fi

session="$(echo "$input" | jq -r '.session_id // ""')"
cwd_in="$(echo "$input" | jq -r '.cwd // ""')"
transcript="$(echo "$input" | jq -r '.transcript_path // ""')"

[ -z "$session" ] && exit 0
[ -z "$cwd_in" ] && cwd_in="$CLAUDE_PROJECT_DIR"

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(dirname "$0")/../..}"

cd "$PROJECT_DIR" 2>/dev/null || exit 0

args=(-m memory.cli prime --session "$session" --cwd "$cwd_in")
[ -n "$transcript" ] && args+=(--transcript "$transcript")

# If python3 fails, exit silently (don't break session start)
python3 "${args[@]}" 2>>/tmp/agentic-brain.log || exit 0
