#!/bin/bash
# Stop hook: kick off reflection in the background so the session ends quickly
# and the existing global git-check Stop hook is not delayed.

set -uo pipefail

input="$(cat)"
stop_active="$(echo "$input" | jq -r '.stop_hook_active // false')"
if [ "$stop_active" = "true" ]; then
  exit 0
fi

session="$(echo "$input" | jq -r '.session_id // ""')"
cwd_in="$(echo "$input" | jq -r '.cwd // ""')"
[ -z "$session" ] && exit 0
[ -z "$cwd_in" ] && cwd_in="${CLAUDE_PROJECT_DIR:-$PWD}"

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(dirname "$0")/../..}"

# Detached background reflection. nohup + disown so it survives session exit.
(
  cd "$PROJECT_DIR" 2>/dev/null || exit 0
  nohup python3 -m memory.cli reflect --session "$session" --cwd "$cwd_in" \
    >> /tmp/agentic-brain.log 2>&1 &
  disown 2>/dev/null || true
) >/dev/null 2>&1

exit 0
