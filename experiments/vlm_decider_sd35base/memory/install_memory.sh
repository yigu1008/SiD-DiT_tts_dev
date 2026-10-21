#!/usr/bin/env bash
# Install these memory notes into Claude Code's per-project memory on THIS machine so a new session recalls them.
# Claude Code keys project memory by the working directory: ~/.claude/projects/<cwd with / -> ->/memory/
# Run from the directory you will start `claude` in (normally the repo root):
#   bash experiments/vlm_decider_sd35base/memory/install_memory.sh            # uses $PWD
#   bash .../install_memory.sh /path/to/project/dir                            # or an explicit project dir
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="${1:-$PWD}"; PROJ="$(cd "$PROJ" && pwd)"
SLUG="$(printf '%s' "$PROJ" | sed 's#/#-#g')"
DEST="${CLAUDE_MEMORY_DIR:-$HOME/.claude/projects/$SLUG/memory}"
mkdir -p "$DEST"
n=0
for f in "$HERE"/*.md; do b="$(basename "$f")"; [ "$b" = "MEMORY.md" ] && continue
  if [ -f "$DEST/$b" ] && ! cmp -s "$f" "$DEST/$b"; then cp "$DEST/$b" "$DEST/$b.bak.$(date +%s)"; fi
  cp "$f" "$DEST/$b"; n=$((n+1)); done
touch "$DEST/MEMORY.md"
added=0
while IFS= read -r line; do [ -z "$line" ] && continue
  key="$(printf '%s' "$line" | sed -E 's/^- \[[^]]*\]\(([^)]*)\).*/\1/')"
  grep -qF "($key)" "$DEST/MEMORY.md" || { printf '%s\n' "$line" >> "$DEST/MEMORY.md"; added=$((added+1)); }
done < "$HERE/MEMORY.md"
echo "installed $n memory files into $DEST; added $added index lines to MEMORY.md (existing entries kept)"
echo "start claude from: $PROJ"
