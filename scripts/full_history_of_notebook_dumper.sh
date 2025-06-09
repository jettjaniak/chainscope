#!/usr/bin/env bash
set -euo pipefail

# ---- User-tunable constants -----------------------------------------------
HEAD_PATH="notebooks/analyze_faithful_autorating.py"
OUT_DIR="notebooks/analyze_faithful_autorating_full_history"
# ---------------------------------------------------------------------------

mkdir -p "$OUT_DIR"

git log --follow --format='%H %ct' -- "$HEAD_PATH" |
while read -r sha epoch; do
  # Human-readable UTC timestamp: jan_2_15_04
  readable=$(date -u -d "@$epoch" +"%b_%-d_%H_%M" | tr 'A-Z' 'a-z')

  # Pick the correct pathname for this commit.
  if git cat-file -e "${sha}:${HEAD_PATH}" 2>/dev/null; then
      path=$HEAD_PATH
  else
      # Fall back: locate any historical .py with the right stem.
      path=$(git ls-tree -r --name-only "$sha" \
               | grep -E '(analyze_faithful_autorating|google_ai_studio)\.py$' \
               | head -n1)
  fi

  # Skip if we somehow didn’t find the file (paranoia guard).
  [[ -z $path ]] && continue

  # Dump the blob.
  out_file="${OUT_DIR}/${epoch}_${readable}.py"
  git show "${sha}:${path}" > "$out_file"
  echo "wrote $out_file"
done
