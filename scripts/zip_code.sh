#!/usr/bin/env bash
# Zips only the CODE of the project (files git knows or would add: tracked + untracked-not-ignored), no results, checkpoints,
# virtualenvs, caches or .git. Run it inside the repo folder:
#   bash scripts/zip_code.sh                 # -> ~/Desktop/terrain-memory-prosthesis-code-<date>.zip
#   bash scripts/zip_code.sh /path/out.zip   # own output path
#   MAXMB=2 bash scripts/zip_code.sh         # skip files bigger than 2 MB (default 5)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
OUT="${1:-$HOME/Desktop/$(basename "$PWD")-code-$(date +%Y%m%d-%H%M).zip}"
MAXMB="${MAXMB:-5}"
LIST="$(mktemp)"
git ls-files --cached --others --exclude-standard | grep -vE '(^|/)(__pycache__|\.venv|venv|node_modules|\.git)/|\.(pyc|zip|npz|pt|pth|ckpt|mp4|gif|png|jpg|jpeg|log)$' | while read -r f; do
  [ -f "$f" ] || continue
  size=$(wc -c < "$f")
  if [ "$size" -gt $((MAXMB * 1024 * 1024)) ]; then echo "skipped (>${MAXMB} MB): $f" >&2; continue; fi
  echo "$f"
done > "$LIST"
rm -f "$OUT"
zip -q "$OUT" -@ < "$LIST"
echo "files: $(wc -l < "$LIST" | tr -d ' ')   size: $(du -h "$OUT" | cut -f1)   -> $OUT"
rm -f "$LIST"
