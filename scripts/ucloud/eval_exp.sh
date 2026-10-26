#!/usr/bin/env bash
# Evaluate the runs of exp_grid.sh (each run with the sensor noise and preview it was trained with; same terrains as eval_stage2.py).
#
#   bash scripts/ucloud/eval_exp.sh                      # every tag that has runs
#   TAGS="hz1 hz2" bash scripts/ucloud/eval_exp.sh       # only some
# Output: $WS/eval_exp/<tag>_<mlp|gru>/stage2_eval.csv + stage2_summary.csv. Options: WORKERS=288 EPISODES=20 CKPTS=1 (last checkpoint only)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WS="${WS:-$(cd "$HERE/../../.." && pwd)}"
[ -f "$WS/.ucloud_env.sh" ] && source "$WS/.ucloud_env.sh"
REPO="$WS/terrain-memory-prosthesis"; RES="$WS/myoassist/rl_train/results"
WORKERS="${WORKERS:-288}"; EPISODES="${EPISODES:-20}"; CKPTS="${CKPTS:-1}"
TAGS="${TAGS:-hz1 hz2 hz3 hz4 fN fD fR g8_32 g32_32 g16_16 g16_64 g4_32 g64_32}"
mkdir -p "$WS/eval_exp"
cd "$WS/myoassist"
for tag in $TAGS; do
  for ctrl in mlp gru; do
    ls -d "$RES"/stage2_P*_"${ctrl}"_s*_"${tag}"_* > /dev/null 2>&1 || continue
    echo "=== eval $tag $ctrl $(date)"
    python "$REPO/scripts/eval_stage2.py" --runs "$RES"/stage2_P*_"${ctrl}"_s*_"${tag}"_* --ckpts "$CKPTS" --episodes "$EPISODES" \
      --workers "$WORKERS" --out "$WS/eval_exp/${tag}_${ctrl}" > "$WS/eval_exp/log_${tag}_${ctrl}.txt" 2>&1 || echo "FAILED eval $tag $ctrl"
  done
done
echo "EVAL DONE $(date)"
