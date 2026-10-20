#!/usr/bin/env bash
# Evaluate the finished noisy-sensor runs (each run is evaluated with the noise it was trained with; terrains as in eval_stage2.py).
#
#   WAIT=1 nohup bash scripts/ucloud/eval_noise.sh > $WS/eval_noise.log 2>&1 &    # waits for "ALL DONE" in the training queue, then evaluates
#   bash scripts/ucloud/eval_noise.sh                                              # evaluate now (whatever runs exist)
#
# Output: $WS/eval_noise/<level>_<mlp|gru>/stage2_eval.csv  (level W/M/S). Options: WORKERS=288 EPISODES=20 CKPTS=8 STAGE1_REF=1
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WS="${WS:-$(cd "$HERE/../../.." && pwd)}"
[ -f "$WS/.ucloud_env.sh" ] && source "$WS/.ucloud_env.sh"
REPO="$WS/terrain-memory-prosthesis"; RES="$WS/myoassist/rl_train/results"
WORKERS="${WORKERS:-288}"; EPISODES="${EPISODES:-20}"; CKPTS="${CKPTS:-8}"; LEVELS="${LEVELS:-W M S}"
if [ "${WAIT:-0}" = 1 ]; then
  until grep -q "ALL DONE" "$WS/noise_logs/queue.log" 2>/dev/null; do sleep 60; done
  echo "training queue finished $(date)"
fi
cd "$WS/myoassist"
for lv in $LEVELS; do
  for ctrl in mlp gru; do
    ls -d "$RES"/stage2_P*_"${ctrl}"_s*_n"${lv}"_* > /dev/null 2>&1 || { echo "no runs for $lv $ctrl"; continue; }
    echo "=== eval $lv $ctrl $(date)"
    python "$REPO/scripts/eval_stage2.py" --runs "$RES"/stage2_P*_"${ctrl}"_s*_n"${lv}"_* --ckpts "$CKPTS" --episodes "$EPISODES" \
      --workers "$WORKERS" --out "$WS/eval_noise/${lv}_${ctrl}" > "$WS/eval_noise/log_${lv}_${ctrl}.txt" 2>&1 || echo "FAILED eval $lv $ctrl"
  done
done
echo "EVAL DONE $(date)"
