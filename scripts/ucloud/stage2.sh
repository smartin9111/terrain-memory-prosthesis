#!/usr/bin/env bash
# Stage 2 (frozen Stage-1 human; only the prosthesis controller learns on per-episode random terrain) on UCloud.
#
#   STAGE1=<session dir> bash scripts/ucloud/stage2.sh measure    # 4 PPO iterations, prints steps/s (~2 min)
#   STAGE1=<session dir> bash scripts/ucloud/stage2.sh full       # ONE run in the background: COND=P1|P2 SEED=0 (default P1 / 0)
#   STAGE1=<session dir> bash scripts/ucloud/stage2.sh batch      # background queue, one run after the other: CONDS="P1 P2" SEEDS="0 1 2"
#
# STAGE1 = the Stage-1 session folder (session_config.json + trained_models/; the last checkpoint is used), e.g.
#   STAGE1=$WS/myoassist/rl_train/results/train_session_20261012-174642
# Options: ENVS=128 (must divide 16384) THREADS=8 TOTAL=20000000 AMP_LO=0.02 AMP_HI=0.06 LOGFREQ=10
#          CONTROLLER=mlp|gru (controllers/registry.py; gru = C2: HISTORY=16 frames, GRU_HIDDEN=32)  PERCEPTION=<name> (default by COND)  DISTANCES="0.2,0.4" (clean preview)  TAG=<label>
# Results: $WS/myoassist/rl_train/results/stage2_<cond>_<controller>_s<seed>_<timestamp>/   (+ log $WS/stage2_<cond>_<controller>_s<seed>_<ts>.log)
# Order inside `batch`: for every seed first P1 then P2 (a stopped queue still leaves complete pairs).
set -euo pipefail
MODE="${1:-}"
[ "$MODE" = measure ] || [ "$MODE" = full ] || [ "$MODE" = batch ] || { sed -n '2,12p' "$0"; exit 1; }
HERE="$(cd "$(dirname "$0")" && pwd)"
WS="${WS:-$(cd "$HERE/../../.." && pwd)}"
# shellcheck disable=SC1091
[ -f "$WS/.ucloud_env.sh" ] && source "$WS/.ucloud_env.sh"
STAGE1="${STAGE1:?set STAGE1=<stage-1 session folder>}"
ENVS="${ENVS:-128}"; THREADS="${THREADS:-8}"; TOTAL="${TOTAL:-20000000}"; LOGFREQ="${LOGFREQ:-10}"
AMP_LO="${AMP_LO:-0.02}"; AMP_HI="${AMP_HI:-0.06}"
CONTROLLER="${CONTROLLER:-mlp}"; HISTORY="${HISTORY:-}"; GRU_HIDDEN="${GRU_HIDDEN:-}"; PERCEPTION="${PERCEPTION:-}"; DISTANCES="${DISTANCES:-}"; TAG="${TAG:-}"
COND="${COND:-P1}"; SEED="${SEED:-0}"; CONDS="${CONDS:-P1 P2}"; SEEDS="${SEEDS:-0 1 2}"
[ $((16384 % ENVS)) -eq 0 ] || { echo "ENVS must divide 16384"; exit 1; }
if [ "$MODE" = measure ]; then TOTAL=65536; LOGFREQ=1000; fi

cd "$WS/myoassist"
run_one() {  # $1=cond $2=seed
  MYOASSIST_NUM_THREADS="$THREADS" python "$WS/terrain-memory-prosthesis/scripts/train_stage2.py" \
    --stage1-session "$STAGE1" --cond "$1" --seed "$2" --timesteps "$TOTAL" --num-envs "$ENVS" \
    --amp-lo "$AMP_LO" --amp-hi "$AMP_HI" --logfreq "$LOGFREQ" --controller "$CONTROLLER" \
    ${PERCEPTION:+--perception "$PERCEPTION"} ${DISTANCES:+--preview-distances "$DISTANCES"} ${TAG:+--tag "$TAG"} \
    ${HISTORY:+--history "$HISTORY"} ${GRU_HIDDEN:+--gru-hidden "$GRU_HIDDEN"}
}
if [ "$MODE" = measure ]; then
  run_one "$COND" "$SEED" 2>&1 | grep -E "fps|total_timesteps|equivalent|Session|Error|Traceback" || true
  echo "== steps/s above (iterations 2-4). TOTAL steps take about TOTAL / fps seconds."
elif [ "$MODE" = full ]; then
  LOG="$WS/stage2_${COND}_${CONTROLLER}_s${SEED}_$(date +%Y%m%d-%H%M%S).log"
  nohup bash -c "$(declare -f run_one); STAGE1='$STAGE1' ENVS=$ENVS THREADS=$THREADS TOTAL=$TOTAL LOGFREQ=$LOGFREQ AMP_LO=$AMP_LO AMP_HI=$AMP_HI CONTROLLER='$CONTROLLER' HISTORY='$HISTORY' GRU_HIDDEN='$GRU_HIDDEN' PERCEPTION='$PERCEPTION' DISTANCES='$DISTANCES' TAG='$TAG' WS='$WS'; run_one $COND $SEED" > "$LOG" 2>&1 &
  echo "started PID $! | log: $LOG"
  echo "session: grep -h 'Session directory' $LOG"
else
  LOG="$WS/stage2_batch_$(date +%Y%m%d-%H%M%S).log"
  nohup bash -c "$(declare -f run_one); STAGE1='$STAGE1' ENVS=$ENVS THREADS=$THREADS TOTAL=$TOTAL LOGFREQ=$LOGFREQ AMP_LO=$AMP_LO AMP_HI=$AMP_HI CONTROLLER='$CONTROLLER' HISTORY='$HISTORY' GRU_HIDDEN='$GRU_HIDDEN' PERCEPTION='$PERCEPTION' DISTANCES='$DISTANCES' TAG='$TAG' WS='$WS'; \
    for s in $SEEDS; do for c in $CONDS; do echo \"=== \$c seed \$s \$(date)\"; run_one \$c \$s || echo \"FAILED \$c \$s\"; done; done; echo ALL DONE" > "$LOG" 2>&1 &
  echo "started batch PID $! | log: $LOG"
  echo "progress: grep -E '^===|Session directory|FAILED|ALL DONE' $LOG"
fi
