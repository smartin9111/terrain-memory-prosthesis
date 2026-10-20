#!/usr/bin/env bash
# Noisy-sensor grid on ONE UCloud machine: LEVELS x CONDS x CONTROLLERS x SEEDS runs, PAR of them at the same time (a queue).
#
#   STAGE1=<stage-1 session dir> bash scripts/ucloud/noise_grid.sh            # start in the background
#   DRY=1 STAGE1=<...> bash scripts/ucloud/noise_grid.sh                      # only print the commands
#
# Needs $WS/channel_stats/channel_std.json (python scripts/channel_stats.py --out $WS/channel_stats ...).
# Levels (edit LEVEL_* below): noise std = alpha x normal per-channel std, delay in control steps (30/s), dropout probability.
# Order: seed by seed (all levels / conditions / controllers of seed 0 first), so a stopped queue leaves complete seeds.
# Re-running this script after the node was lost CONTINUES every existing run (--resume, with its own recorded noise) and starts the missing ones.
#   Only do that when the old queue is dead (pgrep -f train_stage2 shows nothing), otherwise two processes write to one run.
# Results: $WS/myoassist/rl_train/results/stage2_<cond>_<mlp|gru>_s<seed>_n<level>_<stamp>/ ; logs in $WS/noise_logs/
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WS="${WS:-$(cd "$HERE/../../.." && pwd)}"
[ -f "$WS/.ucloud_env.sh" ] && source "$WS/.ucloud_env.sh"
REPO="$WS/terrain-memory-prosthesis"
STAGE1="${STAGE1:?set STAGE1=<stage-1 session folder>}"
STATS="${STATS:-$WS/channel_stats/channel_std.json}"
[ -f "$STATS" ] || { echo "missing $STATS (run scripts/channel_stats.py first)"; exit 1; }
PAR="${PAR:-24}"; ENVS="${ENVS:-32}"; THREADS="${THREADS:-8}"; TOTAL="${TOTAL:-10000000}"; HISTORY="${HISTORY:-16}"
SEEDS="${SEEDS:-0 1 2 3 4 5}"; LEVELS="${LEVELS:-W M S}"; CONDS="${CONDS:-P1 P2}"; CONTROLLERS="${CONTROLLERS:-gru mlp}"
LEVEL_W="0.2 1 0.02"; LEVEL_M="0.5 2 0.05"; LEVEL_S="1.0 3 0.10"   # alpha delay dropout
LOGS="$WS/noise_logs"; mkdir -p "$LOGS"; RES="$WS/myoassist/rl_train/results"

run_job() {  # $1=level $2=cond $3=controller $4=seed
  local lv="$1" cond="$2" ctrl="$3" seed="$4" var="LEVEL_$1" d log
  log="$LOGS/${cond}_${ctrl}_s${seed}_n${lv}.log"
  d=$(ls -d "$RES"/stage2_${cond}_${ctrl}_s${seed}_n${lv}_* 2>/dev/null | tail -n 1 || true)
  cd "$WS/myoassist"
  if [ -n "$d" ] && ls "$d"/trained_models/model_*.zip > /dev/null 2>&1; then
    # the run exists (the queue was stopped, e.g. the node was lost): continue it with the noise it was STARTED with
    eval "$(python - "$d" <<'PY'
import json, sys
env = json.load(open(sys.argv[1] + "/stage2_info.json")).get("perception_env") or {}
for k, v in env.items():
    if k.startswith("MYO_SENSOR_"):
        print(f"export {k}='{v}'")
PY
)"
    MYOASSIST_NUM_THREADS="$THREADS" python "$REPO/scripts/train_stage2.py" --resume "$d" --timesteps "$TOTAL" >> "$log" 2>&1 || echo "FAILED $lv $cond $ctrl $seed (resume)"
    return 0
  fi
  read -r alpha delay drop <<< "${!var}"
  eval "$(python "$REPO/scripts/noise_env.py" "$STATS" --alpha "$alpha" --delay "$delay" --dropout "$drop")"
  MYOASSIST_NUM_THREADS="$THREADS" python "$REPO/scripts/train_stage2.py" --stage1-session "$STAGE1" --cond "$cond" --seed "$seed" \
    --timesteps "$TOTAL" --num-envs "$ENVS" --amp-lo 0.02 --amp-hi 0.06 --logfreq 10 --controller "$ctrl" \
    --tag "n$lv" ${HISTORY:+$( [ "$ctrl" = gru ] && echo "--history $HISTORY" )} \
    > "$log" 2>&1 || echo "FAILED $lv $cond $ctrl $seed"
}
export -f run_job
export RES WS REPO STAGE1 STATS ENVS THREADS TOTAL HISTORY LOGS LEVEL_W LEVEL_M LEVEL_S

JOBS="$LOGS/joblist.txt"; : > "$JOBS"
for s in $SEEDS; do for lv in $LEVELS; do for c in $CONDS; do for k in $CONTROLLERS; do echo "$lv $c $k $s" >> "$JOBS"; done; done; done; done
echo "$(wc -l < "$JOBS") runs, $PAR in parallel, $TOTAL steps each, ENVS=$ENVS THREADS=$THREADS"
if [ "${DRY:-0}" = 1 ]; then cat "$JOBS"; exit 0; fi
nohup bash -c "xargs -P $PAR -L1 bash -c 'run_job \$0 \$1 \$2 \$3' < '$JOBS'; echo ALL DONE" > "$LOGS/queue.log" 2>&1 &
echo "started PID $! | queue log: $LOGS/queue.log | per-run logs: $LOGS/"
echo "progress: ls $WS/myoassist/rl_train/results | grep -c _n ; grep FAILED $LOGS/queue.log"
