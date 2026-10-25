#!/usr/bin/env bash
# Follow-up experiments on ONE UCloud machine (same queue mechanism as noise_grid.sh): which runs are queued is chosen with EXPS.
#
#   EXPS="horizon factors gru" STAGE1=<stage-1 session dir> bash scripts/ucloud/exp_grid.sh      # start in the background
#   DRY=1 EXPS=... STAGE1=<...> bash scripts/ucloud/exp_grid.sh                                    # only print the job list
#
# horizon  how much terrain preview is needed: P2 with the first 1..4 look-ahead points, sensor level P (small realistic noise), C1 and C2, seeds 0-3.
#          (reference points that already exist: no preview = P1 at level P, five points up to 1.0 m = P2 at level P.)   32 runs
# factors  noise / delay / dropout ONE AT A TIME at the size of level Q, P2, C1 and C2, seeds 0-3 (Q itself = the three together).   24 runs
# gru      GRU window length K and hidden size, level P, P2, C2 only, seeds 0-3 (reference K=16, hidden 32 = the existing runs).   16 runs
# Replication seeds are NOT here: LEVELS="P Q" CONDS="P2" SEEDS="4 5 6 7" PAR=16 bash scripts/ucloud/noise_grid.sh   (same tags as the first grid)
#
# Order: seed by seed, so a stopped queue leaves complete seeds. Re-running continues existing runs (--resume with their own recorded
# noise) and starts the missing ones; only do that when the old queue is dead (pgrep -f train_stage2).
# Run directory: $WS/myoassist/rl_train/results/stage2_<cond>_<mlp|gru>_s<seed>_<tag>_<stamp>/ ; logs: $WS/exp_logs/
# Tags: hz1..hz4 (number of preview points), fN fD fR (noise / delay / dropout alone), g8_32 g32_32 g16_16 g16_64 (K_hidden)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WS="${WS:-$(cd "$HERE/../../.." && pwd)}"
[ -f "$WS/.ucloud_env.sh" ] && source "$WS/.ucloud_env.sh"
REPO="$WS/terrain-memory-prosthesis"
STAGE1="${STAGE1:?set STAGE1=<stage-1 session folder>}"
STATS="${STATS:-$WS/channel_stats/channel_std.json}"
[ -f "$STATS" ] || { echo "missing $STATS (run scripts/channel_stats.py first)"; exit 1; }
PAR="${PAR:-24}"; ENVS="${ENVS:-32}"; THREADS="${THREADS:-8}"; TOTAL="${TOTAL:-10000000}"
SEEDS="${SEEDS:-0 1 2 3}"; EXPS="${EXPS:-horizon factors gru}"
# sensor levels as in noise_grid.sh: "alpha delay dropout sigma_ang[rad] sigma_vel[rad/s] sigma_force[body weight]"
LEVEL_P="${LEVEL_P:-0 1 0.01 0.005 0.1 0.03}"
LEVEL_FN="${LEVEL_FN:-0 0 0 0.02 0.4 0.12}"   # noise alone, size of Q
LEVEL_FD="${LEVEL_FD:-0 2 0}"                 # delay alone, size of Q (2 steps = 67 ms)
LEVEL_FR="${LEVEL_FR:-0 0 0.05}"              # dropout alone, size of Q (5 %)
LOGS="$WS/exp_logs"; mkdir -p "$LOGS"; RES="$WS/myoassist/rl_train/results"

run_job() {  # $1=tag $2=sensor-level variable $3=cond $4=controller $5=seed $6=extra train arguments ('@' = space)
  local tag="$1" var="LEVEL_$2" cond="$3" ctrl="$4" seed="$5" extra="${6//@/ }" d log
  log="$LOGS/${cond}_${ctrl}_s${seed}_${tag}.log"
  d=$(ls -d "$RES"/stage2_${cond}_${ctrl}_s${seed}_${tag}_* 2>/dev/null | tail -n 1 || true)
  cd "$WS/myoassist"
  if [ -n "$d" ] && ls "$d"/trained_models/model_*.zip > /dev/null 2>&1; then
    eval "$(python - "$d" <<'PY'
import json, sys
env = json.load(open(sys.argv[1] + "/stage2_info.json")).get("perception_env") or {}
for k, v in env.items():
    if k.startswith("MYO_SENSOR_"):
        print(f"export {k}='{v}'")
PY
)"
    MYOASSIST_NUM_THREADS="$THREADS" python "$REPO/scripts/train_stage2.py" --resume "$d" --timesteps "$TOTAL" >> "$log" 2>&1 || echo "FAILED $tag $cond $ctrl $seed (resume)"
    return 0
  fi
  read -r alpha delay drop sa sv sf <<< "${!var}"
  eval "$(python "$REPO/scripts/noise_env.py" "$STATS" --alpha "$alpha" --delay "$delay" --dropout "$drop" --sigma-ang "${sa:-0}" --sigma-vel "${sv:-0}" --sigma-force "${sf:-0}")"
  MYOASSIST_NUM_THREADS="$THREADS" python "$REPO/scripts/train_stage2.py" --stage1-session "$STAGE1" --cond "$cond" --seed "$seed" \
    --timesteps "$TOTAL" --num-envs "$ENVS" --amp-lo 0.02 --amp-hi 0.06 --logfreq 10 --controller "$ctrl" --tag "$tag" $extra \
    > "$log" 2>&1 || echo "FAILED $tag $cond $ctrl $seed"
}
export -f run_job
export RES WS REPO STAGE1 STATS ENVS THREADS TOTAL LOGS LEVEL_P LEVEL_FN LEVEL_FD LEVEL_FR

STAMP="$(date +%H%M%S)"; JOBS="$LOGS/joblist_$STAMP.txt"; : > "$JOBS"
for s in $SEEDS; do
  for e in $EXPS; do
    case "$e" in
      horizon)
        i=0
        for dist in "0.2" "0.2,0.4" "0.2,0.4,0.6" "0.2,0.4,0.6,0.8"; do
          i=$((i + 1))
          for c in mlp gru; do echo "hz$i P P2 $c $s --preview-distances=$dist" >> "$JOBS"; done
        done ;;
      factors)
        for f in FN FD FR; do
          for c in mlp gru; do echo "f${f#F} $f P2 $c $s -" >> "$JOBS"; done
        done ;;
      gru)
        for v in "8 32" "32 32" "16 16" "16 64"; do
          set -- $v
          echo "g$1_$2 P P2 gru $s --history=$1@--gru-hidden=$2" >> "$JOBS"
        done ;;
      *) echo "unknown experiment: $e"; exit 1 ;;
    esac
  done
done
# '-' = no extra arguments
sed -i 's/ -$/ @/' "$JOBS"
echo "$(wc -l < "$JOBS") runs, $PAR in parallel, $TOTAL steps each, ENVS=$ENVS THREADS=$THREADS"
if [ "${DRY:-0}" = 1 ]; then cat "$JOBS"; exit 0; fi
nohup bash -c "xargs -P $PAR -L1 bash -c 'run_job \$0 \$1 \$2 \$3 \$4 \$5' < '$JOBS'; echo ALL DONE" > "$LOGS/queue_$STAMP.log" 2>&1 &
echo "started PID $! | queue log: $LOGS/queue_$STAMP.log | per-run logs: $LOGS/"
echo "progress: ls $RES | grep -c -E '_(hz|f[NDR]|g[0-9])' ; grep FAILED $LOGS/queue_*.log"
