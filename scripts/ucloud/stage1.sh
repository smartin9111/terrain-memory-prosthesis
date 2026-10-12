#!/usr/bin/env bash
# Stage 1 (human + prosthesis learn to walk on flat ground) on UCloud.
#
#   bash scripts/ucloud/stage1.sh measure    # 4 PPO iterations with the real recipe: prints steps/s (~2 min), then you know the full time
#   bash scripts/ucloud/stage1.sh full       # 30M steps in the background (nohup), log + checkpoints in the persistent folder
#
# Recipe: rollout = ENVS x NSTEPS = 16384, minibatch 8192, 30 epochs, lr 1e-4 (previous SDU project). ENVS must divide 16384
# (16, 32, 64, 128); use roughly the number of physical cores, leave a few vCPUs for the PPO update.
#   ENVS=32 (default)  THREADS=8 (torch threads of the update; MyoAssist's measured optimum)  TOTAL=30000000
# Results: $WS/myoassist/rl_train/results/train_session_<timestamp>/  (checkpoints every 10 rollouts ~ 164k steps)
# Default config: stage1_flat_final.json = OSL physics (hard ankle limit + motor model) + 160 m heightfield ground (25 cm cells)
# + no-flight/foot-force and prosthesis-effort penalties (see docs/stage1_findings.md). Older configs, kept for the history:
# stage1_flat_osl_phys.json (same physics, original reward -> hopping), stage1_flat_osl_stiff.json (limit only), stage1_flat_osl.json (stock). Training goes through scripts/train.py (registers our env ids, then runs run_train.py).
# Optional: PREV=<model.zip> warm start (fine-tune a checkpoint, e.g. after a reward change), TOTAL=<steps> then counts the new steps.
# Optional: CFG=<config.json>, TAG=<name> (goes into the log file name), e.g. the control run
#   CFG=$WS/myoassist/rl_train/train/train_configs/imitation_tutorial_22_separated_net_full_obs.json TAG=control TOTAL=6000000 \
#     ENVS=16 THREADS=2 bash scripts/ucloud/stage1.sh full
set -euo pipefail
MODE="${1:-}"
[ "$MODE" = measure ] || [ "$MODE" = full ] || { sed -n '2,16p' "$0"; exit 1; }

HERE="$(cd "$(dirname "$0")" && pwd)"
WS="${WS:-$(cd "$HERE/../../.." && pwd)}"
# shellcheck disable=SC1091
[ -f "$WS/.ucloud_env.sh" ] && source "$WS/.ucloud_env.sh"
ENVS="${ENVS:-32}"; THREADS="${THREADS:-8}"; TOTAL="${TOTAL:-30000000}"; LOGFREQ="${LOGFREQ:-10}"
[ $((16384 % ENVS)) -eq 0 ] || { echo "ENVS must divide 16384 (16, 32, 64)"; exit 1; }
NSTEPS=$((16384 / ENVS))
CFG="${CFG:-$WS/terrain-memory-prosthesis/configs/experiments/stage1_flat_final.json}"; TAG="${TAG:-stage1}"
if [ "$MODE" = measure ]; then TOTAL=65536; LOGFREQ=1000; fi
# PREV=<model.zip>: warm start (fine-tune) from an earlier checkpoint instead of random init
PREV="${PREV:-}"

cd "$WS/myoassist"   # run_train writes rl_train/results/ relative to the current directory = the persistent folder
CMD=(python "$WS/terrain-memory-prosthesis/scripts/train.py" --config_file_path "$CFG"
     --config.env_params.num_envs "$ENVS" --config.ppo_params.n_steps "$NSTEPS"
     --config.total_timesteps "$TOTAL" --config.logger_params.logging_frequency "$LOGFREQ")
if [ -n "$PREV" ]; then CMD+=(--config.env_params.prev_trained_policy_path "$PREV"); fi

if [ "$MODE" = measure ]; then
  MYOASSIST_NUM_THREADS="$THREADS" "${CMD[@]}" 2>&1 | grep -E "fps|total_timesteps|it/s|Error|Traceback" || true
  echo "== steps/s above (use iterations 2-4). 30M steps take about 30000000 / fps seconds."
else
  LOG="$WS/${TAG}_$(date +%Y%m%d-%H%M%S).log"
  MYOASSIST_NUM_THREADS="$THREADS" nohup "${CMD[@]}" > "$LOG" 2>&1 &
  echo "started PID $! | log: $LOG"
  echo "watch:   tail -f $LOG        (Ctrl-C leaves the training running)"
  echo "session: ls -t $WS/myoassist/rl_train/results | head -1"
  echo "check:   python scripts/pilots/00_stage1_walk_check.py --session $WS/myoassist/rl_train/results/<session> --episodes 5 --max-steps 300 --workers 4"
fi
