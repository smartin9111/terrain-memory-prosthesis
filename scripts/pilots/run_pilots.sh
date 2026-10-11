#!/usr/bin/env bash
# Run the cheap pilots in order and stop at the first FAIL (exit code 2).
#   SESSION=<MyoAssist>/rl_train/results/train_session_XXXX  WORKERS=8  bash scripts/pilots/run_pilots.sh
# 00-02 and 03a need a (partly) trained Stage-1 session. 04 needs nothing. 03b is expensive: run it by hand (see README).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
: "${SESSION:?set SESSION=path/to/stage1/train_session_*}"
CONFIG="${CONFIG:-$HERE/../../configs/experiments/stage1_flat_osl.json}"
WORKERS="${WORKERS:-4}"
OUT="${OUT:-pilot_out}"
run() { echo; echo "######## $*"; python "$@"; rc=$?; if [ $rc -eq 2 ]; then echo "FAIL -> stop"; exit 2; fi; }
run "$HERE/04_check_gru_memory.py" --out "$OUT/04"
run "$HERE/01_check_freeze.py" --config "$CONFIG" --out "$OUT/01"
run "$HERE/00_stage1_walk_check.py" --session "$SESSION" --workers "$WORKERS" --out "$OUT/00"
run "$HERE/02_prosthesis_sensitivity.py" --session "$SESSION" --workers "$WORKERS" --out "$OUT/02"
run "$HERE/03a_terrain_calibration.py" --session "$SESSION" --workers "$WORKERS" --out "$OUT/03a"
echo; echo "Done. Next: 03b (expensive) - see README.md"
