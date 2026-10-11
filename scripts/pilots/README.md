# Pilot validation scripts

Goal: settle the biggest doubts cheaply *before* the long (hours-per-run) trainings. Each script asks one question and ends
with a verdict (PASS / WARN / FAIL), a table (CSV + JSON), a figure and an exit code (0 / 1 / 2). Outputs go to `pilot_out/<id>/`.

Run everything from an activated environment (see the main README, "Setup"). Paths (`--session`, `--config`, `--out`) are
resolved against the directory you start the script from; the scripts switch to the MyoAssist checkout themselves
(its configs use relative paths). The checkout is found automatically (installed `rl_train`, `$MYOASSIST_DIR`, or `../myoassist`).

| # | Script | Question | Needs | Time |
|---|---|---|---|---|
| 04 | `04_check_gru_memory.py` | Does a recurrent PPO learn a memory task that a feed-forward PPO cannot? | nothing (no MuJoCo) | ~5 min |
| 01 | `01_check_freeze.py` | Does the human policy really stay frozen in Stage 2? | a config | ~1 min |
| 00 | `00_stage1_walk_check.py` | Does the human learn to walk with the prosthesis (Stage 1)? | running / finished Stage 1 session | minutes |
| 02 | `02_prosthesis_sensitivity.py` | Does the prosthesis matter, and is the degradation smooth or a cliff? | Stage 1 checkpoint | 10-30 min |
| 03a | `03a_terrain_calibration.py` | Which terrain amplitude leaves something to learn (headroom)? | Stage 1 checkpoint | 10-30 min |
| 03b | `03b_p1_vs_p2_short_train.py` | Does terrain preview (P2) help at all vs. P1 after the same short budget? | Stage 1 checkpoint + band from 03a | hours |

```bash
# cheap ones, stops at the first FAIL
SESSION=../myoassist/rl_train/results/train_session_XXXX WORKERS=8 bash scripts/pilots/run_pilots.sh

python scripts/pilots/00_stage1_walk_check.py --session <session> --workers 8          # works while Stage 1 is still running
python scripts/pilots/01_check_freeze.py --config configs/experiments/stage1_flat_osl.json --stage1 <ckpt.zip>
python scripts/pilots/02_prosthesis_sensitivity.py --session <session> --episodes 20 --workers 8
python scripts/pilots/03a_terrain_calibration.py   --session <session> --episodes 10 --workers 8

# 03b is expensive: plumbing smoke first (minutes), then the real pilot
python scripts/pilots/03b_p1_vs_p2_short_train.py --config configs/experiments/stage1_flat_osl.json --stage1 <ckpt.zip> --smoke
python scripts/pilots/03b_p1_vs_p2_short_train.py --config configs/experiments/stage1_flat_osl.json --stage1 <ckpt.zip> \
       --timesteps 3000000 --seeds 3 --num-envs 16 --parallel 2 --amplitude 0.03        # amplitude from 03a

python scripts/pilots/04_check_gru_memory.py --timesteps 100000
python scripts/pilots/04_check_gru_memory.py --agent my_gru_agent:train_and_eval        # your own GRU: f(make_env, timesteps, seed, n_eval) -> accuracy
```

Long runs (03b, UCloud): `nohup ... &` or tmux. 03b is **resumable**: re-run with the same `--out` and finished runs (`runs/*/result.json`) are skipped.
Cores: `--workers` (00/02/03a) and `--parallel x --num-envs` (03b) should not exceed the machine's cores. Threads are capped to 1, MuJoCo is headless (`MUJOCO_GL=egl`).

## Decision tree

```
04 FAIL  -> the recurrent pipeline cannot learn memory: fix before measuring anything with C2
01 FAIL  -> freezing is broken: P1/P2 and C1/C2 differences would not be purely the prosthesis controller
00 FAIL  -> the human does not learn to walk with the prosthesis: dynamics / reward / reference; everything else waits
00 WARN  -> still learning: keep training, re-run on newer checkpoints
02 WARN  -> (a) zero torque is not worse than baseline: the prosthesis is irrelevant, nothing for the controller to learn
            (b) cliff within +-25 % of nominal torque: the gait is too brittle
03a WARN -> no intermediate amplitude: refine the grid / larger or smaller amplitudes
03b PASS -> the full 2x2 experiment looks worth its compute
03b WARN -> inconclusive (short run, few seeds): NOT evidence against P2
03b FAIL -> neither condition walks on the terrain (lower the amplitude) or the pipeline is broken
```

## Limits (honest)

* 03b is a go/no-go **signal**, not evidence: tiny budget, few seeds, one training terrain (evaluation uses held-out terrains).
* 03a measures a human trained on *flat* ground; the final human also trains on terrain, so the real headroom is smaller.
  Re-run it with a terrain-trained Stage 1.
* Upstream terrain default: 20 m wide heightfield with a 3 m flat start - the walker leaves it after ~10 m (~250 steps).
  We use 60 m (`envs/terrain.py`); set `extent` / `resolution` in the Stage 1/2 training configs too.
* A MyoAssist heightfield is fixed per model: one training run = one terrain (seed). For the final experiment (new terrain
  per episode) the heightfield data has to be regenerated at reset - not written yet.
* 04 uses sb3-contrib's **LSTM** (no GRU there); test your own GRU with the `--agent` hook on the same task.
* Evaluation follows MyoAssist's own evaluator (imitation early-termination off, long episodes): "fell" = pelvis below `safe_height`.
  The random start phase is unseeded upstream; the scripts seed it per episode so conditions are comparable.
* `train_log.json` average episode length / reward is misleading (training episodes terminate early): verdicts come from deterministic evaluation.
* Only 04, 01 and the unit tests were run for real. 00/02/03a/03b were plumbing-tested on untrained smoke checkpoints.
