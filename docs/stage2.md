# Stage 2 - frozen human, prosthesis controller on rough terrain

Research question: how much terrain preview (0.2-1.0 m) does the prosthesis controller need, and does a recurrent (GRU)
controller compensate? Design 2x2: P1/P2 (proprioception only / + terrain preview) x C1/C2 (feedforward MLP / GRU).
This page documents what is implemented for **C1** (MLP) and how P1 and P2 are made comparable. C2 (GRU) is documented in `docs/controllers.md` (same pipeline: `--controller gru`).
Code layout and who may import whom: `docs/architecture.md`. The pieces are chosen by name: `--cond P1|P2`, `--controller mlp`,
`--perception <name>` (default: P1 `none`, P2 `clean_preview`), `--preview-distances 0.2,0.4,...`.

## What is fixed across conditions (so a P1-vs-P2 difference can only come from the preview)

| item | how |
|---|---|
| human | Stage-1 muscle policy + its muscle `log_std`, **frozen** (`controllers/freeze.py`); checked bit-exact after training (smoke: human identical, muscle log_std identical, prosthesis weights changed) |
| start | both conditions start from the Stage-1 policy. P1 = Stage-1 prosthesis actor; P2 = same actor + zero-initialised input columns for the perception block (5 for the default preview). `controllers/warm_start.py::check_equivalent` verifies (random observations, max diff 0) that at step 0 P1, P2 and Stage 1 output the same torque |
| critic | sees the clean preview (privileged block) in both conditions (zero-initialised extra columns) - identical value baseline |
| reward, env physics | exactly the Stage-1 ones (OSL motor model + hard ankle limit, flight / effort penalties) |
| terrain | a new random heightfield **every episode**: amplitude ~ U(2, 6) cm, per-cell noise, 3 m flat zone around the start, 25 cm cells (`environment/episode_terrain.py`); compiled relief = 6 cm |
| seed | the same `--seed` gives P1 and P2 the same numpy/torch seed |
| recipe | PPO settings of the Stage-1 session (rollout 16384, minibatch 8192, 30 epochs, lr 1e-4) |

The preview (P2): `perception/clean_preview.py`, heights of the world geoms at 0.2/0.4/0.6/0.8/1.0 m ahead of the pelvis relative
to the height under the pelvis, in units of 10 cm. **Bug fixed on 12 Oct:** the downward ray used to hit the walker's own
geoms (more than the allowed 12 hops stack up above the pelvis at reset) and returned NaN -> the preview was all zeros at
every reset. Now only world-body geoms are intersected (`mj_rayHfield` / `mju_rayGeom`). Regression test:
`tests/test_terrain_preview.py::test_many_stacked_walker_geoms_do_not_hide_the_ground`. Any *earlier* P1-vs-P2 pilot
result (03b) was obtained with the broken preview and is not valid.

## One command: P1xC1 vs P2xC1 (train, evaluate, figures, videos)

```bash
python scripts/compare_p1_p2.py --stage1-session ../stage1_session --smoke            # plumbing check, ~3-5 min (numbers are meaningless)
python scripts/compare_p1_p2.py --stage1-session ../stage1_session --timesteps 2000000 --out compare_out
```

`--stage1-session` = a folder with the Stage-1 `session_config.json` and `trained_models/model_*.zip` (GitHub Release `stage1-final`).
It trains P1 and P2 (same seed and recipe), evaluates both and the Stage-1 policy on the held-out terrains, and writes into `--out`:

| file | content |
|---|---|
| `report/fig_fall_rate.png` | fall rate vs terrain amplitude, 95 % Wilson intervals, Stage-1 reference, per-seed dots (if several seeds); sinusoidal terrain on the right |
| `report/fig_metrics.png` | survived fraction, distance, speed error, mean prosthesis command vs amplitude |
| `report/fig_training.png` | training reward, training-episode length, policy std, critic explained variance |
| `report/fig_checkpoints.png` | fall rate vs training steps (only with `--ckpts 5`: evaluates 5 checkpoints per run, 5x the evaluation time) |
| `report/summary.md/.csv` | per cell x terrain: n, falls, fall rate [CI], survival, distance, speed error, command |
| `report/differences.md/.csv` | P2 - P1 fall-rate difference with a Newcombe 95 % interval (an interval containing 0 = not separated) |
| `videos/compare_*.mp4` | P1 (left) and P2 (right) on the same terrain and seeds (needs ffmpeg; else the separate videos are in `videos/P1_*`, `videos/P2_*`) |
| `logs/`, `compare_info.json` | full logs, arguments, run folders |

Options: `--seed`, `--episodes` (per terrain instance, default 10), `--ckpts`, `--video-amp`, `--skip-train --p1-run <dir> --p2-run <dir>` (re-evaluate
existing runs), `--no-video`. With more than one training seed run the script once per seed, then analyse them together:
`python scripts/analyze_stage2.py --eval <dir> --runs "<runs>*" --out <report>` (after `eval_stage2.py` over all runs).
Several seeds at once (e.g. on a 144-core node: three processes with `--num-envs 64`, each `--seed N --steps train --out ...`), then ONE
evaluation over all runs: `python scripts/eval_stage2.py --runs "<results>/stage2_P*_mlp_s*_cmp_*" --stage1-session <S1> --episodes 20 --ckpts 5 --out eval_all`
and `python scripts/analyze_stage2.py --eval eval_all --runs "<results>/stage2_P*_mlp_s*_cmp_*" --out report_all` (single seeds appear as dots).
`--steps train,eval,report,video` selects the stages; `--skip-train --p1-run D --p2-run D --steps video` makes the videos of existing runs.
Honest reading: with few episodes the intervals are wide; per-seed dots show the spread between training runs.

## Run (UCloud)

```bash
cd /work/ai3_test_train/terrain-memory-prosthesis && git pull
S1=/work/ai3_test_train/myoassist/rl_train/results/train_session_20261012-174642   # final Stage 1
STAGE1=$S1 bash scripts/ucloud/stage2.sh measure                       # ~2 min, prints steps/s
STAGE1=$S1 COND=P1 SEED=0 TOTAL=20000000 bash scripts/ucloud/stage2.sh full     # CONTROLLER=mlp is the default
STAGE1=$S1 CONDS="P1 P2" SEEDS="0 1 2" bash scripts/ucloud/stage2.sh batch   # queue: seed 0 P1, P2, seed 1 P1, P2 ...
```

Video of a finished run (works for P1 and P2 runs: the perception settings are read from the run's `stage2_info.json`):

```bash
cd /work/ai3_test_train/myoassist
python ../terrain-memory-prosthesis/scripts/pilots/05_render_video.py --session rl_train/results/stage2_P1_mlp_s0_<stamp> \
   --seeds 0,1,2 --terrain random --amplitude 0.04 --max-steps 600 --out ~/videos_p1
```

Evaluation on fixed held-out terrains (flat; random 2/4/6/8 cm x 3 terrain seeds; sinusoidal 4 cm), Stage-1 policy as
reference:

```bash
cd /work/ai3_test_train/myoassist
python ../terrain-memory-prosthesis/scripts/eval_stage2.py --runs "rl_train/results/stage2_P*"   # each run's own perception settings are read from its stage2_info.json \
   --stage1-session $S1 --episodes 10 --workers 16 --out eval_stage2
```

## Open items / caveats for the report

* C2 (GRU, `--controller gru`, `docs/controllers.md`) results with the CLEAN preview and no sensor imperfections (random terrain 6 cm, fall rate, checkpoints above 7.9M steps; C1 and K=16: 6 seeds, K=32 and K=64: 3 seeds): P1 / P2 = C1 0.55 / 0.42, K16 0.52 / 0.41, K32 0.54 / 0.49, K64 0.52 / 0.44. The preview lowers the fall rate by about 11-13 points (borderline at the seed level, p about 0.1); the GRU memory adds nothing measurable on top (K = 16, 32, 64 are indistinguishable from C1). Seeds differ by 0.2-0.35 inside one cell, so 3 seeds are not enough for effects of this size. Learning curves of C1 are flat after about 10M steps. These runs used the layout without the sensor block (tag `stage2-clean-results`); results with sensor imperfections need new runs. The memory is a window of K frames (0.5-2 s), not an unbounded hidden state: say so when reporting it.
* Perception noise / dropout / delay: not implemented (perception owner). Rule: no hidden memory in the perception; dropout = zeros + validity bit; same delay for C1 and C2.
* One Stage-1 seed; prosthesis armature (0.1 kg m^2) is an assumption (sensitivity 0.05-0.2 still to do).
* The 2-6 cm training band is a default taken from the 03a calibration of the blind Stage-1 policy (6 cm: 80 % falls, flat: 0 %); check it against the full 03a table before the long runs - a band where the blind policy never falls leaves no headroom, one where the *frozen human* falls whatever the prosthesis does hides the effect.
* The human keeps its (small) Stage-1 exploration noise during Stage-2 training; evaluation is deterministic.
* 3 training seeds per condition is the minimum for a statement; report per-seed values, not only the mean.

## Continuing a stopped run (`--resume`)

A UCloud job can expire or be restarted in the middle of a long run. The run folder keeps a checkpoint every `--logfreq` PPO iterations
(10 by default, ~164k steps), so nothing is lost but the last few iterations:

```bash
python scripts/train_stage2.py --resume <run dir> --timesteps 10000000     # the NEW TOTAL, not the remaining steps
```

* Everything that defines the run (condition, controller, `--history`, seed, terrain range, `--num-envs`, ...) is read from the run's own
  `stage2_info.json`; only `--timesteps` is yours. The run continues **in the same folder** (`trained_models/`, `train_log.json` is
  appended, `stage2_info.json` gets a `resumed` list with the step it restarted from and the commit).
* Restored: policy, value function, Adam state, step counter. Re-applied: the frozen human. Not restored: the rollout in progress, the
  running episode statistics, and the random streams (a new seed `seed + 1000 * n` is used), so a resumed run is not bit-identical to an
  uninterrupted one. Say so when you report it.
* Works for C1 and C2 (`tests/test_refreeze.py` covers the freezing; both paths were run end to end on the smoke config).
