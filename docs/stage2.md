# Stage 2 - frozen human, prosthesis controller on rough terrain

Research question: how much terrain preview (0.2-1.0 m) does the prosthesis controller need, and does a recurrent (GRU)
controller compensate? Design 2x2: P1/P2 (proprioception only / + terrain preview) x C1/C2 (feedforward MLP / GRU).
This page documents what is implemented for **C1** (MLP) and how P1 and P2 are made comparable. C2 (GRU) is still open.
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

* C2 (GRU actor) is not implemented (`controllers/registry.py` raises NotImplementedError for `gru`; add `controllers/gru.py`). The P2 vs P1 question with C1 comes first.
* Perception noise / dropout / delay: not implemented (perception owner). Rule: no hidden memory in the perception; dropout = zeros + validity bit; same delay for C1 and C2.
* One Stage-1 seed; prosthesis armature (0.1 kg m^2) is an assumption (sensitivity 0.05-0.2 still to do).
* The 2-6 cm training band is a default taken from the 03a calibration of the blind Stage-1 policy (6 cm: 80 % falls, flat: 0 %); check it against the full 03a table before the long runs - a band where the blind policy never falls leaves no headroom, one where the *frozen human* falls whatever the prosthesis does hides the effect.
* The human keeps its (small) Stage-1 exploration noise during Stage-2 training; evaluation is deterministic.
* 3 training seeds per condition is the minimum for a statement; report per-seed values, not only the mean.
