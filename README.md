# Terrain Preview vs. Controller Memory

Research project investigating the trade-off between exteroceptive terrain perception and temporal controller memory for powered ankle prosthesis locomotion in MyoAssist.

## Simulation

- MyoAssist / MuJoCo
- Human model: `myolegs22`
- Prosthesis: `OpenSourceLeg_A_L1`

## Experimental Design

|                     | C1: Feed-forward PPO | C2: Recurrent PPO + GRU |
|---------------------|----------------------|--------------------------|
| P1: Proprioception  | P1C1                 | P1C2                     |
| P2: Terrain Preview | P2C1                 | P2C2                     |

## Perception

- P1: proprioceptive observations only
- P2: proprioception + terrain-height preview

## Controllers

- C1: feed-forward PPO controller
- C2: GRU-based recurrent PPO controller

## Main Research Question

How much exteroceptive terrain preview is required for robust prosthetic locomotion, and can temporal controller memory compensate for incomplete or noisy environmental perception?

## Training Scheme

1. **Stage 1** - the human (muscle policy) and a prosthesis actor learn to walk on flat ground with MyoAssist's own `run_train.py`
   (`configs/experiments/stage1_flat_final.json`: OSL prosthesis physics, no-flight and effort penalties, 160 m heightfield ground;
   recipe of the previous SDU project: 30M steps, ~2.5 h with 128 envs on 96 cores). Results and caveats: `docs/stage1_findings.md`,
   `results/stage1/`. Older configs (`stage1_flat_osl*.json`, `history/`) document the road to it and are not used any more.
2. **Stage 2** - the human is copied into a new policy and **frozen** (`utils/freeze.py`); only the prosthesis controller is trained.
   P1/P2 and C1/C2 therefore differ *only* in the prosthesis controller. The critic sees the terrain preview in both P1 and P2.
3. Terrain: flat -> mild random heightfield (2-5 cm), no stairs at first; fixed training distribution, fixed held-out evaluation terrains.

Before any long training, run the pilots in `scripts/pilots/` (see its README): do the human walk with the prosthesis, is the
freeze correct, does the prosthesis matter, which terrain amplitude leaves headroom, does preview help at all, does the
recurrent pipeline learn memory.

## Setup

Layout: MyoAssist is a sibling clone of this repo (`../myoassist`). Tested with commit `b1baf69da59dc6728515d0977cbc8fbf00b9225a` on Python 3.12 (MyoSuite 2.8.4, MuJoCo 3.4).

```bash
cd terrain-memory-prosthesis
uv venv --python 3.12 && source .venv/bin/activate
(cd ../myoassist && uv pip install -e .)     # plain pip cannot resolve MyoSuite's pins; uv uses MyoAssist's override
uv pip install -e ".[dev]"
pytest                                        # unit tests (no MyoAssist needed for most)
```

`scripts/pilots/*` find MyoAssist automatically (installed `rl_train`, `$MYOASSIST_DIR`, or `../myoassist`).

## Repository Structure

```
configs/experiments/
  p1c1.yaml ... p2c2.yaml        experiment definitions (documentation of the 2x2 design; not parsed by code yet)
  p1c1_myoassist.json            MyoAssist session config: P1xC1 on 6 cm random terrain, 200k-step smoke recipe
  stage1_flat_final.json         MyoAssist session config: Stage 1 FINAL (flat, proprioception-only prosthesis actor, 30M steps)
  stage1_flat_osl*.json, history/  earlier Stage-1 variants (stock physics, hard limit, motor model; fine-tune experiments)
src/terrain_memory_prosthesis/
  perception/terrain_preview.py  P2 exteroception: ray-cast terrain heights ahead of the pelvis (TODO: noise/dropout/delay)
  controllers/                   C1 (MLP) / C2 (GRU) prosthesis controllers (TODO)
  envs/osl_physics.py            OSL prosthesis physics (hard ankle limit, motor torque-speed model) + no-flight penalty; Stage-1 env
  envs/stiff_limit.py            hard ankle limit only (intermediate step, kept for the history)
  envs/terrain.py                MyoAssist terrain dicts (60 m field instead of the 20 m default)
  envs/terrain_preview.py        myoAssistLegImitationExoPreview-v0 = base env + preview block obs[40:45]
  evaluation/reporting.py        Wilson CI, summaries, verdicts / exit codes
  evaluation/rollout.py          seeded deterministic rollouts, prosthesis-command perturbations, parallel evaluation
  utils/myoassist.py             locating MyoAssist, loading configs/checkpoints, building envs/models
  utils/session_config.py        derive P1 / P2 session configs from a base config
  utils/freeze.py                load the Stage-1 human into a Stage-2 policy and freeze it
scripts/                         check_myoassist.py, check_prosthesis_env.py, inspect_osl_model.py, view_prosthesis.py
scripts/pilots/                  pilot validation scripts 00-04 + README
tests/                           unit tests
results/                         experiment outputs (git-ignored, except results/stage1/ = Stage-1 gate results, no checkpoints/videos)
docs/                            project documentation
```

## Development Responsibilities

### Perception
- Proprioceptive observation
- Terrain preview (`perception/terrain_preview.py`: pilot implementation exists, to be reviewed/extended)
- Noise / dropout / delay (not started). Rule: perception carries no hidden memory; dropout = zeros + validity bit; same delay for both controllers

### Control
- Feed-forward PPO (MyoAssist multi-actor policy; Stage 2 trains the `exo_actor`)
- GRU-based recurrent controller (sb3-contrib `RecurrentPPO` is LSTM-only and not compatible with MyoAssist's multi-actor policy: needs its own implementation)
- Controller evaluation (`evaluation/`)

### Shared
- MyoAssist integration (`utils/`, `envs/`)
- Experiments
- Results
- Report
