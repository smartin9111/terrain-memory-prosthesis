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
- C2: GRU-based recurrent controller: the C1 network plus a GRU over the last K observation frames (`docs/controllers.md`)

## Main Research Question

How much exteroceptive terrain preview is required for robust prosthetic locomotion, and can temporal controller memory compensate for incomplete or noisy environmental perception?

## Training Scheme

1. **Stage 1** - the human (muscle policy) and a prosthesis actor learn to walk on flat ground with MyoAssist's own `run_train.py`
   (`configs/experiments/stage1_flat_final.json`: OSL prosthesis physics, no-flight and effort penalties, 160 m heightfield ground;
   recipe of the previous SDU project: 30M steps, ~2.5 h with 128 envs on 96 cores). Results and caveats: `docs/stage1_findings.md`,
   `results/stage1/`. Older configs (`stage1_flat_osl*.json`, `history/`) document the road to it and are not used any more.
2. **Stage 2** - the human is copied into a new policy and **frozen** (`controllers/freeze.py`); only the prosthesis controller is trained
   (`scripts/train_stage2.py`, `docs/stage2.md`). P1/P2 and C1/C2 therefore differ *only* in the prosthesis controller and what it sees.
   The critic sees the clean terrain preview in both P1 and P2.
3. Terrain: flat -> mild random heightfield (2-5 cm), no stairs at first; fixed training distribution, fixed held-out evaluation terrains.

Stage 1 is done once and shared by all four cells (P1C1 ... P2C2); every Stage 2 run starts from its checkpoint
(`docs/architecture.md`, section "Stage 1 and Stage 2").

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

The two axes of the 2x2 design live in separate packages and meet only through the observation vector
(`[40 proprioceptive | perception block | privileged preview for the critic]`). Details, import rules, how to add a variant: **`docs/architecture.md`**.

```
src/terrain_memory_prosthesis/
  common/        shared contract: obs_layout.py (sizes), interfaces.py (PerceptionModel), session_config.py (who sees which obs)
  perception/    PERCEPTION AXIS  registry.py, none.py (P1), clean_preview.py (P2; ray-cast heights 0.2-1.0 m ahead), future: noise/dropout/delay/encodings
  controllers/   CONTROLLER AXIS  registry.py, mlp.py (C1), gru.py + gru_module.py (C2), warm_start.py, freeze.py (frozen human)
  environment/   MyoAssist glue (myoassist.py), terrain.py / episode_terrain.py, osl_physics.py (Stage-1 physics), preview_env.py (Stage-2 env), history.py (C2 observation window)
  training/      stage2.py - the only place that assembles perception + controller + PPO
  evaluation/    rollout.py (seeded deterministic rollouts, parallel), reporting.py (Wilson CI, summaries)
scripts/
  train.py                       Stage 1 (MyoAssist run_train with our env ids)
  train_stage2.py                Stage 2 launcher: --cond P1|P2 --controller mlp|gru --perception NAME --preview-distances ...
  compare_p1_p2.py               ONE COMMAND: train P1+P2, evaluate, figures + tables, side-by-side videos (docs/stage2.md)
  eval_stage2.py                 held-out terrain evaluation of Stage-2 runs
  analyze_stage2.py              figures (fall rate vs amplitude, metrics, training curves) and tables from an evaluation
  ucloud/                        UCloud launchers (stage1.sh, stage2.sh)
  pilots/                        pilot validation scripts 00-06 + README
  check_*.py, inspect_*.py, view_prosthesis.py   sanity / inspection tools
configs/experiments/             p1c1.yaml ... p2c2.yaml (design), p1c1_myoassist.json, stage1_flat_final.json (+ history/)
tests/                           unit tests incl. test_import_boundaries.py (guards the split)
results/                         experiment outputs (git-ignored, except results/stage1/ = Stage-1 gate results, no checkpoints/videos)
docs/                            architecture.md, stage2.md, controllers.md, stage1_findings.md, design_decisions.md, ucloud.md, colleague_tasks.md
```

## Development Responsibilities

One person per axis; shared packages need both reviews (`.github/CODEOWNERS`). Work on `feature/controller-*` / `feature/perception-*` branches.

### Perception (`perception/`)
- P1 (nothing) and P2 (terrain preview); noise / dropout / delay, encoded preview, preview-distance ablation, terrain families
- Rule: perception carries no hidden memory; dropout = zeros + validity bit; same delay for both controllers

### Control (`controllers/`)
- C1 feed-forward PPO (MyoAssist multi-actor policy; Stage 2 trains the `exo_actor`), freezing of the human, Stage 1
- C2 GRU controller: implemented as a GRU over a sliding window of observation frames (sb3-contrib `RecurrentPPO` is LSTM-only and not compatible with MyoAssist's multi-actor policy); design, cost and open choices in `docs/controllers.md`

### Shared
- `common/`, `environment/`, `training/`, `evaluation/` (evaluation owner to be agreed), experiments, results, report
