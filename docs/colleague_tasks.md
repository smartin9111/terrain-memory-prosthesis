# Handover: the perception axis (P1/P2), terrain and evaluation

Project: how much terrain preview (0.2-1.0 m ahead) does a powered ankle prosthesis controller need for robust walking on
rough ground, and does a recurrent controller compensate when the preview is poor? Design 2x2: **P1/P2** (proprioception only /
+ terrain preview) x **C1/C2** (feed-forward PPO / recurrent). Split: Martin = controller axis (C1, C2, freezing of the human,
Stage 1), you = perception axis (P1/P2, terrain, degradation, preview-distance ablation). Roughly equal share of the remaining work.

Read first (30 min): `docs/architecture.md` (who may import whom, the one interface), `docs/stage2.md` (what is fixed across
conditions and why), `docs/design_decisions.md`.

## 1. Where you work

Your package is **`src/terrain_memory_prosthesis/perception/`**. You add a perception variant as one new file with `@register("name")`
(see `perception/clean_preview.py` for the pattern) and train with `--perception name`. You never edit `controllers/`; Martin never edits
`perception/`. `tests/test_import_boundaries.py` enforces that (perception may import `common` only).

| Area | Files |
|---|---|
| Yours | `perception/*`, perception env variables, terrain (`environment/terrain.py`, `episode_terrain.py`), terrain families in `scripts/eval_stage2.py` |
| Shared (both review) | `common/*`, `environment/*` (rest), `training/*`, `evaluation/*`, `docs/` |
| Martin's | `controllers/*`, Stage-1 files (`environment/osl_physics.py`, Stage-1 configs, `docs/stage1_findings.md`) |

**The P1/P2 code already exists and works (interface included) - read it, run the tests, extend it.** P1 = `none` (block size 0), P2 =
`clean_preview` (5 heights at 0.2..1.0 m ahead of the pelvis, relative to the ground under the pelvis, units of 10 cm). The critic always
gets the clean 5 values as a privileged block. History worth knowing: an early version returned NaN (the ray hit the walker's own
geoms at reset) - regression test `tests/test_terrain_preview.py::test_many_stacked_walker_geoms_do_not_hide_the_ground`; and every env
instance used to keep one terrain for life - hence `EpisodeTerrain`.

Check your setup: `python -m pytest -q` (all green). A 1-minute smoke of the whole chain (needs MyoAssist, see `docs/ucloud.md`):
`python scripts/train_stage2.py --stage1-session <session> --cond P2 --smoke`.

## 2. The contract (keep it - it keeps C1-vs-C2 fair)

The only coupling to the controllers is the observation vector `[40 proprioceptive | your block | 5 privileged]`. Rules for every `PerceptionModel`
(`common/interfaces.py`):
1. Fixed `size` (the controller input size is derived from it).
2. **No hidden memory** a controller could exploit: no filtering/smoothing state. A pure delay buffer is fine (sensor property, identical for C1 and C2).
3. Dropout = **zeros + validity bit** (never "repeat the last value").
4. Noise/dropout/delay use their own RNG seeded in `reset(seed)` (same disturbances for every controller in an evaluation).
5. The default clean preview stays bit-identical (reference for the main runs); variants are separate models.
6. Do not change `obs[0:40]`, the reward or the physics (all runs must stay comparable).

## 3. Tasks (the interface task "PerceptionModel" is already done)

**B. Quality of the preview (~14 h).** Noise (sigma in m: 0.005 / 0.01 / 0.02), dropout (p; zeros + validity bit, so the block grows by
the validity bits), delay (0, 2, 5 control steps) as perception models (or wrappers around `clean_preview`). Unit tests: noise std, dropout
rate, delay equals a shifted clean signal, validity bit, reproducibility.

**C. Encoded preview (the mail: "an encoded representation of the upcoming terrain", ~10 h).** At least one encoding besides the raw heights, e.g.
{mean slope, largest step height, roughness} over 0.2-1.0 m; same fixed-size rule. Document why it is a fair P2 variant.

**D. Amount of preview (~6 h + compute).** `--preview-distances` (0.2 / 0.2-0.6 / 0.2-1.0 m) and the control `MYO_PREVIEW_ZERO=1` (same input
size, no information). Run with `scripts/ucloud/stage2.sh` (`DISTANCES=...`, see `docs/ucloud.md`); Martin starts the C1/C2 runs, you the
perception runs - agree the UCloud slots.

**E. Terrain families as held-out test sets (~8 h).** Besides random (2/4/6/8 cm) and sinusoidal: slope (3 / 6 deg), single step (2 / 4 / 6 cm), a
transition flat->rough. Add them to `scripts/eval_stage2.py`; the cell size must stay 25 cm (160 m / 640 cells), see `docs/stage1_findings.md`.

**F. Analysis (~10 h, can start with the smoke output).** `scripts/analyze_stage2.py`: fall rate vs terrain amplitude per cell (P1C1, P2C1, P1C2,
P2C2) with Wilson intervals; per-seed points (never only the mean); summary table (CSV/LaTeX); training curves from `train_log.json`. **The
headline result** is the *minimum preview distance* each controller needs: define it together with Martin (proposal: shortest look-ahead whose
fall rate on the held-out terrains is below a threshold, e.g. 10 %) and produce fall rate vs preview distance, one line per controller.

**G. Literature and report sections (~8 h).** Related work for the perception side (environment-aware prosthesis control, depth cameras /
look-ahead: Xing et al., arXiv 2507.01111; Tschiedel et al. 2020, JNER; Miki et al. 2022 for the belief-encoder idea); methods and results of the perception axis.

To agree together: who owns `evaluation/`, the terrain band (2-6 vs 2-5 cm), the "minimum preview distance" definition.

## 4. Share of the remaining work (estimate)

| | Martin | You |
|---|---|---|
| C2 GRU, C1/C2 training, controller analysis, controller report sections | ~25 % | |
| Degradation, encoding, preview-amount runs, terrain families, analysis, perception report sections | | ~25 % |
| Done so far (Stage 1, Stage 2 framework, structure, first P1/P2 and C1) | shared foundation, by Martin - to be stated as such in the report | |

Workflow: branch `feature/perception-<topic>`, open a PR, keep `pytest` green. Changes in shared packages need Martin's review.
