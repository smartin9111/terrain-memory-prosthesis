# Design decisions (agreed + measured)

* **Frozen human, trained prosthesis only.** Joint training would let the human adapt differently in each condition and confound
  P1/P2 and C1/C2. Mechanism: `utils/freeze.py` (human weights `requires_grad=False`, gradient mask on the muscle part of the single
  shared `log_std`). Verified bitwise by `scripts/pilots/01_check_freeze.py` and `tests/test_freeze_logic.py`.
* **Same critic input everywhere.** The critic sees the preview in P1 and P2; otherwise a better value baseline could explain a P2 gain.
* **C1 vs C2 differ only in memory.** Same PPO settings. Optional third baseline: frame-stacked MLP (memory without recurrence).
  sb3-contrib `RecurrentPPO` (LSTM) does not fit MyoAssist's multi-actor policy: the GRU must be implemented in the prosthesis actor.
* **Perception has no hidden memory.** Dropout = zeros + validity bit; delay identical for C1 and C2.
* **Terrain.** Flat -> mild random heightfield (2-5 cm), no stairs initially. Training terrain distribution fixed; held-out evaluation
  terrains fixed by seed. Upstream heightfield is 20 m wide with a 3 m flat start (the walker leaves it after ~10 m): use 60 m.
* **Terrain preview.** `mujoco.mj_ray` straight down at pelvis_x + d (d = 0.2 ... 1.0 m); must skip the walker's own geoms
  (re-cast from below until the hit geom belongs to the world body). Height relative to the ground under the pelvis, scaled by 0.1 m.
  Cost ~0.3 % of a control step.
* **Compute (previous project).** 30M PPO steps ~ 10.5 h on 64 CPUs / 128 envs (~790 it/s), GPU gives no speedup; PPO had not plateaued.
* **Evaluation.** Deterministic checkpoint evaluation with MyoAssist's evaluator setup, fixed episode seeds; training-log episode
  length is misleading. Report fall rate with Wilson CIs and several training seeds.
