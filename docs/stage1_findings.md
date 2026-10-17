# Stage 1 - what was changed, why, and what was measured

Stage 1 = human (myolegs22) + powered prosthesis (OpenSourceLeg_A_L1) learn to walk on flat ground with MyoAssist's PPO recipe
(30M steps). Its policy is the fixed starting point of all four Stage-2 conditions (P1/P2 x C1/C2), so everything below applies
to all of them equally. Final config: `configs/experiments/stage1_flat_final.json`
(`ENVS=128 bash scripts/ucloud/stage1.sh full`, ~2.5 h on UCloud, ~3400 fps).

## Modelling changes (all applied after the model is built, MyoAssist files are untouched)

| # | Finding (measured) | Change | Where |
|---|---|---|---|
| 1 | The stock OSL ankle limit (+-30 deg) is soft; a 168 Nm motor drives the joint to +110 / -300 deg, the policy exploited it (foot spins round, `scripts/check_ankle_limit.py` test A: stock FAIL) | hard limit, solref time constant 0.002 s | `envs/stiff_limit.py` |
| 2 | No speed limit, no reflected rotor inertia: 50 rad/s joint speed, bang-bang ankle (80 % of the time at the stops) | motor torque-speed line tau = K*u - b*omega with the OSL spec (peak 145.25 Nm, 6.13 rad/s) + reflected inertia **0.1 kg m^2 (ASSUMPTION - not in the OSL docs; sensitivity 0.05..0.2 still to be checked)**; `mj_setConst` is required after changing armature at run time | `envs/osl_physics.py` |
| 3 | A plane-trained gait falls on any heightfield, even at 0.01 mm relief (a heightfield produces more contacts: 7.1 vs 2.7); cell size matters (7.8 cm cells fail, 25 cm cells fine) | Stage 1 trains on a heightfield: random terrain, relief 1e-5 m, **160 m x 160 m, 640x640 cells (25 cm)** | config `terrain` |
| 4 | Hopping/running gait: 23 % of the steps with no foot on the ground, 19 cm pelvis bounce, 1.5 m/s (target 1.25). A 10x higher pelvis_ty imitation weight did nothing (that term is exp(-8 d^2) with d in metres: a 10 cm error costs 8 %) | reward: `-dt` for every step without foot contact (folded into `foot_force_penalty`, weight 8) and `exo_activation_penalty` 0.5 (prosthesis torque was free) | `NoFlightPenaltyMixin` in `envs/osl_physics.py`, config |

## Runs (UCloud, all 128 envs unless noted)

| Run | What | Flight (no foot touching) | Walk 20 s |
|---|---|---|---|
| stock model, plane | soft limit | (not measured) | foot spins round |
| `stage1_flat_osl_stiff` | hard limit only, plane | (46 %) | 12.7 m / 8 s |
| `stage1_flat_osl_phys` 30M | + motor model, hfield, original reward | 22-24 % | 30 m, 1.5 m/s |
| `+ nohop` 8M fine-tune | pelvis_ty weights x10, foot_force 2.0 | 22-23 % | 30-31 m |
| `+ walk` 8M fine-tune | flight penalty 3 | 18-20 % | 30 m |
| `+ walk2` 8M fine-tune | flight penalty 8 + exo effort 0.5 | 14-16 % | 28 m |
| **`stage1_flat_final` from scratch, 30M** | same reward as walk2 | **0-1 %** | 26.6-27.0 m, 1.34 m/s, ankle -18..+30 deg, no fall |

Fine-tuning a hopping policy does not leave the hopping basin in 8M steps; the penalty has to act from the start.

## Final Stage-1 policy (`train_session_20261012-174642`, checkpoint `model_29999104`, 30M steps) - gate results

| Check | Result |
|---|---|
| 05 walk video, 3 seeds x 20 s | no fall; flight 0-1 %; 26.6-27.0 m (1.34 m/s, target 1.25); pelvis bounce 12-15 cm (includes the settling at the start); prosthesis ankle -18..+30 deg, 0 % beyond the limit |
| 00 walk check (10 episodes x 1000 steps, 160 m field) | **PASS**: fall rate 0.00 (Wilson 95 % CI [0, 0.28]), speed 1.34 m/s |
| `check_ankle_limit.py` A (gravity off) / B (policy walking) | phys: PASS / PASS (peak ankle speed 7 rad/s) |
| 03a terrain calibration (flat-trained, blind) | flat baseline fall 0.00; random 6 cm: **80 %** falls |
| 02 prosthesis sensitivity | zero torque: 100 % falls (prosthesis is essential); torque x0.5: 100 %, x0.75: 5 % -> **brittle below ~0.75 of the nominal torque** (WARN) |

The walking policy is *less* robust on rough ground than the earlier hopping one (random 6 cm: 80 % vs 43 % falls), i.e. there is
real headroom for a terrain-aware prosthesis controller already at a few centimetres.

## Evaluation pitfalls (do not repeat)

* **Terrain edge**: the field must be longer than the walked distance (1.3-1.5 m/s x 33 s = up to 50 m). A 100 m field ends at +-50 m;
  all "falls" of an early 00 pilot were walks off the edge (50.4 m). Use `extent >= 160`.
* **`truncated` is not a fall**: eval episodes end when the reference data is used up (random start index); only `terminated` is a fall.
* **Terrain geometry in eval must equal training** (extent 160 / resolution 640 = 25 cm cells); 03a with 60 m / 768 cells gave meaningless falls.
* The training episode length is not a walking measure (early termination); the termination-independent pilot `00` is.
* Checkpoints sort numerically with `ls -v` (not `sort -t_ -k2`, the session path itself contains underscores).

## Terrain headroom of a flat-trained, blind policy (25 cm cells)

Hopping policy (earlier): random 0.5-4 cm no falls, 6 cm 43 %, sinusoidal (period 1 m) 8 cm 80 %. Final walking policy: random 6 cm 80 %
(see the 03a table in `eval_final/03a` for the other amplitudes). Choose the Stage-2 training terrain from the 03a table of the
final policy (target: blind fall rate 20-80 %), optionally slopes (`terrain: "slope"`, `deg`).

## Open

* Pick the Stage-2 terrain band from the full 03a table (`eval_final/03a`).
* Torque brittleness (02): irrelevant while Stage 2 starts from the Stage-1 controller, but it limits how far a learned controller may deviate.
* Armature sensitivity (0.05 / 0.1 / 0.2).
* Per-episode terrain randomisation for Stage 2 (a heightfield is fixed per env instance; same seed in all envs).

## Where the Stage-1 artefacts are

* Code and configs: this repository (tag `stage1-final`). The final policy was trained from `configs/experiments/stage1_flat_final.json`
  (UCloud: `ENVS=128`, `n_steps=128`, 30M steps; the config itself says 64 x 256, same rollout of 16384).
* Final policy: session `train_session_20261012-174642`, checkpoint `model_29999104.zip` (UCloud folder `ai3_test_train`).
  Checkpoints are **not** in git (`*.zip` is ignored); keep a copy as a GitHub Release asset of tag `stage1-final` and one locally.
* Gate results (CSV/JSON/PNG, no videos): `results/stage1/` (from `stage1_results.tgz`, see the README there).

