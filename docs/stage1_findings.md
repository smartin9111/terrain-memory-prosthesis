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
| **`stage1_flat_final` from scratch, 30M** | same reward as walk2 | **1-2 %** (28.9M checkpoint) | 26.8 m, 1.34 m/s, ankle -15..+30 deg |

Fine-tuning a hopping policy does not leave the hopping basin in 8M steps; the penalty has to act from the start.

## Evaluation pitfalls (do not repeat)

* **Terrain edge**: the field must be longer than the walked distance (1.3-1.5 m/s x 33 s = up to 50 m). A 100 m field ends at +-50 m;
  all "falls" of an early 00 pilot were walks off the edge (50.4 m). Use `extent >= 160`.
* **`truncated` is not a fall**: eval episodes end when the reference data is used up (random start index); only `terminated` is a fall.
* **Terrain geometry in eval must equal training** (extent 160 / resolution 640 = 25 cm cells); 03a with 60 m / 768 cells gave meaningless falls.
* The training episode length is not a walking measure (early termination); the termination-independent pilot `00` is.
* Checkpoints sort numerically with `ls -v` (not `sort -t_ -k2`, the session path itself contains underscores).

## Terrain headroom of a flat-trained, blind policy (measured on the *hopping* 30M policy, 25 cm cells; recalibrate with the final one)

random 0.5-4 cm: no falls ("easy"); random 6 cm: 43 % falls; sinusoidal (period 1 m) 8 cm: 80 % falls. A Stage-2 terrain of
2-5 cm random would therefore leave the preview nothing to gain; use the 6-10 cm band (and/or slopes: `terrain: "slope"`, `deg`).

## Open

* Final numbers of `stage1_flat_final` at 30M (00, ankle test B, 03a, 02) -> `eval_final/`.
* Armature sensitivity (0.05 / 0.1 / 0.2).
* Per-episode terrain randomisation for Stage 2 (a heightfield is fixed per env instance; same seed in all envs).
