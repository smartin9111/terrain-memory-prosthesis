# Architecture - how perception (P) and controller (C) stay independent

The 2x2 design has two axes. Each axis lives in its own package and is chosen **by name**; the two meet in exactly one place.

```
                 common/            interfaces + observation layout + session-config rules   (imports nothing internal)
                  /    \
        perception/    controllers/                  P1/P2 + degradations  |  C1/C2 + freezing + warm start
                  \    /
              environment/                           MyoAssist glue, terrain, physics, Preview env (uses perception/ only)
                  |
        evaluation/   training/                      rollouts + reports  |  Stage-2 assembly (the ONLY place that knows both axes)
```

| Package | Owner | Contains | May import |
|---|---|---|---|
| `common/` | both (CODEOWNERS: both) | `obs_layout.py` (sizes), `interfaces.py` (`PerceptionModel`), `session_config.py` (who sees which obs) | nothing internal |
| `perception/` | perception axis | `registry.py`, `none.py` (P1), `clean_preview.py` (P2), future: noise / dropout / delay / encodings | `common` |
| `controllers/` | controller axis | `registry.py`, `mlp.py` (C1), `gru.py` + `gru_module.py` (C2), `warm_start.py`, `freeze.py` | `common` |
| `environment/` | shared | MyoAssist glue, terrain, OSL physics, `preview_env.py`, `history.py` (C2 observation window) | `common`, `perception` |
| `evaluation/` | shared (owner open) | rollouts, perturbations, reporting | `common`, `environment` |
| `training/` | shared | `stage2.py` = Stage-1 policy -> frozen human + controller + perception -> PPO | everything |

`tests/test_import_boundaries.py` enforces the table: a perception module that imports a controller (or the reverse) fails the test, so
the two axes cannot grow into each other by accident.

## Stage 1 and Stage 2

| | Stage 1 | Stage 2 |
|---|---|---|
| What learns | human (muscles) + prosthesis actor, together | prosthesis controller only (human frozen) |
| Terrain | flat (160 m heightfield, 25 cm cells) | new random field every episode (2-6 cm) |
| Entry point | `scripts/train.py` (registers our env ids, then runs MyoAssist's own `run_train.py`); UCloud: `scripts/ucloud/stage1.sh` | `scripts/train_stage2.py` -> `training/stage2.py`; UCloud: `scripts/ucloud/stage2.sh` |
| Code it uses | `environment/osl_physics.py` (OSL ankle limit + motor model, no-flight penalty), config `configs/experiments/stage1_flat_final.json` | `perception/`, `controllers/`, `environment/preview_env.py` |
| Result | checkpoint `train_session_*/trained_models/model_*.zip` (GitHub Release `stage1-final`), gate results in `results/stage1/`, notes in `docs/stage1_findings.md` | `stage2_<cond>_<controller>_s<seed>_<stamp>/` with `stage2_info.json` |

Stage 1 is done once and shared by all four cells of the 2x2; its checkpoint is the start of every Stage 2 run. Stage 1 was not rewritten
into `training/`: it runs MyoAssist's trainer unchanged, only the environment and the config are ours.

## The one interface: the observation vector

```
obs = [ proprioceptive 40 | perception block (n_percept) | privileged clean preview 5 | (C2 only) window of the last K frames ]
          human + prosthesis      prosthesis actor only         critic only
```

* `obs[0:40]` is what the frozen Stage-1 human was trained on. Never reorder or rescale it.
* The perception block is produced by a `PerceptionModel` (`common/interfaces.py`): `name`, `size`, `reset(seed)`, `observe(model, data)`.
  P1 = `"none"` (size 0), P2 = `"clean_preview"` (size = number of distances, default 5).
* The critic always gets the *clean* preview (asymmetric actor-critic), so degrading the perception cannot be blamed on a worse value baseline.
* Layout arithmetic lives only in `common/obs_layout.py`; `common/session_config.py` decides who sees what.

Fairness rules for every perception model (they keep C1 vs C2 comparable): fixed size; no hidden memory (a pure delay buffer is
fine); dropout = zeros + validity bit; own RNG seeded in `reset(seed)`; the default clean preview stays bit-identical.

## Adding a variant (no edits in the other axis)

* Perception: new file `perception/<name>.py` with `@register("<name>")` on a class that satisfies `PerceptionModel`; add the import to
  `_load_builtin()` in `perception/registry.py`. Train with `--perception <name>`.
* Controller: new file `controllers/<name>.py` with `@register("<name>")` on a function `(model, stage1_model, **kw) -> info dict` that
  freezes the human and initialises the prosthesis controller. Train with `--controller <name>`.
* Nothing in `training/` or `scripts/` changes. Every run writes `stage2_info.json` (perception env, controller, git commit);
  `scripts/eval_stage2.py` re-applies the run's own perception settings, so evaluation always sees what training saw.

## Environment variables of the perception (read inside the env workers)

`MYO_PERCEPTION` (model name), `MYO_PREVIEW_DISTANCES` (e.g. `0.2,0.4`), `MYO_PREVIEW_SCALE`, `MYO_PREVIEW_ZERO` (control: same size, no information).
`scripts/train_stage2.py` sets them from its arguments; add yours the same way and list them in `stage2_info.json`.

## Git workflow

`main` is protected by PR review (CODEOWNERS). Branches: `feature/controller-*` (controller axis), `feature/perception-*` (perception axis).
A change in `common/`, `environment/`, `training/` or `evaluation/` needs the other person's review; a change that stays inside your own
package does not need anything else. Keep `pytest` green (it contains the import-boundary test).
