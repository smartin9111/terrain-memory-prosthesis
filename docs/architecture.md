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
| `controllers/` | controller axis | `registry.py`, `mlp.py` (C1), `warm_start.py`, `freeze.py`, future `gru.py` (C2) | `common` |
| `environment/` | shared | MyoAssist glue, terrain, OSL physics, `preview_env.py` | `common`, `perception` |
| `evaluation/` | shared (owner open) | rollouts, perturbations, reporting | `common`, `environment` |
| `training/` | shared | `stage2.py` = Stage-1 policy -> frozen human + controller + perception -> PPO | everything |

`tests/test_import_boundaries.py` enforces the table: a perception module that imports a controller (or the reverse) fails the test, so
the two axes cannot grow into each other by accident.

## The one interface: the observation vector

```
obs = [ clean 40 | corrupted sensor values 40 | validity 40 | perception (n) | privileged preview 5 ]
          human + critic      prosthesis actor         prosthesis actor      prosthesis actor     critic only
```

* `obs[0:40]` remains the clean MyoAssist observation and is never reordered or rescaled. The frozen human actor and critic keep using it.
* `obs[40:80]` contains the sensor-impaired duplicate; `obs[80:120]` contains matching validity bits. The prosthesis actor's existing 22 channels are redirected to corresponding channels in these blocks, so sensor faults do not leak into the human actor.
* The perception block is produced by a `PerceptionModel` (`common/interfaces.py`): `name`, `size`, `reset(seed)`, `observe(model, data)`. P1 = `"none"` (size 0), P2 = `"clean_preview"` (default size 5).
* The critic always gets the *clean* preview, so degrading actor inputs does not also degrade the value baseline.
* Layout arithmetic lives in `common/obs_layout.py`; `common/session_config.py` defines actor/critic routing.

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

`MYO_PERCEPTION` (model name), `MYO_PREVIEW_DISTANCES`, `MYO_PREVIEW_SCALE`, `MYO_PREVIEW_ZERO`, plus `MYO_SENSOR_NOISE_STD`, `MYO_SENSOR_DELAY_STEPS`, and `MYO_SENSOR_DROPOUT_PROBABILITY`. Sensor settings accept scalars or JSON arrays for the 40 channels. Training records the exact settings in `stage2_info.json`, and evaluation reuses them.

## Git workflow

`main` is protected by PR review (CODEOWNERS). Branches: `feature/controller-*` (controller axis), `feature/perception-*` (perception axis).
A change in `common/`, `environment/`, `training/` or `evaluation/` needs the other person's review; a change that stays inside your own
package does not need anything else. Keep `pytest` green (it contains the import-boundary test).
