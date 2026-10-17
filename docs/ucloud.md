# Running Stage 1 on UCloud

Stage 1 (human + prosthesis learn to walk on flat ground, 30M steps) is too long for a laptop (~7-10 h on a Mac). Measured on UCloud: ~2.5 h with ENVS=128. What was changed in Stage 1 and why: `docs/stage1_findings.md`.
This guide runs it on a UCloud CPU machine and keeps all results in a **persistent project folder**.

## One-time: create the persistent workspace folder

1. UCloud -> *Files* -> project **Member files** -> create a folder, e.g. `ai3_test_train`.
2. Upload the project folder `terrain-memory-prosthesis` into it.
   Do **not** upload `.venv`, `rl_train/results`, `pilot_out`, `__pycache__` (large / useless there).

Everything under that folder survives after a job ends. Anything outside it (e.g. `$HOME`, `/tmp`) is lost.

## Per job

1. *Runs* -> *Terminal* app (or any app with a shell), CPU-only machine type if offered (a GPU gives no speed-up;
   billing is per GPU-hour). Choose as many vCPUs as possible (32-64).
2. *Hours*: 8 (the job is killed when the time limit ends; extend it with the +8 h button if needed). *Select folders*: add `ai3_test_train` (without it the folder is not mounted at `/work/ai3_test_train`; a shell started without the folder has 1 core and cannot run evaluations).
3. In the job terminal:

```bash
cd /work/ai3_test_train
bash terrain-memory-prosthesis/scripts/ucloud/setup_ucloud.sh
source .ucloud_env.sh
cd terrain-memory-prosthesis
bash scripts/ucloud/stage1.sh measure
bash scripts/ucloud/stage1.sh full
```

* `setup_ucloud.sh` is idempotent: installs `uv`, clones MyoAssist at the pinned commit into the workspace,
  builds a venv (`$HOME/venv-prosthesis`, rebuilt every job, ~minutes), installs everything, runs the unit tests.
  It writes `$WS/.ucloud_env.sh`; `source` it in every new shell.
* `stage1.sh measure` runs 4 PPO iterations and prints steps/s -> estimated total time. Check it before `full`.
  Environment knobs: `ENVS` (default 32, must divide 16384; 128 gave ~3400 fps on a 96-core machine, the final run used it), `PREV=<model.zip>` (warm start), `THREADS` (8), `TOTAL` (30000000), `LOGFREQ` (10).
* `stage1.sh full` starts training with `nohup` in the background; log: `$WS/stage1_<date>.log`.
  `tail -f` it; closing the terminal does not stop the job (the job time limit does).

## Control run (is the prosthesis the problem?)

Same recipe with MyoAssist's own tutorial config (intact human + exoskeleton), only ~6M steps, in parallel with the main run:

```bash
cd /work/ai3_test_train/terrain-memory-prosthesis
ENVS=32 THREADS=4 bash scripts/ucloud/stage1.sh full
CFG=/work/ai3_test_train/myoassist/rl_train/train/train_configs/imitation_tutorial_22_separated_net_full_obs.json \
  TAG=control TOTAL=6000000 ENVS=16 THREADS=2 bash scripts/ucloud/stage1.sh full
```

Compare `average_num_timestep` in each session's `train_log.json` at the same step count.

## Monitoring

```bash
cd /work/ai3_test_train/terrain-memory-prosthesis
python scripts/pilots/00_stage1_walk_check.py \
  --session ../myoassist/rl_train/results/train_session_<timestamp> --workers 8
```

(With several runs the newest session is not necessarily yours: `grep -h "Session directory" /work/ai3_test_train/<tag>_*.log` gives the session of one run. Newest checkpoint of a session: `ls -v $S/trained_models/model_*.zip | tail -n 1`.)

## Where results are saved

`/work/ai3_test_train/myoassist/rl_train/results/train_session_<timestamp>/`
with `trained_models/model_<steps>.zip` (checkpoints), `train_log.json`, `session_config.json`.
This is inside the persistent folder, so download/inspect them after the job ended.

## Notes / not yet verified

* Setup and `measure` were tested in a sandbox, **not on UCloud itself**. The 2-4 h estimate for 48 vCPUs is an estimate.
* The CPU-only torch wheel index may be unreachable; the script falls back to the default index (larger download, still works).
* Resuming an interrupted Stage 1 is untested. If the time limit hits early, keep the last checkpoint and start a new job
  (set a longer *Hours* rather than relying on resume).
* Commit `uv.lock` once the environment resolves, for reproducible setups.


## Stage 2
See `docs/stage2.md` (commands: `scripts/ucloud/stage2.sh measure|full|batch`, `scripts/eval_stage2.py`).
