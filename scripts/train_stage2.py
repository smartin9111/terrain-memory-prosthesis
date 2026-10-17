#!/usr/bin/env python
"""Stage 2: train ONLY the prosthesis controller on rough terrain; the Stage-1 human (muscle policy) is frozen.

    python scripts/train_stage2.py --stage1-session <myoassist>/rl_train/results/train_session_<stamp> \
        --cond P1|P2 --controller mlp --seed 0 --timesteps 20000000 --num-envs 128

* start      : the Stage-1 policy (latest checkpoint of the session, or --stage1-ckpt). P1 = Stage-1 prosthesis actor as it is,
               P2 = the same actor with ZERO-initialised input columns for the perception block -> at step 0 both compute the
               same torque (checked numerically, see controllers/warm_start.py). The critic sees the clean preview in both.
* human      : frozen (weights + muscle log_std), see controllers/freeze.py.
* controller : --controller NAME from controllers/registry.py ("mlp" = C1; "gru" = C2 once implemented).
* perception : --perception NAME from perception/registry.py (default: P1 -> "none", P2 -> "clean_preview");
               --preview-distances "0.2,0.4" sets MYO_PREVIEW_DISTANCES for the clean preview.
* terrain    : a NEW random field every episode, amplitude ~ U(--amp-lo, --amp-hi) [m], flat safe zone around the start
               (environment/episode_terrain.py). The compiled relief is --amp-hi, cells 25 cm like in Stage 1.
* recipe     : PPO hyper-parameters, reward and env settings are taken from the Stage-1 session_config.json; rollout =
               num_envs x n_steps = 16384 (num_envs must divide 16384).
* output     : <myoassist>/rl_train/results/stage2_<cond>_<controller>_s<seed>_<stamp>/ (same layout as a Stage-1 session) +
               stage2_info.json (arguments, perception settings, git commit: everything needed to reproduce and to evaluate it).
Same --seed => P1 and P2 use the same torch/numpy seed.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pilots"))
import common  # noqa: E402  (before torch: caps thread pools, chdir to the MyoAssist root)

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--root", default=None)
_known, _ = _pre.parse_known_args()
common.init(_known.root or os.environ.get("MYOASSIST_DIR"), threads=int(os.environ.get("MYOASSIST_NUM_THREADS", "8")))
from terrain_memory_prosthesis.environment import osl_physics, preview_env  # noqa: E402,F401  (register env ids; also in SB3 workers)


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage1-session", required=True, help="Stage-1 session folder (session_config.json + trained_models/)")
    ap.add_argument("--stage1-ckpt", default=None, help="Stage-1 checkpoint .zip (default: the last one of the session)")
    ap.add_argument("--cond", required=True, choices=["P1", "P2"])
    ap.add_argument("--controller", default="mlp", help="controller name (controllers/registry.py)")
    ap.add_argument("--perception", default=None, help="perception model name (perception/registry.py); default by --cond")
    ap.add_argument("--preview-distances", default=None, help='look-ahead distances of the clean preview, e.g. "0.2,0.4,0.6"')
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timesteps", type=int, default=20_000_000)
    ap.add_argument("--num-envs", type=int, default=128)
    ap.add_argument("--amp-lo", type=float, default=0.02, help="terrain amplitude range [m] drawn per episode")
    ap.add_argument("--amp-hi", type=float, default=0.06)
    ap.add_argument("--safe-radius", type=float, default=3.0, help="flat zone around the start [m]")
    ap.add_argument("--logfreq", type=int, default=10)
    ap.add_argument("--tag", default="")
    ap.add_argument("--smoke", action="store_true", help="plumbing only: 2 envs, 64 steps/rollout, 256 steps in total")
    ap.add_argument("--root", default=None)
    a = ap.parse_args()
    if a.smoke:
        a.num_envs, a.timesteps, a.logfreq = 2, 256, 1
    return a


if __name__ == "__main__":
    from terrain_memory_prosthesis.training.stage2 import run

    run(parse())
