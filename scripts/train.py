#!/usr/bin/env python
"""Launcher for MyoAssist's rl_train/run_train.py that first registers our extra env ids
(...ExoStiff-v0 = hard prosthesis joint limit, ...ExoOSLPhys-v0 = + OSL motor model, ...ExoPreview-v0). Same CLI as run_train.py:

    python scripts/train.py --config_file_path <cfg.json> --config.total_timesteps 30000000 ...

Registration happens at import time of THIS file (top level), so SB3 spawn/forkserver workers, which re-import it as
__mp_main__, know the ids as well. Must be started from anywhere; it chdirs to the MyoAssist checkout (results land in
<myoassist>/rl_train/results as usual).
"""
import os
import runpy
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pilots"))
import common  # noqa: E402  (also sets thread env vars, chdir to the MyoAssist root, sys.path)

root = common.init(os.environ.get("MYOASSIST_DIR"), threads=int(os.environ.get("MYOASSIST_NUM_THREADS", "8")))
from terrain_memory_prosthesis.environment import osl_physics, preview_env, stiff_limit  # noqa: E402,F401  (register the ids)

if __name__ == "__main__":
    # a relative --config_file_path given by the user is resolved against the directory the launcher was started from
    argv = sys.argv[1:]
    for i, a in enumerate(argv[:-1]):
        if a == "--config_file_path":
            argv[i + 1] = common.resolve(argv[i + 1])
    target = os.path.join(root, "rl_train", "run_train.py")
    sys.argv = [target] + argv
    # exec run_train.py INSIDE this module (not runpy): SB3's workers re-import the *__main__ file*, which must stay this
    # launcher (so the env ids get registered in the workers); runpy.run_path would replace __main__ by run_train.py.
    with open(target) as f:
        code = compile(f.read(), target, "exec")
    exec(code, globals())
