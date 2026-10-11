import importlib.util
import sys
from pathlib import Path

import numpy as np

PILOTS = Path(__file__).resolve().parents[1] / "scripts" / "pilots"


def _load():
    sys.path.insert(0, str(PILOTS))
    spec = importlib.util.spec_from_file_location("pilot04", PILOTS / "04_check_gru_memory.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_toy_env_needs_memory():
    m = _load()
    Env = m.make_env_cls()
    # an agent that only sees the current observation can at best guess: acc ~ 0.5
    acc = m.evaluate(lambda obs, st, ep: (np.random.randint(2), None), Env, 400)
    assert 0.4 < acc < 0.6
    # an agent that remembers the cue is perfect
    memo = {}

    def oracle(obs, st, ep):
        if obs[0] != 0:
            memo["c"] = int(obs[0] > 0)
        return memo["c"], None

    assert m.evaluate(oracle, Env, 100) == 1.0
    # the cue is invisible at the query step
    e = Env(fixed_delay=3)
    o, _ = e.reset(seed=0)
    for _ in range(3):
        o, *_ = e.step(0)
        assert o[0] == 0.0
    o, *_ = e.step(0)
    assert o[1] == 1.0 and o[0] == 0.0
