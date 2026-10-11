import json
import math

from terrain_memory_prosthesis.evaluation.reporting import summarize, verdict, wilson_ci


def test_wilson_ci_bounds():
    lo, hi = wilson_ci(0, 10)
    assert lo == 0.0 or lo < 1e-9
    assert 0.2 < hi < 0.35
    lo, hi = wilson_ci(10, 10)
    assert hi > 0.999 and 0.65 < lo < 0.8
    lo, hi = wilson_ci(5, 10)
    assert lo < 0.5 < hi
    assert all(math.isnan(x) for x in wilson_ci(0, 0))


def _ep(fell, steps):
    return dict(fell=fell, steps=steps, distance=steps * 0.04, speed=1.2, speed_err=0.1, prosthesis_cmd=0.3)


def test_summarize():
    s = summarize([_ep(True, 10), _ep(False, 30)])
    assert s["n"] == 2 and s["falls"] == 1 and s["fall_rate"] == 0.5
    assert s["mean_steps"] == 20 and s["fall_ci_lo"] < 0.5 < s["fall_ci_hi"]


def test_verdict_exit_codes(tmp_path):
    assert verdict(str(tmp_path), "x", "PASS", ["ok"]) == 0
    assert verdict(str(tmp_path), "x", "WARN", ["hmm"]) == 1
    assert verdict(str(tmp_path), "x", "FAIL", ["bad"]) == 2
    assert json.loads((tmp_path / "x_verdict.json").read_text())["status"] == "FAIL"
