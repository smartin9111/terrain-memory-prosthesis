import csv
import math

import pytest

from terrain_memory_prosthesis.evaluation import analysis as A


def _row(run, cond, ctrl, kind, amp, ts, n, falls, seed=0, ckpt=100):
    return dict(run=run, cond=cond, controller=ctrl, seed=seed, ckpt_steps=ckpt, terrain_kind=kind, amplitude=amp, terrain_seed=ts, n=n,
                falls=falls, fall_rate=falls / n, fall_ci_lo=0.0, fall_ci_hi=1.0, steps_frac=1 - falls / n, mean_distance_m=10.0 * (1 - falls / n),
                speed_err_mps=0.2, mean_abs_prosthesis_cmd=0.5, mean_steps=100.0, mean_speed_mps=1.2)


def _rows():
    rows = []
    for ts in (1, 2, 3):
        for amp, f1, f2 in ((0.02, 0, 0), (0.04, 2, 1), (0.06, 4, 2)):
            rows.append(_row("r1", "P1", "mlp", "random", amp, ts, 5, f1))
            rows.append(_row("r2", "P2", "mlp", "random", amp, ts, 5, f2))
            rows.append(_row("s1", "stage1", "", "random", amp, ts, 5, f1))
    rows.append(_row("r1", "P1", "mlp", "flat", 0.0, 0, 5, 0))
    rows.append(_row("r2", "P2", "mlp", "flat", 0.0, 0, 5, 0))
    rows.append(_row("r1", "P1", "mlp", "sinusoidal", 0.04, 0, 5, 0))
    rows.append(_row("r2", "P2", "mlp", "sinusoidal", 0.04, 0, 5, 0))
    return rows


def test_cell_labels():
    assert A.cell_label({"cond": "P1", "controller": "mlp"}) == "P1xC1"
    assert A.cell_label({"cond": "P2", "controller": "gru"}) == "P2xC2"
    assert A.cell_label({"cond": "stage1"}) == "Stage 1"


def test_pool_sums_falls_and_recomputes_interval():
    rows = A.with_labels(_rows())
    p = {r["cell"]: r for r in A.pool([r for r in rows if r["terrain_kind"] == "random" and r["amp_cm"] == 6.0], ("cell",))}
    assert p["P1xC1"]["n"] == 15 and p["P1xC1"]["falls"] == 12 and p["P1xC1"]["fall_rate"] == pytest.approx(0.8)
    assert 0.5 < p["P1xC1"]["fall_ci_lo"] < 0.8 < p["P1xC1"]["fall_ci_hi"] <= 1.0


def test_newcombe_difference():
    d, lo, hi = A.newcombe_diff(12, 15, 6, 15)  # 80 % -> 40 %
    assert d == pytest.approx(-0.4) and lo < d < hi and hi < 0.05  # the interval excludes 0 only barely or not at all: sign checked
    d0, lo0, hi0 = A.newcombe_diff(3, 10, 3, 10)
    assert d0 == 0 and lo0 == pytest.approx(-hi0) and lo0 < 0 < hi0


def test_difference_table_needs_both_cells():
    t = A.summary_table(A.latest_ckpt_rows(_rows()))
    diff = A.difference_table(t)
    assert {r["controller"] for r in diff} == {"C1"}
    row = [r for r in diff if r["terrain"] == "random 6 cm"][0]
    assert row["p1"] == pytest.approx(0.8) and row["p2"] == pytest.approx(0.4) and row["diff_p2_minus_p1"] == pytest.approx(-0.4)
    assert A.difference_table([r for r in t if r["cell"] != "P2xC1"]) == []


def test_latest_ckpt_rows():
    rows = [_row("r1", "P1", "mlp", "flat", 0.0, 0, 5, 1, ckpt=50), _row("r1", "P1", "mlp", "flat", 0.0, 0, 5, 0, ckpt=100)]
    assert [r["ckpt_steps"] for r in A.latest_ckpt_rows(rows)] == [100]


def test_read_rows_numeric(tmp_path):
    p = tmp_path / "e.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run", "cond", "n", "falls", "amplitude"])
        w.writeheader()
        w.writerow(dict(run="a", cond="P1", n="5", falls="2", amplitude="0.04"))
    r = A.read_rows(str(p))[0]
    assert r["n"] == 5.0 and r["amplitude"] == pytest.approx(0.04) and r["run"] == "a"


def test_figures_are_written(tmp_path):
    rows = A.latest_ckpt_rows(_rows())
    A.fig_fall_rate(rows, str(tmp_path / "a.png"))
    A.fig_metrics(rows, str(tmp_path / "b.png"))
    assert (tmp_path / "a.png").stat().st_size > 5000 and (tmp_path / "b.png").stat().st_size > 5000
    assert A.fig_checkpoints(rows, str(tmp_path / "c.png")) is False  # one checkpoint per run: no curve
    assert not math.isnan(A.smooth([1.0, 2.0, 3.0])[-1])
