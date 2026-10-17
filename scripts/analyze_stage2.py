#!/usr/bin/env python
"""Comparison figures + tables from an evaluation made by scripts/eval_stage2.py (and the training logs of the runs).

    python scripts/analyze_stage2.py --eval eval_out --runs "<myoassist>/rl_train/results/stage2_P*" --out report_out

Writes into --out:
    fig_fall_rate.png        fall rate vs terrain amplitude per cell (P1xC1, P2xC1, ...), 95 % Wilson intervals, + Stage-1 reference
    fig_metrics.png          survival, distance, speed error, prosthesis command vs amplitude
    fig_training.png         training curves (reward, episode length, policy std, critic explained variance)
    fig_checkpoints.png      fall rate vs training steps (only if eval_stage2 was run with --ckpts > 1)
    summary.csv / summary.md fall rate [CI], survival, distance, speed error per cell x terrain
    differences.csv / .md    P2 - P1 fall-rate difference with a Newcombe 95 % interval (per controller, per terrain)
Small n means wide intervals: read the interval, not only the point. Per-seed dots show the spread between training seeds.
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
try:
    from terrain_memory_prosthesis.evaluation import analysis as A
except ImportError:  # run from a checkout without `pip install -e .`
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
    from terrain_memory_prosthesis.evaluation import analysis as A


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", required=True, help="folder with stage2_eval.csv (the --out of eval_stage2.py)")
    ap.add_argument("--runs", nargs="*", default=[], help="Stage-2 session folders / globs (for the training curves)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = A.read_rows(os.path.join(a.eval, "stage2_eval.csv"))
    if not rows:
        raise SystemExit("stage2_eval.csv is empty")
    last = A.latest_ckpt_rows(rows)

    A.fig_fall_rate(last, os.path.join(a.out, "fig_fall_rate.png"))
    A.fig_metrics(last, os.path.join(a.out, "fig_metrics.png"))
    runs = A.expand_runs(a.runs)
    if runs:
        A.fig_training(runs, os.path.join(a.out, "fig_training.png"))
    made_ck = A.fig_checkpoints(rows, os.path.join(a.out, "fig_checkpoints.png"))

    table = A.summary_table(last)
    for r in table:
        r["fall_ci"] = f"[{r['fall_ci_lo']:.2f}, {r['fall_ci_hi']:.2f}]"
    cols = ["cell", "terrain", "n", "falls", "fall_rate", "fall_ci", "steps_frac", "mean_distance_m", "speed_err_mps", "mean_abs_prosthesis_cmd"]
    fmt = {c: ".3f" for c in ("fall_rate", "steps_frac", "mean_distance_m", "speed_err_mps", "mean_abs_prosthesis_cmd")}
    write_csv(os.path.join(a.out, "summary.csv"), table)
    diff = A.difference_table(table)
    for r in diff:
        r["ci"] = f"[{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]"
    write_csv(os.path.join(a.out, "differences.csv"), diff)
    with open(os.path.join(a.out, "summary.md"), "w") as f:
        f.write("# Stage-2 evaluation summary\n\nFall rate with 95 % Wilson interval; all terrain seeds and training seeds of a cell pooled.\n\n")
        f.write(A.markdown_table(table, cols, fmt) + "\n")
        if diff:
            f.write("\n## P2 - P1 (fall rate, same controller)\n\nNegative = the preview helps. Newcombe 95 % interval; "
                    "an interval containing 0 means the data do not separate the two.\n\n")
            f.write(A.markdown_table(diff, ["controller", "terrain", "p1", "p2", "diff_p2_minus_p1", "ci", "n_p1", "n_p2"],
                                     {c: ".3f" for c in ("p1", "p2", "diff_p2_minus_p1")}) + "\n")
    print(A.markdown_table(table, cols, fmt))
    if diff:
        print()
        print(A.markdown_table(diff, ["controller", "terrain", "p1", "p2", "diff_p2_minus_p1", "ci", "n_p1", "n_p2"],
                               {c: ".3f" for c in ("p1", "p2", "diff_p2_minus_p1")}))
    print(f"\nwritten to {a.out}: fig_fall_rate.png, fig_metrics.png" + (", fig_training.png" if runs else "")
          + (", fig_checkpoints.png" if made_ck else "") + ", summary.csv/.md, differences.csv")


if __name__ == "__main__":
    main()
