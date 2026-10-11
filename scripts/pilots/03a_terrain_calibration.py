#!/usr/bin/env python
"""03a - Which terrain amplitude gives the controller something to learn (headroom)?

Question : Too flat -> the blind human walks anyway and terrain preview (P2) cannot help; too rough -> nobody walks and
           nothing can be learned. We need an amplitude band where the Stage-1 policy degrades but does not collapse.
Method   : evaluate the Stage-1 policy (deterministic, fixed seeds) on
             - flat ground,
             - random terrain: several amplitudes x several terrain seeds,
             - sinusoidal terrain: several amplitudes,
           N episodes each, long episodes. Band = amplitudes whose fall rate is in [--band-lo, --band-hi].
Verdict  : FAIL  flat baseline already falls (> --baseline-max-fall) -> go back to 00
           WARN  no amplitude in the band: terrain is either always easy (increase amplitudes) or a cliff (refine grid)
           PASS  at least one amplitude in the band
Caveat   : The Stage-1 human here was (probably) trained on FLAT ground. A human trained on the target terrain is
           better than this one, so the real headroom is smaller than measured. Use this to choose the band for the
           Stage-1 terrain curriculum, not as a final number.
Terrain  : upstream default heightfield is 20 m wide (walker leaves it after ~10 m) with a 3 m flat start; we use a
           60 m field (see common.terrain_spec).
Exit code: 0 PASS, 1 WARN, 2 FAIL.

Example:
    python 03a_terrain_calibration.py --session rl_train/results/train_session_X --episodes 10 --workers 8
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--checkpoint", default=None, help="default: latest checkpoint")
    ap.add_argument("--amps", default="0.01,0.02,0.03,0.04,0.06", help="random-terrain amplitudes [m]")
    ap.add_argument("--terrain-seeds", type=int, default=3, help="different random terrains per amplitude")
    ap.add_argument("--sin-amps", default="0.02,0.04")
    ap.add_argument("--sin-period", type=float, default=1.0)
    ap.add_argument("--episodes", type=int, default=10, help="episodes per terrain instance")
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--extent", type=float, default=60.0)
    ap.add_argument("--resolution", type=int, default=768)
    ap.add_argument("--band-lo", type=float, default=0.2)
    ap.add_argument("--band-hi", type=float, default=0.8)
    ap.add_argument("--baseline-max-fall", type=float, default=0.2)
    ap.add_argument("--target-velocity", type=float, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default=None)
    return ap.parse_args()


def floats(s):
    return [float(x) for x in s.split(",") if x.strip()]


def main():
    args = parse()
    root = common.init(args.root)
    out = common.out_dir(args.out, "pilot_out/03a")
    ck = common.resolve(args.checkpoint) if args.checkpoint else common.list_checkpoints(args.session)[-1][1]
    print("checkpoint:", ck)
    T = lambda kind, amp=0.0, seed=0: common.terrain_spec(kind, amp, seed, args.sin_period, args.extent, args.resolution)

    items = [("flat", "flat", 0.0, 0, T("flat"))]
    for a in floats(args.amps):
        for s in range(args.terrain_seeds):
            items.append((f"random a={a:g} s={s}", "random", a, s, T("random", a, s)))
    for a in floats(args.sin_amps):
        items.append((f"sinusoidal a={a:g}", "sinusoidal", a, 0, T("sinusoidal", a)))

    tasks = [
        dict(config=args.session, checkpoint=ck, episodes=args.episodes, max_steps=args.max_steps, seed0=0,
             target_velocity=args.target_velocity, terrain=terr, label=lab, extra={"kind": kind, "amplitude": a, "terrain_seed": s})
        for lab, kind, a, s, terr in items
    ]
    print(f"{len(tasks)} terrains x {args.episodes} episodes, workers={args.workers}", flush=True)
    rows = common.run_parallel(tasks, args.workers, root=root)
    for r in rows:
        r["steps_frac"] = r["mean_steps"] / args.max_steps
    flat_rows = common.strip_episodes(rows)
    common.save_rows(out, "03a_terrains", flat_rows)
    cols = ["label", "n", "fall_rate", "steps_frac", "mean_distance_m", "mean_speed_mps"]
    print()
    common.print_table(flat_rows, cols, {c: ".2f" for c in cols[2:]})

    # aggregate per (kind, amplitude) over terrain seeds
    agg = {}
    for r in rows:
        agg.setdefault((r["kind"], r["amplitude"]), []).append(r)
    summary = []
    for (kind, a), rs in sorted(agg.items()):
        eps = [e for r in rs for e in r["episodes"]]
        s = common.summarize(eps)
        s.update(kind=kind, amplitude=a, terrains=len(rs), steps_frac=s["mean_steps"] / args.max_steps,
                 fall_rate_per_terrain_min=min(r["fall_rate"] for r in rs), fall_rate_per_terrain_max=max(r["fall_rate"] for r in rs))
        summary.append(s)
    common.save_rows(out, "03a_summary", summary)
    print("\nAggregated per amplitude:")
    c2 = ["kind", "amplitude", "terrains", "n", "fall_rate", "fall_rate_per_terrain_min", "fall_rate_per_terrain_max", "steps_frac", "mean_distance_m"]
    common.print_table(summary, c2, {c: ".2f" for c in c2[4:]})

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        for kind in ("random", "sinusoidal"):
            ss = [s for s in summary if s["kind"] == kind]
            if ss:
                ax.errorbar([s["amplitude"] * 100 for s in ss], [s["fall_rate"] for s in ss],
                            yerr=[[s["fall_rate"] - s["fall_ci_lo"] for s in ss], [s["fall_ci_hi"] - s["fall_rate"] for s in ss]], fmt="o-", capsize=3, label=kind)
        fl = [s for s in summary if s["kind"] == "flat"][0]
        ax.axhline(fl["fall_rate"], c="gray", ls=":", label="flat")
        ax.axhspan(args.band_lo, args.band_hi, color="g", alpha=0.1, label="target band")
        ax.set_xlabel("terrain amplitude [cm]")
        ax.set_ylabel("fall rate")
        ax.set_ylim(-0.02, 1.02)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out, "03a_terrain_calibration.png"), dpi=120)
    except Exception as e:
        print("(plot skipped)", e)

    base = [s for s in summary if s["kind"] == "flat"][0]
    rnd = [s for s in summary if s["kind"] == "random"]
    inband = [s for s in summary if s["kind"] != "flat" and args.band_lo <= s["fall_rate"] <= args.band_hi]
    reasons = [f"flat baseline fall rate {base['fall_rate']:.2f}"]
    if base["fall_rate"] > args.baseline_max_fall:
        status = "FAIL"
        reasons.append("flat baseline already falls -> Stage 1 not usable, back to 00")
    elif inband:
        status = "PASS"
        reasons.append("amplitudes in the target band: " + ", ".join(f"{s['kind']} {s['amplitude']*100:g} cm (fall {s['fall_rate']:.2f})" for s in inband))
        reasons.append("pick the band for the training terrain distribution; fix evaluation terrain seeds (those used here are a start)")
    else:
        status = "WARN"
        if rnd and all(s["fall_rate"] < args.band_lo for s in rnd):
            reasons.append("every tested amplitude is easy (fall < band) -> a flat-trained human is not troubled; test larger amplitudes / add slopes")
        elif rnd and all(s["fall_rate"] > args.band_hi for s in rnd):
            reasons.append("every tested amplitude is too hard -> test smaller amplitudes")
        else:
            reasons.append("a cliff between neighbouring amplitudes: refine the grid between them")
    reasons.append("caveat: Stage-1 human not trained on rough terrain -> real headroom (after terrain training) is smaller")
    return common.verdict(out, "03a_terrain_calibration", status, reasons, {"checkpoint": ck})


if __name__ == "__main__":
    sys.exit(main())
