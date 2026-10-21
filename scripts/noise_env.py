#!/usr/bin/env python
"""Print `export` lines for a sensor-imperfection level, scaled per channel.

    eval "$(python scripts/noise_env.py channel_stats/channel_std.json --alpha 0.1 --delay 1 --dropout 0.02)"

alpha = noise std relative to each channel's normal variation (0 = no noise). With --sigma-ang/--sigma-vel/--sigma-force (physical units, alpha ignored) the noise is absolute per sensor group. Sets MYO_SENSOR_NOISE_STD (JSON list of 40 numbers),
MYO_SENSOR_DELAY_STEPS and MYO_SENSOR_DROPOUT_PROBABILITY (a scalar). Nothing is printed for a zero value, so alpha 0 / delay 0 /
dropout 0 is the clean sensor.
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stats")
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--sigma-ang", type=float, default=0.0, help="absolute noise std of the 8 joint angles [rad] (obs 0:8)")
    ap.add_argument("--sigma-vel", type=float, default=0.0, help="absolute noise std of the 9 joint velocities [rad/s] (obs 8:17)")
    ap.add_argument("--sigma-force", type=float, default=0.0, help="absolute noise std of the 4 foot/toe contact forces [body weight] (obs 35:39)")
    ap.add_argument("--delay", type=int, default=0)
    ap.add_argument("--dropout", type=float, default=0.0)
    a = ap.parse_args()
    if a.sigma_ang > 0 or a.sigma_vel > 0 or a.sigma_force > 0:  # absolute (physical) noise per sensor group; the other channels stay clean
        std = [0.0] * 40
        std[0:8] = [a.sigma_ang] * 8
        std[8:17] = [a.sigma_vel] * 9
        std[35:39] = [a.sigma_force] * 4
        print("export MYO_SENSOR_NOISE_STD='" + json.dumps(std) + "'")
    elif a.alpha > 0:
        std = json.load(open(a.stats))["std"]
        print("export MYO_SENSOR_NOISE_STD='" + json.dumps([round(a.alpha * s, 6) for s in std]) + "'")
    if a.delay > 0:
        print(f"export MYO_SENSOR_DELAY_STEPS={a.delay}")
    if a.dropout > 0:
        print(f"export MYO_SENSOR_DROPOUT_PROBABILITY={a.dropout}")


if __name__ == "__main__":
    main()
