#!/usr/bin/env python
"""Print `export` lines for a sensor-imperfection level, scaled per channel.

    eval "$(python scripts/noise_env.py channel_stats/channel_std.json --alpha 0.1 --delay 1 --dropout 0.02)"

alpha = noise std relative to each channel's normal variation (0 = no noise). Sets MYO_SENSOR_NOISE_STD (JSON list of 40 numbers),
MYO_SENSOR_DELAY_STEPS and MYO_SENSOR_DROPOUT_PROBABILITY (a scalar). Nothing is printed for a zero value, so alpha 0 / delay 0 /
dropout 0 is the clean sensor.
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stats")
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--delay", type=int, default=0)
    ap.add_argument("--dropout", type=float, default=0.0)
    a = ap.parse_args()
    if a.alpha > 0:
        std = json.load(open(a.stats))["std"]
        print("export MYO_SENSOR_NOISE_STD='" + json.dumps([round(a.alpha * s, 6) for s in std]) + "'")
    if a.delay > 0:
        print(f"export MYO_SENSOR_DELAY_STEPS={a.delay}")
    if a.dropout > 0:
        print(f"export MYO_SENSOR_DROPOUT_PROBABILITY={a.dropout}")


if __name__ == "__main__":
    main()
