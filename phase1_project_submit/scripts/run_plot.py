#!/usr/bin/env python3
"""Generate all Phase 1 figures for a run.

    python scripts/run_plot.py --tag phase1_v1
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sysid import config, dataio, evaluate, physics, viz     


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="run")
    p.add_argument("--window", type=float, nargs=2, default=None,
                   help="zoom the time series, e.g. --window 0 1")
    p.add_argument("--config", default=None)
    args = p.parse_args()

    cfg = config.load_config(args.config)
    data = dataio.load(cfg)
    ref = physics.least_squares_reference(data, cfg)

    made = []
    made.append(viz.timeseries(cfg, args.tag, window=args.window))
    made.append(viz.convergence(cfg, args.tag, reference=ref))
    made.append(viz.spectrum(cfg, data, args.tag))

    cmp_path = Path(cfg["output"]["dir"]) / f"comparison_{args.tag}.json"
    if cmp_path.exists():
        with open(cmp_path) as f:
            made.append(viz.comparison_bars(cfg, json.load(f), args.tag))

    for m in made:
        if m:
            print(f"  {m}")


if __name__ == "__main__":
    main()
