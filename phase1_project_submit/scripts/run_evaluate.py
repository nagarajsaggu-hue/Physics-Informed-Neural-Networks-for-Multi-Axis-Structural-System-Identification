#!/usr/bin/env python3
"""Compare a trained result against the least-squares and ANSYS references.

    python scripts/run_evaluate.py --tag phase1_v1
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sysid import config, dataio, evaluate                   


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="run")
    p.add_argument("--config", default=None)
    args = p.parse_args()

    cfg = config.load_config(args.config)
    data = dataio.load(cfg)
    evaluate.run(cfg, data, tag=args.tag)


if __name__ == "__main__":
    main()
