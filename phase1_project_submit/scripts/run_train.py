#!/usr/bin/env python3
"""Train the Phase 1 PINN.

    python scripts/run_train.py --tag smoke --iterations 2000
    python scripts/run_train.py --tag phase1_v1
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sysid import config, dataio, train                      


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="run", help="name for this run's outputs")
    p.add_argument("--iterations", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--config", default=None)
    args = p.parse_args()

    cfg = config.load_config(args.config)
    data = dataio.load(cfg)

    print(f"Loaded {data.n} samples, "
          f"t = {data.t.min():.2f} to {data.t.max():.2f} s")
    print(f"Mass  {cfg['physics']['mass']:,.0f} kg  "
          f"[{data.meta.get('mass_status', 'no status')}]")
    print()

    train.run(cfg, data, tag=args.tag,
              iterations=args.iterations, seed=args.seed)


if __name__ == "__main__":
    main()
