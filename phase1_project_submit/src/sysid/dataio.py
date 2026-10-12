"""Loading and non-dimensionalising the Phase 1 dataset.

The CSV holds displacement, velocity, acceleration and force against time.
Only DISPLACEMENT is used as a training target: the velocity and acceleration
columns exported by ANSYS are not accurate derivatives of the displacement
column (correlation with da/dt is 0.88), and fitting with them produces
negative - that is, unphysical - damping. The PINN obtains its derivatives by
automatic differentiation from the network instead.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import load_json


@dataclass
class Dataset:
    """One loaded and scaled Phase 1 record."""

    t: np.ndarray          # seconds, shape (N, 1)
    u: np.ndarray          # metres, shape (N, 1)
    u_hat: np.ndarray      # dimensionless displacement
    f: np.ndarray          # newtons, shape (N, 1)
    dt: float
    meta: dict
    modal: dict

    @property
    def n(self) -> int:
        return len(self.t)


def _match_column(columns, *keywords):
    """Find a column containing all keywords, case-insensitively.

    Tolerant matching so that a re-exported CSV with slightly different
    headers still loads rather than failing at the first bracket.
    """
    for col in columns:
        low = col.lower()
        if all(k.lower() in low for k in keywords):
            return col
    raise KeyError(f"No column matching {keywords} in {list(columns)}")


def load(cfg) -> Dataset:
    """Load the CSV and metadata described by cfg, apply the time window,
    and non-dimensionalise."""
    df = pd.read_csv(cfg["data"]["csv"])

    t_col = _match_column(df.columns, "time")
    u_col = cfg["data"]["observation_column"]  # Strictly obeys the active YAML block
    f_col = _match_column(df.columns, "force")

    t = df[t_col].to_numpy(dtype=np.float64)
    u = df[u_col].to_numpy(dtype=np.float64)
    f = df[f_col].to_numpy(dtype=np.float64)

    t_min = cfg["data"].get("t_min") or t.min()
    t_max = cfg["data"].get("t_max")
    t_max = t.max() if t_max is None else float(t_max)
    keep = (t >= t_min - 1e-12) & (t <= t_max + 1e-12)
    t, u, f = t[keep], u[keep], f[keep]

    if len(t) < 50:
        raise ValueError(
            f"Only {len(t)} samples left after the time window "
            f"[{t_min}, {t_max}] - widen it in config.yaml."
        )

    dt_all = np.diff(t)
    dt = float(np.median(dt_all))
    if np.ptp(dt_all) > 1e-6 * dt:
        print(f"  WARNING: time step is not uniform "
              f"(spread {np.ptp(dt_all):.3e} s)")

    u_hat = u / cfg["scaling"]["displacement"]

    return Dataset(
        t=t.reshape(-1, 1),
        u=u.reshape(-1, 1),
        u_hat=u_hat.reshape(-1, 1),
        f=f.reshape(-1, 1),
        dt=dt,
        meta=load_json(cfg["data"]["meta"]) or {},
        modal=load_json(cfg["data"]["modal"]) or {},
    )


if __name__ == "__main__":
    from .config import load_config

    cfg = load_config()
    d = load(cfg)
    print(f"samples      : {d.n}")
    print(f"time         : {d.t.min():.3f} to {d.t.max():.3f} s, dt = {d.dt:.5f}")
    print(f"displacement : {d.u.min():.4e} to {d.u.max():.4e} m")
    print(f"scaled u_hat : {d.u_hat.min():.4f} to {d.u_hat.max():.4f}")
    print(f"force        : peak {np.abs(d.f).max():,.1f} N")
    print(f"mass (meta)  : {d.meta.get('generalized_mass_kg')} kg "
          f"[{d.meta.get('mass_status', 'no status')}]")
