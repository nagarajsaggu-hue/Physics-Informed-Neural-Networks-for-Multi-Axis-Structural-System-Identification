"""Figures for Phase 1.

Four plots, each answering one question:

  timeseries   - does the network reproduce the measured displacement?
  convergence  - did the identified parameters settle, or are they drifting?
  spectrum     - is the dominant line the drive frequency or a natural one?
  comparison   - do the three methods agree on K?
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # no display on a compute node
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _fig_dir(cfg):
    d = Path(cfg["output"]["dir"]) / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def timeseries(cfg, tag, window=None):
    """Predicted against measured displacement, with the error below."""
    path = Path(cfg["output"]["dir"]) / f"prediction_{tag}.csv"
    df = pd.read_csv(path)
    t = df["Time (s)"].to_numpy()
    meas = df["u_measured (m)"].to_numpy()
    pred = df["u_predicted (m)"].to_numpy()
    err = df["error (m)"].to_numpy()

    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                           gridspec_kw={"height_ratios": [2, 1]})

    sel = slice(None)
    if window:
        sel = (t >= window[0]) & (t <= window[1])

    ax[0].plot(t[sel], meas[sel], lw=1.4, label="measured (ANSYS)")
    ax[0].plot(t[sel], pred[sel], lw=1.0, ls="--", label="PINN")
    ax[0].set_ylabel("displacement (m)")
    ax[0].legend(loc="upper right")
    rmse = float(np.sqrt(np.mean(err ** 2)))
    # Dynamically extract the direction from the config
    obs_col = cfg["data"].get("observation_column", "X")
    if "_Y" in obs_col:
        direction = "Y"
    elif "_Z" in obs_col:
        direction = "Z"
    else:
        direction = "X"
        
    ax[0].set_title(f"Node 3575, {direction} direction - RMSE {rmse:.3e} m")
    ax[0].grid(alpha=0.3)

    ax[1].plot(t[sel], err[sel], lw=0.9, color="crimson")
    ax[1].axhline(0, color="black", lw=0.6)
    ax[1].set_ylabel("error (m)")
    ax[1].set_xlabel("time (s)")
    ax[1].grid(alpha=0.3)

    fig.tight_layout()
    out = _fig_dir(cfg) / f"timeseries_{tag}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def convergence(cfg, tag, reference=None):
    """Identified K and C against training iteration."""
    path = Path(cfg["output"]["dir"]) / f"convergence_{tag}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for i, (col, label) in enumerate(
            [("K_N_per_m", "K (N/m)"), ("C_N_s_per_m", "C (N.s/m)")]):
        ax[i].plot(df["iteration"], df[col], lw=1.3)
        if reference and reference.get(["K", "C"][i]) is not None:
            ax[i].axhline(reference[["K", "C"][i]], color="crimson",
                          ls="--", lw=1.0, label="least squares")
            ax[i].legend()
        ax[i].set_xlabel("iteration")
        ax[i].set_ylabel(label)
        ax[i].grid(alpha=0.3)
    ax[0].set_title("Parameter convergence")

    fig.tight_layout()
    out = _fig_dir(cfg) / f"convergence_{tag}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def spectrum(cfg, data, tag):
    """Displacement spectrum with the drive and natural frequencies marked."""
    u = data.u.ravel()
    n = len(u)
    freqs = np.fft.rfftfreq(n, data.dt)
    mag = np.abs(np.fft.rfft(u * np.hanning(n)))
    mag = mag / mag.max()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.semilogy(freqs, np.maximum(mag, 1e-6), lw=1.1)
    ax.axvline(cfg["physics"]["force_frequency"], color="crimson", ls="--",
               lw=1.2, label=f"drive {cfg['physics']['force_frequency']:.1f} Hz")

    modal = data.modal or {}
    for m in (modal.get("modes") or [])[:3]:
        ax.axvline(m["frequency_hz"], color="seagreen", ls=":", lw=1.0)
    if modal.get("modes"):
        ax.plot([], [], color="seagreen", ls=":", label="ANSYS modal")

    ax.set_xlim(0, min(60, freqs.max()))
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("normalised amplitude")
    ax.set_title("Displacement spectrum")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = _fig_dir(cfg) / f"spectrum_{tag}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def comparison_bars(cfg, comparison, tag):
    """Bar chart of K from each method."""
    rows = [r for r in comparison["rows"] if r["K_N_per_m"]]
    if len(rows) < 2:
        return None

    names = [r["method"] for r in rows]
    vals = [r["K_N_per_m"] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, vals, color=["#4C6FBF", "#C0504D", "#6FA96F"][:len(rows)])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3e}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("K (N/m)")
    ax.set_title("Identified stiffness by method")
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    out = _fig_dir(cfg) / f"comparison_{tag}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
