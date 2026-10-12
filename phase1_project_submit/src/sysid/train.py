"""PINN training for Phase 1.

The network maps time to scaled displacement. Two scalars, k_hat and c_hat,
are trained alongside the weights. The loss has two terms:

    physics   - the equation of motion residual at collocation points
    data      - the difference between the network and the measured
                displacement at the sample times

Only displacement is used as data. Velocity and acceleration come from
automatic differentiation.
"""

import json
import math
import time
from pathlib import Path

import numpy as np

from . import physics


def run(cfg, data, tag="run", iterations=None, seed=0):
    """Train and return a dict of identified parameters and diagnostics."""
    import torch
    import deepxde as dde

    dde.config.set_random_seed(seed)

    iterations = iterations or cfg["training"]["iterations"]
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- trainable parameters, in scaled units ------------------------
    # Stored as RAW values and passed through softplus in the residual, so
    # stiffness and damping cannot become negative. The init values below are
    # inverse-softplus of the requested starting points, so k_hat_init in
    # config.yaml still means what it says.
    positive = bool(cfg["training"].get("enforce_positive", True))
    k_init = float(cfg["training"]["k_hat_init"])
    c_init = float(cfg["training"]["c_hat_init"])
    if positive:
        k_raw = dde.Variable(physics.inverse_softplus(k_init))
        c_raw = dde.Variable(physics.inverse_softplus(c_init))
        print("Parameters constrained positive (softplus)")
    else:
        k_raw = dde.Variable(k_init)
        c_raw = dde.Variable(c_init)

    residual = physics.make_residual(cfg, k_raw, c_raw, positive=positive)

    t_lo = float(data.t.min())
    t_hi = float(data.t.max())
    geom = dde.geometry.TimeDomain(t_lo, t_hi)

    # --- hold out part of the data ------------------------------------
    # DeepXDE evaluates its "test loss" on the training points, so the loss it
    # reports is a training fit, not evidence of generalisation. Holding out
    # samples the network never sees gives an honest error figure.
    #
    # The held-out points are spread evenly through the record rather than
    # taken as a block at the end, so the check covers the whole time span.
    holdout_every = int(cfg["training"].get("holdout_every", 5))
    if holdout_every > 1:
        idx = np.arange(data.n)
        test_mask = (idx % holdout_every) == (holdout_every // 2)
        train_mask = ~test_mask
    else:
        train_mask = np.ones(data.n, dtype=bool)
        test_mask = np.zeros(data.n, dtype=bool)

    t_train, u_train = data.t[train_mask], data.u_hat[train_mask]
    t_test, u_test = data.t[test_mask], data.u_hat[test_mask]
    print(f"Training on {train_mask.sum()} points, "
          f"holding out {test_mask.sum()} for validation")

    observation = dde.icbc.PointSetBC(t_train, u_train, component=0)

    problem = dde.data.PDE(
        geom,
        residual,
        [observation],
        num_domain=int(cfg["training"]["num_domain"]),
        num_boundary=2,
        anchors=t_train,
    )

    layers = list(cfg["network"]["layers"])
    ff = cfg["network"].get("fourier_features") or {}
    use_ff = bool(ff.get("enabled", False))

    if use_ff:
        # FOURIER FEATURES
        # A plain network fed raw time cannot represent many oscillations: with
        # Glorot-scale weights the first layer spans roughly one cycle over the
        # input range, and gradient descent will not climb from there to 190
        # cycles. Zero output becomes a stable local minimum - which is exactly
        # what a stuck, flat loss looks like.
        #
        # Supplying sin and cos of time at a spread of frequencies means the
        # network only has to learn amplitudes. The frequencies are given, not
        # discovered.
        freqs = np.linspace(float(ff.get("f_min", 5.0)),
                            float(ff.get("f_max", 40.0)),
                            int(ff.get("n", 64)))
        freq_t = torch.tensor(freqs, dtype=torch.float32).reshape(1, -1)

        def feature_transform(t):
            ang = 2.0 * np.pi * t * freq_t.to(t.device)
            return torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)

        layers[0] = 2 * len(freqs)
        print(f"Fourier features: {len(freqs)} frequencies, "
              f"{freqs[0]:.1f}-{freqs[-1]:.1f} Hz, input dim {layers[0]}")
    else:
        # Plain normalised time in [-1, 1]. Derivatives are still taken with
        # respect to real time, so the physics is unchanged.
        mid = 0.5 * (t_lo + t_hi)
        half = max(0.5 * (t_hi - t_lo), 1e-12)

        def feature_transform(t):
            return (t - mid) / half

    net = dde.nn.FNN(
        layers,
        cfg["network"]["activation"],
        cfg["network"]["initializer"],
    )
    net.apply_feature_transform(feature_transform)

    # Warn if the record contains more oscillations than a plain network can
    # reasonably represent.
    if not use_ff:
        cycles = cfg["physics"]["force_frequency"] * (t_hi - t_lo)
        if cycles > 30:
            print(f"  WARNING: {cycles:.0f} oscillations in the training "
                  f"window. Plain networks struggle beyond about 30.")
            print("  Either reduce data.t_max in config.yaml, or set "
                  "network.fourier_features.enabled: true")

    model = dde.Model(problem, net)
    model.compile(
        "adam",
        lr=float(cfg["training"]["learning_rate"]),
        loss_weights=list(cfg["training"]["loss_weights"]),
        external_trainable_variables=[k_raw, c_raw],
    )

    var_file = out_dir / f"convergence_{tag}.dat"
    tracker = dde.callbacks.VariableValue(
        [k_raw, c_raw], period=100, filename=str(var_file), precision=8
    )

    print(f"Training {iterations} iterations, tag '{tag}' ...")
    t0 = time.time()
    model.train(iterations=iterations, callbacks=[tracker], display_every=1000)
    runtime = time.time() - t0
    print(f"Training finished in {runtime / 60:.1f} min")

    # --- identified parameters ----------------------------------------
    k_val = physics.read_parameter(k_raw, positive=positive)
    c_val = physics.read_parameter(c_raw, positive=positive)
    K, C = physics.to_physical(k_val, c_val, cfg)
    mass = cfg["physics"]["mass"]

    result = {
        "tag": tag,
        "identified": {"K_N_per_m": K, "C_N_s_per_m": C},
        "fixed_inputs": {
            "mass_kg": mass,
            "force_amplitude_N": cfg["physics"]["force_amplitude"],
            "force_frequency_hz": cfg["physics"]["force_frequency"],
        },
        "derived": physics.derived(K, C, mass),
        "scaled": {"k_hat": k_val, "c_hat": c_val,
                   "m_hat": physics.scaled_mass(cfg)},
        "run": {
            "iterations": iterations,
            "runtime_seconds": runtime,
            "seed": seed,
            "n_samples": data.n,
            "t_range": [t_lo, t_hi],
        },
    }

    # --- prediction over the measured times ---------------------------
    u_hat_pred = model.predict(data.t).reshape(-1)
    u_pred = u_hat_pred * cfg["scaling"]["displacement"]
    u_meas = data.u.reshape(-1)
    error = u_pred - u_meas

    rmse = float(np.sqrt(np.mean(error ** 2)))
    denom = float(np.sqrt(np.mean(u_meas ** 2)))
    result["fit"] = {
        "rmse_m": rmse,
        "relative_rmse": rmse / denom if denom > 0 else float("nan"),
        "max_abs_error_m": float(np.max(np.abs(error))),
        "r_squared": float(
            1.0 - np.sum(error ** 2) / np.sum((u_meas - u_meas.mean()) ** 2)
        ),
        "note": "computed over ALL samples, training and held out",
    }

    # --- honest generalisation figure ---------------------------------
    if test_mask.sum() > 0:
        e_tr = error[train_mask]
        e_te = error[test_mask]
        d_te = float(np.sqrt(np.mean(u_meas[test_mask] ** 2)))
        rmse_te = float(np.sqrt(np.mean(e_te ** 2)))
        result["validation"] = {
            "n_train": int(train_mask.sum()),
            "n_holdout": int(test_mask.sum()),
            "rmse_train_m": float(np.sqrt(np.mean(e_tr ** 2))),
            "rmse_holdout_m": rmse_te,
            "relative_rmse_holdout": rmse_te / d_te if d_te > 0 else float("nan"),
            "r_squared_holdout": float(
                1.0 - np.sum(e_te ** 2)
                / np.sum((u_meas[test_mask] - u_meas[test_mask].mean()) ** 2)
            ),
            "note": "held-out points were never shown to the network",
        }
    result["run"]["enforce_positive"] = positive
    result["run"]["holdout_every"] = holdout_every

    import pandas as pd

    pd.DataFrame({
        "Time (s)": data.t.ravel(),
        "u_measured (m)": u_meas,
        "u_predicted (m)": u_pred,
        "error (m)": error,
    }).to_csv(out_dir / f"prediction_{tag}.csv", index=False)

    _tidy_convergence(var_file, out_dir / f"convergence_{tag}.csv", cfg,
                      positive_log=positive)

    with open(out_dir / f"identified_{tag}.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nIdentified  K = {K:.6e} N/m")
    print(f"            C = {C:.6e} N.s/m")
    print(f"Derived   f_n = {result['derived']['f_n_hz']:.4f} Hz")
    print(f"         zeta = {result['derived']['zeta'] * 100:.4f} %")
    print(f"Fit      RMSE = {rmse:.4e} m "
          f"({result['fit']['relative_rmse'] * 100:.3f} % of RMS, all points)")
    if "validation" in result:
        v = result["validation"]
        print(f"HELD OUT RMSE = {v['rmse_holdout_m']:.4e} m "
              f"({v['relative_rmse_holdout'] * 100:.3f} %), "
              f"R2 = {v['r_squared_holdout']:.6f}  "
              f"[{v['n_holdout']} unseen points]")
    print(f"\nWrote {out_dir}/identified_{tag}.json")
    return result


def _tidy_convergence(dat_path, csv_path, cfg, positive_log=True):
    """Convert DeepXDE's variable log into a CSV in physical units."""
    if not Path(dat_path).exists():
        return
    rows = []
    with open(dat_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                step, rest = line.split(" ", 1)
                vals = json.loads(rest.replace("'", '"'))
                if positive_log:
                    vals = [math.log1p(math.exp(v)) if v < 30 else v
                            for v in vals]
                K, C = physics.to_physical(vals[0], vals[1], cfg)
                rows.append({"iteration": int(step),
                             "k_hat": vals[0], "c_hat": vals[1],
                             "K_N_per_m": K, "C_N_s_per_m": C})
            except (ValueError, json.JSONDecodeError, IndexError):
                continue
    if rows:
        import pandas as pd
        pd.DataFrame(rows).to_csv(csv_path, index=False)
