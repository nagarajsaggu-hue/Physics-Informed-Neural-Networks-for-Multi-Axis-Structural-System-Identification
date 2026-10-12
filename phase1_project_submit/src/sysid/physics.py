"""The equation of motion, in physical and scaled form.

PHYSICAL EQUATION
-----------------
        M u'' + C u' + K u = F(t)

with M known (the generalized mass from modal analysis) and K, C identified.

Damping is a single scalar C. It is NOT decomposed into Rayleigh alpha and
beta: for a single degree of freedom, C = alpha*K + beta*M is one equation in
two unknowns, so alpha and beta are not separately identifiable. Any pair on
that line fits the data identically. They become identifiable only with two or
more distinct modes.

SCALED EQUATION
---------------
With  u = U0 * u_hat  and  F = F0 * F_hat,  dividing through by F0 gives

        m_hat u_hat'' + c_hat u_hat' + k_hat u_hat = F_hat(t)

        m_hat = M * U0 / F0
        c_hat = C * U0 / F0
        k_hat = K * U0 / F0

Time is left in seconds, so derivatives are with respect to real time.
"""

import math


def scaled_mass(cfg) -> float:
    """m_hat: the known mass in scaled units."""
    return (cfg["physics"]["mass"] * cfg["scaling"]["displacement"]
            / cfg["scaling"]["force"])


def to_scaled(k_physical, c_physical, cfg):
    """Physical (K, C) -> scaled (k_hat, c_hat)."""
    s = cfg["scaling"]["displacement"] / cfg["scaling"]["force"]
    return k_physical * s, c_physical * s


def to_physical(k_hat, c_hat, cfg):
    """Scaled (k_hat, c_hat) -> physical (K, C) in N/m and N.s/m."""
    s = cfg["scaling"]["force"] / cfg["scaling"]["displacement"]
    return float(k_hat) * s, float(c_hat) * s


def derived(k_physical, c_physical, mass):
    """Natural frequency and damping ratio from identified K and C.

    These are arithmetic, not identified quantities. They are also the most
    robust outputs: both depend on K/M and C/sqrt(K*M), so an error in the
    assumed mass largely cancels.
    """
    if k_physical <= 0 or mass <= 0:
        return {"omega_n_rad_s": float("nan"),
                "f_n_hz": float("nan"),
                "zeta": float("nan")}
    omega = math.sqrt(k_physical / mass)
    return {
        "omega_n_rad_s": omega,
        "f_n_hz": omega / (2.0 * math.pi),
        "zeta": c_physical / (2.0 * math.sqrt(k_physical * mass)),
    }


def force_physical(t, cfg):
    """F(t) in newtons. Numpy version, for reference and plotting."""
    import numpy as np

    a = cfg["physics"]["force_amplitude"]
    f = cfg["physics"]["force_frequency"]
    return a * np.sin(2.0 * np.pi * f * t)


def inverse_softplus(y):
    """x such that softplus(x) = y. Used to initialise raw parameters.

    softplus(x) = log(1 + exp(x)), so x = log(exp(y) - 1). Computed in a form
    that does not overflow for large y.
    """
    if y <= 0:
        raise ValueError(f"softplus output must be positive, got {y}")
    if y > 30:
        return y                      # softplus(x) ~ x for large x
    return math.log(math.expm1(y))


def make_residual(cfg, k_raw, c_raw, positive=True):
    """Build the DeepXDE residual function for the scaled equation.

    k_raw and c_raw are dde.Variable objects trained alongside the network
    weights.

    POSITIVITY
    ----------
    With positive=True the parameters enter as softplus(raw), so stiffness and
    damping cannot go negative no matter what the optimiser does. This matters:
    without it, C is free to wander into negative values during training, which
    is physically meaningless - energy would be created rather than dissipated.
    Runs on this dataset have been seen to dip to C = -8e4 before recovering.
    Making negative values unreachable removes the possibility entirely.
    """
    import torch
    import deepxde as dde

    m_hat = scaled_mass(cfg)
    omega = 2.0 * math.pi * cfg["physics"]["force_frequency"]
    f_scale = cfg["physics"]["force_amplitude"] / cfg["scaling"]["force"]

    if positive:
        def k_of(v):
            return torch.nn.functional.softplus(v)
        c_of = k_of
    else:
        def k_of(v):
            return v
        c_of = k_of

    def residual(t, u):
        du_dt = dde.grad.jacobian(u, t)
        d2u_dt2 = dde.grad.hessian(u, t)
        f_hat = f_scale * torch.sin(omega * t)
        return (m_hat * d2u_dt2
                + c_of(c_raw) * du_dt
                + k_of(k_raw) * u
                - f_hat)

    return residual


def read_parameter(raw, positive=True):
    """Read a trained dde.Variable back as a scaled parameter value."""
    import torch

    v = raw.detach()
    if positive:
        v = torch.nn.functional.softplus(v)
    return float(v.cpu().numpy())


def least_squares_reference(data, cfg):
    """Classical fit of M u'' + C u' + K u = F with M fixed.

    This is the reference the PINN must reproduce. Derivatives are computed
    spectrally from the displacement (exact for a band-limited signal) rather
    than taken from the CSV's velocity and acceleration columns, which are not
    consistent with the displacement.

    Returns K, C, R^2 and their standard errors.
    """
    import numpy as np

    t = data.t.ravel()
    u = data.u.ravel()
    f = data.f.ravel()
    n = len(t)
    dt = data.dt

    freqs = np.fft.rfftfreq(n, dt)
    jw = 2j * np.pi * freqs
    spec = np.fft.rfft(u)
    v = np.fft.irfft(spec * jw, n=n)
    a = np.fft.irfft(spec * jw ** 2, n=n)

    # Trim the ends: spectral derivatives assume periodicity, so the first and
    # last few percent of the record are contaminated by the wrap-around.
    edge = max(10, n // 20)
    sl = slice(edge, -edge)

    mass = cfg["physics"]["mass"]
    A = np.column_stack([u[sl], v[sl]])
    b = f[sl] - mass * a[sl]
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    k, c = float(sol[0]), float(sol[1])

    resid = b - A @ sol
    ss_res = float(resid @ resid)
    ss_tot = float(np.sum((b - b.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    dof = max(1, len(b) - 2)
    cov = (ss_res / dof) * np.linalg.inv(A.T @ A)
    se = np.sqrt(np.diag(cov))

    out = {"K": k, "C": c, "r_squared": r2,
           "K_stderr": float(se[0]), "C_stderr": float(se[1]),
           "mass_used": mass}
    out.update(derived(k, c, mass))
    return out
