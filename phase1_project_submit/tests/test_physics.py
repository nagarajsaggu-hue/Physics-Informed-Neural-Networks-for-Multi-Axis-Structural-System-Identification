"""Tests for the scaling and the equation of motion.

These catch the errors that are otherwise invisible: a scaling factor applied
the wrong way round produces a PINN that simply will not converge, with no
message explaining why.

    python -m pytest tests/ -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sysid import config, dataio, physics                    


@pytest.fixture(scope="module")
def cfg():
    """Full record, regardless of the training window in config.yaml.

    data.t_max exists to make training tractable, not to describe the dataset.
    These tests check the data and the physics, so they always use everything.
    """
    c = config.load_config()
    c["data"]["t_min"] = None
    c["data"]["t_max"] = None
    return c


def test_scaling_round_trip(cfg):
    """Physical -> scaled -> physical must return the original value."""
    K, C = 8.2e8, 2.8e4
    k_hat, c_hat = physics.to_scaled(K, C, cfg)
    K_back, C_back = physics.to_physical(k_hat, c_hat, cfg)
    assert K_back == pytest.approx(K, rel=1e-12)
    assert C_back == pytest.approx(C, rel=1e-12)


def test_scaled_mass_consistent(cfg):
    """m_hat must equal M * U0 / F0."""
    expected = (cfg["physics"]["mass"] * cfg["scaling"]["displacement"]
                / cfg["scaling"]["force"])
    assert physics.scaled_mass(cfg) == pytest.approx(expected, rel=1e-12)


def test_derived_quantities():
    """f_n and zeta from known K, C, M."""
    K, C, M = 8.206e8, 2.833e4, 56767.0
    d = physics.derived(K, C, M)
    assert d["f_n_hz"] == pytest.approx(np.sqrt(K / M) / (2 * np.pi), rel=1e-9)
    assert d["zeta"] == pytest.approx(C / (2 * np.sqrt(K * M)), rel=1e-9)
    assert 0 < d["zeta"] < 0.1, "damping ratio should be small and positive"


def test_derived_scale_invariance():
    """f_n and zeta are unchanged if K, C and M are all scaled together.

    This is why they are the most trustworthy outputs: an error in the assumed
    mass largely cancels out of both.
    """
    a = physics.derived(8.2e8, 2.8e4, 56767.0)
    s = 1.4
    b = physics.derived(8.2e8 * s, 2.8e4 * s, 56767.0 * s)
    assert a["f_n_hz"] == pytest.approx(b["f_n_hz"], rel=1e-9)
    assert a["zeta"] == pytest.approx(b["zeta"], rel=1e-9)


def test_force_matches_csv(cfg):
    """The analytic force must reproduce the exported ANSYS force column."""
    data = dataio.load(cfg)
    f_model = physics.force_physical(data.t.ravel(), cfg)
    f_file = data.f.ravel()
    rel = np.max(np.abs(f_model - f_file)) / cfg["physics"]["force_amplitude"]
    assert rel < 0.01, (
        f"analytic force differs from the CSV by {rel:.4f} - check "
        "force_amplitude and force_frequency in config.yaml"
    )


def test_least_squares_reference_sane(cfg):
    """The classical fit must give positive stiffness and positive damping."""
    data = dataio.load(cfg)
    ref = physics.least_squares_reference(data, cfg)
    assert ref["K"] > 0, "negative stiffness is unphysical"
    assert ref["C"] > 0, (
        "negative damping is unphysical - usually caused by using the CSV's "
        "velocity/acceleration columns instead of derivatives of displacement"
    )
    assert ref["r_squared"] > 0.95, (
        f"the 1-DOF model explains only R^2 = {ref['r_squared']:.4f} of the "
        "data; a single degree of freedom may not be enough"
    )


def test_residual_small_at_reference(cfg):
    """The reference K and C must nearly satisfy the equation of motion.

    Evaluated numerically, without the network, so this tests the physics
    rather than the training.
    """
    data = dataio.load(cfg)
    ref = physics.least_squares_reference(data, cfg)

    u = data.u.ravel()
    f = data.f.ravel()
    n = len(u)
    freqs = np.fft.rfftfreq(n, data.dt)
    jw = 2j * np.pi * freqs
    spec = np.fft.rfft(u)
    v = np.fft.irfft(spec * jw, n=n)
    a = np.fft.irfft(spec * jw ** 2, n=n)

    edge = max(10, n // 20)
    sl = slice(edge, -edge)
    resid = (cfg["physics"]["mass"] * a[sl] + ref["C"] * v[sl]
             + ref["K"] * u[sl] - f[sl])
    rel = np.sqrt(np.mean(resid ** 2)) / np.sqrt(np.mean(f[sl] ** 2))
    print(f"\n  residual / applied force = {rel:.4f}")

    # With the modal generalized mass (56767 kg) this sits near 0.10. The
    # transient data's own preferred mass is about 86000 kg, and the gap
    # between the two is why the residual does not go lower. That is a
    # modelling question, not a code fault, so the threshold here is set to
    # catch genuine breakage rather than to enforce a perfect 1-DOF fit.
    assert rel < 0.20, (
        f"residual is {rel:.4f} of the applied force. Above 0.2 means the "
        "single-DOF model does not describe this data at all - check the "
        "mass, the force definition, and the observation column."
    )


def test_inverse_softplus_round_trip():
    """inverse_softplus must invert softplus exactly.

    If this is wrong, k_hat_init in config.yaml no longer means what it says
    and the run starts from a different place than intended.
    """
    import math
    for y in (1e-4, 0.01, 1.0, 7.21, 25.0):
        x = physics.inverse_softplus(y)
        back = math.log1p(math.exp(x)) if x < 30 else x
        assert back == pytest.approx(y, rel=1e-9)


def test_inverse_softplus_rejects_nonpositive():
    with pytest.raises(ValueError):
        physics.inverse_softplus(0.0)
    with pytest.raises(ValueError):
        physics.inverse_softplus(-1.0)
