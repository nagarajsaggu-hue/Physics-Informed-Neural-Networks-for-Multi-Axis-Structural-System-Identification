"""Tests for dataset loading."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sysid import config, dataio                          


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


def test_loads(cfg):
    d = dataio.load(cfg)
    assert d.n > 100
    assert d.t.shape == (d.n, 1)
    assert d.u.shape == (d.n, 1)


def test_no_missing_values(cfg):
    d = dataio.load(cfg)
    for name, arr in [("t", d.t), ("u", d.u), ("f", d.f)]:
        assert not np.isnan(arr).any(), f"{name} contains NaN"


def test_uniform_time_step(cfg):
    d = dataio.load(cfg)
    steps = np.diff(d.t.ravel())
    assert np.ptp(steps) < 1e-6 * d.dt


def test_scaling_makes_order_one(cfg):
    """u_hat should be order 1 - that is the point of scaling."""
    d = dataio.load(cfg)
    peak = np.abs(d.u_hat).max()
    assert 0.05 < peak < 50, (
        f"scaled displacement peaks at {peak:.3f}; adjust "
        "scaling.displacement in config.yaml"
    )


def test_metadata_present(cfg):
    """The mass must be resolved before results are trustworthy."""
    d = dataio.load(cfg)
    assert d.meta, "metadata JSON missing - copy it from the ANSYS machine"
    status = d.meta.get("mass_status", "")
    if "UNRESOLVED" in status:
        pytest.skip("mass unresolved; results are pipeline tests only")


def test_displacement_actually_varies(cfg):
    d = dataio.load(cfg)
    assert np.ptp(d.u) > 1e-12, "displacement is flat - wrong node?"
