"""Aberrations carry their unit in the key at the public boundary: nm for lengths, degrees for azimuths."""

import math

import numpy as np
import pytest

from quantem.gpu.direct_ptycho.units import (
    aberrations_from_engine,
    aberrations_to_engine,
    higher_order_arrays,
    refine_lock_to_engine,
    search_ranges_to_engine,
    validate_aberrations,
)

PUBLIC = {"C10_nm": -8.0, "C12_nm": 4.0, "phi12_deg": 30.0}

# ---


def test_public_keys_convert_once_to_engine_angstrom_and_radians():
    engine = aberrations_to_engine(PUBLIC)
    assert engine == {"C10": -80.0, "C12": 40.0, "phi12": math.radians(30.0)}
    assert aberrations_from_engine(engine) == pytest.approx(PUBLIC)
    assert validate_aberrations(None) == {"C10_nm": 0.0, "C12_nm": 0.0, "phi12_deg": 0.0}


def test_keys_without_a_unit_are_refused_not_read_as_nm_or_radians():
    """The earlier C10 / C12 / phi12 mapping held radians under phi12; it is named in the error, never converted."""
    with pytest.raises(ValueError, match=r"unknown \['C10', 'C12', 'phi12'\]"):
        validate_aberrations({"C10": -8.0, "C12": 4.0, "phi12": 0.5})
    # a higher order is a preview input: a fit or a fixed reconstruction would drop it without a message
    with pytest.raises(ValueError, match=r"unknown \['C21_nm'\]"):
        validate_aberrations(PUBLIC | {"C21_nm": 40.0})
    assert validate_aberrations(PUBLIC | {"C21_nm": 40.0}, higher_order=True)["C21_nm"] == 40.0


def test_higher_orders_are_named_keys_packed_in_krivanek_order():
    assert higher_order_arrays(PUBLIC) is None
    magnitudes, angles = higher_order_arrays(
        PUBLIC | {"C21_nm": 40.0, "phi21_deg": 90.0, "C30_nm": 1500.0, "C56_nm": 4.0e5, "phi56_deg": -45.0}
    )
    expected_magnitudes = np.zeros(14, np.float32)
    expected_magnitudes[[0, 1, 2, 4, 13]] = [-8.0, 4.0, 40.0, 1500.0, 4.0e5]
    expected_angles = np.zeros(14, np.float32)
    expected_angles[[1, 2, 13]] = np.radians([30.0, 90.0, -45.0])
    np.testing.assert_array_equal(magnitudes, expected_magnitudes)
    np.testing.assert_array_equal(angles, expected_angles)


class _DeviceArray(np.ndarray):
    """A host array with CuPy's ``get(out=...)``: what a CUDA backend hands the session for a preview."""

    def get(self, out):
        out[...] = self
        return out


@pytest.mark.parametrize("compute_loss", [False, True])
def test_higher_orders_reach_cuda_and_mps_in_angstrom_from_one_public_input(monkeypatch, compute_loss):
    """The session converts nm to Angstrom once, before the backend: for one public mapping CUDA and MPS are handed
    the same 14 magnitudes in Angstrom and azimuths in radians at the coherent backend boundary.
    """
    from types import SimpleNamespace

    from quantem.gpu.direct_ptycho import workflow

    received = {}

    class Recorder:
        def __init__(self, name):
            self.name = name

        def preview_upsampled(self, aberrations, *, compute_loss, higher_order_magnitudes=None, higher_order_angles=None, **kwargs):
            assert "phase_estimator" not in kwargs  # SSB has one estimator, the phase of the mean wave
            received[self.name] = (aberrations, higher_order_magnitudes, higher_order_angles)
            return np.zeros((2, 2), np.float32).view(_DeviceArray), 0.5 if compute_loss else None

    monkeypatch.setattr(workflow, "resolve", lambda _backend: "mps")
    # A CUDA session copies its device phase into a pinned host buffer it owns (phase.get(out=host)); the double's
    # pinned memory is a bytearray and its device array a NumPy array with CuPy's get(out=...).
    monkeypatch.setattr(workflow, "cp", SimpleNamespace(asnumpy=np.asarray, cuda=SimpleNamespace(alloc_pinned_memory=bytearray)))

    def session(name):
        ssb = workflow.SSB(np.zeros((2, 2, 3, 3), np.uint16), voltage_kV=300.0, semiangle_mrad=30.0,
                           scan_sampling_A=(0.2, 0.2), detector_sampling_mrad=(1.0, 1.0))
        ssb.backend = name          # the branch preview takes; the arrays recorded below are the backend boundary
        monkeypatch.setattr(ssb, "_backend", lambda recorder=Recorder(name): recorder)
        return ssb

    public = PUBLIC | {"C21_nm": 4.0, "phi21_deg": 30.0, "C30_nm": 1000.0}
    for name in ("cuda", "mps"):
        session(name).preview(public, compute_loss=compute_loss)
    (cuda_coefs, cuda_magnitudes, cuda_angles), (mps_coefs, mps_magnitudes, mps_angles) = received["cuda"], received["mps"]
    assert cuda_coefs == mps_coefs == {"C10": -80.0, "C12": 40.0, "phi12": math.radians(30.0)}
    np.testing.assert_array_equal(mps_magnitudes, cuda_magnitudes)
    np.testing.assert_array_equal(mps_angles, cuda_angles)
    assert mps_magnitudes.dtype == np.float32 and mps_magnitudes.shape == (14,)
    # C10 -8 nm, C12 4 nm, C21 4 nm, C30 1000 nm as Angstrom in slots 0, 1, 2 and 4
    np.testing.assert_array_equal(mps_magnitudes[[0, 1, 2, 4]], np.float32([-80.0, 40.0, 40.0, 10000.0]))
    np.testing.assert_array_equal(mps_angles[[1, 2]], np.float32([math.radians(30.0)] * 2))
    # Zero higher-order magnitudes reuse the two-term kernels, including retained angles.
    session("mps").preview(PUBLIC | {"C21_nm": 0.0}, compute_loss=compute_loss)
    coefs, magnitudes, angles = received["mps"]
    assert magnitudes is angles is None
    assert coefs == {"C10": -80., "C12": 40., "phi12": float(np.float32(math.radians(30.)))}


def test_search_ranges_reach_the_engines_in_angstrom_under_angstrom_names():
    """The engines' range keys name their unit: C10_A, C12_A. They were C10_nm and C12_nm while holding Angstrom."""
    engine = search_ranges_to_engine({"C10_nm": (-20.0, 20.0), "C12_nm": 4.0, "phi12_deg": [-90.0, 90.0]})
    assert engine == {"C10_A": (-200.0, 200.0), "C12_A": 40.0, "phi12_deg": (-90.0, 90.0)}
    with pytest.raises(ValueError, match="C10_nm, C12_nm and phi12_deg"):
        search_ranges_to_engine({"C10": (-20.0, 20.0)})
    assert refine_lock_to_engine(["C12_nm", "phi12_deg"]) == ["C12", "phi12"]
    with pytest.raises(ValueError, match="refine_lock"):
        refine_lock_to_engine(["C12"])
