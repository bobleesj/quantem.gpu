"""Aberrations carry their unit in the key at the public boundary: nm for lengths, degrees for azimuths."""

import math

import numpy as np
import pytest

from quantem.gpu.ssb.units import (
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


def test_search_ranges_reach_the_engines_in_angstrom_under_angstrom_names():
    """The engines' range keys name their unit: C10_A, C12_A. They were C10_nm and C12_nm while holding Angstrom."""
    engine = search_ranges_to_engine({"C10_nm": (-20.0, 20.0), "C12_nm": 4.0, "phi12_deg": [-90.0, 90.0]})
    assert engine == {"C10_A": (-200.0, 200.0), "C12_A": 40.0, "phi12_deg": (-90.0, 90.0)}
    with pytest.raises(ValueError, match="C10_nm, C12_nm and phi12_deg"):
        search_ranges_to_engine({"C10": (-20.0, 20.0)})
    assert refine_lock_to_engine(["C12_nm", "phi12_deg"]) == ["C12", "phi12"]
    with pytest.raises(ValueError, match="refine_lock"):
        refine_lock_to_engine(["C12"])
