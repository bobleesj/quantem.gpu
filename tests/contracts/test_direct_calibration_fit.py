"""A saved calibration becomes the fit a series needs: ``DirectPtycho.load_calibration``.

The record is the model (aberrations, sample tilt, depth spread, rotation). The loader computes the fit
coordinates from it, so a calibration whose aberrations were edited in ShowPtycho loads too, and takes the search
bounds from the record's ``max_order`` and ``higher_order_limit``.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from quantem.gpu.direct_ptycho.calibration import CALIBRATION_SCHEMA, capture_calibration, load_calibration
from quantem.gpu.direct_ptycho.fifth_order_fit import fifth_order_bounds
from quantem.gpu.direct_ptycho.fifth_order_parameters import Coordinates, coordinates, public_params
from quantem.gpu.direct_ptycho.results import DirectPtychoResult
from quantem.gpu.optics.physics import wavelength_A_from_kV

# Two Arina masters of one area, taken one after the other (512 x 512 scans): the fitted scan and the next one.
FIRST = Path(os.environ.get("QUANTEM_DIRECT_ARINA_MASTER", ""))
NEXT = Path(os.environ.get("QUANTEM_DIRECT_ARINA_NEXT_MASTER", ""))
STEP_A = float(os.environ.get("QUANTEM_DIRECT_SCAN_SAMPLING_A", "0.264"))
REAL = dict(voltage_kV=300.0, semiangle_mrad=30.0, scan_sampling_A=(STEP_A, STEP_A))
WAVELENGTH_A, SEMIANGLE_RAD = wavelength_A_from_kV(300.0), 0.030


def encoder():
    """The fit's own map from coordinates to named aberrations, at 300 kV and 30 mrad."""
    encode = Coordinates()
    encode.p = SimpleNamespace(wavelength=WAVELENGTH_A, semiangle_rad=SEMIANGLE_RAD)
    return encode


def test_coordinates_invert_the_named_aberrations_of_every_order():
    z = np.random.default_rng(3).uniform(-1.5, 1.5, 28)
    z[5] = abs(z[5]) + 0.3                          # a depth spread is positive
    aberrations, tilt, depth = public_params(encoder(), z, 5)
    rebuilt = coordinates(aberrations, tilt, depth, WAVELENGTH_A, SEMIANGLE_RAD)
    # the named higher orders pass through the kernels' float32 packing: 1e-6 of a radian at the aperture edge
    np.testing.assert_allclose(rebuilt, z, atol=2e-6)
    np.testing.assert_allclose(rebuilt[:6], z[:6], atol=1e-12)


def saved_record(tmp_path, **changes):
    """A calibration written by ``fit.save_calibration`` for a fit that ended on known coordinates."""
    z = np.random.default_rng(4).uniform(-1.0, 1.0, 28)
    z[5] = 0.8
    aberrations, tilt, depth = public_params(encoder(), z, 5)
    source = tmp_path / "scan.bin"
    source.write_bytes(b"disposable acquisition identity")
    fit = DirectPtychoResult(
        object_wave=np.ones((8, 8), np.complex64), backend="cuda", source_path=str(source), aberrations=aberrations,
        sample_tilt_mrad=tuple(float(value) for value in tilt), depth_spread_nm=depth, num_bright_field_pixels=17,
        voltage_kV=300.0, semiangle_mrad=30.0, scan_detector_rotation_deg=171.25, upsample=1,
        reconstruction_method="phase", reconstruction_options={
            "phase_damp": 0.3, "smoothness_inv_A": 1.0, "detector_weighting": "uniform"},
        optimizer={"max_order": 4, "higher_order_limit": 2.0, "coordinates": z.tolist(),
                   "bounds": fifth_order_bounds(2.0, max_order=4).tolist()},
    )
    session = SimpleNamespace(
        source_kind="detector", source_path=str(source), input_scan_shape=(4, 4), scan_region=None,
        source_dtype="uint16", source_bytes=128, source_detector_bin=1, voltage_kV=300.0, semiangle_mrad=30.0,
        scan_sampling_A=(0.5, 0.5), detector_sampling_mrad=(0.5, 0.5), scan_detector_rotation_flipped=False,
        bright_field_radius_px=2.0, bright_field_center_px=(1.5, 1.5), bright_field_intensity_threshold=0.3,
        source_provenance={}, source_storage_path=None, calibration_path=None, source_manifest_path=None,
    )
    capture_calibration(session, fit)
    path = fit.save_calibration(tmp_path / "calibration" / "scan.json")
    if changes:
        record = json.loads(path.read_text())
        for key, value in changes.items():
            if value is KeyError:
                del record[key]
            else:
                record[key] = value
        path.write_text(json.dumps(record))
    return fit, path


def test_a_loaded_calibration_is_the_fit_a_series_reads(tmp_path):
    fit, path = saved_record(tmp_path)
    loaded = load_calibration(path)
    assert loaded.object_wave is None and loaded.saved_path == path
    assert loaded.aberrations == fit.aberrations
    assert loaded.sample_tilt_mrad == fit.sample_tilt_mrad and loaded.depth_spread_nm == fit.depth_spread_nm
    assert loaded.scan_detector_rotation_deg == 171.25 and loaded.scan_detector_rotation_flipped is False
    np.testing.assert_allclose(loaded.optimizer["coordinates"], fit.optimizer["coordinates"], atol=2e-6)
    assert loaded.optimizer["bounds"] == fit.optimizer["bounds"]     # the fit's own: order 4, +-2 rad
    assert loaded.optimizer["max_order"] == 4 and loaded.optimizer["higher_order_limit"] == 2.0
    geometry = loaded.calibration_context["session_identity"]["geometry"]
    assert geometry["bright_field_center_px"] == [1.5, 1.5] and geometry["detector_sampling_mrad"] == [0.5, 0.5]
    assert loaded._phase_settings == {"phase_damp": 0.3, "smoothness_inv_A": 1.0, "detector_weighting": "uniform"}
    assert loaded.reconstruction_method == "phase"


def test_a_calibration_edited_in_the_viewer_loads_with_its_edited_aberrations(tmp_path):
    """ShowPtycho writes the aberrations as its sliders stand, names no fit settings for a scan it did not fit, and
    has no acquisition identity for a scan held in GPU memory."""
    fit, path = saved_record(tmp_path)
    edited = {**fit.aberrations, "C10_nm": fit.aberrations["C10_nm"] + 3.0, "C50_nm": 0.0, "C52_nm": 0.0,
              "C54_nm": 0.0, "C56_nm": 0.0}
    record = json.loads(path.read_text())
    record.update(aberrations=edited, session_identity=None, reconstruction={"method": "phase_of_mean"})
    del record["max_order"], record["higher_order_limit"]
    path.write_text(json.dumps(record))
    loaded = load_calibration(path)
    assert loaded.aberrations["C10_nm"] == edited["C10_nm"]
    assert loaded.optimizer["coordinates"][0] == pytest.approx(edited["C10_nm"] / 10)
    np.testing.assert_allclose(loaded.optimizer["coordinates"][1:21], fit.optimizer["coordinates"][1:21], atol=2e-6)
    assert not np.any(loaded.optimizer["coordinates"][21:])           # the seven fifth-order coordinates, set to zero
    # no fit settings in the record: the highest order it holds, and the default limit
    assert loaded.optimizer["bounds"] == fifth_order_bounds(20.0, max_order=4).tolist()
    assert loaded.calibration_context["session_identity"]["geometry"]["voltage_kV"] == 300.0
    assert not hasattr(loaded, "_phase_settings")                    # no phase method saved: the series' own


def test_a_fit_without_tilt_holds_tilt_and_depth(tmp_path):
    _, path = saved_record(tmp_path, sample_tilt_mrad=None, depth_spread_nm=None)
    loaded = load_calibration(path)
    assert loaded.sample_tilt_mrad is None and loaded.depth_spread_nm is None
    assert loaded.optimizer["coordinates"][3:6] == [0.0, 0.0, 0.0]
    assert loaded.optimizer["bounds"][3:6] == [[0.0, 0.0]] * 3


def test_other_records_are_refused_with_what_is_expected(tmp_path):
    _, path = saved_record(tmp_path, schema="quantem.direct-calibration/8")
    with pytest.raises(ValueError, match=CALIBRATION_SCHEMA):
        load_calibration(path)
    _, path = saved_record(tmp_path, scan_region=[0, 2, 0, 4])
    with pytest.raises(ValueError, match="whole scan"):
        load_calibration(path)


def test_save_calibration_names_the_file_after_the_scan_and_needs_a_fit(tmp_path):
    from quantem.gpu import DirectPtycho

    written = []

    class Fit:
        def save_calibration(self, path):
            written.append(path)
            return path

    session = DirectPtycho.__new__(DirectPtycho)
    session._input_dataset = SimpleNamespace(metadata={"direct_calibration_destination": "/elsewhere/calibration/scan_7.json"})
    session.source_path, session._aberration_search = None, None
    with pytest.raises(ValueError, match="no fit to save"):
        session.save_calibration(tmp_path)
    session._aberration_search = Fit()
    assert session.save_calibration(tmp_path) == tmp_path / "calibration" / "scan_7.json"

    class Viewer:
        def save_calibration(self, path):
            written.append(("viewer", path))
            return path

    viewer = Viewer()
    session._viewer = lambda: viewer                # show() keeps a weak reference to its ShowPtycho
    session._input_dataset, session.source_path = None, str(tmp_path / "data" / "scan_9_master.h5")
    assert session.save_calibration(tmp_path) == tmp_path / "calibration" / "scan_9_master.json"
    assert written == [tmp_path / "calibration" / "scan_7.json",
                       ("viewer", tmp_path / "calibration" / "scan_9_master.json")]


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    """A real scan fitted through fifth order, its calibration saved and loaded, and the next scan of the area."""
    if not FIRST.is_file() or not NEXT.is_file():
        pytest.skip("set QUANTEM_DIRECT_ARINA_MASTER and QUANTEM_DIRECT_ARINA_NEXT_MASTER to two Arina masters "
                    "of one area, taken one after the other")
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("No CUDA device")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("No CUDA runtime")
    from quantem.gpu import DirectPtycho, io

    scan, following = io.load(FIRST, verbose=False), io.load(NEXT, verbose=False)
    ptycho = DirectPtycho(scan, **REAL)
    fit = ptycho.aberrations(show=False, verbose=False)
    path = ptycho.save_calibration(tmp_path_factory.mktemp("results"))
    yield ptycho, fit, DirectPtycho.load_calibration(path), following
    ptycho.close()
    following.close()
    scan.close()


def test_the_loaded_fit_has_the_fits_coordinates_and_bounds(fitted):
    _, fit, loaded, _ = fitted
    np.testing.assert_allclose(loaded.optimizer["coordinates"], fit.optimizer["coordinates"], atol=1e-5)
    assert loaded.optimizer["bounds"] == fit.optimizer["bounds"]
    assert loaded.aberrations == pytest.approx(fit.aberrations, rel=1e-6, abs=1e-9)
    assert loaded.scan_detector_rotation_deg == fit.scan_detector_rotation_deg


# What "the same" means for the next scan's phase (rad; its standard deviation is about 0.04 rad) and its
# defocus and astigmatism (nm). refit="none" fits nothing, so it is exact. A refit starts from coordinates that
# passed through the calibration's named aberrations (1e-6 rad at the aperture edge), and a full refit moves 28
# of them, the fifth orders barely constrained by one scan.
SAME = {"none": (0.0, 0.0), "defocus": (1e-5, 1e-3), "full": (2e-4, 0.02)}


@pytest.mark.parametrize("refit", ["none", "defocus", "full"])
def test_a_series_from_the_loaded_fit_is_the_series_from_the_fit(fitted, refit):
    ptycho, fit, loaded, following = fitted
    phase_tolerance, optics_tolerance = SAME[refit]
    phases, optics = [], []
    for reference in (fit, loaded):
        series = ptycho.series(fit=reference, refit=refit, method="phase", phase_damp=0.3, show=False, verbose=False)
        series.add(following)
        phases.append(series.images["phase"][-1].get())
        optics.append(dict(series.aberrations[-1]))
    assert float(np.abs(phases[0] - phases[1]).max()) <= phase_tolerance
    for name in ("C10_nm", "C12_nm"):
        assert abs(optics[0][name] - optics[1][name]) <= optics_tolerance
    ours, theirs = (coordinates(values, None, None, WAVELENGTH_A, SEMIANGLE_RAD) for values in optics)
    # every aberration, as radians of phase at the aperture edge
    assert float(np.abs(ours - theirs).max()) <= (0.0 if refit == "none" else 0.02)


def test_run_reconstructs_with_the_loaded_fit(fitted):
    ptycho, fit, loaded, _ = fitted
    ours = ptycho.run(loaded, method="phase", phase_damp=0.3, upsample=1, compute_loss=False, save_calibration=False,
                       verbose=False)
    theirs = ptycho.run(fit, method="phase", phase_damp=0.3, upsample=1, compute_loss=False, save_calibration=False,
                       verbose=False)
    assert float(abs(ours.phase - theirs.phase).max()) <= 1e-5
