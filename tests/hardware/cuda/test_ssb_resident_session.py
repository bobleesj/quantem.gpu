"""An SSB session on an acquisition already resident on the GPU: a region of its scan, its saved fit and its previews.

At a microscope the data are streamed and there is no file: SSB takes the dataset ``io.load`` returns, decodes the
bright-field columns of ``scan_region`` only, finds the disk on the whole scan's mean pattern (``mean_pattern`` when the
caller has it), and files a saved fit under the data's identity. The tilt fit polishes with Newton steps
(``refinement="newton"``), can skip the standard comparison (``compare_standard=False``) and, with no trials, refines
the previous scan's fit. Synthetic Poisson counts with a bright-field disk; skips without a CUDA device.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.slow

cp = pytest.importorskip("cupy")

SETTINGS = dict(backend="cuda", voltage_kV=300.0, semiangle_mrad=30.0, scan_sampling_A=(0.3, 0.3), detector_sampling_mrad=(3.0, 3.0))


def _require_cuda():
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("No CUDA device")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("No CUDA runtime")


@pytest.fixture
def acquisition(tmp_path):
    """A 128 x 128 scan of 32 x 32 Poisson patterns with a bright-field disk, loaded encoded on the GPU."""
    _require_cuda()
    import h5py
    import hdf5plugin

    from quantem.gpu.io import load

    rng = np.random.default_rng(7)
    counts = rng.poisson(40.0, size=(128, 128, 32, 32)).astype(np.uint16)
    rows, cols = np.meshgrid(np.arange(32) - 15.5, np.arange(32) - 15.5, indexing="ij")
    counts[:, :, np.hypot(rows, cols) < 10] += 200
    source = tmp_path / "synthetic_master.h5"
    with h5py.File(source, "w") as handle:
        handle.create_dataset("entry/data/data", data=counts.reshape(-1, 32, 32), chunks=(1, 32, 32),
                              **hdf5plugin.Bitshuffle(cname="lz4"))
    with load(str(source), scan_shape=(128, 128), verbose=False) as data:
        yield data


def test_a_region_session_holds_the_region_of_the_whole_scans_bright_field(acquisition):
    """scan_region decodes those positions only, with the disk of the whole scan: the same pixels, centre and radius."""
    from quantem.gpu import SSB, detector

    with SSB(acquisition, **SETTINGS) as whole:
        crop, center, radius = cp.asnumpy(whole._data), whole.bright_field_center_px, whole.bright_field_radius_px
    mean = detector.mean(acquisition)
    with SSB(acquisition, scan_region=(16, 80, 32, 96), mean_pattern=mean, **SETTINGS) as region:
        np.testing.assert_array_equal(cp.asnumpy(region._data), crop[16:80, 32:96])
        assert region.bright_field_center_px == center and region.bright_field_radius_px == radius
        assert region.scan_region == (16, 80, 32, 96)
    assert acquisition.read(scan_region=(0, 1, 0, 1)).shape[:2] == (1, 1)      # still open, still the caller's


def test_a_saved_tilt_fit_of_resident_data_is_found_by_the_data(acquisition, tmp_path):
    """No file identifies a resident acquisition: a second session on the same data reuses the fit by its identity."""
    from quantem.gpu import SSB

    settings = dict(tilt=True, num_trials=8, refinement="newton", compare_standard=False, check_rotation=False,
                    save_dir=tmp_path / "fit", verbose=False)
    with SSB(acquisition, **SETTINGS) as first:
        fitted = first.find_aberrations(**settings)
        assert first.source_path is None and first.source_kind == "resident"
    with SSB(acquisition, **SETTINGS) as second:
        again = second.find_aberrations(**settings)
    assert not fitted.reused and again.reused
    assert again.aberrations == fitted.aberrations and again.sample_tilt_mrad == fitted.sample_tilt_mrad
    assert fitted.tilt_fit_gain is None and fitted.refinement == "newton"
    with SSB(acquisition, scan_region=(0, 64, 0, 64), **SETTINGS) as region:
        assert not region.find_aberrations(**settings).reused      # a region is other data


def test_no_trials_polishes_the_previous_fit_and_native_previews_stay_on_the_gpu(acquisition):
    """num_trials=0 starts from the session's fit and searches nothing; output='native' is the CuPy phase of the preview."""
    from quantem.gpu import SSB

    with SSB(acquisition, **SETTINGS) as ssb:
        fitted = ssb.find_aberrations(tilt=True, num_trials=8, refinement="newton", compare_standard=False,
                                      check_rotation=False, verbose=False)
        again = ssb.find_aberrations(tilt=True, num_trials=0, refinement="newton", compare_standard=False,
                                     check_rotation=False, verbose=False)
        assert again.num_trials == 0 and again.trial_records == []
        assert abs(again.aberrations["C10_nm"] - fitted.aberrations["C10_nm"]) < 0.5
        options = dict(sample_tilt_mrad=again.sample_tilt_mrad, depth_spread_nm=again.depth_spread_nm,
                       phase_estimator="phase_of_mean", compute_loss=False)
        native, _ = ssb.preview(dict(again.aberrations), output="native", **options)
        host, _ = ssb.preview(dict(again.aberrations), **options)
        assert isinstance(native, cp.ndarray)
        np.testing.assert_array_equal(cp.asnumpy(native).astype(np.float32), host)
    with SSB(acquisition, **SETTINGS) as standard, pytest.raises(ValueError, match="newton"):
        standard.find_aberrations(refinement="newton", verbose=False)



def test_native_previews_are_their_own_arrays_and_a_region_refit_starts_from_assigned_optics(acquisition, monkeypatch):
    """Each native preview survives the next one (the mean-phase path reuses an engine buffer), and a region refit with no
    trials starts from the aberrations a caller assigned to the session, not from zero."""
    from quantem.gpu import SSB

    with SSB(acquisition, **SETTINGS) as ssb:
        options = dict(phase_estimator="mean_phase", compute_loss=False)
        first, _ = ssb.preview({"C10_nm": 1.0, "C12_nm": 0.0, "phi12_deg": 0.0}, output="native", **options)
        kept = cp.asnumpy(first)
        ssb.preview({"C10_nm": 30.0, "C12_nm": 0.0, "phi12_deg": 0.0}, output="native", **options)
        np.testing.assert_array_equal(cp.asnumpy(first), kept)
    starts = []
    fit_tilt = SSB._fit_tilt

    def recorded(self, *args, **kwargs):
        starts.append(dict(self.aberrations))
        return fit_tilt(self, *args, **kwargs)
    monkeypatch.setattr(SSB, "_fit_tilt", recorded)
    with SSB(acquisition, **SETTINGS) as ssb:
        ssb.aberrations = {"C10_nm": 4.0, "C12_nm": 1.5, "phi12_deg": 20.0}
        ssb.sample_tilt_mrad, ssb.depth_spread_nm = (1.0, -2.0), 30.0
        ssb.find_aberrations(tilt=True, num_trials=0, refinement="newton", compare_standard=False, check_rotation=False,
                             scan_region=(0, 128, 0, 128), verbose=False)
    assert starts == [{"C10_nm": 4.0, "C12_nm": 1.5, "phi12_deg": 20.0}]

_OPENED: list[str] = []
_WATCH = {"folder": None}


def _note_open(event, arguments):
    """Record every file the interpreter opens under the watched folder (audit hooks cannot be removed, so it idles)."""
    if event == "open" and _WATCH["folder"] is not None and str(arguments[0]).startswith(_WATCH["folder"]):
        _OPENED.append(str(arguments[0]))


def test_calibration_from_a_resident_acquisition_opens_no_file(tmp_path, monkeypatch):
    """One read: with a real acquisition resident on the GPU, the mean pattern, virtual images, DPC, the disk fit, the SSB
    session, the saved tilt fit, the phase image and the column sign open nothing in the acquisition's folder, through
    Python or through h5py. The session decodes the same bright-field crop as SSB.open(path)."""
    source = Path(os.environ.get("QUANTEM_SSB_ARINA_MASTER", ""))
    if not source.is_file():
        pytest.skip("set QUANTEM_SSB_ARINA_MASTER to a real Arina master file")
    _require_cuda()
    import h5py

    from quantem.gpu import SSB, detector, dpc
    from quantem.gpu.io import load
    from quantem.gpu.ssb.results import column_sign

    settings = dict(SETTINGS, scan_sampling_A=(0.264, 0.264), detector_sampling_mrad=(0.5554, 0.5554), scan_detector_rotation_deg=158.9)
    by_path = SSB.open(str(source), **settings)
    crop, center = cp.asnumpy(by_path._data), by_path.bright_field_center_px
    by_path.close()
    del by_path
    cp.get_default_memory_pool().free_all_blocks()
    opened_by_h5py = []
    original = h5py.File.__init__

    def counting(self, name, *args, **kwargs):
        opened_by_h5py.append(str(name))
        return original(self, name, *args, **kwargs)
    fit = dict(tilt=True, num_trials=8, refinement="newton", compare_standard=False, save_dir=tmp_path / "fit", verbose=False)
    with load(str(source), verbose=False) as data:
        sys.addaudithook(_note_open)
        _WATCH["folder"] = str(source.resolve().parent)
        monkeypatch.setattr(h5py.File, "__init__", counting)
        try:
            mean = detector.mean(data)
            detector.bf(data)
            detector.adf(data)
            dpc.run(data)
            detector.fit_probe(mean)
            session = SSB(data, **settings)
            fitted = session.find_aberrations(**fit)
            with SSB(data, **settings) as second:      # a new session on the same data finds the saved fit by the data's identity
                again = second.find_aberrations(**fit)
            phase, _ = session.preview(dict(fitted.aberrations), sample_tilt_mrad=fitted.sample_tilt_mrad,
                                       depth_spread_nm=fitted.depth_spread_nm, compute_loss=False)
            column_sign(phase)
            with SSB(data, scan_region=(64, 256, 128, 320), mean_pattern=mean, **settings) as region:      # its own positions only
                region_counts, region_center = cp.asnumpy(region._data), region.bright_field_center_px
        finally:
            _WATCH["folder"] = None
        assert data.read(scan_region=(0, 1, 0, 1)).shape[:2] == (1, 1)      # still open, still the caller's
    assert _OPENED == [] and opened_by_h5py == []
    assert again.reused and session.source_path is None
    np.testing.assert_array_equal(cp.asnumpy(session._data), crop)
    assert session.bright_field_center_px == center
    np.testing.assert_array_equal(region_counts, crop[64:256, 128:320])
    assert region_center == center
    session.close()
