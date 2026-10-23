"""MPS SSB after a rotation change: the same answer as a session prepared at that rotation."""

import math
import platform

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64", reason="MPS SSB requires Apple silicon"
)


def test_rotation_change_keeps_every_aperture_pixel_with_anisotropic_sampling():
    """With rows and columns sampled differently, set_rotation gives what a session prepared at the new rotation gives.

    A rotation keeps each bright-field pixel's scattering angle but turns the aperture's soft edge, one detector pixel
    wide along k, and that width changes with direction when the samplings differ. Preparation stored only the pixels
    inside the aperture at the first rotation, so after set_rotation the pixels that had entered it were missing while
    the bright-field count still included them.
    """
    pytest.importorskip("mlx.core")
    from quantem.gpu import SSB

    rng = np.random.default_rng(5)
    rows, cols = np.indices((64, 64)) - 31.5
    counts = rng.poisson(30.0, (128, 128, 64, 64)).astype(np.uint16)
    counts[:, :, np.hypot(rows, cols) < 30] += 300
    # the selection (30 px) reaches past the aperture (25 mrad: 25 px along rows, 17 px along columns)
    settings = dict(backend="mps", voltage_kV=300.0, semiangle_mrad=25.0, scan_sampling_A=(0.3, 0.3), detector_sampling_mrad=(1.0, 1.5),
                    bright_field_radius_px=30.0, bright_field_center_px=(31.5, 31.5))
    aberrations = {"C10_nm": 4.0, "C12_nm": 1.0, "phi12_deg": math.degrees(0.4)}

    with SSB(counts, scan_detector_rotation_deg=0.0, **settings) as turned:
        turned.reconstruct(aberrations=aberrations)
        turned.set_rotation(45.0)
        phase, loss = turned.preview(aberrations)
    with SSB(counts, scan_detector_rotation_deg=45.0, **settings) as prepared:
        expected, expected_loss = prepared.preview(aberrations)

    np.testing.assert_array_equal(phase, expected)
    assert loss == expected_loss


def _astigmatic_crystal(n=64, scan_A=0.25, C10_A=-80.0, C12_A=25.0, phi12_deg=20.0):
    """Noise-free 4D-STEM of a periodic phase object (bright columns) under a 300 kV, 30 mrad probe with astigmatism.

    The scan steps are the object pixels, so each frame is the far field of the object times the probe shifted by whole
    pixels. Returns the frames (a dose of 1e5 electrons) and the detector sampling in mrad.
    """
    energy = 300e3
    wavelength_A = 12.2643 / math.sqrt(energy * (1 + energy / (2 * 511e3)))
    k = np.fft.fftfreq(n, scan_A)
    kr, kc = np.meshgrid(k, k, indexing="ij")
    alpha, azimuth = np.hypot(kr, kc) * wavelength_A, np.arctan2(kc, kr)
    chi = math.pi / wavelength_A * alpha**2 * (C10_A + C12_A * np.cos(2 * (azimuth - math.radians(phi12_deg))))
    probe = np.fft.ifft2((alpha < 30e-3) * np.exp(-1j * chi))
    rows, cols = np.indices((n, n)) * scan_A
    field, a = n * scan_A, 4.0
    potential = np.zeros((n, n))
    for i in range(int(field / a)):
        for j in range(int(field / a)):
            for offset, weight in ((0.0, 1.0), (0.5, 0.6)):
                dr = (rows - (i + offset) * a + field / 2) % field - field / 2
                dc = (cols - (j + offset) * a + field / 2) % field - field / 2
                potential += weight * np.exp(-(dr**2 + dc**2) / (2 * 0.35**2))
    transmission = np.exp(0.4j * potential)
    frames = np.empty((n, n, n, n), np.float32)
    for i in range(n):
        for j in range(n):
            frames[i, j] = np.fft.fftshift(np.abs(np.fft.fft2(np.roll(probe, (i, j), axis=(0, 1)) * transmission)) ** 2)
    frames *= 1e5 / frames[0, 0].sum()
    return frames, wavelength_A / (n * scan_A) * 1e3


def _settings(detector_mrad, **extra):
    return dict(backend="mps", voltage_kV=300.0, semiangle_mrad=30.0, scan_sampling_A=(0.25, 0.25),
                detector_sampling_mrad=(detector_mrad, detector_mrad), **extra)


@pytest.fixture
def estimate_180_degrees_off(monkeypatch):
    """The centre-of-mass estimate lands on the wrong 180-degree branch: 180 degrees for a crystal simulated at 0."""
    from quantem.gpu.ssb import workflow

    monkeypatch.setattr(workflow, "_estimate_rotation", lambda *args: (180.0, False))


def test_an_estimate_180_degrees_off_keeps_the_branch_with_positive_columns(estimate_180_degrees_off):
    """Both branches are fitted independently; the dark-field columns have positive phase only on the true one.

    Replaces the polarity check that moved the branch only for a negative median-centred skew and refitted from
    the mapped twin. Here the probe is near focus (C10 -3 nm), so the low-angle dark field peaks on the columns.
    """
    pytest.importorskip("mlx.core")
    from quantem.gpu import SSB

    frames, detector_mrad = _astigmatic_crystal(C10_A=-30.0, C12_A=10.0)
    with SSB(frames, scan_detector_rotation_deg=0.0, **_settings(detector_mrad)) as session:
        direct = session.find_aberrations(search="standard", num_trials=40, check_rotation=False, verbose=False)
    with SSB(frames, **_settings(detector_mrad)) as session:
        assert session.scan_detector_rotation_source == "estimated"
        checked = session.find_aberrations(search="standard", num_trials=40, verbose=False)
        assert session.scan_detector_rotation_deg == 0.0

    branches = checked.rotation_branches
    assert [branch.scan_detector_rotation_deg for branch in branches.branches] == [180.0, 0.0]
    assert branches.decided_by == "columns" and branches.chosen == branches.kept == 1
    assert branches.branches[1].column_z >= 3.0 and branches.branches[0].column_z <= -3.0
    assert branches.branches[1].column_positive_fraction > 0.9
    assert checked.scan_detector_rotation_deg == 0.0
    for name in ("C10_nm", "C12_nm"):
        assert checked.aberrations[name] == pytest.approx(direct.aberrations[name], abs=0.05)
    assert checked.aberrations["C10_nm"] == pytest.approx(-3.0, abs=0.1)
    assert checked.loss == pytest.approx(direct.loss, rel=1e-3)
    assert {record["search"] for record in checked.trial_records} == {0, 1}


def test_unresolved_columns_leave_the_choice_to_the_phase_tail(estimate_180_degrees_off):
    """With a dark field that shows no columns (no contrast), the larger mean-centred phase skewness decides."""
    pytest.importorskip("mlx.core")
    from quantem.gpu import SSB
    from quantem.gpu.ssb.results import phase_skewness

    frames, detector_mrad = _astigmatic_crystal()
    with SSB(frames, **_settings(detector_mrad)) as session:
        session._dark_field = np.ones(frames.shape[:2], np.float32)
        checked = session.find_aberrations(search="standard", num_trials=40, verbose=False)
    branches = checked.rotation_branches
    assert branches.decided_by == "phase_tail" and checked.scan_detector_rotation_deg == 0.0
    assert branches.num_columns == 0 and "no atomic columns located" in branches.summary
    assert branches.branches[1].phase_skewness == pytest.approx(phase_skewness(checked.phase), abs=1e-6)
    assert branches.branches[0].phase_skewness < 0.0 < 1.0 < branches.branches[1].phase_skewness
    assert checked.aberrations["C10_nm"] == pytest.approx(-8.0, abs=0.15)
    assert checked.aberrations["C12_nm"] == pytest.approx(2.5, abs=0.1)


def test_reconstruct_keeps_the_branch_comparison_of_the_fit_it_applies(estimate_180_degrees_off, capsys):
    """reconstruct() after a branch comparison reuses the kept fit and carries its evidence; no third search."""
    pytest.importorskip("mlx.core")
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from quantem.gpu import SSB

    frames, detector_mrad = _astigmatic_crystal(C10_A=-30.0, C12_A=10.0)
    with SSB(frames, **_settings(detector_mrad)) as session:
        fitted = session.find_aberrations(search="standard", num_trials=40, verbose=False)
        assert session.reconstruct() is fitted
        for result in (session.reconstruct(), session.reconstruct(fitted, upsampling_factor=2)):
            assert result.rotation_branches is fitted.rotation_branches
            report = result.report().iloc[0]
            assert report["Suggested rotation (deg)"] == 180.0 and report["SSB-selected rotation (deg)"] == 0.0
            assert report["180° branch by"] == "atomic columns"
            assert isinstance(result.show("rotation"), Figure)
        changed = session.reconstruct(aberrations={"C10_nm": fitted.aberrations["C10_nm"] + 1.0})
        assert changed.rotation_branches is None
        with pytest.raises(ValueError, match="compare_rotation_branches"):
            changed.show("rotation")
    capsys.readouterr()
    with SSB(frames, **_settings(detector_mrad)) as fresh:
        auto = fresh.reconstruct()
    assert auto.rotation_branches is not None and auto.scan_detector_rotation_deg == 0.0
    out = capsys.readouterr().out
    assert "SSB: kept 0.0°: atomic columns positive" in out and out.count("SSB: kept") == 1


def test_branch_panels_are_each_fit_at_the_auto_factor_while_the_fit_stays_native(estimate_180_degrees_off):
    """Both branch panels are reconstructed at 2x auto (0.25 Å) from the session's G(q, k); fit and tests stay 1x.

    Each panel is the session's own preview at that branch's rotation, aberrations, tilt and depth spread; the session
    is back on the kept fit afterwards. show_trials reconstructs at the same auto factor.
    """
    pytest.importorskip("mlx.core")
    import matplotlib

    matplotlib.use("Agg")
    from quantem.gpu import SSB

    frames, detector_mrad = _astigmatic_crystal(C10_A=-30.0, C12_A=10.0)
    with SSB(frames, **_settings(detector_mrad)) as session:
        fitted = session.find_aberrations(search="standard", num_trials=40, verbose=False)
        branches = fitted.rotation_branches
        assert fitted.upsampling_factor == 1 and fitted.phase.shape == (64, 64)
        assert all(branch.phase.shape == (64, 64) for branch in branches.branches)
        tests = [(branch.column_z, branch.phase_skewness) for branch in branches.branches]
        panels = branches.upsampled_phases()
        assert branches.upsampling_factor == 2 and branches.object_sampling_A == (0.125, 0.125)
        assert session.scan_detector_rotation_deg == 0.0 and session.reconstruct() is fitted
        for branch, panel in zip(branches.branches, panels, strict=True):
            assert panel.shape == (128, 128)
            session.set_rotation(branch.scan_detector_rotation_deg)
            expected, _ = session.preview(dict(branch.aberrations), sample_tilt_mrad=branch.sample_tilt_mrad,
                                          depth_spread_nm=branch.depth_spread_nm, upsampling_factor=2,
                                          compute_loss=False)
            np.testing.assert_array_equal(panel, expected)
            # the native samples of the 2x grid are the scan positions the columns were located at
            native = branch.phase - branch.phase.mean()
            assert np.corrcoef(panel[::2, ::2].ravel(), native.ravel())[0, 1] > 0.9
        session.set_rotation(fitted.scan_detector_rotation_deg)
        assert [(branch.column_z, branch.phase_skewness) for branch in branches.branches] == tests
        figure = fitted.show("rotation")
        assert [axis.images[0].get_array().shape[:2] for axis in figure.axes if axis.images] == [(128, 128)] * 2
        trials = session.show_trials(best=2)
        assert "2× auto" in trials._suptitle.get_text()
        assert {axis.images[0].get_array().shape[:2] for axis in trials.axes if axis.images} == {(128, 128)}


def test_a_region_fit_keeps_its_branch_panels_after_its_region_session_closes(tmp_path):
    """find_aberrations(scan_region=...) fits in a region session that closes: the panels come with the fit and record.

    A bright-field disk shifted by a smooth phase object, 128 x 128 positions at 0.25 Å (2x auto); fit quality is not
    the point, only that the panels are the region's, reconstructed before its G(q, k) is gone, and saved.
    """
    pytest.importorskip("mlx.core")
    from quantem.gpu import SSB
    from quantem.gpu.ssb.persistence import load_result

    rows, cols = np.indices((32, 32))
    counts = np.zeros((128, 128, 32, 32), np.uint16)
    counts[..., (rows - 15.5) ** 2 + (cols - 15.5) ** 2 <= 8.0**2] = 40
    shift = np.round(3 * np.sin(np.linspace(0, 6 * np.pi, 128))[:, None] * np.cos(np.linspace(0, 4 * np.pi, 128))[None, :])
    for row in range(128):
        for col in range(128):
            counts[row, col] = np.roll(counts[row, col], int(shift[row, col]), axis=1)
    counts += np.random.default_rng(0).integers(0, 3, counts.shape, dtype=np.uint16)
    master = tmp_path / "scan_master.h5"
    master.write_bytes(b"master")   # an array session names its source to be saved
    settings = dict(backend="mps", voltage_kV=300.0, semiangle_mrad=30.0, scan_sampling_A=0.25,
                    detector_sampling_mrad=(3.75, 3.75), scan_detector_rotation_flipped=False, source_path=str(master))
    with SSB(counts, **settings) as session:
        fitted = session.find_aberrations(search="standard", scan_region=(0, 128, 0, 128), tilt=False, num_trials=8, refinement=None,
                                          save_dir=tmp_path / "results", verbose=False)
    branches = fitted.rotation_branches
    assert fitted.fit_scan_region == (0, 128, 0, 128) and branches.upsampling_factor == 2
    assert [panel.shape for panel in branches.upsampled_phases()] == [(256, 256)] * 2
    paths = (fitted.saved_path, fitted.saved_path.with_suffix(".json"))
    reloaded = load_result(paths=paths, signature=fitted.metadata["signature"], backend="mps").rotation_branches
    assert reloaded.upsampling_factor == 2
    for saved, panel in zip(reloaded.upsampled_phases(), branches.upsampled_phases(), strict=True):
        np.testing.assert_array_equal(saved, panel)


def test_a_given_rotation_is_never_flipped(capsys):
    """A rotation the user gives is fitted once and kept, even 180 degrees off; the comparison only reports.

    Before 2026-10-20 reconstruct() moved a given 180 degrees to 0 here (and a given 169.888 to 349.888 on data).
    """
    pytest.importorskip("mlx.core")
    from quantem.gpu import SSB

    frames, detector_mrad = _astigmatic_crystal(C10_A=-30.0, C12_A=10.0)
    with SSB(frames, scan_detector_rotation_deg=180.0, **_settings(detector_mrad)) as session:
        assert session.scan_detector_rotation_source == "given"
        result = session.reconstruct()
        assert result.scan_detector_rotation_deg == 180.0 and result.rotation_branches is None
        assert "SSB: kept" not in capsys.readouterr().out
        branches = session.compare_rotation_branches(num_trials=40, verbose=False)
        assert branches.kept_rotation_deg == 180.0 and branches.chosen_rotation_deg == 0.0
        assert branches.summary.startswith("kept the given 180.0°; the evidence selects 0.0°")
        assert session.scan_detector_rotation_deg == 180.0
        kept = session.reconstruct()
        assert kept.scan_detector_rotation_deg == 180.0 and kept.rotation_branches is branches
        assert session.compare_rotation_branches(num_trials=40, verbose=False, apply=True).kept_rotation_deg == 0.0
        assert session.scan_detector_rotation_deg == 0.0
        with pytest.raises(TypeError, match="check_rotation"):
            session.compare_rotation_branches(check_rotation=False)


def test_joint_fit_takes_a_c10_range_and_holds_c12_at_zero():
    """The tutorial's constrained joint fit runs: a C10 range, C12 held at 0, the others fitted.

    Before 2026-10-20 search_ranges was refused with tilt=True and fixed_aberrations needed all three coefficients
    with check_rotation=False.
    """
    pytest.importorskip("mlx.core")
    from quantem.gpu import SSB

    frames, detector_mrad = _astigmatic_crystal()
    settings = dict(backend="mps", voltage_kV=300.0, semiangle_mrad=30.0, scan_sampling_A=(0.25, 0.25),
                    detector_sampling_mrad=(detector_mrad, detector_mrad), scan_detector_rotation_deg=0.0)
    with SSB(frames, **settings) as session:
        held = session.find_aberrations(search="standard", tilt=True, num_trials=40, verbose=False,
                                        search_ranges={"C10_nm": (-50, 50)}, fixed_aberrations={"C12_nm": 0.0})
        assert held.aberrations["C12_nm"] == 0.0 and held.aberrations["phi12_deg"] == 0.0
        # a given rotation is fitted once: no branch comparison
        assert -50.0 <= held.aberrations["C10_nm"] <= 50.0 and held.rotation_branches is None
        bounded = session.find_aberrations(search="standard", tilt=True, num_trials=40, verbose=False,
                                           search_ranges={"C10_nm": (-12, -4), "C12_nm": (0, 5)})
        assert -12.0 <= bounded.aberrations["C10_nm"] <= -4.0 and 0.0 <= bounded.aberrations["C12_nm"] <= 5.0
        assert bounded.aberrations["C10_nm"] == pytest.approx(-8.0, abs=0.1)
        fixed_defocus = session.find_aberrations(search="standard", tilt=True, num_trials=40, verbose=False,
                                                 fixed_aberrations={"C10_nm": -8.0})
        assert fixed_defocus.aberrations["C10_nm"] == pytest.approx(-8.0, abs=1e-6)


@pytest.mark.parametrize("constraints,match", [
    (dict(search_ranges={"phi12_deg": (0, 30)}), "phi12_deg over its whole period"),
    (dict(fixed_aberrations={"phi12_deg": 10.0}), "phi12_deg over its whole period"),
    (dict(fixed_aberrations={"C12_nm": 2.0}), "C12_nm from 0"),
    (dict(search_ranges={"C12_nm": (1, 5)}), "C12_nm from 0"),
    (dict(search_ranges={"C10_nm": (5, -5)}), "low <= high"),
    (dict(search_ranges={"C10_nm": (0, 5)}, fixed_aberrations={"C10_nm": 1.0}), "both"),
    (dict(fixed_aberrations={"C10_nm": 1.0, "C12_nm": 2.0, "phi12_deg": 3.0}), "check_rotation=False"),
    (dict(refine_lock=["C12_nm"]), "fixed_aberrations"),
])
def test_unsupported_joint_constraints_say_what_to_pass(constraints, match):
    from quantem.gpu import SSB

    session = SSB.__new__(SSB)  # validation runs before any data is touched
    session._device = None
    with pytest.raises(ValueError, match=match):
        SSB.find_aberrations(session, search="standard", tilt=True, verbose=False, **constraints)


@pytest.mark.parametrize("pins,lock,refinement", [
    ({"C12_nm": 0.0, "phi12_deg": 0.0}, ["C12_nm", "phi12_deg"], "nelder-mead"),
    ({"C12_nm": 1.5, "phi12_deg": 20.0}, ["C12_nm", "phi12_deg"], "nelder-mead"),
    ({"C12_nm": 1.5, "phi12_deg": 20.0}, None, None),
])
def test_search_range_pins_hold_their_values_on_mps(pins, lock, refinement):
    """A coefficient pinned in search_ranges keeps its value in the result, as on CUDA.

    Before 2026-10-20 MPS evaluated every trial at the pins but reported (and reconstructed) the search's start for
    them: C12 5.0 nm and phi12 0 degrees, whatever was pinned.
    """
    pytest.importorskip("mlx.core")
    from quantem.gpu import SSB

    frames, detector_mrad = _astigmatic_crystal()
    settings = dict(backend="mps", voltage_kV=300.0, semiangle_mrad=30.0, scan_sampling_A=(0.25, 0.25),
                    detector_sampling_mrad=(detector_mrad, detector_mrad), scan_detector_rotation_deg=0.0)
    with SSB(frames, **settings) as session:
        fitted = session.find_aberrations(search="standard", tilt=False, num_trials=30, search_ranges=pins, refine_lock=lock,
                                          refinement=refinement,
                                          check_rotation=False, verbose=False)
        assert fitted.aberrations["C12_nm"] == pytest.approx(pins["C12_nm"], abs=1e-9)
        assert fitted.aberrations["phi12_deg"] == pytest.approx(pins["phi12_deg"], abs=1e-9)
        expected, _ = session.preview(fitted.aberrations, compute_loss=False)
        np.testing.assert_allclose(fitted.phase, expected, atol=1e-5)


@pytest.mark.parametrize("lock", [["C12_nm"], None])
def test_holding_c12_at_zero_still_fits_the_defocus(lock):
    """search_ranges={"C12_nm": 0.0} holds the astigmatism and still fits C10 over its default range.

    Before this fix MPS returned C10 = C12 = phi12 = 0 exactly: the omitted C10 was held at the start, and the pinned
    start (no correction at all) was a degenerate candidate no trial could beat (on the public gold scan loss 0.0201,
    below the real optimum 0.0445, with the wrong 180-degree branch kept).
    """
    pytest.importorskip("mlx.core")
    from quantem.gpu import SSB

    frames, detector_mrad = _astigmatic_crystal(C12_A=0.0)  # truth C10 -8 nm, no astigmatism
    settings = dict(backend="mps", voltage_kV=300.0, semiangle_mrad=30.0, scan_sampling_A=(0.25, 0.25),
                    detector_sampling_mrad=(detector_mrad, detector_mrad), scan_detector_rotation_deg=0.0)
    with SSB(frames, **settings) as session:
        fitted = session.find_aberrations(search="standard", tilt=False, num_trials=40, search_ranges={"C12_nm": 0.0}, refine_lock=lock,
                                          check_rotation=False, verbose=False)
    assert fitted.aberrations["C10_nm"] == pytest.approx(-8.0, abs=0.5)
    assert fitted.aberrations["C12_nm"] == 0.0 and fitted.aberrations["phi12_deg"] == 0.0


@pytest.mark.parametrize("estimate_deg", [2.0, 357.8])
def test_refine_rotation_recovers_a_two_degree_offset(monkeypatch, estimate_deg):
    """The centre-of-mass estimate is about 2 degrees off a crystal simulated at 0 degrees; the refinement finds 0.

    The branch test keeps the estimate's branch, then the 1-D search polishes the fit at each angle from the nearest
    angle's fit (no new search) and keeps the angle of the highest agreement.
    """
    pytest.importorskip("mlx.core")
    from quantem.gpu import SSB
    from quantem.gpu.ssb import workflow

    monkeypatch.setattr(workflow, "_estimate_rotation", lambda *args: (estimate_deg, False))
    frames, detector_mrad = _astigmatic_crystal(C10_A=-30.0, C12_A=10.0)
    with SSB(frames, **_settings(detector_mrad)) as session:
        fit = session.find_aberrations(search="standard", num_trials=40, refine_rotation=True, verbose=False)
        assert session.scan_detector_rotation_deg == fit.scan_detector_rotation_deg
        assert session.reconstruct() is fit
    assert fit.rotation_branches.kept == 0 and fit.rotation_branches.decided_by == "columns"
    assert fit.rotation_refined and fit.rotation_estimate_deg == estimate_deg
    assert abs((fit.scan_detector_rotation_deg + 180.0) % 360.0 - 180.0) < 0.1
    record = fit.rotation_refinement
    assert record["refined_agreement"] == fit.sample_agreement > record["estimate_agreement"]
    assert record["evaluations"] <= 12
    assert fit.aberrations["C10_nm"] == pytest.approx(-3.0, abs=0.1)
    assert len([trial for trial in fit.trial_records if trial["stage"] == "search"]) == 80


def test_a_given_rotation_is_fitted_once_and_not_refined():
    pytest.importorskip("mlx.core")
    from quantem.gpu import SSB

    frames, detector_mrad = _astigmatic_crystal(C10_A=-30.0, C12_A=10.0)
    with SSB(frames, scan_detector_rotation_deg=2.0, **_settings(detector_mrad)) as session:
        fit = session.find_aberrations(search="standard", num_trials=40, refine_rotation=True, verbose=False)
    assert fit.scan_detector_rotation_deg == 2.0 and fit.rotation_branches is None
    assert (fit.rotation_refined, fit.rotation_estimate_deg, fit.rotation_refinement) == (False, None, None)


@pytest.mark.slow
def test_gold_refined_rotation(capsys):
    """Gold, estimated rotation, the default tilt-corrected fit of both branches, then the refinement on the kept one.

    On an M5 Max: kept 358.31 (phase tail), refined to 1.24 degrees, agreement +9.8%, 11 polishes in 12.5 s.
    """
    pytest.importorskip("mlx.core")
    from quantem.gpu import SSB, io
    from tests.hardware.mps.test_ssb_thick_torch_reference import _gold_harness

    harness = _gold_harness()
    master = harness.gold_master()
    if master is None:
        pytest.skip("the gold scan is not in the Hugging Face cache (tests/benchmarks/mps_ssb.py run --download fetches it)")
    with io.load(str(master), hot_pixel_correction=harness.HOT_PIXELS, verbose=False) as data:
        harness.calibrate(data)
        with SSB(data) as ssb:
            fit = ssb.find_aberrations(search="standard", refine_rotation=True)
            assert ssb.scan_detector_rotation_deg == fit.scan_detector_rotation_deg
    record = fit.rotation_refinement
    kept = fit.rotation_branches.branches[fit.rotation_branches.kept]
    assert fit.rotation_refined and fit.rotation_estimate_deg == kept.scan_detector_rotation_deg == record["estimate_deg"]
    change = (fit.scan_detector_rotation_deg - fit.rotation_estimate_deg + 180.0) % 360.0 - 180.0
    assert 0.05 < abs(change) < 10.0
    assert fit.rotation_estimate_deg == pytest.approx(358.3108, abs=1e-3)
    assert fit.scan_detector_rotation_deg == pytest.approx(1.24, abs=0.05)
    assert record["refined_agreement"] == fit.sample_agreement > record["estimate_agreement"]
    assert 3 <= record["evaluations"] <= 15
    assert fit.timings["rotation_refinement_s"] == record["seconds"]
    out = capsys.readouterr().out
    assert out.count("SSB: refined rotation ") == 1 and out.index("SSB: kept ") < out.index("SSB: refined rotation ")
    report = fit.report().iloc[0]
    assert report["Estimated rotation (deg)"] == round(fit.rotation_estimate_deg, 3)
    assert report["SSB-selected rotation (deg)"] == round(fit.scan_detector_rotation_deg, 3)
