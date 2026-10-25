"""GPU-owned adaptive search, precision, and six-parameter Newton contracts."""

import math
import platform

import numpy as np
import pytest


@pytest.fixture(params=["mps", "cuda"])
def device(request):
    """Require the physical backend named by the parametrized test."""
    if request.param == "mps":
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            pytest.skip("Metal requires Apple silicon")
        pytest.importorskip("mlx.core")
    else:
        cp = pytest.importorskip("cupy")
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip("CUDA hardware is unavailable")
    from quantem.gpu.ssb._resident_search import _Device

    return _Device(request.param)


def test_probability_precision_and_random_stream(device):
    """Bounded scalar references qualify decisions, never full CPU SSB."""
    from scipy.special import ndtr, ndtri

    from quantem.gpu.ssb._resident_search import _random

    checks = [
        ("cdf", np.linspace(-10, 10, 41), ndtr),
        (
            "inverse_cdf",
            np.r_[
                np.logspace(-15, -1, 20), 0.25, 0.5, 0.75, 1 - np.logspace(-15, -1, 20)
            ],
            ndtri,
        ),
        ("exp", np.linspace(-40, 40, 100), np.exp),
        ("log", np.logspace(-15, 15, 100), np.log),
        ("sin", np.linspace(-math.pi, math.pi, 100), np.sin),
        ("cos", np.linspace(-math.pi, math.pi, 100), np.cos),
    ]
    for name, values, reference in checks:
        packed = device.pack(values)
        output = device.run(
            "test_resident_" + name,
            {"values": packed},
            [packed.shape],
            f"if(index>=COUNT)return;store_real(out0,int(index),{name}_r(load_real(values,int(index))));",
            count=len(values),
            constants=f"\n#define COUNT {len(values)}\n",
        )[0]
        np.testing.assert_allclose(
            device.host(output), reference(values), rtol=3e-14, atol=3e-15
        )
    actual = device.host(_random(device, 11, 200, 168)).ravel()
    np.testing.assert_array_equal(
        actual, np.random.RandomState(11).random_sample(len(actual))
    )


@pytest.mark.parametrize("fixed", [False, True])
def test_adaptive_tpe_preserves_trial_sequence(device, fixed):
    """Fixed score history isolates TPE equivalence from model rounding."""
    from quantem.gpu.ssb._resident_search import _search
    from quantem.gpu.ssb.thick_sample_fit import PARAMETERS, fit_sample_search

    positions = {"gpu": 0, "reference": 0}

    def values(rows, key):
        offset = positions[key]
        positions[key] += len(rows)
        return np.sin(np.arange(offset, offset + len(rows)) * 0.7) + 2

    lower = [-300.0, 0.0, -math.pi / 2, -25.0, -25.0, 20.0]
    upper = [300.0, 200.0, math.pi / 2, 25.0, 25.0, 600.0]
    fixed_values = {"C10": 100.0, "C12": 60.0, "phi12": 0.2} if fixed else None
    if fixed:
        lower[:3] = upper[:3] = list(fixed_values.values())
    points, _ = _search(
        device,
        lambda trial: device.pack(values(trial, "gpu")),
        device.pack(lower + upper),
        200,
        11,
        fixed=fixed,
    )
    expected = fit_sample_search(
        lambda *args: 1.0,
        objective_batch=lambda rows: values(rows, "reference"),
        trials=200,
        seed=11,
        polish_starts=0,
        compare_standard=False,
        polish="newton",
        verbose=False,
        fixed_aberrations=fixed_values,
    )
    reference = np.array(
        [
            [row["params"][key] for key in PARAMETERS]
            for row in expected["trial_records"]
        ]
    )
    np.testing.assert_allclose(device.host(points), reference, atol=1e-8, rtol=1e-10)


def test_joint_newton_recovers_solution_without_host_optimizer(device, monkeypatch):
    """Every search and Newton point stays resident for a full 200-trial fit."""
    import scipy.optimize

    from quantem.gpu.ssb import thick_sample_fit
    from quantem.gpu.ssb._resident_search import fit_resident

    def forbidden(*args, **kwargs):
        raise AssertionError("The joint fitter called a CPU numerical optimizer")

    monkeypatch.setattr(thick_sample_fit, "fit_sample_search", forbidden)
    monkeypatch.setattr(thick_sample_fit, "newton_polish", forbidden)
    monkeypatch.setattr(scipy.optimize, "minimize", forbidden)
    monkeypatch.setattr(np.linalg, "eigh", forbidden)

    def objective(trial):
        return device.run(
            "test_resident_quadratic",
            {"trial": trial},
            [(len(trial), 3)],
            r"""
if(index>=COUNT)return;int b=int(index);Real sum(0);
Real target[7]={Real(100),Real(60),Real(1),Real(0),Real(150),Real(0.003f),Real(-0.004f)};
Real scales[7]={Real(50),Real(50),Real(1),Real(1),Real(100),Real(0.01f),Real(0.01f)};
for(int d=0;d<7;d++){Real delta=(Real(trial[b*7+d])-target[d])/scales[d];sum=sum+delta*delta;}
store_real(out0,b,Real(100)-sum);
""",
            count=len(trial),
            constants=f"\n#define COUNT {len(trial)}\n",
        )[0]

    result = fit_resident(
        objective,
        backend=device.backend,
        trials=200,
        compare_standard=True,
        verbose=False,
    )
    assert len(result["trial_records"]) == 200
    assert abs(result["fit"] - 100) < 1e-7
    np.testing.assert_allclose(
        [result["tilt_row_mrad"], result["tilt_col_mrad"]], [3, -4], atol=1e-4
    )
    assert abs(result["C10"] - 100) < 0.01
    assert abs(result["C12"] - 60) < 0.02
    assert abs(result["depth_spread_A"] - 150) < 0.01
    assert result["optimizer"]["device"] == device.backend
    assert result["optimizer"]["host_numerical_search"] is False
    assert result["optimizer"]["newton_starts"]
    assert result["standard_fit"] > 0
    assert result["gain"] > 1
    resumed = fit_resident(
        objective,
        backend=device.backend,
        trials=0,
        start=[result[name] for name in thick_sample_fit.PARAMETERS],
        compare_standard=False,
        verbose=False,
    )
    assert resumed["trial_records"] == []
    assert abs(resumed["fit"] - result["fit"]) < 1e-7


def test_model_scores_and_public_joint_fit_remain_on_device(device, monkeypatch):
    """Use the complete synthetic scan/BF and block the previous host fitter."""
    from quantem.gpu import SSB
    from quantem.gpu.ssb import thick_sample_fit
    from quantem.gpu.ssb._resident_search import _trial

    rng = np.random.default_rng(7)
    counts = rng.poisson(40.0, size=(128, 128, 32, 32)).astype(np.uint16)
    rr, cc = np.meshgrid(np.arange(32) - 15.5, np.arange(32) - 15.5, indexing="ij")
    counts[:, :, np.hypot(rr, cc) < 10] += 200
    rows = rng.uniform(
        [-300, 0, -1.5, -25, -25, 20], [300, 200, 1.5, 25, 25, 600], (84, 6)
    )
    with SSB(
        counts,
        backend=device.backend,
        voltage_kV=300.0,
        semiangle_mrad=30.0,
        scan_sampling_A=(0.3, 0.3),
        detector_sampling_mrad=(3.0, 3.0),
        scan_detector_rotation_deg=0.0,
    ) as ssb:
        backend = ssb._backend()
        backend.cache_rotation(0.0)
        expected = backend.sample_agreement(rows, (0.2, 0.9))
        trial = _trial(device, device.pack(rows))
        if device.backend == "mps":
            from quantem.gpu.ssb.mps import resident_fit

            actual = device.host(resident_fit.evaluate(backend._prepared, trial))
            # Both eight-point search batches and the initial three-start
            # Newton design retain exactly the same model and reduction order
            # when the memory planner schedules independent tiles together.
            for count in (8, 84):
                concurrent = device.host(
                    resident_fit.evaluate(backend._prepared, trial[:count])
                )
                with monkeypatch.context() as limited:
                    limited.setattr(resident_fit, "_FIT_PENDING_BYTES", 1)
                    serial = device.host(
                        resident_fit.evaluate(backend._prepared, trial[:count])
                    )
                np.testing.assert_array_equal(concurrent, serial)
        else:
            actual = device.host(backend._get_accelerator().thick.fit_resident(trial))
        # Model kernels remain float32. Only final band reduction gains
        # precision on CUDA; Metal already used an FP64 host reduction.
        np.testing.assert_allclose(actual, expected, rtol=2e-6)

        def forbidden(*args, **kwargs):
            raise AssertionError(
                "Public joint fitting called the previous host optimizer"
            )

        monkeypatch.setattr(thick_sample_fit, "fit_sample_search", forbidden)
        module = __import__(
            f"quantem.gpu.ssb.{device.backend}.optimizer", fromlist=["fit_sample"]
        )
        monkeypatch.setattr(module, "fit_sample_search", forbidden)
        result = ssb.find_aberrations(search="primary",
            tilt=True,
            num_trials=200,
            refinement="newton",
            compare_standard=False,
            check_rotation=False,
            verbose=False,
        )
        assert len(result.trial_records) == 200
        assert result.optimizer["device"] == device.backend
        assert not result.optimizer["host_numerical_search"]
        assert result.phase.shape == (128, 128)


@pytest.mark.parametrize("transposed", [False, True])
def test_resident_region_matches_reopened_counts_and_keeps_parent(
    device, tmp_path, transposed
):
    """A moved rectangle reuses counts and retains the full detector calibration."""
    from quantem.gpu import SSB, io

    rng = np.random.default_rng(19)
    counts = rng.poisson(5.0, (128, 128, 24, 24)).astype(np.uint16)
    source = tmp_path / "scan.npy"
    np.save(source, counts)
    geometry = {
        "backend": device.backend,
        "voltage_kV": 200.0,
        "semiangle_mrad": 25.0,
        "scan_sampling_A": (0.3, 0.3),
        "detector_sampling_mrad": (3.0, 3.5),
        "bright_field_center_px": (12.2, 10.8),
        "bright_field_radius_px": 6.5,
        "scan_detector_rotation_flipped": transposed,
        "scan_detector_rotation_deg": 17.0,
        "aberrations": {"C10_nm": 8.0, "C12_nm": 3.0, "phi12_deg": 20.0},
    }
    with SSB.open(str(source), **geometry) as parent:
        reference = dict(parent.aberrations)
        resident = parent._data
        for bounds in ((9, 105, 13, 111), (12, 124, 2, 114)):
            with parent.select_region(bounds) as selected:
                assert selected.scan_region == bounds
                assert selected.source_path == str(source)
                assert selected.bright_field_center_px == parent.bright_field_center_px
                if device.backend == "mps":
                    assert (
                        selected._data.untyped_storage().data_ptr()
                        == resident.untyped_storage().data_ptr()
                    )
                else:
                    assert selected._data.base is not None
                actual, _ = selected.preview(
                    reference,
                    sample_tilt_mrad=(3.0, -2.0),
                    depth_spread_nm=15.0,
                    compute_loss=False,
                )
                with (
                    io.load(
                        str(source), backend=device.backend, verbose=False
                    ) as acquisition,
                    SSB(
                        acquisition,
                        scan_region=bounds,
                        source_path=str(source),
                        **geometry,
                    ) as reopened,
                ):
                    expected, _ = reopened.preview(
                        reference,
                        sample_tilt_mrad=(3.0, -2.0),
                        depth_spread_nm=15.0,
                        compute_loss=False,
                    )
                np.testing.assert_array_equal(actual, expected)
                selected.aberrations = {**reference, "C10_nm": 20.0}
            assert parent._data is resident
            assert parent.aberrations == reference


def test_loaded_metadata_and_explicit_ssb_overrides_keep_counts(device, tmp_path):
    """Load, inspect, then reconstruct with metadata or explicit calibration."""
    import copy

    from quantem.gpu import SSB, io

    counts = np.random.default_rng(17).poisson(4, (128, 128, 24, 24)).astype(np.uint16)
    source = tmp_path / "scan.npy"
    np.save(source, counts)
    with io.load(source, backend=device.backend, verbose=False) as data:
        data.metadata.update(
            voltage_kV=200.0,
            semiangle_mrad=25.0,
            scan_sampling_A=(0.3, 0.3),
            detector_sampling_mrad=(3.0, 3.0),
        )
        original = copy.deepcopy(data.metadata)
        assert data.metadata_table().loc["Accelerating voltage", "Value"] == 200.0
        geometry = {
            "backend": device.backend,
            "bright_field_center_px": (12.0, 12.0),
            "bright_field_radius_px": 6.5,
            "scan_detector_rotation_deg": 17.0,
        }
        optics = {"C10_nm": 8.0, "C12_nm": 0.0, "phi12_deg": 0.0}
        with SSB(data, voltage_kV=300.0, **geometry) as ssb:
            assert ssb.voltage_kV == 300.0
            assert ssb.scan_sampling_A == (0.3, 0.3)
            assert ssb.detector_sampling_mrad == (3.0, 3.0)
            assert ssb.source_path == data.metadata["source_path"]
            assert ssb.scan_detector_rotation_deg == 17.0
            table = ssb.calibration()
            assert table.loc["Accelerating voltage", "File value"] == 200.0
            assert table.loc["Accelerating voltage", "Value used"] == 300.0
            assert table.loc["Accelerating voltage", "Source"] == "Explicit override"
            assert table.loc["Scan step (row, col)", "Source"] == "Recorded"
            assert table.loc["Scan-detector rotation", "File value"] == "Not recorded"
            assert table.loc["Scan-detector rotation", "Source"] == "Explicit override"
            actual, _ = ssb.preview(optics, compute_loss=False)
            with SSB.open(
                str(source),
                voltage_kV=300.0,
                semiangle_mrad=25.0,
                scan_sampling_A=(0.3, 0.3),
                detector_sampling_mrad=(3.0, 3.0),
                **geometry,
            ) as reference:
                expected, _ = reference.preview(optics, compute_loss=False)
            np.testing.assert_array_equal(actual, expected)
        assert data.metadata == original
        np.testing.assert_array_equal(data[0, 0].cpu().numpy(), counts[0, 0])


def test_launch_calibration_keeps_saved_defaults_estimates_and_equal_overrides(device):
    """Inspect file values and launch choices without changing recorded calibration."""
    import copy

    from quantem.gpu import SSB, io
    from quantem.gpu.formats.qem.metadata import (
        acquisition_metadata,
        effective_metadata,
    )
    from quantem.gpu.io.metadata_display import display_value

    counts = np.random.default_rng(3).poisson(4, (128, 128, 24, 24)).astype(np.uint16)
    metadata = {"voltage_kV": 200.0, "semiangle_mrad": 25.0,
                "scan_sampling_A": (0.3, 0.3), "scan_detector_rotation_deg": 12.0}
    scientific = acquisition_metadata(counts.shape, metadata)
    scientific["calibration_overrides"] = {
        "electron_source/accelerating_voltage": {
            "value": 300.0, "unit": "kV", "provenance": "user_override",
            "evidence": "microscope calibration",
        },
    }
    metadata = {**effective_metadata(metadata, scientific), "scientific_metadata": scientific}
    original = copy.deepcopy(metadata)
    data = io.Dataset4dstemGPU(counts, metadata)
    geometry = {"backend": device.backend, "bright_field_center_px": (12., 12.),
                "bright_field_radius_px": 6.5}
    with SSB(data, **geometry) as ssb:
        table = ssb.calibration()
        assert table.loc["Accelerating voltage", "File value"] == 300.0
        assert table.loc["Accelerating voltage", "Source"] == "Overridden"
        assert table.loc["Scan-detector rotation", "File value"] == 12.0
        assert table.loc["Scan-detector rotation", "Value used"] == 12.0
        assert table.loc["Scan-detector rotation", "Source"] == "Recorded"
        assert table.loc["Rotation flipped", "Value used"] == "no"
        assert table.loc["Detector step (row, col)", "Value used"] == "Pending probe fit"
        assert ssb._compute is None
        ssb.preview({"C10_nm": 8., "C12_nm": 0., "phi12_deg": 0.}, compute_loss=False)
        table = ssb.calibration()
        assert table.loc["Detector step (row, col)", "Source"] == "Estimated"
        # the table rounds for reading (six significant figures); the session keeps the full value
        assert display_value(ssb.detector_sampling_mrad) == table.loc["Detector step (row, col)", "Value used"]
        prepared = ssb._compute
        ssb.preview({"C10_nm": 9., "C12_nm": 0., "phi12_deg": 0.}, compute_loss=False)
        assert ssb._compute is prepared
        assert display_value(ssb.detector_sampling_mrad) == table.loc["Detector step (row, col)", "Value used"]
        assert all(value > 0 for value in table.loc["Detector step (row, col)", "Value used"])
        ssb.set_rotation(9.0)
        assert ssb.calibration().loc["Scan-detector rotation", "Source"] == "Modified in session"
    with SSB(data, voltage_kV=300.0, scan_detector_rotation_deg=0.0, **geometry) as ssb:
        table = ssb.calibration()
        assert table.loc["Accelerating voltage", "Source"] == "Explicit override"
        assert table.loc["Scan-detector rotation", "Source"] == "Explicit override"
    assert data.metadata == original
