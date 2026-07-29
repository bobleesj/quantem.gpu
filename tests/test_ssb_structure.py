"""Architecture tests for the backend-neutral SSB package."""
from __future__ import annotations

import inspect
import json
from pathlib import Path


def test_public_ssb_api_has_one_workflow() -> None:
    """C1: scientist API, expect one session with explicit operations."""

    from quantem.gpu import SSB

    for name in (
        "open",
        "from_array",
        "fit",
        "reconstruct",
        "preview",
        "close",
    ):
        assert hasattr(SSB, name)
    for stale_name in (
        "load_evidence",
        "run",
        "optimize_aberrations",
        "compute_backend",
        "free",
        "write_evidence",
    ):
        assert not hasattr(SSB, stale_name)


def test_ssb_open_keeps_detector_storage_dtype_explicit() -> None:
    """C1b: raw HDF5 loading defaults native and permits an explicit dtype."""

    from quantem.gpu import SSB

    signature = inspect.signature(SSB.open)
    assert signature.parameters["dtype"].default is None


def test_every_public_ssb_entry_point_defaults_to_full_bright_field() -> None:
    """C1c: all scientist entry points use the complete automatic BF disk."""

    from quantem.gpu import SSB

    for entry_point in (SSB, SSB.open, SSB.from_array):
        signature = inspect.signature(entry_point)
        assert signature.parameters["bf_intensity_threshold"].default == 0.0


def test_ssb_auto_backend_never_falls_back_to_cpu(monkeypatch) -> None:
    """C1d: unavailable GPU execution fails instead of changing science."""

    import pytest

    from quantem.gpu.ssb import workflow

    monkeypatch.setattr(workflow, "select_device", lambda requested: "cpu")

    with pytest.raises(RuntimeError, match="CPU is test-only"):
        workflow._resolve_backend("auto")


def test_cuda_brightfield_detection_stays_on_gpu() -> None:
    """C1e: CUDA probe fitting must not use the host detector reduction."""

    backend = Path(
        "src/quantem/gpu/ssb/compute/cuda/backend.py"
    ).read_text(encoding="utf-8")
    source = backend.split("def _compute_bf_mask(", 1)[1].split(
        "def _compute_bf_mask_from_mean_dp(", 1
    )[0]

    assert "from quantem.gpu.detector import dp_mean" not in source
    assert "frames.sum(axis=0" in source
    assert "dtype=sum_dtype" in source


def test_reference_512_contract_matches_public_optimizer_defaults() -> None:
    """C1f: the canonical real-data run pins defaults and exact results."""

    from quantem.gpu import SSB
    from quantem.gpu.ssb.compute import SSBPrecision

    contract = json.loads(
        Path("tests/fixtures/ssb_reference_512_mps.json").read_text(
            encoding="utf-8"
        )
    )
    signature = inspect.signature(SSB.fit)
    optimizer = contract["optimizer"]
    precision = SSBPrecision()

    assert signature.parameters["trials"].default == optimizer["trials"]
    assert signature.parameters["refinement"].default == optimizer["refinement"]
    assert signature.parameters["seed"].default == optimizer["seed"]
    assert contract["brightfield"]["policy"] == "automatic_full_disk"
    assert contract["source"]["scan_shape"] == [512, 512]
    assert contract["source"]["detector_shape"] == [192, 192]
    assert contract["hardware"] == {
        "system_class": "reference Apple Silicon laptop",
        "model": "MacBook Pro Mac17,2",
        "chip": "Apple M5",
        "gpu_cores": 10,
        "unified_memory_gb": 24,
    }
    assert contract["performance"]["retained"]["community_signoff"][
        "recorded_evaluations"
    ] == optimizer["recorded_evaluations"]
    assert precision.real_dtype == contract["precision"]["real_dtype"]
    assert precision.complex_dtype == contract["precision"]["complex_dtype"]


def test_removed_ssb_modules_and_result_types_stay_absent() -> None:
    """C1g: deleted compatibility modules must not return to the package."""

    root = Path("src/quantem/gpu/ssb")
    stale_modules = (
        "api.py",
        "backend.py",
        "batch_optuna.py",
        "engine.py",
        "fft_common.py",
        "fft128.py",
        "fft256.py",
        "fft512.py",
        "fft1024.py",
        "mps.py",
        "mps_engine.py",
        "integration.py",
        "preprocess.py",
        "reconstruction.py",
    )
    assert not any((root / name).exists() for name in stale_modules)

    import quantem.gpu.ssb as ssb

    assert ssb.__all__ == ["SSB", "SSBResult"]
    assert not hasattr(ssb, "AberrationFit")
    assert not hasattr(ssb, "PhaseEvaluation")


def test_every_backend_exposes_the_same_size_modules() -> None:
    """C2: backend tree, expect explicit 128/256/512/1024 modules."""

    root = Path("src/quantem/gpu/ssb/compute")
    suffix = {"cuda": ".py", "mps": ".py", "webgpu": ".ts"}
    for backend, extension in suffix.items():
        kernels = root / backend / "kernels"
        for size in (128, 256, 512, 1024):
            assert (kernels / f"fft{size}{extension}").is_file()


def test_backend_ownership_is_symmetric() -> None:
    """C2b: each backend owns one backend, optimizer, and kernel tree."""

    root = Path("src/quantem/gpu/ssb/compute")
    for backend, extension in {
        "cuda": ".py",
        "mps": ".py",
        "webgpu": ".ts",
    }.items():
        folder = root / backend
        assert (folder / f"backend{extension}").is_file()
        assert (folder / f"optimizer{extension}").is_file()
        assert (folder / "kernels").is_dir()


def test_mps_registry_names_all_native_scan_sizes() -> None:
    """C3: MPS registry, expect deterministic native-size dispatch."""

    from quantem.gpu.ssb.compute.mps.kernels import MPS_FFT_CONFIGS

    assert tuple(MPS_FFT_CONFIGS) == (128, 256, 512, 1024)
    assert all(config.size == size for size, config in MPS_FFT_CONFIGS.items())


def test_mps_source_companion_precedes_historical_calibration(tmp_path) -> None:
    """C4: source-local exact columns, expect authoritative source ordering."""

    from quantem.gpu.ssb.workflow import _mps_brightfield_sources

    export = tmp_path / "current"
    source = export / "source" / "scan_master.h5"
    source.parent.mkdir(parents=True)
    source.touch()
    historical = tmp_path / "historical" / "calibration.json"

    candidates = _mps_brightfield_sources(str(source), str(historical))

    assert candidates == (export, source.parent, historical)


def test_mps_fit_calibration_restores_exact_export_brightfield(tmp_path) -> None:
    """C4: fit-only calibration, expect sibling exact BF geometry first."""

    from quantem.gpu.ssb.workflow import _mps_brightfield_sources

    source = tmp_path / "raw" / "scan_master.h5"
    source.parent.mkdir()
    source.touch()
    fit_calibration = tmp_path / "result" / "calibration.json"
    exact_calibration = fit_calibration.parent / "snapshots" / "cal.json"
    exact_calibration.parent.mkdir(parents=True)
    exact_calibration.write_text("{}", encoding="utf-8")

    candidates = _mps_brightfield_sources(str(source), str(fit_calibration))

    assert candidates == (
        exact_calibration,
        source.parent,
        fit_calibration,
    )


def test_webgpu_compute_has_no_widget_owned_names() -> None:
    """C5: browser compute, expect SSB ownership rather than UI ownership."""

    source = Path(
        "src/quantem/gpu/ssb/compute/webgpu/backend.ts"
    ).read_text(encoding="utf-8")

    assert "export class WebGPUSSBBackend" in source
    assert "implements SSBProtocol" in source
    assert "async fit" in source
    assert "ShowPtycho" not in source
    assert "showPtycho" not in source
    assert "showptycho" not in source
