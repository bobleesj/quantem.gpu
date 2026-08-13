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


def test_ssb_operations_share_saved_result_controls() -> None:
    """C1b2: fit and fixed-probe reconstruction share persistence vocabulary."""

    from quantem.gpu import SSB

    for operation in (SSB.fit, SSB.reconstruct):
        signature = inspect.signature(operation)
        assert signature.parameters["save_to"].default is None
        assert signature.parameters["force"].default is False


def test_ssb_series_uses_inclusive_human_frame_bounds() -> None:
    """Series selection should not expose Python's half-open range contract."""
    from quantem.gpu import SSB

    parameters = inspect.signature(SSB.reconstruct_series).parameters

    assert "first_frame" in parameters
    assert "last_frame" in parameters
    assert "frames" not in parameters
    assert "screening_directory" not in parameters
    assert "fixed_probe" not in parameters
    assert parameters["probe_reference_frame"].default is None


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

    monkeypatch.setattr(workflow, "resolve", lambda requested: "cpu")

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


def test_reconstruct_series_reuses_only_exact_results(
    tmp_path,
    capsys,
    monkeypatch,
) -> None:
    """The series API should route reuse through exact operation validation."""
    import json

    import numpy as np

    from quantem.gpu import SSB, SSBResult

    raw = tmp_path / "raw"
    screen = raw / "quantem" / "screen"
    raw.mkdir()
    screen.mkdir(parents=True)
    reference = "91mm_30mrad_151"
    aberrations = {"C10": -88.0, "C12": 53.0, "phi12": -0.78}
    for index in range(1, 4):
        (raw / f"91mm_30mrad_{150 + index}_master.h5").write_bytes(b"raw")
        dataset = screen / f"91mm_30mrad_{150 + index}"
        dataset.mkdir()
        settings = {
            "voltage_kV": 80.0,
            "semiangle_mrad": 30.0,
            "scan_sampling_A": 0.2,
            "rotation_angle_deg": 173.0,
            "bf_radius": 55.0,
            "aberrations": aberrations,
            "loss": 0.01 * index,
        }
        (dataset / "config.json").write_text(
            json.dumps({"computed": {"ssb": settings}})
        )
        np.save(dataset / "ssb_phase.npy", np.full((8, 9), 2.5))
        np.save(dataset / "ssb_phase_locked.npy", np.full((8, 9), 2.5))
        np.save(dataset / "bf.npy", np.full((8, 9), 20 + index))
        np.save(dataset / "df.npy", np.full((8, 9), 30 + index))
        (dataset / "ssb_phase_locked_meta.json").write_text(
            json.dumps({"ref": reference, "loss": 0.1 * index})
        )

    class FakeSession:
        def __init__(self, source):
            self.index = int(Path(source).stem.rsplit("_", 2)[-2]) - 150

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def _result(self, phase):
            return SSBResult(
                object_wave=np.exp(
                    1j * np.full((8, 9), phase, dtype=np.float32)
                ).astype(np.complex64),
                backend="cuda",
                aberrations=aberrations,
                rotation_angle_deg=173.0,
                loss=0.01 * self.index,
                reused=True,
            )

        def fit(self, **kwargs):
            assert kwargs["save_to"].name == "ssb-fit"
            return self._result(0.1 * self.index)

        def reconstruct(self, fitted, **kwargs):
            assert fitted == aberrations
            assert kwargs["save_to"].name == "ssb-locked"
            return self._result(0.3 + 0.1 * self.index)

    monkeypatch.setattr(
        SSB,
        "open",
        classmethod(lambda cls, source, **kwargs: FakeSession(source)),
    )

    independent = SSB.reconstruct_series(
        raw,
        first_frame=151,
        last_frame=153,
        progress=False,
    )
    fixed_probe_result = SSB.reconstruct_series(
        raw,
        first_frame=151,
        last_frame=153,
        probe_reference_frame=151,
    )
    assert "Checking exact saved SSB results" in capsys.readouterr().out

    np.testing.assert_allclose(independent.phase[:, 0, 0], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(fixed_probe_result.phase[:, 0, 0], [0.4, 0.5, 0.6])
    np.testing.assert_array_equal(
        fixed_probe_result.bright_field[:, 0, 0],
        [21, 22, 23],
    )
    np.testing.assert_array_equal(
        fixed_probe_result.dark_field[:, 0, 0],
        [31, 32, 33],
    )
    assert fixed_probe_result.frames == (151, 152, 153)
    assert fixed_probe_result.datasets == (
        "91mm_30mrad_151",
        "91mm_30mrad_152",
        "91mm_30mrad_153",
    )
    assert not hasattr(fixed_probe_result, "fixed_probe")
    assert fixed_probe_result.probe_reference_frame == 151
    assert fixed_probe_result.probe_reference_dataset == reference
    assert [row["result"] for row in independent.records] == [
        "saved",
        "saved",
        "saved",
    ]
    assert [row["result"] for row in fixed_probe_result.records] == [
        "saved",
        "saved",
        "saved",
    ]
    independent_metadata = independent.metadata().data.set_index("Setting")["Value"]
    assert independent_metadata["Probe mode"] == "independent fits"
    assert "Probe reference frame" not in independent_metadata
    assert "Probe reference dataset" not in independent_metadata
    metadata = fixed_probe_result.metadata().data.set_index("Setting")["Value"]
    assert metadata["Requested backend"] == "auto"
    assert metadata["First frame"] == 151
    assert metadata["Last frame"] == 153
    assert metadata["Frame count"] == 3
    assert metadata["First dataset"] == "91mm_30mrad_151"
    assert metadata["Last dataset"] == "91mm_30mrad_153"
    assert metadata["Probe mode"] == "fixed probe"
    assert metadata["Probe reference frame"] == 151
    assert metadata["Probe reference dataset"] == reference
    assert metadata["Trials"] == 200
    assert metadata["Refinement"] == "nelder-mead"
    assert metadata["Results reused"] == 3
    assert fixed_probe_result.alignment == {
        "normalization": "median_mad",
        "pad_fraction": 3.0 / 32.0,
        "upsample_factor": 50,
        "running_avg_frames": 12.0,
    }


def test_reconstruct_series_rejects_reversed_frame_bounds(tmp_path) -> None:
    """Inclusive frame bounds should fail clearly when reversed."""
    import pytest

    from quantem.gpu import SSB

    raw = tmp_path / "raw"
    screen = raw / "quantem" / "screen"
    raw.mkdir()
    screen.mkdir(parents=True)

    with pytest.raises(ValueError, match="first_frame must be at most last_frame"):
        SSB.reconstruct_series(
            raw,
            first_frame=3,
            last_frame=1,
            progress=False,
        )


def test_series_discovery_keeps_raw_acquisitions_after_partial_screening(
    tmp_path,
) -> None:
    """One saved screen result must not hide later raw acquisitions."""
    from quantem.gpu.ssb.workflow import _series_paths

    raw = tmp_path / "raw"
    raw.mkdir()
    for frame in (52, 53, 54):
        (raw / f"scan_{frame}_master.h5").touch()
    screened = raw / "quantem" / "screen" / "scan_52"
    screened.mkdir(parents=True)
    (screened / "config.json").write_text("{}", encoding="utf-8")

    _, paths = _series_paths(raw)

    assert tuple(path.name for path in paths) == ("scan_52", "scan_53", "scan_54")


def test_series_bounds_are_acquisition_ids_not_list_positions(
    tmp_path,
    monkeypatch,
) -> None:
    """Non-consecutive endpoint IDs should include intervening acquisitions."""
    import numpy as np

    from quantem.gpu import SSB, SSBResult

    raw = tmp_path / "raw"
    screen = raw / "quantem" / "screen"
    screen.mkdir(parents=True)
    for index, frame in enumerate((100, 250, 900), start=1):
        (raw / f"scan_{frame}_master.h5").write_bytes(b"raw")
        dataset = screen / f"scan_{frame}"
        dataset.mkdir()
        (dataset / "config.json").write_text(
            json.dumps(
                {
                    "computed": {
                        "ssb": {
                            "voltage_kV": 300.0,
                            "semiangle_mrad": 30.0,
                            "scan_sampling_A": 0.5,
                            "rotation_angle_deg": 0.0,
                            "aberrations": {
                                "C10": 1.0,
                                "C12": 2.0,
                                "phi12": 0.3,
                            },
                            "loss": 0.1,
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        np.save(dataset / "ssb_phase.npy", np.full((2, 3), index))
        np.save(dataset / "bf.npy", np.zeros((2, 3), dtype=np.float32))
        np.save(dataset / "df.npy", np.zeros((2, 3), dtype=np.float32))

    class FakeSession:
        def __init__(self, source):
            frame = int(Path(source).stem.rsplit("_", 2)[-2])
            self.index = {100: 1, 250: 2, 900: 3}[frame]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def fit(self, **kwargs):
            return SSBResult(
                object_wave=np.exp(
                    1j * np.full((2, 3), 0.1 * self.index, dtype=np.float32)
                ).astype(np.complex64),
                backend="cuda",
                aberrations={"C10": 1.0, "C12": 2.0, "phi12": 0.3},
                rotation_angle_deg=0.0,
                loss=0.1,
            )

    monkeypatch.setattr(
        SSB,
        "open",
        classmethod(lambda cls, source, **kwargs: FakeSession(source)),
    )

    result = SSB.reconstruct_series(
        raw,
        first_frame=100,
        last_frame=900,
        progress=False,
    )

    assert result.frames == (100, 250, 900)
    np.testing.assert_allclose(result.phase[:, 0, 0], (0.1, 0.2, 0.3))


def test_reconstruct_series_initializes_missing_live_results(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    """A raw series should run without a user-facing screening-directory step."""
    from types import SimpleNamespace

    import numpy as np

    from quantem.gpu import SSB, SSBResult
    from quantem.gpu import screening

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "scan_1_master.h5").write_bytes(b"raw")
    (raw / "dataset.yaml").write_text(
        """
microscope:
  voltage_kV: 300
  semiangle_mrad: 30
  screen_rotation_deg: 12
calibrations:
  mag: {scan_sampling_A: 0.5}
files:
  1: {mag: mag}
""".strip()
    )

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def fit(self, **kwargs):
            return SSBResult(
                object_wave=np.full((3, 4), 1 + 1j, dtype=np.complex64),
                backend="cuda",
                aberrations={"C10": 1.0, "C12": 2.0, "phi12": 0.3},
                rotation_angle_deg=12.0,
                loss=0.25,
                bf_radius=8.0,
            )

    monkeypatch.setattr(
        SSB,
        "open",
        classmethod(lambda cls, source, **kwargs: FakeSession()),
    )
    zeros = np.zeros((3, 4), dtype=np.float32)
    monkeypatch.setattr(
        screening,
        "prepare",
        lambda *args, **kwargs: SimpleNamespace(
            mean_dp=np.zeros((5, 6), dtype=np.float32),
            bright_field=np.ones((3, 4), dtype=np.float32),
            dark_field=np.full((3, 4), 2.0, dtype=np.float32),
            com_row=zeros,
            com_col=zeros,
            dpc_phase=zeros,
        ),
    )

    result = SSB.reconstruct_series(
        raw,
        first_frame=1,
        last_frame=1,
    )
    assert (
        "Checking exact saved SSB results"
        in capsys.readouterr().out
    )

    assert result.phase.shape == (1, 3, 4)
    assert np.all(result.bright_field == 1.0)
    assert np.all(result.dark_field == 2.0)
    assert result.master_names == ("scan_1_master.h5",)
    assert result.probe_reference_frame is None
    assert result.records[0]["result"] == "computed"
    config = raw / "quantem" / "screen" / "scan_1" / "config.json"
    assert config.is_file()
    assert (config.parent / "bf.npy").is_file()
    assert (config.parent / "df.npy").is_file()


def test_ssb_series_show_starts_with_companion_views_hidden(monkeypatch) -> None:
    """Show3D should retain BF/DF views while opening on SSB phase."""
    import sys
    from types import SimpleNamespace

    import numpy as np

    from quantem.gpu.ssb.results import SSBSeriesResult

    captured = {}

    def show3d(*arrays, **kwargs):
        captured["arrays"] = arrays
        captured["kwargs"] = kwargs
        return "viewer"

    monkeypatch.setitem(
        sys.modules,
        "quantem.widget",
        SimpleNamespace(Show3D=show3d),
    )
    stack = np.zeros((2, 3, 4), dtype=np.float32)
    result = SSBSeriesResult(
        phase=stack,
        bright_field=stack + 1,
        dark_field=stack + 2,
        frames=(1, 2),
        datasets=("a", "b"),
        master_names=("a_master.h5", "b_master.h5"),
        probe_reference_frame=1,
        probe_reference_dataset="a",
        records=(),
        source_directory=Path("raw"),
        results_directory=Path("raw/quantem/screen"),
        requested_backend="cuda",
        trials=200,
        refinement="nelder-mead",
    )

    assert result.show() == "viewer"
    assert len(captured["arrays"]) == 3
    assert captured["kwargs"]["panel_titles"] == (
        "Fixed probe from F1 | a",
        "Bright field",
        "Dark field",
    )
    assert captured["kwargs"]["hidden_panels"] == (
        "Bright field",
        "Dark field",
    )


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
