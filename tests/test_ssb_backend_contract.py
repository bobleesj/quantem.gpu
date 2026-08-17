from __future__ import annotations

import builtins
import importlib
import inspect
from types import SimpleNamespace

import numpy as np

from tests.ssb_precision import PRECISION


def test_mps_ssb_chunked_source_never_imports_torch(monkeypatch) -> None:
    """Packaged local MPS wraps exact chunks without the omitted Torch runtime."""
    from quantem.gpu.detector.compute.mps import kernels as mps_kernels
    from quantem.gpu.ssb.compute.mps.engine import _as_chunked_frames

    class FakeMPSChunked4DSTEM:
        pass

    class FakeMetalVirtualImage:
        row_prefix_enabled = False

        def __init__(self, chunks, row_prefix=False):
            self.chunks = chunks
            self.row_prefix_enabled = row_prefix

        def gather_columns_float32(self, rows, cols, *, out=None):
            values = np.concatenate(
                [np.asarray(chunk)[:, rows, cols] for chunk in self.chunks],
                axis=0,
            ).T.astype(np.float32, copy=False)
            if out is None:
                return values
            out[...] = values
            return out

    monkeypatch.setattr(mps_kernels, "MetalVirtualImage", FakeMetalVirtualImage)
    native_uint16 = np.asarray(
        [
            [[0, 53], [7, 11]],
            [[13, 17], [19, 23]],
        ],
        dtype=np.uint16,
    )
    assert int(native_uint16.max()) == 53
    assert int(np.count_nonzero(native_uint16 > 255)) == 0
    working = native_uint16.astype(np.uint8)
    np.testing.assert_array_equal(working.astype(np.uint16), native_uint16)
    source = FakeMPSChunked4DSTEM()
    source.chunks = [working]
    source.metadata = {
        "scan_shape": (1, 2),
        "source_dtype": "uint16",
        "dtype": "uint8",
    }
    source.det_bin = 1
    source.fast_chunks = None
    source.fast_det_bin = None
    source.detector_sum = None
    source.row_prefix = False
    mps_io = importlib.import_module("quantem.gpu.io.backends.mps")
    monkeypatch.setattr(
        mps_io,
        "MPSChunked4DSTEM",
        FakeMPSChunked4DSTEM,
        raising=False,
    )
    imported = builtins.__import__

    def reject_torch(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("the packaged local-MPS SSB path imported torch")
        return imported(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_torch)
    frames = _as_chunked_frames(source)
    gathered = frames.columns_float32(
        np.asarray([0, 1], dtype=np.int32),
        np.asarray([1, 0], dtype=np.int32),
    )

    assert frames.dtype == np.dtype(np.uint8)
    assert frames.metadata["source_dtype"] == "uint16"
    assert frames.metadata["dtype"] == "uint8"
    assert frames.torch_dtype is None
    assert isinstance(frames[0], np.ndarray)
    np.testing.assert_array_equal(frames[0], working[0])
    np.testing.assert_array_equal(
        gathered,
        np.asarray([[53, 17], [7, 19]], dtype=np.float32),
    )


class _Backend:
    backend = "mps"
    scan_shape = (4, 4)
    detector_shape = (3, 3)
    num_bf = 1

    precision = PRECISION

    def fit(self, **kwargs):
        raise NotImplementedError

    def reconstruct_result(self, aberrations, *, compute_loss=True):
        del compute_loss
        raise NotImplementedError

    def preview(self, aberrations, **kwargs):
        del kwargs
        return self.reconstruct(
            aberrations["C10"], aberrations["C12"], aberrations["phi12"]
        ), None

    def cache_rotation(self, rotation_rad: float) -> None:
        self.rotation_rad = float(rotation_rad)

    def reconstruct(self, c10: float, c12: float, phi12: float):
        return np.zeros(self.scan_shape, dtype=np.float32)

    def reconstruct_with_loss(self, c10: float, c12: float, phi12: float):
        return self.reconstruct(c10, c12, phi12), 0.0

    def reconstruct_full(self, mags_m, angles_rad):
        return np.zeros(self.scan_shape, dtype=np.float32)

    def reconstruct_full_with_loss(self, mags_m, angles_rad):
        return self.reconstruct_full(mags_m, angles_rad), 0.0

    @staticmethod
    def phase_to_numpy(phase) -> np.ndarray:
        return np.asarray(phase, dtype=np.float32)

    def browser_state(self):
        raise NotImplementedError

    def preview_context(self, num_bf: int):
        return None

    def export_brightfield(self, data, path_stem):
        return None

    def close(self) -> None:
        pass


def _selection():
    from quantem.gpu.ssb.bf_selector import BrightfieldDisk

    return BrightfieldDisk(
        rows=np.asarray([1], dtype=np.int32),
        cols=np.asarray([1], dtype=np.int32),
        center_row_col=(1.0, 1.0),
        radius_px=1.0,
        detected_radius_px=1.0,
        detector_shape=(3, 3),
    )


def test_public_ssb_has_one_backend_neutral_signature() -> None:
    """C1: one stateful API owns explicit units and backend choice."""
    from quantem.gpu import SSB

    constructor = inspect.signature(SSB)
    shared = {
        "backend",
        "voltage_kV",
        "semiangle_mrad",
        "scan_sampling_A",
        "det_sampling",
        "aberrations",
        "rotation_angle_deg",
        "bf_radius",
    }
    assert shared <= constructor.parameters.keys()
    assert constructor.parameters["backend"].default == "auto"
    reconstruct = inspect.signature(SSB.reconstruct)
    assert reconstruct.parameters["compute_loss"].default is True


def test_mps_reconstruct_honors_compute_loss(monkeypatch) -> None:
    """MPS must reuse retained evidence and compute loss only when requested."""
    from quantem.gpu.ssb.compute.mps import backend as mps_backend

    requested = []
    prepared = object()

    def object_wave(candidate, **_kwargs):
        assert candidate is prepared
        return np.ones((2, 2), dtype=np.complex64)

    def reconstruct(candidate, **kwargs):
        assert candidate is prepared
        requested.append(kwargs["compute_loss"])
        return None, 0.25, np.zeros((2, 2), dtype=np.float32)

    class FakeMlx:
        @staticmethod
        def clear_cache():
            requested.append("clear")

    monkeypatch.setattr(mps_backend, "_object_fourier_sum_dynamic", object_wave)
    monkeypatch.setattr(mps_backend, "_reconstruct_prepared", reconstruct)
    monkeypatch.setattr(mps_backend, "_require_mlx", lambda: FakeMlx)
    backend = mps_backend.MpsSSBBackend.__new__(mps_backend.MpsSSBBackend)
    backend._prepared = prepared
    backend._chunk_bf = 4
    backend._phase_chunk_bf = 2
    backend._voltage_kV = 300.0
    backend._semiangle_mrad = 30.0
    backend._scan_sampling = (1.0, 1.0)
    backend._rotation_angle_deg = 0.0
    backend._selection = SimpleNamespace(
        size=3,
        center_row_col=(1.0, 1.0),
        radius_px=1.0,
        detected_radius_px=1.0,
    )

    aberrations = {"C10": 0.0, "C12": 0.0, "phi12": 0.0}
    without_loss = backend.reconstruct_result(aberrations, compute_loss=False)
    with_loss = backend.reconstruct_result(aberrations, compute_loss=True)
    assert without_loss.loss is None
    assert with_loss.loss == 0.25
    assert requested == ["clear", True, "clear"]


def test_mps_rotation_retargets_geometry_without_repreparing_source(
    monkeypatch,
) -> None:
    """Changed rotation must retain the existing source FFT evidence object."""
    from quantem.gpu.ssb.compute.mps import backend as mps_backend

    prepared = object()
    calls = []
    monkeypatch.setattr(
        mps_backend,
        "_retarget_prepared_rotation",
        lambda candidate, **kwargs: calls.append((candidate, kwargs)),
    )

    class FakeMlx:
        @staticmethod
        def clear_cache():
            calls.append("clear")

    monkeypatch.setattr(mps_backend, "_require_mlx", lambda: FakeMlx)
    backend = mps_backend.MpsSSBBackend.__new__(mps_backend.MpsSSBBackend)
    backend._prepared = prepared
    backend._rotation_angle_deg = 0.0
    backend._selection = object()
    backend._fit_preview_phase = np.ones((1,), dtype=np.float32)
    backend._fit_preview_loss = 1.0
    backend._fit_preview_aberrations = {"C10": 1.0}

    backend.cache_rotation(np.deg2rad(3.0))

    assert backend._prepared is prepared
    assert calls[0][0] is prepared
    assert calls[0][1]["selection"] is backend._selection
    assert np.isclose(calls[0][1]["rotation_angle_deg"], 3.0)
    assert calls[1] == "clear"
    assert backend._fit_preview_phase is None


def test_mps_rotation_geometry_keeps_exact_source_fft_identity() -> None:
    """Rotation changes k geometry while retaining the exact prepared G(q,k)."""
    from quantem.gpu.ssb.compute.mps.engine import _retarget_prepared_rotation

    class FakeMlx:
        float32 = np.float32

        array = staticmethod(np.asarray)
        sqrt = staticmethod(np.sqrt)
        where = staticmethod(np.where)
        clip = staticmethod(np.clip)

        @staticmethod
        def eval(*_values):
            pass

    source_fft = object()
    prepared = SimpleNamespace(
        mx=FakeMlx,
        g_qk=source_fft,
        qx=np.zeros((1, 2, 1), dtype=np.float32),
        qy=np.zeros((1, 1, 2), dtype=np.float32),
        wavelength=1.0,
        semiangle_rad=1.0,
        ang_y_rad=1.0,
        ang_x_rad=1.0,
        alpha_k2=None,
        bf_storage_indices_np=None,
    )
    selection = SimpleNamespace(
        rows=np.asarray([2], dtype=np.int32),
        cols=np.asarray([1], dtype=np.int32),
        center_row_col=(1.0, 1.0),
    )

    _retarget_prepared_rotation(prepared, selection=selection, rotation_angle_deg=90.0)

    assert prepared.g_qk is source_fft
    np.testing.assert_allclose(prepared.kx_np, [0.0], atol=1e-7)
    np.testing.assert_allclose(prepared.ky_np, [1.0], atol=1e-7)


def test_default_aberrations_remain_distinct_from_an_explicit_zero_start(
    monkeypatch,
) -> None:
    """Backends may apply their canonical fit start only when none was given."""
    from quantem.gpu.ssb import workflow

    monkeypatch.setattr(workflow, "_resolve_backend", lambda _backend: "mps")
    data = np.zeros((2, 2, 2, 2), dtype=np.uint8)
    common = {
        "backend": "mps",
        "voltage_kV": 300.0,
        "semiangle_mrad": 30.0,
        "scan_sampling_A": 1.0,
    }

    default = workflow.SSB.from_array(data, **common)
    explicit = workflow.SSB.from_array(
        data,
        aberrations={"C10": 0.0, "C12": 0.0, "phi12": 0.0},
        **common,
    )

    assert default.aberrations == explicit.aberrations
    assert not default._aberrations_explicit
    assert explicit._aberrations_explicit


def test_ssb_result_owns_optimization_and_reconstruction_metadata() -> None:
    """C2: one result owns both optimization and object-wave output."""
    from quantem.gpu import SSBResult

    selection = _selection()
    result = SSBResult(
        object_wave=np.ones((2, 2), dtype=np.complex64),
        backend="mps",
        aberrations={"C10": 1.0, "C12": 2.0, "phi12": 0.1},
        rotation_angle_deg=3.0,
        loss=0.25,
        num_bf=selection.size,
        bf_center=selection.center_row_col,
        bf_radius=selection.radius_px,
        detected_bf_radius=selection.detected_radius_px,
        n_trials=200,
    )

    assert result.num_bf == 1
    assert result.bf_center == (1.0, 1.0)
    assert result.bf_radius == 1.0
    assert result.phase.dtype == np.float32


def test_export_state_separates_logical_and_aperture_active_bf_counts() -> None:
    """Logical BF membership and nonzero probe support are distinct evidence."""
    from quantem.gpu.ssb.bf_selector import BrightfieldDisk
    from quantem.gpu.ssb.compute.protocol import SSBExportState

    selection = BrightfieldDisk(
        rows=np.asarray([0, 1, 2], dtype=np.int32),
        cols=np.asarray([1, 1, 1], dtype=np.int32),
        center_row_col=(1.0, 1.0),
        radius_px=1.1,
        detected_radius_px=1.1,
        detector_shape=(3, 3),
    )
    values = np.ones(3, dtype=np.float32)
    state = SSBExportState(
        backend="mps",
        scan_shape=(2, 2),
        brightfield=selection,
        kx_bf=values,
        ky_bf=values,
        qx_1d=np.ones(2, dtype=np.float32),
        qy_1d=np.ones(2, dtype=np.float32),
        aperture_k=np.asarray([1.0, 0.0, 0.25], dtype=np.float32),
        alpha_k2=values,
        cos2phi_k=values,
        sin2phi_k=values,
        wavelength_A=0.02,
        semiangle_rad=0.03,
        angular_sampling_rad=(0.001, 0.001),
        sampling_A=(0.264, 0.264),
        dc_value=0j,
    )

    assert state.num_bf == 3
    assert state.active_num_bf == 2


def test_float32_precision_contract_is_strict_and_backend_neutral() -> None:
    """C3: every backend advertises immutable float32 storage."""
    from quantem.gpu.ssb.compute import SSBPrecision

    precision = PRECISION
    assert isinstance(precision, SSBPrecision)
    assert precision.real_dtype == "float32"
    assert precision.complex_dtype == "complex64"
    assert not hasattr(precision, "phase_rtol")


def test_compute_backend_contract_is_runtime_checkable() -> None:
    """C4: UI adapters consume the common compute protocol."""
    from quantem.gpu.ssb.compute import SSBProtocol

    backend = _Backend()

    assert isinstance(backend, SSBProtocol)


def test_backend_specific_fit_entry_points_are_not_public() -> None:
    """C5: callers cannot select a different scientific API by backend."""
    import quantem.gpu as qg

    assert not any(
        name.lower().endswith(("_cuda", "_mps", "_webgpu")) for name in qg.__all__
    )


def test_public_ssb_exports_are_backend_neutral() -> None:
    """C6: root SSB exports contain no transport or implementation types."""

    from quantem.gpu import ssb

    assert set(ssb.__all__) == {
        "SSB",
        "SSBResult",
    }


def test_preview_context_is_consumed_by_public_workflow() -> None:
    """Live previews never need backend-private context handling."""

    from quantem.gpu import SSB

    events: list[str] = []

    class Context:
        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, *_args):
            events.append("exit")

    class Session(SSB):
        @property
        def _backend_protocol(self):
            return self._test_backend

    session = Session.__new__(Session)
    session._test_backend = _Backend()
    phase, loss = session.preview(
        {"C10": 0.0, "C12": 0.0, "phi12": 0.0},
        context=Context(),
    )

    assert events == ["enter", "exit"]
    assert phase.dtype == np.float32
    assert loss is None
