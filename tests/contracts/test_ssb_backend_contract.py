import inspect
import math
from types import SimpleNamespace

import numpy as np
import pytest

from tests.parity.ssb_precision import PRECISION


class _Backend:
    backend = "mps"
    scan_shape = (4, 4)
    detector_shape = (3, 3)
    num_bright_field_pixels = 1

    precision = PRECISION

    def fit(self, **kwargs):
        raise NotImplementedError

    def cache_rotation(self, scan_detector_rotation_rad: float) -> None:
        self.scan_detector_rotation_rad = float(scan_detector_rotation_rad)

    def browser_state(self):
        raise NotImplementedError

    def preview_upsampled(self, aberrations, **kwargs):
        assert "phase_estimator" not in kwargs
        return np.zeros(self.scan_shape, dtype=np.float32), None

    def fit_sample(self, **options):
        raise NotImplementedError

    def preview_context(self, num_bright_field_pixels: int):
        return None

    def export_brightfield(self, data, path_stem):
        return None

    def close(self) -> None:
        pass


def _selection():
    from quantem.gpu.direct_ptycho.brightfield import BrightfieldDisk

    return BrightfieldDisk(
        rows=np.asarray([1], dtype=np.int32),
        cols=np.asarray([1], dtype=np.int32),
        bright_field_center_px=(1.0, 1.0),
        bright_field_radius_px=1.0,
        bright_field_edge_radius_px=1.0,
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
        "detector_sampling_mrad",
        "aberrations",
        "scan_detector_rotation_deg",
        "bright_field_radius_px",
    }
    assert shared <= constructor.parameters.keys()
    assert constructor.parameters["backend"].default == "auto"
    reconstruct = inspect.signature(SSB.reconstruct)
    assert reconstruct.parameters["compute_loss"].default is True


def test_mps_preview_honors_compute_loss(monkeypatch) -> None:
    """MPS returns the SSB phase at the output factor and evaluates the native phase variance only when asked."""
    from quantem.gpu.direct_ptycho.mps import backend as mps_backend

    calls = []
    prepared = object()

    def upsampled(candidate, **kwargs):
        assert candidate is prepared and "phase_estimator" not in kwargs
        calls.append(("phase", kwargs["upsampling_factor"]))
        return np.ones((4, 4), dtype=np.float32)

    def variance(c10, c12, phi12):
        calls.append("loss")
        return np.zeros((2, 2), dtype=np.float32), 0.25

    monkeypatch.setattr(mps_backend, "_reconstruct_upsampled", upsampled)
    backend = mps_backend.MpsSSBBackend.__new__(mps_backend.MpsSSBBackend)
    backend._prepared = prepared
    backend._fit_preview_phase = None
    backend._fit_preview_loss = None
    backend._fit_preview_aberrations = None
    backend.reconstruct_with_loss = variance

    aberrations = {"C10": 0.0, "C12": 0.0, "phi12": 0.0}
    phase, without_loss = backend.preview_upsampled(aberrations, upsampling_factor=2, compute_loss=False)
    _, with_loss = backend.preview_upsampled(aberrations, upsampling_factor=2, compute_loss=True)
    assert phase.shape == (4, 4)
    assert without_loss is None
    assert with_loss == 0.25
    assert calls == [("phase", 2), ("phase", 2), "loss"]


def test_mps_rotation_retargets_geometry_without_repreparing_source(
    monkeypatch,
) -> None:
    """Changed rotation must retain the existing source FFT evidence object."""
    from quantem.gpu.direct_ptycho.mps import backend as mps_backend

    prepared = object()
    calls = []
    monkeypatch.setattr(
        mps_backend,
        "retarget_prepared_rotation",
        lambda candidate, **kwargs: calls.append((candidate, kwargs)),
    )

    class FakeMlx:
        @staticmethod
        def clear_cache():
            calls.append("clear")

    monkeypatch.setattr(mps_backend, "require_mlx", lambda: FakeMlx)
    backend = mps_backend.MpsSSBBackend.__new__(mps_backend.MpsSSBBackend)
    backend._prepared = prepared
    backend._scan_detector_rotation_deg = 0.0
    backend._selection = object()
    backend._fit_preview_phase = np.ones((1,), dtype=np.float32)
    backend._fit_preview_loss = 1.0
    backend._fit_preview_aberrations = {"C10": 1.0}

    backend.cache_rotation(np.deg2rad(3.0))

    assert backend._prepared is prepared
    assert calls[0][0] is prepared
    assert calls[0][1]["selection"] is backend._selection
    assert np.isclose(calls[0][1]["scan_detector_rotation_deg"], 3.0)
    assert calls[1] == "clear"
    assert backend._fit_preview_phase is None


def test_mps_rotation_geometry_keeps_exact_source_fft_identity() -> None:
    """Rotation changes k geometry while retaining the exact prepared G(q,k)."""
    from quantem.gpu.direct_ptycho.mps.prepared import retarget_prepared_rotation

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
        bright_field_center_px=(1.0, 1.0),
    )

    retarget_prepared_rotation(prepared, selection=selection, scan_detector_rotation_deg=90.0)

    assert prepared.g_qk is source_fft
    np.testing.assert_allclose(prepared.kx_np, [0.0], atol=1e-7)
    np.testing.assert_allclose(prepared.ky_np, [1.0], atol=1e-7)


def test_mps_bright_field_frequencies_round_once() -> None:
    """k is the float64 offset, scaling and rotation rounded once to float32, never from a float32-rounded centre."""
    from quantem.gpu.direct_ptycho.mps.prepared import bright_field_frequencies

    rows, cols = (index.ravel().astype(np.int32) for index in np.indices((24, 24)))
    center = (99.68301, 95.12007)
    selection = SimpleNamespace(rows=rows + 88, cols=cols + 84, bright_field_center_px=center)
    reciprocal, angle = 0.75e-3 / 0.0196875, -math.radians(174.0)
    row = (selection.rows.astype(np.float64) - center[0]) * reciprocal
    col = (selection.cols.astype(np.float64) - center[1]) * reciprocal
    expected = (row * math.cos(angle) + col * math.sin(angle), -row * math.sin(angle) + col * math.cos(angle))

    actual = bright_field_frequencies(selection, 0.75e-3, 0.75e-3, 0.0196875, 174.0)

    for value, reference in zip(actual, expected):
        assert value.dtype == np.float32
        np.testing.assert_array_equal(value, reference.astype(np.float32))


def test_default_aberrations_remain_distinct_from_an_explicit_zero_start(
    monkeypatch,
) -> None:
    """Backends may apply their canonical fit start only when none was given."""
    from quantem.gpu.direct_ptycho import workflow

    monkeypatch.setattr(workflow, "resolve", lambda _backend: "mps")
    data = np.zeros((2, 2, 2, 2), dtype=np.uint8)
    common = {
        "backend": "mps",
        "voltage_kV": 300.0,
        "semiangle_mrad": 30.0,
        "scan_sampling_A": (1.0, 1.0),
    }

    default = workflow.SSB(data, **common)
    explicit = workflow.SSB(
        data,
        aberrations={"C10_nm": 0.0, "C12_nm": 0.0, "phi12_deg": 0.0},
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
        aberrations={"C10_nm": 1.0, "C12_nm": 2.0, "phi12_deg": math.degrees(0.1)},
        scan_detector_rotation_deg=3.0,
        loss=0.25,
        num_bright_field_pixels=selection.size,
        bright_field_center_px=selection.bright_field_center_px,
        bright_field_radius_px=selection.bright_field_radius_px,
        bright_field_edge_radius_px=selection.bright_field_edge_radius_px,
        num_trials=200,
    )

    assert result.num_bright_field_pixels == 1
    assert result.bright_field_center_px == (1.0, 1.0)
    assert result.bright_field_radius_px == 1.0
    assert result.phase.dtype == np.float32


def test_export_state_separates_logical_and_aperture_active_bf_counts() -> None:
    """Logical BF membership and nonzero probe support are distinct evidence."""
    from quantem.gpu.direct_ptycho.brightfield import BrightfieldDisk
    from quantem.gpu.direct_ptycho.contract import SSBExportState

    selection = BrightfieldDisk(
        rows=np.asarray([0, 1, 2], dtype=np.int32),
        cols=np.asarray([1, 1, 1], dtype=np.int32),
        bright_field_center_px=(1.0, 1.0),
        bright_field_radius_px=1.1,
        bright_field_edge_radius_px=1.1,
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

    assert state.num_bright_field_pixels == 3


def test_float32_precision_contract_is_strict_and_backend_neutral() -> None:
    """C3: every backend advertises immutable float32 storage."""
    from quantem.gpu.direct_ptycho.contract import SSBPrecision

    precision = PRECISION
    assert isinstance(precision, SSBPrecision)
    assert precision.real_dtype == "float32"
    assert precision.complex_dtype == "complex64"
    assert not hasattr(precision, "phase_rtol")


def test_compute_backend_contract_lists_every_backend_call_of_the_workflow() -> None:
    """``SSB`` reaches its CUDA or MPS backend only through ``SSBProtocol`` members."""
    import ast

    from quantem.gpu.direct_ptycho import workflow
    from quantem.gpu.direct_ptycho.contract import SSBProtocol

    called = set()
    for node in ast.walk(ast.parse(inspect.getsource(workflow))):
        if isinstance(node, ast.Attribute) and (
            (isinstance(node.value, ast.Name) and node.value.id == "backend")
            or (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "_backend")
        ):
            called.add(node.attr)
    members = set(vars(SSBProtocol)) | set(SSBProtocol.__annotations__)
    assert {"preview_upsampled", "fit_sample"} <= called
    assert called <= members


def test_backend_specific_fit_entry_points_are_not_public() -> None:
    """C5: callers cannot select a different scientific API by backend."""
    import quantem.gpu as qg

    assert not any(
        name.lower().endswith(("_cuda", "_mps", "_webgpu")) for name in qg.__all__
    )


def test_public_ssb_exports_are_backend_neutral() -> None:
    """C6: root SSB exports contain no transport or implementation types; the one constant is a physical range."""

    from quantem.gpu import ssb

    assert set(ssb.__all__) == {
        "DEPTH_SPREAD_RANGE_NM",
        "DirectPtycho",
        "RotationBranch",
        "RotationBranches",
        "RotationDefocusComparison",
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
        def _backend(self):
            return self._test_backend

    session = Session.__new__(Session)
    session._device = None
    session._streamed = None
    session._test_backend = _Backend()
    session.backend = session._test_backend.backend
    phase, loss = session.preview(
        {"C10_nm": 0.0, "C12_nm": 0.0, "phi12_deg": 0.0},
        context=Context(),
    )

    assert events == ["enter", "exit"]
    assert phase.dtype == np.float32
    assert loss is None
