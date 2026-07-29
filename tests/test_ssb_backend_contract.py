from __future__ import annotations

import inspect

import numpy as np

from tests.ssb_precision import PRECISION


class _Backend:
    backend = "mps"
    scan_shape = (4, 4)
    detector_shape = (3, 3)
    num_bf = 1

    precision = PRECISION

    def fit(self, **kwargs):
        raise NotImplementedError

    def reconstruct_result(self, aberrations):
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
        name.lower().endswith(("_cuda", "_mps", "_webgpu"))
        for name in qg.__all__
    )


def test_public_ssb_exports_are_backend_neutral() -> None:
    """C6: root SSB exports contain no transport or implementation types."""

    import quantem.gpu.ssb as ssb

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
