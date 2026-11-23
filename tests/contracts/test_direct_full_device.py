"""An encoded scan's whole-scan session reads batches of detector columns (ANS first, never a dense decode); its damped
phase is the dense one; when even the batches cannot fit, the error says how much memory is free and how much is needed."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from quantem.gpu import DirectPtycho
from quantem.gpu.direct_ptycho import workflow
from quantem.gpu.direct_ptycho.cuda import streamed
from quantem.gpu.io.dataset import Dataset4dstem, resident_metadata
from quantem.gpu.resident.cuda.counts import StreamedCounts

SETTINGS = {
    "backend": "cuda", "voltage_kV": 300., "semiangle_mrad": 30., "scan_sampling_A": (.3, .35),
    "detector_sampling_mrad": (6., 7.), "bright_field_center_px": (5.15, 5.6), "bright_field_radius_px": 5.1,
    "scan_detector_rotation_deg": 27.,
}
ABERRATIONS = {"C10_nm": 4.25, "C12_nm": 1.5, "phi12_deg": 17.}


def _encoded(scan_shape=(128, 128), detector_shape=(12, 12)):
    """A small encoded acquisition of positive counts, in irregular chunks."""
    values = cp.random.RandomState(9).poisson(15, size=(*scan_shape, *detector_shape)).astype(cp.uint16)
    source = StreamedCounts(values.shape, values.dtype, np.ones(detector_shape, bool))
    frames = values.reshape(-1, *detector_shape)
    for first in range(0, frames.shape[0], 1777):
        source.append(cp.ascontiguousarray(frames[first:first + 1777]))
    return Dataset4dstem(source, resident_metadata(values.shape, values.dtype, source.nbytes, backend="cuda"))


def _never(data) -> bool:
    """The device holds the dense spectrum."""
    return False


def _always(data) -> bool:
    """The device cannot hold the dense spectrum."""
    return True


def _phase(data, exceeds, monkeypatch, method, upsample, **strengths):
    """The damped phase of ``data`` on a device that can, or cannot, hold its dense spectrum."""
    monkeypatch.setattr(workflow, "_exceeds_device", exceeds)
    with DirectPtycho(data, **SETTINGS) as session:
        assert (session._streamed is not None) == (exceeds is _always)
        result = session.run(aberrations=ABERRATIONS, method=method, upsample=upsample, compute_loss=False,
                             save_calibration=False, verbose=False, **strengths)
        return cp.asnumpy(cp.asarray(result.phase))


@pytest.mark.parametrize("method, upsample, strengths", [
    ("phase", 1, {"phase_damp": 0.3}), ("phase", 3, {"phase_damp": 0.05}),
    ("complex", 2, {"phase_damp": 0.3, "amplitude_damp": 0.2}),
])
def test_the_damped_phase_on_a_full_device_is_the_dense_one(monkeypatch, method, upsample, strengths):
    with _encoded() as data:
        dense = _phase(data, _never, monkeypatch, method, upsample, **strengths)
        batched = _phase(data, _always, monkeypatch, method, upsample, **strengths)
    assert batched.shape == dense.shape == (128 * upsample, 128 * upsample)
    assert float(dense.std()) > 0
    np.testing.assert_allclose(batched, dense, atol=2e-5 * float(np.abs(dense).max()) + 1e-7)


def test_too_little_memory_for_the_batches_says_how_much_is_free_and_needed(monkeypatch):
    monkeypatch.setattr(workflow, "_exceeds_device", _always)

    def one_megabyte() -> int:
        """A device with 1 MB free."""
        return 1_000_000

    monkeypatch.setattr(streamed, "free_device_bytes", one_megabyte)
    with _encoded() as data, DirectPtycho(data, **SETTINGS) as session:
        with pytest.raises(streamed.DeviceMemoryError, match="GB of free GPU memory; 0.00 GB is free") as raised:
            session.run(aberrations=ABERRATIONS, method="phase", upsample=3, compute_loss=False,
                        save_calibration=False, verbose=False)
    assert raised.value.free_bytes == 1_000_000 and raised.value.needed_bytes > 1_000_000
    assert isinstance(raised.value, MemoryError)


def test_an_encoded_scan_is_never_decoded_dense():
    """ANS first (11-23): with room on the device or without, the whole-scan session reads the encoded counts in
    batches and says nothing about memory."""
    with _encoded() as data, DirectPtycho(data, **SETTINGS) as session:
        assert session._streamed is not None and session._streamed.full_scan_only
