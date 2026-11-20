"""A scaled uint16 result on a CUDA device (a merged tilt series) saved as .qem and reopened unchanged."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
if not cp.cuda.runtime.getDeviceCount():
    pytest.skip("needs a CUDA device", allow_module_level=True)

import torch

from quantem.gpu import io


def scaled(seed=0, shape=(6, 8, 16, 16)):
    """A small float32 acquisition held as calibrated scaled uint16 codes on the device."""
    values = np.random.default_rng(seed).gamma(2.0, 3.0, shape).astype(np.float32)
    return io.load(cp.asarray(values), dtype="scaled_uint16", device="cuda:0", verbose=False), values


def test_a_scaled_result_round_trips_through_qem_with_its_calibration_and_records(tmp_path):
    held, values = scaled()
    held.metadata["maped_merge"] = {"version": 1, "source_count": 13, "scan_box": [0, 6, 0, 8]}
    path = tmp_path / "merged.qem"
    io.save(path, held, verbose=False)
    info = io.inspect(path)
    assert info.ready and tuple(info.scan_shape) == (6, 8) and tuple(info.detector_shape) == (16, 16)
    with io.load(path, device="cuda:0", verbose=False) as back:
        assert tuple(back.shape) == (6, 8, 16, 16) and back.metadata["maped_merge"]["source_count"] == 13
        assert back.metadata["precision"]["storage"] == "scaled_uint16"
        mine, theirs = held.read(output="torch"), back.read(output="torch")
        assert torch.equal(mine, theirs)                         # the codes as they were, not re-encoded
        assert float((theirs.cpu() - torch.from_numpy(values)).abs().max()) <= held.metadata["precision"]["max_abs_error"] + 1e-6
    with pytest.raises(FileExistsError):
        io.save(path, held, verbose=False)


def test_a_damaged_scaled_qem_is_refused(tmp_path):
    held, _ = scaled(1)
    path = tmp_path / "merged.qem"
    io.save(path, held, verbose=False)
    data = bytearray(path.read_bytes())
    data[-5] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="checksum"):
        io.load(path, device="cuda:0", verbose=False)
