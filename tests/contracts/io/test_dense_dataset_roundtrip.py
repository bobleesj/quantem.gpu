"""Scientific algorithms can return dense datasets without losing IO support."""

import pytest
import torch

from quantem.gpu import io


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
def test_dense_float_dataset_regions_and_hdf5_roundtrip(tmp_path, device):
    """Signed fractional values and calibration survive region reads and saving."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("requires MPS")
    values_t = (torch.arange(5 * 7 * 8 * 8, device=device).float() / 7 - 100).reshape(5, 7, 8, 8)
    metadata = {
        "representation": "dense", "working_shape": list(values_t.shape),
        "working_dtype": "float32", "scan_sampling_A": [0.4, 0.6],
        "name": "synthetic float merge",
        "maped_merge": {"version": 1, "scan_box": [1, 4, 2, 6]},
    }
    dataset = io.Dataset4dstemGPU(values_t, metadata)
    region_t = dataset.read(scan_region=(1, 4, 2, 6), detector_region=(2, 6, 1, 7))
    assert torch.equal(region_t, values_t[1:4, 2:6, 2:6, 1:7])
    assert region_t.device == values_t.device
    path = tmp_path / "merged_master.h5"
    io.save(path, dataset, backend=device, verbose=False)
    with io.load(path, device=device, verbose=False) as reopened:
        assert torch.equal(reopened.read(), values_t)
        assert reopened.sampling[:2] == (0.4, 0.6)
        assert reopened.name == metadata["name"]
        assert reopened.metadata["maped_merge"] == metadata["maped_merge"]
