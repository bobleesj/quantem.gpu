"""Scientific algorithms can return dense datasets without losing IO support."""

import h5py
import pytest
import torch

from quantem.gpu import io


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_float_blocks_save_without_changing_any_bits(tmp_path, device):
    """A bounded algorithm result reopens exactly, including fractional negatives."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("requires MPS")
    values_t = (torch.arange(5 * 7 * 8 * 8, device=device).float() / 7 - 100).reshape(5, 7, 8, 8)

    class Result:
        shape = values_t.shape
        dtype = "float32"
        def __init__(self):
            self.save_metadata = {"quantem_maped_merge_v1": '{"version": 1}'}

        def blocks(self):
            yield values_t[:2].flatten(0, 1)
            yield values_t[2:].flatten(0, 1)

    path = tmp_path / "merged_master.h5"
    io.save(path, Result(), backend=device, batch_size=5, frames_per_file=13)
    with io.load(path, device=device, verbose=False) as reopened:
        assert torch.equal(reopened.read().view(torch.int32), values_t.view(torch.int32))
        assert reopened.metadata["maped_merge"] == {"version": 1}


def test_cpu_masked_counts_keep_tilt_identity_and_correction(tmp_path):
    """A masked detector sentinel is excluded and its correction stays explicit."""
    counts_t = torch.arange(4 * 4 * 8 * 8).reshape(4, 4, 8, 8).to(torch.uint16)
    counts_t[:, :, 2, 3] = 65535
    path = tmp_path / "tilt_-17.0x_0.0y_master.h5"
    io.save(path, counts_t, backend="cpu", verbose=False)
    mask_t = torch.zeros((8, 8), dtype=torch.uint32)
    mask_t[2, 3] = 16
    with h5py.File(path, "a") as handle:
        handle.create_dataset(
            "entry/instrument/detector/detectorSpecific/pixel_mask", data=mask_t.numpy()
        )
    with io.load(path, device="cpu", hot_pixel_correction="zero", verbose=False) as loaded:
        expected_t = counts_t.clone()
        expected_t[:, :, 2, 3] = 0
        assert torch.equal(loaded.read(), expected_t)
        assert loaded.metadata["source_path"] == str(path)
        correction = loaded.metadata["hot_pixel_correction"]
        assert correction["method"] == "zero" and correction["applied"]
        assert correction["coordinates_row_col"] == [[2, 3]]


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
