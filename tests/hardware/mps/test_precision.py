"""Metal precision parity against a small NumPy scientific oracle."""

import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("QUANTEM_MPS_PRECISION_TEST") != "1",
    reason="Set QUANTEM_MPS_PRECISION_TEST=1 on an Apple GPU host.",
)


@pytest.mark.parametrize("dtype", ["float16", "scaled_uint16"])
def test_mps_precision_matches_numpy_oracle(tmp_path, dtype):
    from quantem.gpu import io
    from quantem.gpu.detector import prepare

    values = (np.sin(np.arange(8 * 8 * 16 * 16, dtype=np.float32)) * 1000).reshape(
        8, 8, 16, 16
    )
    source = tmp_path / "source.npy"
    np.save(source, values)
    loaded = io.load(source, dtype=dtype, backend="mps", verbose=False)
    report = loaded.metadata["precision"]
    if dtype == "float16":
        expected = values.astype(np.float16).astype(np.float32)
        tolerance = np.finfo(np.float16).eps * np.maximum(1, np.abs(values))
    else:
        expected = (
            np.rint((values - report["offset"]) / report["scale"])
            .clip(0, 65535)
            * report["scale"]
            + report["offset"]
        ).astype(np.float32)
        tolerance = np.full(values.shape, report["scale"] * 1.1, np.float32)
    observed = prepare(loaded).reduce_frames([0], "mean")
    np.testing.assert_allclose(observed, expected[0, 0], rtol=0, atol=float(np.max(tolerance[0, 0])))
    assert report["values"] == values.size
    assert report["range_scope"] == "complete source"
    copied = tmp_path / "copy_master.h5"
    io.save(copied, loaded, backend="mps", verbose=False, wait=True)
    reopened = io.load(copied, backend="mps", verbose=False)
    np.testing.assert_allclose(
        prepare(reopened).reduce_frames([0], "mean"),
        expected[0, 0],
        rtol=0,
        atol=float(np.max(tolerance[0, 0])),
    )
    loaded.close()
    reopened.close()


def test_mps_direct_tensor_conversion_stays_on_device(tmp_path):
    import torch

    from quantem.gpu import io

    values = torch.linspace(-100, 100, 8 * 8 * 16 * 16, device="mps").reshape(
        8, 8, 16, 16
    )
    output = tmp_path / "tensor_master.h5"
    io.save(output, values, dtype="scaled_uint16", backend="mps", verbose=False, wait=True)
    assert output.is_file()
    with io.load(output, backend="mps", verbose=False) as loaded:
        assert loaded.metadata["precision"]["values"] == values.numel()
        assert loaded.data.device.type == "mps"
