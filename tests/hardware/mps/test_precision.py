"""Metal precision parity against a small NumPy scientific oracle."""

from copy import deepcopy
import os

import numpy as np
import pytest

from tests.parity.precision_fixture import (
    encode_precision_reference,
    make_precision_fixture,
    precision_error_reference,
    restore_precision_reference,
)

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
    calibration = report["regions"][0] if report.get("version") == 2 else report
    if dtype == "float16":
        expected = values.astype(np.float16).astype(np.float32)
        tolerance = np.finfo(np.float16).eps * np.maximum(1, np.abs(values))
    else:
        expected = (
            np.rint((values - calibration["offset"]) / calibration["scale"])
            .clip(0, 65535)
            * calibration["scale"]
            + calibration["offset"]
        ).astype(np.float32)
        tolerance = np.full(values.shape, calibration["scale"] * 1.1, np.float32)
    observed = prepare(loaded).reduce_frames([0], "mean")
    np.testing.assert_allclose(observed, expected[0, 0], rtol=0, atol=float(np.max(tolerance[0, 0])))
    assert report["values"] == values.size
    assert report["range_scope"] == ("automatic regions" if report.get("version") == 2 else "complete source")
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


@pytest.mark.parametrize("dtype", ["float16", "scaled_uint16"])
def test_mps_precision_products_match_shared_numpy_oracle(tmp_path, dtype):
    """Frames and reductions obey the same public contract as CUDA."""

    from quantem.gpu import io
    from quantem.gpu.detector import prepare

    original = make_precision_fixture()
    source = tmp_path / "shared_oracle.npy"
    np.save(source, original)
    with io.load(source, dtype=dtype, backend="mps", verbose=False) as loaded:
        report = loaded.metadata["precision"]
        calibration = report["regions"][0] if report.get("version") == 2 else report
        assert report["intensity_min"] == float(original.min())
        assert report["intensity_max"] == float(original.max())
        assert report["values"] == original.size
        assert report["range_scope"] == ("automatic regions" if report.get("version") == 2 else "complete source")
        if dtype == "scaled_uint16":
            assert calibration["scale"] == (
                float(original.max()) - float(original.min())
            ) / 65535
        blocks = []
        for block in loaded.data.encoded_blocks():
            try:
                blocks.append(block.get())
            finally:
                block.release()
        encoded = np.concatenate(blocks).reshape(original.shape)
        np.testing.assert_array_equal(
            encoded, encode_precision_reference(original, report)
        )
        expected = restore_precision_reference(original, report)
        errors = precision_error_reference(original, expected)
        session = prepare(loaded)

        assert loaded.data.numel() == original.size
        np.testing.assert_array_equal(
            loaded.data[17].cpu().numpy(), expected.reshape(-1, 7, 9)[17]
        )
        np.testing.assert_array_equal(
            loaded.data[2, 5].cpu().numpy(), expected[2, 5]
        )

        for index in (0, 17, original.shape[0] * original.shape[1] - 1):
            np.testing.assert_array_equal(
                session.frame(index), expected.reshape(-1, 7, 9)[index]
            )

        indices = [17, 0, 17, 29, 6]
        np.testing.assert_allclose(
            session.reduce_frames(indices, "mean"),
            expected.reshape(-1, 7, 9)[indices].mean(axis=0),
            rtol=3e-6,
            atol=2e-5,
        )
        np.testing.assert_allclose(
            session.mean_dp(), expected.mean(axis=(0, 1)), rtol=3e-6, atol=2e-5
        )
        mask = ((np.indices((7, 9)).sum(axis=0) % 3) == 0).astype(np.float32)
        np.testing.assert_allclose(
            session.masked_sum(mask),
            (expected * mask).sum(axis=(2, 3)),
            rtol=3e-6,
            atol=2e-4,
        )
        assert report["rmse"] == pytest.approx(errors["rmse"], rel=3e-5, abs=1e-7)
        assert report["max_abs_error"] == pytest.approx(
            errors["max_abs_error"], rel=3e-5, abs=1e-7
        )
        for field in ("positive_to_zero", "changed", "overflow"):
            assert report[field] == errors[field]


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


def test_fused_scaled_uint16_measurement_matches_separate_metal_passes():
    from quantem.gpu.io.backends.mps.precision import (
        encode,
        encode_measure,
        measure,
        restore,
        upload,
    )

    values = np.linspace(-123.25, 987.75, 32771, dtype=np.float32)
    source = upload(values)
    report = {
        "storage": "scaled_uint16",
        "intensity_min": float(values.min()),
        "intensity_max": float(values.max()),
        "scale": float(values.max() - values.min()) / 65535,
        "offset": float(values.min()),
        "values": 0,
        "squared_error": 0.0,
        "max_abs_error": 0.0,
        "positive_to_zero": 0,
        "changed": 0,
        "overflow": 0,
    }
    separate_report = deepcopy(report)
    fused_report = deepcopy(report)
    encoded = restored = fused = None
    try:
        encoded = encode(source, separate_report)
        restored = restore(encoded, separate_report)
        measure(source, restored, separate_report)
        fused = encode_measure(source, fused_report)
        np.testing.assert_array_equal(fused.get(), encoded.get())
        assert fused_report == separate_report
    finally:
        for value in (fused, restored, encoded, source):
            if value is not None:
                value.release()


def test_scaled_uint16_rounds_near_half_steps_like_numpy():
    """Large codes retain the true rounding direction near a half step."""
    from quantem.gpu.io.backends.mps.precision import encode, upload

    values = np.arange(1024, dtype=np.float32) * 0.125 - 10.75
    report = {
        "storage": "scaled_uint16",
        "intensity_min": float(values.min()),
        "intensity_max": float(values.max()),
        "scale": (float(values.max()) - float(values.min())) / 65535,
        "offset": float(values.min()),
    }
    source = upload(values)
    encoded = None
    try:
        encoded = encode(source, report)
        np.testing.assert_array_equal(
            encoded.get(), encode_precision_reference(values, report)
        )
    finally:
        if encoded is not None:
            encoded.release()
        source.release()


def test_tensor_range_preserves_extrema_and_rejects_nonfinite_values():
    """Saving a strided scientific tensor preserves extrema and rejects invalid data."""
    import torch

    from quantem.gpu.io.backends.mps.precision import tensor_range

    values = torch.linspace(-123.75, 997.25, 33 * 37, device="mps").reshape(33, 37)
    for selected in (values, values[:, ::2], values.T):
        assert tensor_range(selected) == (float(selected.amin()), float(selected.amax()))
    for nonfinite in (float("nan"), float("inf"), float("-inf")):
        invalid = values.clone()
        invalid[16, 18] = nonfinite
        with pytest.raises(ValueError, match="finite intensities"):
            tensor_range(invalid)


def test_tensor_range_rejects_subnormal_bit_patterns():
    """The fused range check retains the saved-precision subnormal policy."""
    import torch

    from quantem.gpu.io.backends.mps.precision import tensor_range

    for bits in (1, 0x7fffff, -2147483647):
        values = torch.tensor([0, bits, 0x3f800000], device="mps", dtype=torch.int32)
        with pytest.raises(ValueError, match="subnormal"):
            tensor_range(values.view(torch.float32))


def test_direct_ans_mean_matches_prior_decoded_reduction_exactly():
    """Direct encoded means retain the old compensated order across ANS intervals."""
    import torch

    from quantem.gpu import io
    from quantem.gpu.io.backends.mps.precision import (
        MetalArray,
        _dispatch,
        _part_buffers,
    )

    values = torch.arange(17 * 513 * 4 * 8, device="mps").reshape(17, 513, 4, 8)
    values = ((values * 17) % 787).float() / 13 - 7
    with io.load(values, backend="mps", dtype="scaled_uint16", verbose=False) as loaded:
        source = loaded.data
        reference = MetalArray(source.det_shape, np.float32)
        actual = source.mean_dp()
        try:
            for index, part in enumerate(source.parts):
                parameters, calibration = source._params(part)
                parameters[0] = parameters[1]
                parameters[8] = int(index > 0)
                parameters[9] = source.n_frames
                with _part_buffers(part) as buffers:
                    _dispatch("mean", [*buffers, reference], parameters, calibration)
            np.testing.assert_array_equal(actual.get(), reference.get())
        finally:
            reference.release()
            actual.release()
