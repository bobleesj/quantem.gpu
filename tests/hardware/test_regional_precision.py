"""Calibrated storage parity for CUDA and MPS scientist workflows."""

import os

import numpy as np
import pytest

from tests.parity.precision_fixture import (
    encode_precision_reference,
    precision_error_reference,
    restore_precision_reference,
)

BACKEND = os.environ.get("QUANTEM_REGIONAL_BACKEND")
pytestmark = pytest.mark.skipif(
    BACKEND not in {"cuda", "mps"},
    reason="Select QUANTEM_REGIONAL_BACKEND=cuda or mps.",
)


class GeneratedScan:
    """A one-pass GPU source with three different intensity ranges."""

    def __init__(self, values):
        import torch

        self.tensor = torch.as_tensor(values, device=BACKEND)
        self.shape, self.dtype = values.shape, values.dtype
        self.passes = 0

    def blocks(self):
        self.passes += 1
        if self.passes > 1:
            raise AssertionError("The scientific source must be evaluated only once.")
        for first, stop in ((0, 2), (2, 4), (4, 5)):
            yield self.tensor[first:stop].flatten(0, 1)


def fixture():
    values = np.random.default_rng(72).uniform(-4, 7, (5, 6, 8, 8)).astype(np.float32)
    values[:2] *= 100
    values[4:] = 3.125
    values[0, 0, 0, 0] = 0
    return values


def test_generated_scan_roundtrip_and_calibrated_products(tmp_path):
    from quantem.gpu import io

    original = fixture()
    source = GeneratedScan(original)
    with io.load(
        source, dtype="scaled_uint16", backend=BACKEND, verbose=False
    ) as loaded:
        report = loaded.metadata["precision"]
        expected = np.empty_like(original).reshape(-1, 8, 8)
        raw = original.reshape(-1, 8, 8)
        for region in report["regions"]:
            first, stop = region["first_frame"], region["stop_frame"]
            expected[first:stop] = restore_precision_reference(raw[first:stop], region)
        expected = expected.reshape(original.shape)
        observed = loaded.read().cpu().numpy()
        np.testing.assert_array_equal(observed, expected)
        oracle = precision_error_reference(original, expected)
        assert report["rmse"] == pytest.approx(oracle["rmse"], rel=2e-6)
        assert report["max_abs_error"] == oracle["max_abs_error"]
        for key in ("positive_to_zero", "changed", "overflow"):
            assert report[key] == oracle[key]
        assert source.passes == 1
        np.testing.assert_array_equal(
            loaded.read(scan_region=(1, 5, 2, 5), detector_region=(1, 6, 2, 8)).cpu(),
            expected[1:5, 2:5, 1:6, 2:8],
        )
        mean = loaded.data.mean_dp()
        mean = mean.get() if hasattr(mean, "get") else mean
        np.testing.assert_allclose(mean, expected.mean((0, 1)), rtol=2e-6, atol=3e-5)
        mask = np.zeros((8, 8), np.float32)
        mask[1:6, 2:8] = 1
        bf = loaded.data.masked_sum_native(mask)
        bf = bf.get() if hasattr(bf, "get") else bf
        np.testing.assert_allclose(
            bf, (expected * mask).sum((-2, -1)), rtol=2e-6, atol=2e-4
        )
        path = tmp_path / "scaled_master.h5"
        io.save(path, loaded, backend=BACKEND, verbose=False)
        with io.load(path, backend=BACKEND, verbose=False) as reopened:
            np.testing.assert_array_equal(reopened.read().cpu(), expected)
            assert reopened.metadata["precision"] == report
        with io.load(
            path,
            backend=BACKEND,
            scan_region=(1, 5, 2, 5),
            detector_region=(1, 6, 2, 8),
            verbose=False,
        ) as selected:
            np.testing.assert_array_equal(
                selected.read().cpu(), expected[1:5, 2:5, 1:6, 2:8]
            )
        with io.load(path, dtype="float16", backend=BACKEND, verbose=False) as reduced:
            np.testing.assert_array_equal(reduced.read().cpu(), expected.astype(np.float16).astype(np.float32))
            assert reduced.metadata["precision"]["prior_conversion"]["version"] == 2
        # The returned Torch allocation remains valid after subsequent reads/close.
        retained = loaded.read(scan_region=(1, 2, 0, 1))
    np.testing.assert_array_equal(retained.cpu(), expected[1:2, :1])


def test_same_float_fixture_codes_match_numpy():
    from quantem.gpu import io

    original = fixture()
    with io.load(
        GeneratedScan(original), dtype="scaled_uint16", backend=BACKEND, verbose=False
    ) as loaded:
        blocks = list(loaded.data.encoded_blocks())
        codes = np.concatenate([block.get() for block in blocks])
        flat = original.reshape(-1, 8, 8)
        for region in loaded.metadata["precision"]["regions"]:
            first, stop = region["first_frame"], region["stop_frame"]
            np.testing.assert_array_equal(
                codes[first:stop], encode_precision_reference(flat[first:stop], region)
            )


def test_legacy_global_archive_still_restores_units(tmp_path):
    import json
    import torch
    from quantem.gpu import io

    original = fixture()
    low, high = float(original.min()), float(original.max())
    report = dict(
        version=1,
        storage="scaled_uint16",
        source_dtype="float32",
        source_shape=list(original.shape),
        intensity_min=low,
        intensity_max=high,
        offset=low,
        scale=(high - low) / 65535,
        range_scope="complete source",
        complete=True,
        values=original.size,
        clipped=0,
    )
    expected = restore_precision_reference(original, report)
    report.update(precision_error_reference(original, expected))
    codes = torch.as_tensor(
        encode_precision_reference(original, report), device=BACKEND
    )
    path = tmp_path / "legacy_master.h5"
    io.save(
        path,
        codes,
        metadata={"quantem_precision_v1": json.dumps(report)},
        backend=BACKEND,
        verbose=False,
    )
    with io.load(path, backend=BACKEND, verbose=False) as loaded:
        np.testing.assert_array_equal(loaded.read().cpu(), expected)
        assert loaded.metadata["precision"]["version"] == 1


def test_generated_save_is_single_pass_and_reloads(tmp_path):
    from quantem.gpu import io

    original = fixture()
    source = GeneratedScan(original)
    path = tmp_path / "streamed_master.h5"
    io.save(path, source, dtype="scaled_uint16", backend=BACKEND, verbose=False)
    assert source.passes == 1
    with io.load(path, backend=BACKEND, verbose=False) as loaded:
        expected = restore_precision_reference(original, loaded.metadata["precision"])
        np.testing.assert_array_equal(loaded.read().cpu(), expected)


@pytest.mark.parametrize("magnitude", [1.0, 1e-20, 1e30])
def test_ties_and_wide_ranges_match_numpy(magnitude):
    import torch
    from quantem.gpu import io

    values = np.array([0, 65535, 0.5, 1.5, 2.5, 32766.5, 32767.5, 65534.5], np.float32)
    values = np.concatenate(
        [
            values,
            np.nextafter(values, np.float32(np.inf)),
            np.nextafter(values, np.float32(-np.inf)),
        ]
    )
    values = (
        ((values.astype(np.float64) - 32768) * magnitude)
        .astype(np.float32)
        .reshape(1, 3, 2, 4)
    )
    with io.load(
        torch.as_tensor(values, device=BACKEND),
        dtype="scaled_uint16",
        backend=BACKEND,
        verbose=False,
    ) as loaded:
        expected = restore_precision_reference(values, loaded.metadata["precision"])
        np.testing.assert_array_equal(loaded.read().cpu(), expected)


def test_calibrated_center_of_mass():
    from quantem.gpu import io

    original = np.abs(fixture()) + 1
    with io.load(
        GeneratedScan(original), dtype="scaled_uint16", backend=BACKEND, verbose=False
    ) as loaded:
        values = restore_precision_reference(original, loaded.metadata["precision"])
        denominator = values.sum((-2, -1), dtype=np.float64)
        col, row = loaded.data.center_of_mass()
        np.testing.assert_allclose(
            col.get(),
            (values * np.arange(8)[None, :]).sum((-2, -1)) / denominator,
            rtol=2e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            row.get(),
            (values * np.arange(8)[:, None]).sum((-2, -1)) / denominator,
            rtol=2e-6,
            atol=1e-6,
        )
