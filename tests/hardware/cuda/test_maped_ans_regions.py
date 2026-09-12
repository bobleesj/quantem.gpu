"""Bounded MAPED merging directly from exact resident ANS counts."""

import os
from types import SimpleNamespace

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
pytestmark = pytest.mark.skipif(
    os.environ.get("QUANTEM_CUDA_ANS_TEST") != "1",
    reason="Set QUANTEM_CUDA_ANS_TEST=1 in an owned CUDA test window.",
)

from quantem.gpu._compact.streamed import StreamedCounts
from quantem.gpu._maped.cuda import _merge_regions
from quantem.gpu.detector import prepare
from quantem.gpu.io.backends.cuda._ans import CudaPackedResidentCounts
from quantem.gpu.maped import merge_to_scaled_h5


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_streamed_ans_decodes_selected_scan_ranges(dtype):
    """Selected ranges remain exact across entropy chunk boundaries."""

    shape = (6, 300, 3, 4)
    maximum = 251 if dtype == np.uint8 else 4093
    values = (
        cp.arange(np.prod(shape), dtype=cp.uint64) % maximum
    ).astype(dtype).reshape(shape)
    source = StreamedCounts(shape, dtype)
    flat = values.reshape(-1, *shape[2:])
    try:
        for first, stop in ((0, 700), (700, 1500), (1500, 1800)):
            source.append(cp.ascontiguousarray(flat[first:stop]))
        for first, stop in ((510, 515), (695, 705), (1490, 1510)):
            observed = source.decode_scan_range_device(first, stop)
            assert bool(cp.all(observed == flat[first:stop]))
    finally:
        source.release()


def test_maped_ans_regions_match_masked_bitpacked_regions():
    """ANS MAPED applies detector hot-pixel masks without changing raw counts."""

    import torch

    shape = (8, 9, 6, 7)
    ans_sources = []
    packed_sources = []
    for tilt in range(3):
        values = (
            (cp.arange(np.prod(shape), dtype=cp.uint64) * (tilt + 3) + tilt * 17)
            % 2000
        ).astype(cp.uint16).reshape(shape)
        mask = np.zeros(shape[2:], dtype=np.uint8)
        mask[tilt + 1, tilt + 2] = 1
        values[:, :, tilt + 1, tilt + 2] = np.uint16(65535)
        ans = StreamedCounts(shape, np.uint16)
        flat = values.reshape(-1, *shape[2:])
        ans.append(cp.ascontiguousarray(flat[:30]))
        ans.append(cp.ascontiguousarray(flat[30:]))
        assert int(ans.decode_scan_range_device(0, 1)[0, tilt + 1, tilt + 2]) == 65535
        ans_sources.append(
            SimpleNamespace(shape=shape, data=ans, metadata={"pixel_mask": mask})
        )
        corrected = values.copy()
        corrected[:, :, tilt + 1, tilt + 2] = 0
        packed = CudaPackedResidentCounts.from_array(
            cp.ascontiguousarray(corrected.reshape(-1, *shape[2:])), shape
        )
        packed_sources.append(
            SimpleNamespace(shape=shape, data=packed, metadata={})
        )

    real_shifts = torch.tensor(
        [[0.0, 0.0], [-1.25, 0.6], [0.75, -1.4]], device="cuda"
    )
    diffraction_shifts = torch.tensor(
        [[0.0, 0.0], [0.4, -0.7], [-0.25, 0.5]], device="cuda"
    )
    try:
        expected = list(
            _merge_regions(
                packed_sources,
                real_shifts,
                diffraction_shifts,
                scans_per_region=11,
            )
        )
        observed = list(
            _merge_regions(
                ans_sources,
                real_shifts,
                diffraction_shifts,
                scans_per_region=11,
            )
        )
        assert [first for first, _ in observed] == [first for first, _ in expected]
        for (_, observed_region), (_, expected_region) in zip(
            observed, expected, strict=True
        ):
            assert bool(cp.all(observed_region == expected_region))
    finally:
        for source in ans_sources + packed_sources:
            source.data.release()


def test_maped_ans_writes_reopenable_scaled_result(tmp_path):
    """The backend owns range measurement, encoding, writing, and reopening."""

    import torch

    shape = (5, 6, 4, 6)
    values = (
        cp.arange(np.prod(shape), dtype=cp.uint16).reshape(shape) % 173
    )
    source = StreamedCounts(shape, np.uint16)
    source.append(cp.ascontiguousarray(values.reshape(-1, *shape[2:])))
    loaded = SimpleNamespace(shape=shape, data=source, metadata={})
    shifts = torch.zeros((1, 2), device="cuda")
    expected = cp.concatenate(
        [region for _, region in _merge_regions([loaded], shifts, shifts, 7)]
    )
    path = tmp_path / "merged_master.h5"
    try:
        result = merge_to_scaled_h5([loaded], shifts, shifts, path)
        report = result.metadata["precision"]
        assert report["storage"] == "scaled_uint16"
        assert report["range_scope"] == "complete merged output"
        assert report["clipped"] == 0
        assert result.metadata["maped_merge"]["backend"] == "cuda"
        assert result.metadata["maped_merge"]["gpu_encode_seconds"] >= 0
        assert result.metadata["maped_merge"]["reopen_seconds"] >= 0
        session = prepare(result)
        for index in (0, 7, shape[0] * shape[1] - 1):
            np.testing.assert_allclose(
                session.frame(index),
                expected[index].get(),
                rtol=0,
                atol=report["scale"],
            )
        result.close()
    finally:
        source.release()


def test_maped_releases_owned_sources_before_packed_reopen(tmp_path):
    """Owned ANS storage does not overlap the packed result loader."""

    import torch

    shape = (5, 6, 4, 6)
    values = cp.arange(np.prod(shape), dtype=cp.uint16).reshape(shape) % 173
    source = StreamedCounts(shape, np.uint16)
    source.append(cp.ascontiguousarray(values.reshape(-1, *shape[2:])))
    loaded = SimpleNamespace(
        shape=shape,
        data=source,
        metadata={"representation": "ans"},
        close=source.release,
    )
    shifts = torch.zeros((1, 2), device="cuda")
    result = merge_to_scaled_h5(
        [loaded],
        shifts,
        shifts,
        tmp_path / "merged_master.h5",
        release_sources_before_reopen=True,
    )
    try:
        assert source.is_released
        assert result.metadata["maped_merge"][
            "released_sources_before_reopen"
        ]
    finally:
        result.close()
