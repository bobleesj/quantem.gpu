"""Scaled uint16 ``.qem`` results: validator, CPU reference and Metal reader agree exactly."""

import hashlib
import json
import platform
import struct

import numpy as np
import pytest

from quantem.gpu import io
from quantem.gpu.formats.qem import metadata
from quantem.gpu.formats.qem.reference import _integer_arrays
from quantem.gpu.formats.qem.validation import validate_qem

SHAPE = (40, 32, 6, 5)
# Two calibration regions, each saved as one chunk; the first spans two codec blocks.
REGIONS = ((0, 700, 0.03125, -2.0), (700, 1280, 0.0078125, 5.5))


def _write(path, codes, attributes=None):
    """Mirror the native saver: three 8-byte aligned arrays per region chunk."""
    frames = codes.reshape(-1, *SHAPE[2:])
    body, chunks = bytearray(), []
    for number, (first, stop, _, _) in enumerate(REGIONS):
        arrays = []
        encoded = _integer_arrays(frames[first:stop], np.ones(SHAPE[2:], bool))[:3]
        for data, itemsize in zip(encoded, (1, 4, 1)):
            body.extend(b"\0" * (-len(body) % 8))
            arrays.append(dict(offset=len(body), count=len(data) // itemsize))
            body.extend(data)
        chunks.append(dict(first=first, scans=stop - first, region=number, arrays=arrays))
    regions = [
        dict(first_frame=first, stop_frame=stop, scale=scale, offset=offset,
             storage="scaled_uint16", intensity_min=offset,
             intensity_max=offset + 65535 * scale)
        for first, stop, scale, offset in REGIONS
    ]
    scientific = metadata.acquisition_metadata(SHAPE, {})
    scientific["processing"] = [dict(
        operation="scaled_uint16_quantization", changes_measurements=True,
        description="Regional scaled uint16: intensity = float32(code x scale + offset).",
    )]
    header = dict(
        container=metadata.CONTAINER, container_version=metadata.CONTAINER_VERSION,
        version=1, codec="scaled-uint16-column-rans-v1", profile="scaled-uint16-column-rans-v1",
        interval=512, shape=list(SHAPE), dtype="uint16", attributes=attributes or {}, chunks=chunks,
        scientific_metadata=scientific, bytes=len(body),
        sha256=[hashlib.sha256(bytes(body)).hexdigest()],
        intensity_calibration=dict(
            storage="scaled_uint16", version=2, complete=True, source_shape=list(SHAPE),
            source_dtype="float32", changed=1, rmse=0.0, max_abs_error=0.0, regions=regions,
        ),
    )
    blob = json.dumps(header, sort_keys=True).encode()
    path.write_bytes(
        metadata.MAGIC + struct.pack("<QQ", len(blob), 56 + len(blob))
        + hashlib.sha256(blob).digest() + blob + bytes(body)
    )


@pytest.fixture
def scaled(tmp_path):
    rng = np.random.default_rng(7)
    codes = rng.poisson(3.0, SHAPE).astype(np.uint16)
    codes[..., 0, 0] = 0  # all-zero streams
    codes[..., 5, 4] = 9  # constant streams
    codes[3, :, 2, 2] = rng.integers(0, 65536, SHAPE[1])  # escaped and literal values
    path = tmp_path / "merged.qem"
    _write(path, codes)
    expected = np.empty(SHAPE, np.float32)
    for first, stop, scale, offset in REGIONS:
        expected.reshape(-1, 30)[first:stop] = codes.reshape(-1, 30)[first:stop] * scale + offset
    return path, codes, expected


def test_validator_and_cpu_reference_restore_regional_calibration(scaled):
    path, _, expected = scaled
    report = validate_qem(path)
    assert report["codec_layout"] == "verified"
    assert report["measurements_changed_by"] == ["scaled_uint16_quantization"]
    data = io.load(path, backend="cpu")
    assert np.array_equal(np.asarray(data.data).view(np.uint32), expected.view(np.uint32))
    assert data.metadata["precision"]["regions"][1]["scale"] == REGIONS[1][2]


def test_merge_records_read_the_same_from_a_qem_and_an_hdf5_master(tmp_path):
    """A merged result's two records come back parsed under ``maped_merge`` and ``maped_summary`` from either container.

    The scaled ``.qem`` stores them as JSON text in its header ``attributes``, the HDF5 master as root attributes
    of the same names. The CPU ``.qem`` reader returned neither, so the record of which tilts were merged was
    lost on that path.
    """
    import h5py

    merge = {"version": 1, "source_count": 7, "sources": ["tilt_00", "tilt_01"]}
    summary = {"version": 1, "mean_bright_field": {"operation": "arithmetic_mean"}}
    stored = {"quantem_maped_merge_v1": json.dumps(merge), "quantem_maped_summary_v1": json.dumps(summary)}
    codes = np.random.default_rng(7).poisson(3.0, SHAPE).astype(np.uint16)
    _write(tmp_path / "merged.qem", codes, stored)
    with h5py.File(tmp_path / "merged_master.h5", "w") as handle:
        handle["entry/data/data"] = np.zeros((2, 3, 4, 5), np.uint16)
        handle.attrs.update(stored)
    from_qem = io.load(tmp_path / "merged.qem", backend="cpu").metadata
    from_master = io.inspect(tmp_path / "merged_master.h5").metadata
    assert from_qem["maped_merge"] == from_master["maped_merge"] == merge
    assert from_qem["maped_summary"] == from_master["maped_summary"] == summary
    assert "attributes" not in from_qem
    # the header-only inspection of the same file names the merge too
    inspected = io.inspect(tmp_path / "merged.qem").metadata
    assert (inspected["maped_merge"], inspected["maped_summary"]) == (merge, summary)
    # text that is not a version-1 record is left out by both readers, not returned raw
    _write(tmp_path / "other.qem", codes, {"quantem_maped_merge_v1": json.dumps({"version": 2}),
                                           "quantem_maped_summary_v1": "not json"})
    other = io.load(tmp_path / "other.qem", backend="cpu").metadata
    assert "maped_merge" not in other and "maped_summary" not in other


def test_corrupt_or_truncated_scaled_qem_is_refused(scaled, tmp_path):
    path, _, _ = scaled
    raw = bytearray(path.read_bytes())
    raw[-3] ^= 1
    corrupt = tmp_path / "corrupt.qem"
    corrupt.write_bytes(raw)
    with pytest.raises(ValueError, match="checksum"):
        validate_qem(corrupt)
    truncated = tmp_path / "truncated.qem"
    truncated.write_bytes(raw[:-8])
    with pytest.raises(ValueError):
        validate_qem(truncated)


@pytest.mark.skipif(platform.system() != "Darwin", reason="Metal reader needs an Apple GPU.")
def test_metal_reader_is_bitwise_and_masks_are_exact(scaled, tmp_path):
    pytest.importorskip("Metal")
    path, codes, expected = scaled
    data = io.load(path, backend="mps")
    source = data.data
    assert data.metadata["resident_profile"] == "scaled-uint16-column-rans-v1"
    assert source.shape == SHAPE and len(source.parts) == 2
    for index in (0, 511, 512, 699, 700, 1279):
        frame = source.frame(index).reshape(-1)
        assert np.array_equal(frame.view(np.uint32), expected.reshape(-1, 30)[index].view(np.uint32))
    mask = np.zeros(SHAPE[2:], bool)
    mask[1:5, 1:4] = True
    image = source.masked_sum(mask).get().reshape(-1)
    counts = codes.reshape(-1, 30)[:, mask.reshape(-1)].sum(axis=1, dtype=np.uint64)
    reference = np.empty(counts.size, np.float32)
    for first, stop, scale, offset in REGIONS:
        reference[first:stop] = counts[first:stop] * scale + offset * mask.sum()
    assert np.array_equal(image.view(np.uint32), reference.view(np.uint32))
    raw = bytearray(path.read_bytes())
    raw[-3] ^= 1
    corrupt = tmp_path / "corrupt.qem"
    corrupt.write_bytes(raw)
    with pytest.raises(ValueError, match="checksum"):
        io.load(corrupt, backend="mps")
