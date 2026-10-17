"""Keep parallel authenticated CUDA restores exact, bounded and fail-closed."""

import os

import numpy as np
import pytest

from quantem.gpu import io
from quantem.gpu.formats.qem import snapshot
from quantem.gpu.io import qem


@pytest.fixture
def acquisition(tmp_path, monkeypatch, request):
    if os.environ.get("QEM_TEST_BACKEND") != "cuda":
        pytest.skip("Set QEM_TEST_BACKEND=cuda on a physical CUDA device.")
    # Small checksum blocks exercise repeated staging-slot reuse and a tail
    # without making the regression fixture a large acquisition.
    block_bytes = getattr(request, "param", 8192)
    monkeypatch.setattr(snapshot, "BLOCK_BYTES", block_bytes)
    monkeypatch.setattr(qem, "BLOCK_BYTES", block_bytes)
    values = np.random.default_rng(13).integers(
        0, 65536, (32, 32, 8, 12), dtype=np.uint16
    )
    values[0, 0, 0, 0] = 0
    values[-1, -1, -1, -1] = 65535
    original = tmp_path / "original.npy"
    saved = tmp_path / "acquisition.qem"
    np.save(original, values)
    with io.load(original, backend="cuda", verbose=False) as source:
        io.save(saved, source)
    header, start = snapshot.read_header(saved)
    if block_bytes == 8192:
        assert len(header["sha256"]) > 8
    else:
        assert len(header["sha256"]) == 1
    assert header["bytes"] % snapshot.BLOCK_BYTES
    return saved, values, header, start


@pytest.mark.parametrize("acquisition", [8192, 64 << 20], indirect=True)
def test_parallel_restore_reuses_slots_without_changing_counts(acquisition):
    path, values, header, _ = acquisition
    workers = min(4, len(header["sha256"]))
    for _ in range(3):
        with io.load(path, backend="cuda", verbose=False) as loaded:
            actual = loaded.data.decode_scan_range_device(0, 1024).get()
            np.testing.assert_array_equal(actual.reshape(values.shape), values)
            assert loaded.dtype == values.dtype
            timings = loaded.metadata["load_timings"]
            assert timings["verified_read_workers"] == workers
            assert timings["pinned_staging_bytes"] == workers * min(
                snapshot.BLOCK_BYTES, header["bytes"]
            )
            assert timings["verified_encoded_bytes"] == header["bytes"]
        assert loaded.data.is_released and not loaded.data.chunks


@pytest.mark.parametrize("location", ["first", "middle", "last"])
def test_corruption_releases_resident_and_allows_retry(
    acquisition, monkeypatch, location
):
    path, values, header, start = acquisition
    block = {
        "first": 0,
        "middle": len(header["sha256"]) // 2,
        "last": len(header["sha256"]) - 1,
    }[location]
    offset = start + block * snapshot.BLOCK_BYTES
    with path.open("r+b") as handle:
        handle.seek(offset)
        original = handle.read(1)
        handle.seek(offset)
        handle.write(bytes([original[0] ^ 1]))
    released = []
    original_release = qem.StreamedCounts.release

    def release(source):
        original_release(source)
        released.append(source)

    monkeypatch.setattr(qem.StreamedCounts, "release", release)
    with pytest.raises(ValueError, match=f"checksum mismatch in block {block}"):
        io.load(path, backend="cuda", verbose=False)
    assert len(released) == 1
    assert released[0].is_released and not released[0].chunks
    with path.open("r+b") as handle:
        handle.seek(offset)
        handle.write(original)
    with io.load(path, backend="cuda", verbose=False) as loaded:
        actual = loaded.data.decode_scan_range_device(0, 1024).get()
        np.testing.assert_array_equal(actual.reshape(values.shape), values)
