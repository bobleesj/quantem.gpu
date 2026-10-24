"""Save, authenticate and query exact integer counts through parallel Metal IO."""

import numpy as np
import pytest
import torch

from quantem.gpu import detector, io
from quantem.gpu.formats.qem import snapshot
from quantem.gpu.io import qem


@pytest.fixture
def acquisition(tmp_path, monkeypatch, request):
    pytest.importorskip("Metal")
    if not torch.backends.mps.is_available():
        pytest.skip("Physical MPS device required.")
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
    with io.load(original, backend="mps", verbose=False) as source:
        io.save(saved, source)
    header, start = snapshot.read_header(saved)
    return saved, values, header, start


@pytest.mark.parametrize("acquisition", [8192, 64 << 20], indirect=True)
def test_parallel_roundtrip_preserves_counts_and_indexed_products(acquisition):
    path, values, _, _ = acquisition
    mask = np.zeros(values.shape[-2:], dtype=bool)
    mask[::2, ::2] = True
    for _ in range(3):
        with io.load(path, backend="mps", verbose=False) as loaded:
            actual = loaded.data.decode_scan_range_torch(0, 1024).cpu().numpy()
            np.testing.assert_array_equal(actual.reshape(values.shape), values)
            np.testing.assert_array_equal(
                detector.prepare(loaded).masked_sum(mask),
                values[:, :, mask].sum(-1),
            )
        assert loaded.data.is_released and not loaded.data.chunks


@pytest.mark.parametrize("location", ["first", "middle", "last"])
def test_corruption_releases_resident_and_allows_retry(acquisition, monkeypatch, location):
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
    original_release = qem.MPSStreamedCounts.release

    def release(source):
        original_release(source)
        released.append(source)

    monkeypatch.setattr(qem.MPSStreamedCounts, "release", release)
    with pytest.raises(ValueError, match=f"checksum mismatch in block {block}"):
        io.load(path, backend="mps", verbose=False)
    assert len(released) == 1
    assert released[0].is_released and not released[0].chunks
    with path.open("r+b") as handle:
        handle.seek(offset)
        handle.write(original)
    with io.load(path, backend="mps", verbose=False) as loaded:
        actual = loaded.data.decode_scan_range_torch(0, 1024).cpu().numpy()
        np.testing.assert_array_equal(actual.reshape(values.shape), values)
