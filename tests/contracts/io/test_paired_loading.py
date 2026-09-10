"""Original HDF5 and saved resident forms through ``io.load(representation="paired")``."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
if not cp.cuda.runtime.getDeviceCount():
    pytest.skip("paired loading needs a CUDA device", allow_module_level=True)

from quantem.gpu import detector, io
from quantem.gpu._compact.paired import PairedCounts
from quantem.gpu.io.save import save_compressed_arina_h5


def _acquisition(tmp_path, name, seed):
    """A 32x32 scan of 24x24 uint16 counts written as a four-shard Arina master."""
    rng = np.random.default_rng(seed)
    data = rng.poisson(0.6, (32, 32, 24, 24)).astype(np.uint16)
    data[:, :, 3, :] = rng.integers(0, 65536, (32, 32, 24), dtype=np.uint16)  # literal streams
    data[:, :, 5, 5] = 65535
    master = tmp_path / f"{name}_master.h5"
    save_compressed_arina_h5(master, data, dtype="u16", batch_size=256, frames_per_file=256)
    return master, data


def test_original_h5_streams_into_exact_paired_sources(tmp_path):
    first, counts = _acquisition(tmp_path, "first", 1)
    second, other = _acquisition(tmp_path, "second", 2)
    loaded = io.load(first, backend="cuda", representation="paired", dtype="native", apply_mask=False, verbose=False)
    assert isinstance(loaded.data, PairedCounts)
    assert loaded.metadata["representation"] == "paired"
    assert loaded.metadata["resident_profile"] == "paired-polar-counts-v1"
    assert loaded.metadata["load_timings"]["shards"] == 4
    assert bool(cp.array_equal(loaded.data.decode_chunk(0), cp.asarray(counts.reshape(1024, 24, 24))).get())
    series = io.load([first, second], backend="cuda", representation="paired", dtype="native", apply_mask=False, verbose=False)
    assert [item.metadata["shape"] if "shape" in item.metadata else item.data.shape for item in series] == [(32, 32, 24, 24)] * 2
    session = detector.prepare([item.data for item in series])
    mask = detector.detector_mask((12.25, 11.75), 2.5, 9.0, (24, 24), dtype=np.float64)
    expected = np.stack([counts[..., mask].sum(axis=-1, dtype=np.uint64), other[..., mask].sum(axis=-1, dtype=np.uint64)])
    np.testing.assert_array_equal(session.masked_sum(mask, output="native").get().reshape(2, 32, 32), expected)
    np.testing.assert_array_equal(session.frame(1023, output="native").get().reshape(2, 24, 24), np.stack([counts[31, 31], other[31, 31]]))


def test_saved_form_is_detected_and_reopens_without_decoding(tmp_path):
    master, counts = _acquisition(tmp_path, "saved", 3)
    loaded = io.load(master, backend="cuda", representation="paired", dtype="native", apply_mask=False, verbose=False)
    saved = tmp_path / "saved.paired"
    loaded.data.save(saved)
    assert io.DataRepresentation.detect_source(saved) is io.DataRepresentation.PAIRED
    reopened = io.load(saved, backend="cuda", dtype="native", apply_mask=False, verbose=False)
    assert reopened.metadata["representation"] == "paired"
    for before, after in zip(loaded.data.chunks[0].arrays, reopened.data.chunks[0].arrays):
        assert bool(cp.array_equal(before, after).get())
    assert bool(cp.array_equal(reopened.data.decode_chunk(0), cp.asarray(counts.reshape(1024, 24, 24))).get())
    with pytest.raises(ValueError, match="representation='paired'"):
        io.load(saved, backend="cuda", representation="dense")


def test_lean_loader_admits_a_series_with_few_slots(tmp_path):
    """The tail of a full device streams through two 512-scan buffers and three staging slots."""
    from quantem.gpu.io import PairedLoader

    masters = [_acquisition(tmp_path, f"lean{i}", 10 + i)[0] for i in range(3)]
    with PairedLoader(rolling_scans=512, rings=2, slots=3, readers=2, capacity=4 * 1024**2) as loader:
        names = [str(path) for path, source, timings in loader.load_many(masters, scan_shape=(32, 32)) if source.ready_scans == 1024 and timings["chunks"] == 2]
    assert names == [str(m) for m in masters]

