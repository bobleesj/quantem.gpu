"""Whole encoded chunks saved as parts cut across scans, and a scan opened again from its parts."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
if not cp.cuda.runtime.getDeviceCount():
    pytest.skip("needs a CUDA device", allow_module_level=True)

from quantem.gpu import io
from quantem.gpu.streaming import LiveEncoder, chunks_of, dataset_of

CHUNK = LiveEncoder.CHUNK_PATTERNS


def streamed(seed):
    """A 2 x CHUNK scan received in two batches (two chunks), and its counts."""
    counts = np.random.default_rng(seed).integers(0, 4096, (2 * CHUNK, 12, 12), dtype=np.uint16)
    encoder = LiveEncoder((2, CHUNK), (12, 12), device="cuda:0", metadata={"series": f"series_{seed}"})
    for first in (0, CHUNK):
        encoder.add(first, cp.asarray(counts[first:first + CHUNK]))
    return encoder, counts


def test_a_part_cut_across_two_scans_is_an_ordinary_file_and_the_scan_opens_from_its_parts(tmp_path):
    (one, counts_one), (two, counts_two) = streamed(1), streamed(2)
    parts = [
        dataset_of(chunks_of(one.source, 0, CHUNK), (1, CHUNK), like=one, metadata={"first_index": 0}),
        dataset_of(chunks_of(one.source, CHUNK, 2 * CHUNK) + chunks_of(two.source, 0, CHUNK), (1, 2 * CHUNK), like=one,
                   metadata={"first_index": CHUNK}),
        dataset_of(chunks_of(two.source, CHUNK, 2 * CHUNK), (1, CHUNK), like=two, metadata={"first_index": 3 * CHUNK}),
    ]
    for number, part in enumerate(parts):
        io.save(tmp_path / f"part-{number}.qem", part, verbose=False)
    one.source.release(), two.source.release()          # the receiver lets go: the parts are on disk
    del parts
    loaded = [io.load(tmp_path / f"part-{number}.qem", device="cuda:0", hot_pixel_correction="none", verbose=False)
              for number in range(3)]
    assert [tuple(part.shape) for part in loaded] == [(1, CHUNK, 12, 12), (1, 2 * CHUNK, 12, 12), (1, CHUNK, 12, 12)]
    assert loaded[1].metadata["first_index"] == CHUNK
    np.testing.assert_array_equal(loaded[1].data.decode_scan_range_device(CHUNK - 3, CHUNK + 3).get(),
                                  np.concatenate([counts_one[-3:], counts_two[:3]]))
    second = dataset_of(chunks_of(loaded[1].data, CHUNK, 2 * CHUNK) + chunks_of(loaded[2].data, 0, CHUNK), (2, CHUNK),
                        like=loaded[1], metadata={"series": "series_2"})
    assert tuple(second.shape) == (2, CHUNK, 12, 12) and second.metadata["series"] == "series_2"
    np.testing.assert_array_equal(second.data.decode_scan_range_device(0, 2 * CHUNK).get(), counts_two)


def test_a_chunk_is_never_split():
    encoder, _ = streamed(3)
    with pytest.raises(ValueError, match="whole chunks"):
        chunks_of(encoder.source, 10, CHUNK)
    with pytest.raises(ValueError, match="needs"):
        dataset_of(chunks_of(encoder.source, 0, CHUNK), (2, CHUNK), like=encoder)
