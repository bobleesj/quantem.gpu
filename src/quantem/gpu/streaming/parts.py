"""Whole encoded chunks as a dataset of their own: a part of a stream, or a scan put together from parts.

A receiver encodes each scan in chunks that never change once made. ``dataset_of`` gives any run of whole chunks as
an ordinary ``Dataset4dstem`` that shares their memory, so a stream can be saved in parts cut by pattern index while
it arrives (``io.save`` of a 1 x patterns dataset), and a scan can be opened again from the parts that hold it.

>>> part = dataset_of(chunks_of(encoder.source, 0, 8192), (1, 8192), like=encoder)       # doctest: +SKIP
>>> io.save("part.qem", part)                                                            # doctest: +SKIP
>>> scan = dataset_of(chunks_of(first.data, 4096, 8192) + chunks_of(second.data, 0, 4096), (64, 128), like=first)  # doctest: +SKIP
"""

from __future__ import annotations

import math

from quantem.gpu.formats.qem.snapshot import INTEGER_CODEC
from quantem.gpu.io import arrays
from quantem.gpu.io.dataset import Dataset4dstem, resident_metadata
from quantem.gpu.resident.cuda.counts import Chunk, StreamedCounts


def chunks_of(source, first: int, stop: int) -> list:
    """The whole chunks of an encoded source that hold patterns ``first`` to ``stop - 1``. Both ends must fall on
    chunk boundaries: a chunk is never split."""
    first, stop = int(first), int(stop)
    found = [chunk for chunk in source.chunks if chunk.first >= first and chunk.first + chunk.scans <= stop]
    covered = sum(chunk.scans for chunk in found)
    if covered != stop - first or (found and found[0].first != first):
        edges = sorted({chunk.first for chunk in source.chunks} | {chunk.first + chunk.scans for chunk in source.chunks})
        raise ValueError(f"Patterns {first} to {stop} do not fall on whole chunks; this source's chunks meet at {edges[:8]}"
                         f"{'...' if len(edges) > 8 else ''}.")
    return found


def dataset_of(chunks, scan_shape, *, like, metadata: dict | None = None) -> Dataset4dstem:
    """A dataset of ``scan_shape`` whose counts are these whole chunks, in order, sharing their memory.

    ``like`` is the live encoder or the dataset the chunks came from: its detector, dtype, valid pixels, pixel mask
    and correction record are the new dataset's. ``metadata`` is added to (and wins over) what ``like`` carries.
    Releasing ``like`` afterwards does not free the chunks: this dataset holds them.
    """
    import cupy as cp

    source = like.source if hasattr(like, "corrector") else like.data
    scan_shape = tuple(int(side) for side in scan_shape)
    chunks = list(chunks)
    if sum(chunk.scans for chunk in chunks) != math.prod(scan_shape):
        raise ValueError(f"The chunks hold {sum(chunk.scans for chunk in chunks)} patterns; a {scan_shape[0]} x "
                         f"{scan_shape[1]} scan needs {math.prod(scan_shape)}.")
    shape = (*scan_shape, *source.shape[2:])
    with cp.cuda.Device(source.device):
        counts = StreamedCounts(shape, source.dtype, source.valid_pixels)
        at = 0
        for chunk in chunks:
            counts.chunks.append(Chunk(at, chunk.scans, chunk.arrays))
            at += chunk.scans
        counts.ready_scans = at
    if hasattr(like, "corrector"):                  # a live encoder: as its own finish() describes the counts
        record = dict(like.metadata)
        record.update(pixel_mask=like.pixel_mask, source_dtype=source.dtype.name, source_read_passes=1,
                      resident_profile=INTEGER_CODEC, **arrays.correction_metadata(like.corrector.record))
    else:
        record = dict(like.metadata)
    record.update(resident_metadata(shape, source.dtype, counts.nbytes, backend="cuda"),
                  source_logical_tensor_bytes=math.prod(shape) * source.dtype.itemsize,
                  index_bytes=counts.index_nbytes)
    record.update(metadata or {})
    return Dataset4dstem(counts, record)
