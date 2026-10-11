"""Generate small disposable QEM acquisitions for the browser GPU parity gate."""

import json
from pathlib import Path
import sys

import h5py
import hdf5plugin
import numpy as np

from quantem.gpu.formats.qem.reference import save_array


def generate(directory: Path) -> None:
    """Cover every integer stream mode, checkpoints, chunks, and a tail."""
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for dtype in (np.uint8, np.uint16):
        rng = np.random.default_rng(43)
        data = np.zeros((17, 33, 2, 3), dtype=dtype)
        columns = data.reshape(-1, 6)
        high = np.iinfo(dtype).max
        columns[:, 1] = high
        columns[:, 2] = rng.integers(0, int(high) + 1, len(columns), dtype=dtype)
        columns[:, 3] = rng.poisson(3, len(columns)).astype(dtype)
        columns[[0, 255, 256, 511, 512, 560], 4] = 77
        columns[:, 5] = rng.poisson(2, len(columns)).astype(dtype)
        columns[::43, 5] = high
        name = np.dtype(dtype).name + "-qem-modes"
        save_array(directory / (name + ".qem"), data, chunk_scans=512)
        data.tofile(directory / (name + ".bin"))
        records.append(dict(name=name, shape=list(data.shape), dtype=np.dtype(dtype).name, block_frames=512))
        # Two blocks in one chunk: once staged on the GPU, the second block starts
        # mid-word (byte 1425 of the uint8 payload).
        blocks = np.dtype(dtype).name + "-qem-blocks"
        save_array(directory / (blocks + ".qem"), data, chunk_scans=1024)
        data.tofile(directory / (blocks + ".bin"))
        records.append(dict(name=blocks, shape=list(data.shape), dtype=np.dtype(dtype).name, block_frames=512))
    zeros = np.zeros((17, 33, 2, 3), dtype=np.uint16)
    save_array(directory / "zero.qem", zeros, chunk_scans=512)
    zeros.tofile(directory / "zero.bin")
    records.append(dict(name="zero", shape=list(zeros.shape), dtype="uint16", block_frames=512))
    save_array(directory / "float.qem", np.ones((2, 3, 2, 3), dtype=np.float32))
    second = data.copy()
    second[..., 0, 0] = 3
    save_array(directory / "uint16-series-second.qem", second, chunk_scans=512)
    second.tofile(directory / "uint16-series-second.bin")
    save_array(directory / "shape-mismatch.qem", second[:-1], chunk_scans=512)
    valid = np.ones((2, 3), bool)
    valid[0, 1] = False
    save_array(directory / "validity-mismatch.qem", second, metadata={"valid_pixels": valid}, chunk_scans=512)
    save_array(
        directory / "saturated.qem",
        np.full((1, 1, 256, 256), 65535, dtype=np.uint16),
    )
    (directory / "cases.json").write_text(json.dumps(records))


def generate_h5(directory: Path) -> None:
    """Write Arina-layout bitshuffle-LZ4 acquisitions whose counts span all 16 bit planes.

    Each case is a master file with a pixel mask and two data files of one-frame chunks, plus
    the raw values as ``<name>.bin`` for exact host sums. Counts above 255, the dtype maximum
    on every pixel of one frame, and a masked hot pixel make a decoder that drops high bit
    planes, clips, or keeps the masked pixel fail. uint32 and float32 cover the 32-plane
    decoders.
    """
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for name, dtype, detector, frames in (
        ("h5-u16-64", np.uint16, (64, 64), 300),
        ("h5-u16-128", np.uint16, (128, 128), 120),
        ("h5-u8-64", np.uint8, (64, 64), 90),
        ("h5-u32-64", np.uint32, (64, 64), 60),
        ("h5-f32-64", np.float32, (64, 64), 60),
    ):
        rng = np.random.default_rng(7)
        high = np.finfo(dtype).max if dtype == np.float32 else np.iinfo(dtype).max
        counts = rng.poisson(3, (frames, *detector)).astype(dtype)
        if dtype == np.float32:
            counts += rng.normal(0, 1, counts.shape).astype(dtype)
        counts[0] = high
        counts[1:, detector[0] // 2, detector[1] // 2] = rng.integers(0, min(int(high), 2**31 - 1) + 1, frames - 1)
        counts[1::7, 3, 4] = min(300, high)
        counts[2::11, -2, -3] = high
        bad = np.zeros(detector, np.uint32)
        bad[1, 2] = 1
        counts[:, 1, 2] = high
        with h5py.File(directory / f"{name}_master.h5", "w") as master:
            specific = master.create_group("entry/instrument/detector/detectorSpecific")
            specific["pixel_mask"] = bad
            specific["ntrigger"] = np.array([frames], np.uint32)
        split = frames // 2 + 1
        for index, part in enumerate((counts[:split], counts[split:]), start=1):
            with h5py.File(directory / f"{name}_data_{index:06d}.h5", "w") as data:
                data.create_dataset(
                    "entry/data/data", data=part, chunks=(1, *detector),
                    **hdf5plugin.Bitshuffle(nelems=0, cname="lz4"),
                )
        counts.tofile(directory / f"{name}.bin")
        records.append(dict(name=name, dtype=np.dtype(dtype).name, frames=frames,
                            detector=list(detector), bad_pixels=[int(np.flatnonzero(bad)[0])]))
    (directory / "h5-cases.json").write_text(json.dumps(records))


if __name__ == "__main__":
    generate(Path(sys.argv[1]))
    generate_h5(Path(sys.argv[1]))
