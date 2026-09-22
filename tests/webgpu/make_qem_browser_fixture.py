"""Generate small disposable QEM acquisitions for the browser GPU parity gate."""

import json
from pathlib import Path
import sys

import numpy as np

from quantem.gpu.io._qem_reference import save_array


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


if __name__ == "__main__":
    generate(Path(sys.argv[1]))
