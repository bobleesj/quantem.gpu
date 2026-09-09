"""Encode native count fixtures for the browser parity workflow."""

import argparse
import json
from pathlib import Path

import numpy as np

from quantem.gpu import io
from quantem.gpu.io._ans import ANSFile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    directory = parser.parse_args().directory
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(12345)
    counts = np.empty((65, 64, 2, 2), dtype=np.uint16)
    flat = counts.reshape(-1, 4)
    flat[:, 0] = np.arange(len(flat)) % 512
    flat[:, 1] = rng.integers(0, 65536, len(flat), dtype=np.uint16)
    flat[:, 2] = 65535
    flat[:, 3] = np.arange(len(flat)) % 2 * 255
    small = rng.integers(0, 256, (9, 11, 2, 3), dtype=np.uint8)
    companion = counts.copy()
    companion_flat = companion.reshape(-1, 4)
    companion_flat[:, 0] = (companion_flat[:, 0] + 37) % 512
    companion_flat[:, 3] = 3
    other_dtype = rng.integers(0, 256, counts.shape, dtype=np.uint8)
    cases = []
    for name, data, block_frames in (
        ("uint16-model512-tail", counts, 4096),
        ("uint8-block17-tail", small, 17),
        ("uint16-series-second", companion, 4096),
        ("uint16-profile-mismatch", counts, 2048),
        ("uint8-dtype-mismatch", other_dtype, 4096),
    ):
        saved = io.save(
            directory / f"{name}.ans", data, format="quantem", compression="ans", backend="cpu",
            batch_size=block_frames,
        )
        with ANSFile(saved.path) as source:
            if data.dtype == np.uint16 and block_frames == 4096:
                assert np.diff(source.arrays["context_offsets"]).max() > 256
                assert np.any(source.arrays["literal"])
            decoded = np.concatenate([
                source.decode_block_reference(index)
                for index in range((len(data.reshape(-1, *data.shape[2:])) + block_frames - 1) // block_frames)
            ])
            assert np.array_equal(decoded.reshape(data.shape), data)
        data.tofile(directory / f"{name}.bin")
        cases.append(dict(name=name, shape=data.shape, dtype=data.dtype.name, block_frames=block_frames))
    (directory / "cases.json").write_text(json.dumps(cases), encoding="utf-8")


if __name__ == "__main__":
    main()
