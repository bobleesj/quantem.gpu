"""Generate candidate synthetic conformance files; never overwrite frozen files."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from quantem.gpu import io
from quantem.gpu.io._qem_reference import read_envelope
from quantem.gpu.io.qem_validation import validate_qem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for name, shape, dtype, batch in (
        ("u8-interval-boundary", (3, 179, 2, 3), np.uint8, 1024),
        ("u16-multiple-chunks", (3, 179, 9, 9), np.uint16, 512),
        ("float32-special-bits", (1, 2, 128, 128), np.float32, 1),
    ):
        if dtype == np.float32:
            patterns = np.array(
                [
                    0,
                    0x80000000,
                    0x7F800000,
                    0xFF800000,
                    0x7FC01234,
                    0x3E800000,
                    0xBE800000,
                    1,
                ],
                dtype=np.uint32,
            )
            data = np.resize(patterns, shape).view(np.float32)
        else:
            data = (
                (np.arange(np.prod(shape), dtype=np.uint64) % 19)
                .astype(dtype)
                .reshape(shape)
            )
            frames = data.reshape(-1, shape[2] * shape[3])
            frames[:, 0] = 0
            frames[:, 1] = 7
            frames[:, 2] = 0
            frames[511, 2] = 128
            frames[512, 2] = 1
            frames[536, 2] = 3
            frames[:, 3] = np.arange(len(frames)) % 2 * np.iinfo(dtype).max
        path = args.output / f"{name}.qem"
        metadata = dict(
            source_metadata={"fixture": name, "data_origin": "deterministic synthetic"},
            scan_sampling_A=[0.4, 0.6],
            detector_sampling=[0.02, 0.03],
            detector_sampling_unit="1/angstrom",
            voltage_kV=300,
        )
        io.save(path, data, metadata=metadata, backend="cpu", batch_size=batch)
        np.save(args.output / f"{name}.npy", data)
        header, _ = read_envelope(path)
        with io.load(path, backend="cpu") as reopened:
            assert reopened.data.tobytes() == data.tobytes()
        records.append(
            dict(
                name=name,
                shape=list(shape),
                dtype=np.dtype(dtype).name,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                counts_sha256=hashlib.sha256(data.tobytes()).hexdigest(),
                scientific_metadata=header["scientific_metadata"],
                validation=validate_qem(path),
            )
        )
    (args.output / "manifest.json").write_text(
        json.dumps(
            dict(
                license="MIT",
                origin="synthetic",
                specification="0.0.1",
                entries=records,
            ),
            indent=2,
        )
        + "\n"
    )
    print(f"Verified {len(records)} candidate references in {args.output}")


if __name__ == "__main__":
    main()
