"""Create small synthetic references using the production native Metal exporter.

Run on Apple silicon: python scripts/build_qem_references.py NEW_DIRECTORY
Never uses private acquisitions or replaces an existing reference bundle.
"""

import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np


def main() -> None:
    folder = Path(sys.argv[1])
    folder.mkdir(parents=True, exist_ok=False)
    entries = []
    for dtype in ("uint8", "uint16"):
        shape = (3, 5, 16, 16)
        index = np.arange(np.prod(shape), dtype=np.uint64).reshape(shape)
        counts = (
            (index * 17 + index // 7) % (256 if dtype == "uint8" else 65536)
        ).astype(dtype)
        counts[:, :, 0, 0] = 0
        counts[:, :, 0, 1] = np.iinfo(dtype).max
        original = folder / f"{dtype}.npy"
        saved = folder / f"{dtype}.qem"
        np.save(original, counts)
        subprocess.run(
            [
                "bash",
                "scripts/check_qem_reference.sh",
                str(original),
                str(saved),
                "--write",
            ],
            check=True,
        )
        # Scrub only the synthetic build-machine locator, keeping encoded payload
        # bytes and all scientific fields unchanged. Recompute the header digest.
        data = saved.read_bytes()
        length, start = struct.unpack("<QQ", data[8:24])
        header = json.loads(data[56 : 56 + length])
        header["metadata"]["source_path"] = original.name
        blob = json.dumps(
            header, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        saved.write_bytes(
            data[:8]
            + struct.pack("<QQ", len(blob), 56 + len(blob))
            + hashlib.sha256(blob).digest()
            + blob
            + data[start:]
        )
        entries.append(
            dict(
                name=dtype,
                shape=list(shape),
                dtype=dtype,
                counts_sha256=hashlib.sha256(counts.tobytes()).hexdigest(),
                scientific_metadata=header["scientific_metadata"],
                files={
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (original, saved)
                },
            )
        )
    (folder / "manifest.json").write_text(
        json.dumps(
            dict(
                bundle_version=1,
                synthetic=True,
                license="MIT",
                generator="scripts/build_qem_references.py",
                entries=entries,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
