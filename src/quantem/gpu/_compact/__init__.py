"""Private exact prepared-series storage and detector queries.

Use quantem.gpu.io.load and quantem.gpu.detector.prepare.
"""

import hashlib

FORMAT = "compact-prepared-series-v1"
LAYOUT_FORMAT = "compact-series-records-v1"
QUERY_ABI = "compact-series-v1"

# Retain exact recognition of pre-publication labels without distributing
# private source identities. These hashes identify labels, not payloads.
_LEGACY_LABEL_HASHES = {
    FORMAT: "db5e0e20da8d2d45374f29f99b1c02f955335444a0a26869a1302285e5b834ca",
    LAYOUT_FORMAT: "893ce4319aaaae5a69d897d984a6836f2cd2148a70fbb183091d7e03df4394c0",
}


def _matches_format(value: object, expected: str) -> bool:
    """Recognize the canonical label or its exact historical equivalent."""
    return isinstance(value, str) and (
        value == expected
        or hashlib.sha256(value.encode()).hexdigest() == _LEGACY_LABEL_HASHES[expected]
    )


def implementation_id() -> str:
    """Hash the installed implementation, including compiled-source inputs."""
    from pathlib import Path

    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted((*root.glob("*.py"), *root.glob("kernels/*"))):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()
