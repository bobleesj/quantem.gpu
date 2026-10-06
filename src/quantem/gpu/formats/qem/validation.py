"""GPU-free integrity checks for portable QuantEM data files.

Run ``python -m quantem.gpu.formats.qem.validation acquisition.qem``. This checks
the saved bytes, not their decoded scientific values or source completeness.
"""

import argparse
import json
from pathlib import Path

from quantem.gpu.formats.qem.snapshot import (
    BLOCK_BYTES,
    FLOAT_CODEC,
    INTEGER_CODEC,
    SCALED_CODEC,
    read_envelope,
    read_header,
    validate_declared_processing,
    validate_float_layout,
    verify_body,
)


def validate_qem(path: str | Path) -> dict[str, object]:
    """Verify the envelope, metadata contract and every body checksum.

    Parameters
    ----------
    path : str or pathlib.Path
        Existing QEM file; its signature, not its suffix, identifies the format.

    Returns
    -------
    dict
        JSON-compatible integrity report. Codec-layout and decoded-parity
        status are separate: checksums never establish scientific correctness.

    Raises
    ------
    ValueError
        The envelope, metadata, file length or a checksum is invalid.
    NotImplementedError
        The codec has no qualified validation route.

    Examples
    --------
    >>> report = validate_qem("counts.qem")  # doctest: +SKIP
    >>> report["integrity"]  # doctest: +SKIP
    'verified'
    """
    path = Path(path)
    before = path.stat()
    header, start = read_envelope(path)
    validate_declared_processing(header)
    shape = header.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 4
        or any(type(size) is not int or not 0 < size < 1 << 24 for size in shape)
    ):
        raise ValueError("Invalid QEM shape; expected four positive integer axes.")
    codec = header.get("codec")
    if codec not in (INTEGER_CODEC, FLOAT_CODEC, SCALED_CODEC):
        raise NotImplementedError(
            f"QEM codec {codec!r} is not supported by this validator."
        )
    body_bytes = header.get("bytes")
    if (
        type(body_bytes) is not int
        or body_bytes <= 0
        or start + body_bytes != before.st_size
    ):
        raise ValueError(
            "QEM file length disagrees with metadata; recopy the file."
        )
    checksums = header.get("sha256")
    if (
        not isinstance(checksums, list)
        or len(checksums) != (body_bytes - 1) // BLOCK_BYTES + 1
    ):
        raise ValueError("Incomplete QEM checksum table; recopy the file.")
    with path.open("rb") as handle:
        verify_body(handle, header, start)
        if codec == FLOAT_CODEC:
            validate_float_layout(handle, header, start)
        else:
            # The same geometry and span check the loaders run.
            read_header(path)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError(
            "QEM file changed during validation; retry with a stable copy."
        )
    scientific = header["scientific_metadata"]
    return dict(
        container_version=header["container_version"],
        codec=codec,
        shape=shape,
        dtype=header.get("dtype"),
        file_bytes=before.st_size,
        integrity="verified",
        codec_layout="verified",
        decoded_parity="not_checked",
        metadata_coverage=scientific.get("source_metadata_coverage", "unknown"),
        processing=[record["operation"] for record in scientific["processing"]],
        measurements_changed_by=[
            record["operation"] for record in scientific["processing"]
            if record["changes_measurements"]
        ],
        calibrated_axes=[
            axis["name"] for axis in scientific["axes"] if "sampling" in axis
        ],
        normalized_fields=sorted(scientific.get("electron_microscope", {})),
        override_fields=sorted(scientific.get("calibration_overrides", {})),
    )


def main(argv: list[str] | None = None) -> int:
    """Validate files without loading a GPU; emit one JSON report per path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args(argv)
    failed = False
    for path in args.paths:
        try:
            print(
                json.dumps(dict(path=str(path), **validate_qem(path)), sort_keys=True)
            )
        except (OSError, ValueError, NotImplementedError, TypeError, KeyError) as error:
            failed = True
            print(
                json.dumps(dict(path=str(path), integrity="failed", error=str(error)))
            )
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
