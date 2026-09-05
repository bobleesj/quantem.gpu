"""Source-preserving preparation of trusted browse-service bindings."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

from quantem.gpu.io._compact_h5 import CompactH5Index
from quantem.gpu.io.integrity import _manifest_bytes, _seal_source


def prepare_browse_source(
    master: str | Path,
    source: str | Path,
    destination: str | Path,
    *,
    expected_source_sha256: str,
) -> Path:
    """Verify a qualified packed source and publish its deployment registry.

    This one-time preparation reads every source byte and writes only a small
    integrity manifest and registry into a new directory. Neither input is
    changed or copied. Raw-HDF5 packing and detector calibration are separate
    scientific preparation steps, never hidden inside loading or this binding.

    Parameters
    ----------
    master
        Catalogued acquisition master identifying this dataset to clients.
    source
        Already qualified, immutable lossless-packed source for that master.
        Calibration and prepared moments, when present, remain unchanged.
    destination
        New deployment directory. Existing directories are never replaced.
    expected_source_sha256
        Independently recorded SHA-256 of the complete packed source.

    Returns
    -------
    pathlib.Path
        Published registry accepted by ``quantem-gpu serve --compact-sources``.

    Raises
    ------
    ValueError
        If source integrity, raw reconstruction, or acquisition shape fails.
    FileExistsError
        If the destination already exists; choose a new preparation directory.

    Examples
    --------
    >>> from quantem.gpu.remote import prepare_browse_source
    >>> registry = prepare_browse_source(
    ...     "sample_master.h5", "sample-prepared.h5", "sample-deployment",
    ...     expected_source_sha256=qualified_source_sha256,
    ... )
    """
    from quantem.gpu.io import inspect

    master_path = Path(master).expanduser().resolve(strict=True)
    source_path = Path(source).expanduser().resolve(strict=True)
    destination_path = Path(destination).expanduser().absolute()
    if destination_path.exists():
        raise FileExistsError(
            f"Refusing to replace {destination_path}; choose a new deployment directory."
        )
    seal = _seal_source(source_path, expected_source_sha256)
    index = CompactH5Index.from_file(source_path)
    index.require_raw_reconstruction()
    acquisition = inspect(master_path)
    if acquisition.scan_shape is None or acquisition.detector_shape is None:
        raise ValueError(
            "Master has no complete 4D shape; qualify its acquisition metadata first."
        )
    master_shape = (*acquisition.scan_shape, *acquisition.detector_shape)
    if tuple(index.shape) != tuple(master_shape):
        raise ValueError(
            f"Packed source shape {index.shape} differs from master {master_shape}; "
            "select the qualified source for this acquisition."
        )
    manifest = _manifest_bytes(seal)
    row = {
        "master": str(master_path),
        "compact": str(source_path),
        "expected_whole_file_sha256": seal.whole_file_sha256,
        "chunk_integrity_manifest": "source.integrity.json",
        "expected_chunk_integrity_manifest_sha256": hashlib.sha256(
            manifest
        ).hexdigest(),
    }
    registry = (
        json.dumps(
            {
                "schema": "quantem.gpu.compact-browse-sources/v1",
                "sources": [row],
            },
            indent=2,
        )
        + "\n"
    ).encode()
    destination_path.mkdir()
    try:
        (destination_path / "source.integrity.json").write_bytes(manifest)
        partial = destination_path / "sources.json.partial"
        with partial.open("xb") as stream:
            stream.write(registry)
            stream.flush()
            os.fsync(stream.fileno())
        result = destination_path / "sources.json"
        os.replace(partial, result)
    except BaseException:
        # Only this call's exclusively created output directory is removed.
        shutil.rmtree(destination_path)
        raise
    return result
