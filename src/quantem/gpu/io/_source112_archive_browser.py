"""Export preserved source112 metadata without reading or rewriting payloads."""

import errno
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile

import numpy as np


_FORMAT = "source112-tans1024-pair-v1"
_SHAPE = [66, 512, 512, 192, 192]
_COMPONENTS = {"dense", "dense_offsets", "sparse", "sparse_offsets"}
_GLOBALS = {
    "codec__decoding": ((81, 1024), "<u4"),
    "model_ids": ((264, 36864), "|u1"),
    "planner__cache_map": ((36864,), "<i4"),
    "planner__hardware": ((36864,), "|b1"),
    "planner__valid": ((36864,), "|i1"),
}


def _digest(data: bytes | memoryview) -> str:
    return hashlib.sha256(data).hexdigest()


def _valid_digest(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _relative_file(root: Path, name: str) -> Path:
    """Resolve an archive path while keeping it inside its archive directory."""
    if (not isinstance(name, str) or not name or "\\" in name
            or any(char in name for char in ":?#")
            or name.startswith("/")
            or any(part in {"", ".", ".."} for part in name.split("/"))):
        raise ValueError(f"Use an archive-relative file path; got {name!r}.")
    relative = PurePosixPath(name)
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Archive file escapes its directory: {name!r}.")
    return path


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(
            f"Require {name} to be an integer >= {minimum}; got {value!r}."
        )
    return value


def _metadata(root: Path) -> tuple[dict, dict[str, np.ndarray], str]:
    """Validate the complete archive manifest and authenticated global arrays."""
    checkpoint_bytes = (root / "checkpoint.json").read_bytes()
    checkpoint = json.loads(checkpoint_bytes)
    layout = checkpoint["layout"]
    if ((not isinstance(checkpoint["format"], str) or not checkpoint["format"].endswith("-prepared-source254-v1"))
            or checkpoint["complete"] is not True
            or (not isinstance(layout["format"], str) or not layout["format"].endswith("-resident-source112-index180-v1"))
            or layout["complete"] is not True
            or layout["shape"] != _SHAPE
            or layout["source_dtype"] != "<u2"
            or layout["byte_order"] != "little"
            or layout["chunk_scans"] != 16384
            or layout["stream_scans"] != 512
            or checkpoint["index_rebuild"]["source_codec"] != _FORMAT):
        raise ValueError("Select a complete native 66-acquisition source112 archive.")
    acquisitions = checkpoint["original_acquisitions"]["acquisitions"]
    if (len(acquisitions) != 66
            or any(not _valid_digest(item.get("source_identity_sha256"))
                   for item in acquisitions)):
        raise ValueError(
            "Preserve all 66 original acquisition identities and SHA-256 digests."
        )
    metadata_path = _relative_file(root, checkpoint["global_state_file"])
    metadata_bytes = metadata_path.read_bytes()
    if _digest(metadata_bytes) != checkpoint["global_state_file_sha256"]:
        raise ValueError(
            "Global-state NPZ differs from its archive digest; "
            "restore the original metadata."
        )
    with np.load(io.BytesIO(metadata_bytes), allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in _GLOBALS}
    for name, values in arrays.items():
        expected_shape, expected_dtype = _GLOBALS[name]
        spec = checkpoint["global_state"][name]
        if (values.shape != expected_shape or values.dtype.str != expected_dtype
                or list(values.shape) != spec["shape"]
                or values.dtype.str != spec["dtype"]
                or values.nbytes != spec["nbytes"]
                or _digest(values.tobytes(order="C")) != spec["sha256"]):
            raise ValueError(
                f"Global array {name} differs from its native shape, dtype or digest."
            )
    valid = arrays["planner__valid"]
    if np.any((valid != 0) & (valid != 1)):
        raise ValueError("Detector validity must contain exactly 36864 binary values.")
    mapping = arrays["planner__cache_map"]
    hardware = arrays["planner__hardware"]
    dense = np.flatnonzero(mapping < 0).astype("<u4")
    selected = np.flatnonzero(mapping >= 0)
    if (dense.size != 17466 or selected.size != 19398
            or not np.array_equal(np.sort(mapping[selected]), np.arange(19398))
            or np.any(mapping[hardware] >= 0)):
        raise ValueError(
            "Dense and sparse columns must partition all 36864 detector pixels."
        )
    models = arrays["model_ids"]
    if (np.any((models[:, dense] >= 81) & (models[:, dense] != 255))
            or np.any(models[:, hardware] != 255)):
        raise ValueError(
            "Preserve valid tANS models and literal hardware-count streams."
        )
    sparse = np.empty(19398, dtype="<u4")
    sparse[mapping[selected]] = selected
    exported = {
        "dense_columns": dense,
        "sparse_columns": sparse,
        "model_ids": models,
        "decoding": arrays["codec__decoding"],
        "hardware": hardware.astype("u1"),
        "valid": valid.astype("u1"),
    }
    return checkpoint, exported, _digest(checkpoint_bytes)


def _validate_records(root: Path, layout: dict) -> list[Path]:
    """Validate record metadata and shard sizes using stat, never payload reads."""
    alignment = _integer(layout["component_alignment"], "component alignment", 1)
    record_alignment = _integer(layout["record_alignment"], "record alignment", 1)
    files = layout["files"]
    if not files:
        raise ValueError("The source112 archive has no data shards.")
    paths = [_relative_file(root, item["name"]) for item in files]
    if len(set(paths)) != len(paths):
        raise ValueError("Source112 shard paths must be distinct.")
    for item, path in zip(files, paths, strict=True):
        size = _integer(item["nbytes"], "shard size", 1)
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError(
                f"Restore complete data shard {item['name']!r}; "
                "its size differs from the checkpoint."
            )
    chunks = layout["chunks"]
    if len(chunks) != 1056:
        raise ValueError("Preserve all 1056 source records, in acquisition/scan order.")
    ends = [0] * len(paths)
    for index, record in enumerate(chunks):
        if (record["chunk"] != index or record["acquisition"] != index // 16
                or record["first_scan"] != (index % 16) * 16384
                or record["scan_count"] != 16384
                or not _valid_digest(record["sha256"])):
            raise ValueError(
                f"Source record {index} has inconsistent acquisition, scan or hash identity."
            )
        shard = _integer(record["shard"], "shard index")
        start = _integer(record["file_offset"], "record offset")
        size = _integer(record["record_bytes"], "record size", 1)
        if (shard >= len(paths) or size > 256 << 20
                or start % record_alignment or size % record_alignment
                or start < ends[shard] or start + size > files[shard]["nbytes"]):
            raise ValueError(
                f"Source record {index} overlaps or exceeds its data shard."
            )
        ends[shard] = start + size
        names = set()
        component_end = 0
        for component in record["components"]:
            name = component["name"]
            offset = _integer(component["offset"], "component offset")
            count = _integer(component["nbytes"], "component size")
            if (name in names or component["dtype"] != "<u4"
                    or count % 4 or offset % alignment
                    or offset < component_end or offset + count > size
                    or component["shape"] != [count // 4]):
                raise ValueError(
                    f"Source record {index} has invalid component {name!r}."
                )
            names.add(name)
            component_end = offset + count
            if name in {"dense_offsets", "sparse_offsets"}:
                expected_words = (
                    17466 * 9 if name == "dense_offsets" else 19398 * 9 + 1
                )
                if count != expected_words * 4:
                    raise ValueError(
                        f"Source record {index} has invalid {name} extent."
                    )
        if not _COMPONENTS.issubset(names):
            raise ValueError(
                f"Source record {index} is missing a native source component."
            )
    if ends != [item["nbytes"] for item in files]:
        raise ValueError(
            "Data shard sizes must end at their final preserved source record."
        )
    return paths


def _export_source112_browser(path: str | Path, output: str | Path) -> Path:
    """Export authenticated globals and link every original encoded record.

    Only checkpoint.json and the global NPZ are read. Payload file sizes and
    record/component bounds are checked, but payload SHA-256 digests are carried
    forward without verification. No decoding, reencoding or GPU import occurs.
    The original checkpoint is retained verbatim as a nested manifest object.
    Payloads are hardlinked so browser directory grants include regular files.
    Choose a new output directory on the archive's filesystem, outside the
    immutable archive. Cross-filesystem linking fails without copying payloads.
    """
    root = Path(path).resolve()
    output = Path(output).resolve()
    if output == root or output.is_relative_to(root):
        raise ValueError(
            "Choose an output directory outside the preserved source archive."
        )
    if output.exists():
        raise FileExistsError(
            f"Output already exists: {output}. Choose a new export directory."
        )
    checkpoint, arrays, checkpoint_sha256 = _metadata(root)
    shards = _validate_records(root, checkpoint["layout"])
    filenames = {
        "dense_columns": "dense-columns.u32",
        "sparse_columns": "sparse-columns.u32",
        "model_ids": "model-ids.u8",
        "decoding": "decoding.u32",
        "hardware": "hardware.u8",
        "valid": "valid.u8",
    }
    reserved = {"manifest.json", *filenames.values()}
    shard_names = [item["name"] for item in checkpoint["layout"]["files"]]
    if any(name in reserved for name in shard_names):
        raise ValueError("Archive shard names collide with browser metadata filenames.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}-", dir=output.parent
    ) as temporary:
        destination = Path(temporary)
        descriptors = {}
        for name, values in arrays.items():
            raw = values.tobytes(order="C")
            (destination / filenames[name]).write_bytes(raw)
            descriptors[name] = {
                "file": filenames[name], "dtype": values.dtype.str,
                "shape": list(values.shape), "nbytes": len(raw),
                "sha256": _digest(raw),
            }
        for name, source in zip(shard_names, shards, strict=True):
            link = destination / name
            link.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source, link)
            except OSError as error:
                if error.errno == errno.EXDEV:
                    raise ValueError(
                        "Browser folder grants require regular payload files. "
                        "Choose an export directory on the same filesystem as "
                        "the archive so payloads can be hardlinked without copying."
                    ) from error
                raise
        manifest = {
            "format": _FORMAT,
            "shape": _SHAPE,
            "dtype": "<u2",
            "layout": checkpoint["layout"],
            "globals": descriptors,
            "original_acquisitions": checkpoint["original_acquisitions"],
            "source_checkpoint": checkpoint,
            "provenance": {
                "checkpoint_sha256": checkpoint_sha256,
                "global_state_file_sha256": checkpoint["global_state_file_sha256"],
                "metadata_only": True,
                "payload_hashes_verified": False,
                "payload_policy": "unchanged-original-shards",
            },
        }
        (destination / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        destination.rename(output)
    return output / "manifest.json"
