#!/usr/bin/env python3
"""Audit exact 4D-STEM source identity and uint8 admission without mutation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Iterator

import hdf5plugin  # noqa: F401 - registers the bitshuffle/LZ4 HDF5 filter
import h5py
import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def filters(dataset: h5py.Dataset) -> list[dict[str, Any]]:
    creation = dataset.id.get_create_plist()
    result = []
    for index in range(creation.get_nfilters()):
        filter_id, flags, values, name = creation.get_filter(index)
        if isinstance(name, bytes):
            name = name.decode(errors="replace")
        result.append(
            {
                "id": int(filter_id),
                "flags": int(flags),
                "name": str(name),
                "client_values": [int(value) for value in values],
            }
        )
    return result


def scalar_metadata(file: h5py.File, path: str) -> dict[str, Any] | None:
    if path not in file:
        return None
    dataset = file[path]
    value = dataset[()]
    if isinstance(value, bytes):
        value = value.decode(errors="replace")
    elif isinstance(value, np.generic):
        value = value.item()
    units = dataset.attrs.get("units")
    if isinstance(units, bytes):
        units = units.decode(errors="replace")
    return {"value": value, "units": units}


def source_entries(master_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    calibration_paths = [
        "/entry/instrument/detector/beam_center_x",
        "/entry/instrument/detector/beam_center_y",
        "/entry/instrument/detector/detector_distance",
        "/entry/instrument/detector/x_pixel_size",
        "/entry/instrument/detector/y_pixel_size",
        "/entry/instrument/detector/count_time",
        "/entry/instrument/detector/frame_time",
    ]
    entries: list[dict[str, Any]] = []
    with h5py.File(master_path, "r") as master:
        group = master.get("/entry/data")
        if isinstance(group, h5py.Group):
            for name in sorted(group):
                link = group.get(name, getlink=True)
                if isinstance(link, h5py.ExternalLink):
                    entries.append(
                        {
                            "master_link": f"/entry/data/{name}",
                            "path": (master_path.parent / link.filename).resolve(),
                            "dataset_path": link.path,
                        }
                    )
        if not entries and "/entry/data/data" in master:
            entries.append(
                {
                    "master_link": "/entry/data/data",
                    "path": master_path,
                    "dataset_path": "/entry/data/data",
                }
            )
        calibration = {
            path: value
            for path in calibration_paths
            if (value := scalar_metadata(master, path)) is not None
        }
        pixel_mask_summary = None
        mask = master.get("/entry/instrument/detector/detectorSpecific/pixel_mask")
        if isinstance(mask, h5py.Dataset):
            array = np.ascontiguousarray(mask[...])
            pixel_mask_summary = {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "nonzero": int(np.count_nonzero(array)),
                "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
            }
    if not entries:
        raise ValueError(f"no /entry/data source found in {master_path}")
    return entries, {"calibration": calibration, "pixel_mask": pixel_mask_summary}


def batches(
    dataset: h5py.Dataset,
    batch_frames: int,
) -> Iterator[np.ndarray[Any, Any]]:
    if dataset.ndim == 4:
        for row in range(dataset.shape[0]):
            for column in range(0, dataset.shape[1], batch_frames):
                yield np.ascontiguousarray(
                    dataset[row, column : min(column + batch_frames, dataset.shape[1])]
                )
    elif dataset.ndim == 3:
        for start in range(0, dataset.shape[0], batch_frames):
            yield np.ascontiguousarray(
                dataset[start : min(start + batch_frames, dataset.shape[0])]
            )
    else:
        raise ValueError(f"source dataset must be 3D or 4D, got {dataset.shape}")


def audit(master_path: Path, batch_frames: int) -> dict[str, Any]:
    started = time.perf_counter()
    entries, metadata = source_entries(master_path)
    missing = [str(entry["path"]) for entry in entries if not entry["path"].is_file()]
    if missing:
        raise FileNotFoundError(f"missing source files: {missing}")
    unique_paths = list(
        dict.fromkeys([master_path, *(entry["path"] for entry in entries)])
    )
    hashes = {path: sha256_file(path) for path in unique_paths}

    decoded_values = 0
    values_above_255 = 0
    observed_minimum: int | None = None
    observed_maximum: int | None = None
    dtype: np.dtype[Any] | None = None
    detector_shape: tuple[int, int] | None = None
    frame_count = 0
    logical_digest = hashlib.sha256()
    prepared_digest = hashlib.sha256()
    source_files = []
    for ordinal, entry in enumerate(entries):
        with h5py.File(entry["path"], "r") as source_file:
            dataset = source_file[entry["dataset_path"]]
            candidate_dtype = np.dtype(dataset.dtype)
            if candidate_dtype.kind != "u":
                raise ValueError(f"source dtype must be unsigned integer, got {candidate_dtype}")
            candidate_detector = tuple(map(int, dataset.shape[-2:]))
            if dtype is None:
                dtype = candidate_dtype
                detector_shape = candidate_detector
            if candidate_dtype != dtype or candidate_detector != detector_shape:
                raise ValueError("source files do not share dtype and detector geometry")
            frames = int(np.prod(dataset.shape[:-2]))
            local_minimum: int | None = None
            local_maximum: int | None = None
            local_above = 0
            for array in batches(dataset, batch_frames):
                minimum = int(array.min())
                maximum = int(array.max())
                local_minimum = minimum if local_minimum is None else min(local_minimum, minimum)
                local_maximum = maximum if local_maximum is None else max(local_maximum, maximum)
                local_above += int(np.count_nonzero(array > 255))
                decoded_values += int(array.size)
                logical_digest.update(
                    array.astype(candidate_dtype.newbyteorder("<"), copy=False).tobytes(order="C")
                )
                prepared_digest.update(array.astype(np.uint8, copy=False).tobytes(order="C"))
            observed_minimum = (
                local_minimum if observed_minimum is None else min(observed_minimum, local_minimum)
            )
            observed_maximum = (
                local_maximum if observed_maximum is None else max(observed_maximum, local_maximum)
            )
            values_above_255 += local_above
            frame_count += frames
            source_files.append(
                {
                    "ordinal": ordinal,
                    "master_link": entry["master_link"],
                    "path": str(entry["path"]),
                    "dataset_path": entry["dataset_path"],
                    "file_bytes": entry["path"].stat().st_size,
                    "sha256": hashes[entry["path"]],
                    "shape": list(dataset.shape),
                    "dtype": str(candidate_dtype),
                    "chunks": list(dataset.chunks) if dataset.chunks else None,
                    "filters": filters(dataset),
                    "observed_minimum": local_minimum,
                    "observed_maximum": local_maximum,
                    "values_above_255": local_above,
                }
            )

    assert dtype is not None and detector_shape is not None
    side = math.isqrt(frame_count)
    scan_shape = [side, side] if side * side == frame_count else [frame_count]
    shape = [*scan_shape, *detector_shape]
    expected_values = frame_count * math.prod(detector_shape)
    complete = decoded_values == expected_values
    uint8_admitted = bool(
        complete
        and observed_minimum is not None
        and observed_minimum >= 0
        and observed_maximum is not None
        and observed_maximum <= 255
        and values_above_255 == 0
    )
    identity_payload = {
        "schema": "quantem.gpu.ordered-hdf5-source/v1",
        "master_sha256": hashes[master_path],
        "ordered_sources": [
            {
                "ordinal": item["ordinal"],
                "sha256": item["sha256"],
                "bytes": item["file_bytes"],
            }
            for item in source_files
        ],
    }
    source_identity = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema": "quantem.gpu.android-vulkan-real-fixture-audit/v1",
        "status": "ADMITTED" if uint8_admitted else "REJECTED",
        "real_data": True,
        "master": {
            "path": str(master_path),
            "file_bytes": master_path.stat().st_size,
            "sha256": hashes[master_path],
        },
        "source_identity_sha256": source_identity,
        "source_shape": shape,
        "axis_order": ["scan_row", "scan_column", "detector_row", "detector_column"],
        "source_dtype": str(dtype),
        "source_logical_bytes": expected_values * dtype.itemsize,
        "master_plus_sources_file_bytes": sum(path.stat().st_size for path in unique_paths),
        "scan_bin": 1,
        "detector_bin": 1,
        "crop": None,
        "range_audit": {
            "complete": complete,
            "decoded_values": decoded_values,
            "observed_minimum": observed_minimum,
            "observed_maximum": observed_maximum,
            "values_above_255": values_above_255,
            "logical_source_sha256": logical_digest.hexdigest(),
            "uint8_working_representation_admitted": uint8_admitted,
            "prepared_uint8_sha256": prepared_digest.hexdigest() if uint8_admitted else None,
        },
        "master_metadata": metadata,
        "source_files": source_files,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("master", type=Path)
    parser.add_argument("--batch-frames", type=int, default=128)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = audit(arguments.master.resolve(), arguments.batch_frames)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(rendered, end="")
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
