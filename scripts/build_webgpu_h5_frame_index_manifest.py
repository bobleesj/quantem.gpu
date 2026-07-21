#!/usr/bin/env python
"""Build WebGPU HDF5 frame-index manifests from chunked detector data files.

The manifest stores HDF5 chunk byte offsets and detector metadata only. It does
not copy detector pixels or compressed LZ4 payloads. The WebGPU block-index and
selected-block sidecar builders use this as their source of truth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--data-template", default="data_{index:06d}.h5")
    parser.add_argument("--data-files", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def _find_stack(group: h5py.Group) -> h5py.Dataset:
    if "entry/data/data" in group:
        ds = group["entry/data/data"]
        if isinstance(ds, h5py.Dataset):
            return ds
    for value in group.values():
        if isinstance(value, h5py.Dataset) and value.ndim == 3:
            return value
        if isinstance(value, h5py.Group):
            try:
                return _find_stack(value)
            except KeyError:
                pass
    raise KeyError("Could not find a 3-D detector stack dataset.")


def _src_dtype(dtype: Any) -> str:
    text = str(dtype)
    if "uint8" in text:
        return "uint8"
    if "uint32" in text:
        return "uint32"
    if "float32" in text:
        return "float32"
    return "uint16"


def _read_one(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as file:
        ds = _find_stack(file)
        if ds.ndim != 3:
            raise ValueError(f"{path.name}: expected a 3-D detector stack, got {ds.shape}.")
        n_frames, det_rows, det_cols = (int(v) for v in ds.shape)
        if not ds.chunks or int(ds.chunks[0]) != 1:
            raise ValueError(
                f"{path.name}: expected one frame per compressed chunk, got chunks={ds.chunks}."
            )
        src_dtype = _src_dtype(ds.dtype)
        offsets = [
            int(ds.id.get_chunk_info_by_coord((frame, 0, 0)).byte_offset)
            for frame in range(n_frames)
        ]
    return {
        "detRows": det_rows,
        "detCols": det_cols,
        "nFrames": n_frames,
        "srcDtype": src_dtype,
        "frameOffsets": offsets,
    }


def main() -> None:
    args = _parse_args()
    manifest: dict[str, Any] = {}
    for index in range(1, args.data_files + 1):
        name = args.data_template.format(index=index)
        manifest[name] = _read_one(args.input_dir / name)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(manifest, separators=(",", ":")) + "\n", encoding="utf-8")
    total_frames = sum(int(row["nFrames"]) for row in manifest.values())
    print(json.dumps({"files": len(manifest), "totalFrames": total_frames}, indent=2))


if __name__ == "__main__":
    main()
