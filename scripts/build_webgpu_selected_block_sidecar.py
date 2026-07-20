#!/usr/bin/env python
"""Build WebGPU selected-block bslz4 sidecars from HDF5 data files.

The sidecar keeps native bslz4/LZ4 streams for only the detector bitshuffle
blocks touched by a detector mask. It is a derived acceleration artifact: the
browser can load the exact raw evidence needed by a BF/DF/ADF product without
reading unrelated detector blocks from the native HDF5 file.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path
from typing import Any

import numpy as np

MAGIC = b"QBSLZ4S1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-index-json", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-template", default="data_{index:06d}.h5")
    parser.add_argument("--output-template", default="data_{index:06d}_selected.qbslz4")
    parser.add_argument("--data-files", type=int, required=True)
    parser.add_argument("--center-row", type=float, required=True)
    parser.add_argument("--center-col", type=float, required=True)
    parser.add_argument("--radius", type=float, required=True)
    parser.add_argument("--summary-json", type=Path)
    return parser.parse_args()


def _read_be32(data: bytes | bytearray | memoryview, offset: int) -> int:
    return int.from_bytes(data[offset : offset + 4], "big", signed=False)


def _source_bytes(dtype: str) -> int:
    if dtype == "uint8":
        return 1
    if dtype in {"uint32", "float32"}:
        return 4
    return 2


def _selected_blocks(det_rows: int, det_cols: int, block_elems: int, center_row: float, center_col: float, radius: float) -> list[int]:
    yy, xx = np.ogrid[:det_rows, :det_cols]
    mask = (yy - center_row) ** 2 + (xx - center_col) ** 2 <= radius * radius
    pixels = np.flatnonzero(mask.ravel())
    return sorted({int(p // block_elems) for p in pixels})


def _write_sidecar(
    path: Path,
    *,
    meta: dict[str, Any],
    block_meta: np.ndarray,
    compressed: bytearray,
) -> None:
    payload = json.dumps(meta, separators=(",", ":")).encode("utf-8")
    header = bytearray()
    header += MAGIC
    header += struct.pack("<II", len(payload), int(block_meta.size))
    pad = (-len(payload)) % 4
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        fh.write(header)
        fh.write(payload)
        if pad:
            fh.write(b"\0" * pad)
        fh.write(block_meta.astype("<u4", copy=False).tobytes(order="C"))
        fh.write(compressed)


def _build_one(
    data_path: Path,
    output_path: Path,
    index: dict[str, Any],
    selected: list[int] | None,
    center_row: float,
    center_col: float,
    radius: float,
) -> dict[str, Any]:
    data = data_path.read_bytes()
    det_rows = int(index["detRows"])
    det_cols = int(index["detCols"])
    n_frames = int(index["nFrames"])
    src_dtype = str(index["srcDtype"])
    offsets = [int(v) for v in index["frameOffsets"]]
    src_bytes = _source_bytes(src_dtype)
    block_bytes = _read_be32(data, offsets[0] + 8)
    block_elems = block_bytes // src_bytes
    n_blocks = math.ceil(det_rows * det_cols / block_elems)
    selected_blocks = selected or _selected_blocks(det_rows, det_cols, block_elems, center_row, center_col, radius)
    compressed = bytearray()
    block_meta = np.empty((n_frames * len(selected_blocks), 2), dtype=np.uint32)
    row = 0
    selected_set = set(selected_blocks)
    for frame, addr in enumerate(offsets[:n_frames]):
        pos = int(addr) + 12
        for block in range(n_blocks):
            clen = _read_be32(data, pos)
            stream_start = pos + 4
            if block in selected_set:
                block_meta[row, 0] = len(compressed)
                block_meta[row, 1] = clen
                compressed.extend(data[stream_start : stream_start + clen])
                row += 1
            pos = stream_start + clen
    if row != block_meta.shape[0]:
        raise RuntimeError(f"Internal sidecar row mismatch for {data_path.name}: {row} != {block_meta.shape[0]}")
    meta = {
        "detRows": det_rows,
        "detCols": det_cols,
        "nFrames": n_frames,
        "srcDtype": src_dtype,
        "blockElems": block_elems,
        "selectedBlockIds": selected_blocks,
        "center": [center_row, center_col],
        "radius": radius,
    }
    _write_sidecar(output_path, meta=meta, block_meta=block_meta.reshape(-1), compressed=compressed)
    return {
        "input": data_path.name,
        "output": output_path.name,
        "frames": n_frames,
        "selected_blocks": selected_blocks,
        "input_bytes": len(data),
        "sidecar_bytes": output_path.stat().st_size,
        "selected_compressed_bytes": len(compressed),
    }


def main() -> None:
    args = _parse_args()
    manifest = json.loads(args.frame_index_json.read_text(encoding="utf-8"))
    rows = []
    selected: list[int] | None = None
    for index in range(1, args.data_files + 1):
        name = args.data_template.format(index=index)
        output_name = args.output_template.format(index=index)
        frame_index = manifest[name]
        row = _build_one(
            args.input_dir / name,
            args.output_dir / output_name,
            frame_index,
            selected,
            args.center_row,
            args.center_col,
            args.radius,
        )
        selected = row["selected_blocks"]
        rows.append(row)
    summary = {
        "kind": "webgpu-selected-block-sidecar-build",
        "data_files": args.data_files,
        "center": [args.center_row, args.center_col],
        "radius": args.radius,
        "selected_blocks": selected or [],
        "total_input_bytes": sum(int(r["input_bytes"]) for r in rows),
        "total_sidecar_bytes": sum(int(r["sidecar_bytes"]) for r in rows),
        "total_selected_compressed_bytes": sum(int(r["selected_compressed_bytes"]) for r in rows),
        "files": rows,
    }
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
