#!/usr/bin/env python
"""Build WebGPU HDF5 block-index sidecars.

The sidecar stores only deterministic bslz4 block offsets and compressed
lengths. It does not copy detector pixels or LZ4 payload bytes; the browser still
loads the original HDF5 data file and runs the same WebGPU decompressor.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path
from typing import Any

import numpy as np

MAGIC = b"QH5IDX01"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-index-json", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-template", default="data_{index:06d}.h5")
    parser.add_argument("--output-template", default="data_{index:06d}.qh5idx")
    parser.add_argument("--data-files", type=int, required=True)
    parser.add_argument(
        "--frames-per-chunk",
        type=int,
        default=0,
        help="Decoded-frame chunk size. The default mirrors the browser 1 GiB output cap.",
    )
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


def _write_sidecar(path: Path, *, meta: dict[str, Any], block_meta: np.ndarray) -> None:
    payload = json.dumps(meta, separators=(",", ":")).encode("utf-8")
    pad = (-len(payload)) % 4
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        fh.write(MAGIC)
        fh.write(struct.pack("<II", len(payload), int(block_meta.size)))
        fh.write(payload)
        if pad:
            fh.write(b"\0" * pad)
        fh.write(block_meta.astype("<u4", copy=False).tobytes(order="C"))


def _build_one(
    data_path: Path,
    output_path: Path,
    index: dict[str, Any],
    frames_per_chunk: int,
) -> dict[str, Any]:
    data = data_path.read_bytes()
    det_rows = int(index["detRows"])
    det_cols = int(index["detCols"])
    det_size = det_rows * det_cols
    n_frames = int(index["nFrames"])
    src_dtype = str(index["srcDtype"])
    offsets = [int(v) for v in index["frameOffsets"]]
    if len(offsets) < n_frames:
        raise ValueError(f"{data_path.name} has {len(offsets)} offsets, expected {n_frames}.")
    src_bytes = _source_bytes(src_dtype)
    block_bytes = _read_be32(data, offsets[0] + 8)
    block_elems = block_bytes // src_bytes
    n_blocks = math.ceil(det_size / block_elems)
    frame_step = frames_per_chunk or max(1, (1024 * 1024 * 1024) // det_size)
    chunks: list[dict[str, int]] = []
    chunk_meta: list[np.ndarray] = []
    meta_offset_words = 0
    total_compressed = 0
    for start in range(0, n_frames, frame_step):
        stop = min(n_frames, start + frame_step)
        meta_abs = np.empty(((stop - start) * n_blocks, 2), dtype=np.uint64)
        range_start = 2**63 - 1
        range_end = 0
        row = 0
        for frame in range(start, stop):
            addr = offsets[frame]
            range_start = min(range_start, addr)
            pos = addr + 12
            for _block in range(n_blocks):
                clen = _read_be32(data, pos)
                meta_abs[row, 0] = pos + 4
                meta_abs[row, 1] = clen
                row += 1
                pos += 4 + clen
            range_end = max(range_end, pos)
        meta_abs[:, 0] -= np.uint64(range_start)
        if int(meta_abs[:, 0].max(initial=0)) > np.iinfo(np.uint32).max:
            raise ValueError(f"{data_path.name} chunk compressed span exceeds the WebGPU uint32 metadata limit.")
        meta = meta_abs.astype(np.uint32, copy=False)
        words = int(meta.size)
        chunks.append(
            {
                "startFrame": start,
                "nFrames": stop - start,
                "rangeStart": int(range_start),
                "rangeEnd": int(range_end),
                "metaOffsetWords": meta_offset_words,
                "metaWords": words,
            }
        )
        meta_offset_words += words
        total_compressed += int(range_end - range_start)
        chunk_meta.append(meta.reshape(-1))
    block_meta = np.concatenate(chunk_meta) if chunk_meta else np.empty(0, dtype=np.uint32)
    meta_json = {
        "detRows": det_rows,
        "detCols": det_cols,
        "nFrames": n_frames,
        "srcDtype": src_dtype,
        "blockElems": block_elems,
        "nBlocksPerFrame": n_blocks,
        "chunks": chunks,
    }
    _write_sidecar(output_path, meta=meta_json, block_meta=block_meta)
    return {
        "input": data_path.name,
        "output": output_path.name,
        "frames": n_frames,
        "chunks": len(chunks),
        "input_bytes": len(data),
        "index_bytes": output_path.stat().st_size,
        "covered_compressed_bytes": total_compressed,
        "block_meta_words": int(block_meta.size),
    }


def main() -> None:
    args = _parse_args()
    manifest = json.loads(args.frame_index_json.read_text(encoding="utf-8"))
    rows = []
    for index in range(1, args.data_files + 1):
        name = args.data_template.format(index=index)
        output_name = args.output_template.format(index=index)
        rows.append(
            _build_one(
                args.input_dir / name,
                args.output_dir / output_name,
                manifest[name],
                max(0, int(args.frames_per_chunk)),
            )
        )
    summary = {
        "kind": "webgpu-h5-block-index-sidecar-build",
        "data_files": args.data_files,
        "total_input_bytes": sum(int(row["input_bytes"]) for row in rows),
        "total_index_bytes": sum(int(row["index_bytes"]) for row in rows),
        "total_covered_compressed_bytes": sum(int(row["covered_compressed_bytes"]) for row in rows),
        "files": rows,
    }
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
