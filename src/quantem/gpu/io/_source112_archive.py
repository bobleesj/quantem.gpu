"""Bounded migration of the preserved source112 archive, independent of viewers."""

import hashlib
import json
from pathlib import Path

import numpy as np

from ._ans import _write_ans_blocks


def _sha(values) -> str:
    return hashlib.sha256(memoryview(values).cast("B")).hexdigest()


class _Source112Archive:
    """Read immutable source254 archive records on an explicitly selected GPU.

    Only four source components are read. No index180 query products, native
    viewer process, original encoder module or implicit device are used.
    """

    def __init__(self, path: str | Path, *, device: int):
        self.root = Path(path)
        self.manifest = json.loads((self.root / "checkpoint.json").read_text())
        manifest = self.manifest
        layout = manifest["layout"]
        if ((not isinstance(manifest["format"], str) or not manifest["format"].endswith("-prepared-source254-v1"))
                or manifest["complete"] is not True
                or (not isinstance(layout["format"], str) or not layout["format"].endswith("-resident-source112-index180-v1"))
                or layout["shape"] != [66, 512, 512, 192, 192]
                or manifest["index_rebuild"]["source_codec"] != "source112-tans1024-pair-v1"):
            raise ValueError("Require a complete preserved source254/source112 archive.")
        metadata_path = self.root / manifest["global_state_file"]
        if hashlib.sha256(metadata_path.read_bytes()).hexdigest() != manifest["global_state_file_sha256"]:
            raise ValueError("Source112 global-state file differs from its archive digest.")
        names = ("codec__decoding", "model_ids", "planner__cache_map", "planner__hardware")
        with np.load(metadata_path, allow_pickle=False) as saved:
            arrays = {name: saved[name] for name in names}
        for name, values in arrays.items():
            spec = manifest["global_state"][name]
            if (list(values.shape) != spec["shape"] or values.dtype.str != spec["dtype"]
                    or _sha(values) != spec["sha256"]):
                raise ValueError(f"Source112 global array differs from its archive digest: {name}.")
        mapping = arrays["planner__cache_map"]
        hardware = arrays["planner__hardware"]
        dense = np.flatnonzero(mapping < 0).astype(np.int32)
        sparse = np.empty(19398, np.int32)
        selected = np.flatnonzero(mapping >= 0)
        if (mapping.shape != (36864,) or len(dense) != 17466 or len(selected) != 19398
                or not np.array_equal(np.sort(mapping[selected]), np.arange(19398))
                or np.any(mapping[hardware.astype(bool)] >= 0)):
            raise ValueError("Source112 dense and sparse columns must partition the native detector.")
        sparse[mapping[selected]] = selected
        self.models = arrays["model_ids"]
        self.decoding = arrays["codec__decoding"]
        if self.models.shape != (264, 36864) or self.decoding.shape != (81, 1024):
            raise ValueError("Source112 model dimensions differ from source112.")
        if (np.any((self.models[:, dense] >= 81) & (self.models[:, dense] != 255))
                or np.any(self.models[:, hardware.astype(bool)] != 255)):
            raise ValueError("Source112 source must preserve hardware counts in literal streams.")
        import cupy as cp

        self.device = cp.cuda.Device(device)
        source = Path(__file__).parent / "backends/cuda/_source112_archive.cu"
        with self.device:
            self.module = cp.RawModule(code=source.read_text(), options=("--std=c++11",))
            self.dense = cp.asarray(dense)
            self.sparse = cp.asarray(sparse)
            self.tables = cp.asarray(self.decoding)
        self._chunk = None
        self._components = None

    def _record(self, chunk: int):
        import cupy as cp

        layout = self.manifest["layout"]
        record = layout["chunks"][chunk]
        if (record["chunk"] != chunk or record["scan_count"] != 16384
                or record["acquisition"] != chunk // 16
                or record["first_scan"] != (chunk % 16) * 16384
                or not 0 < record["record_bytes"] <= (256 << 20)):
            raise ValueError("Source112 chunk identity differs from its position in the archive.")
        path = self.root / layout["files"][record["shard"]]["name"]
        with path.open("rb") as stream:
            stream.seek(record["file_offset"])
            raw = stream.read(record["record_bytes"])
        if len(raw) != record["record_bytes"] or _sha(raw) != record["sha256"]:
            raise ValueError(f"Source112 source record {chunk} differs from its archive digest.")
        components = {}
        for spec in record["components"]:
            name = spec["name"]
            if name not in {"dense", "dense_offsets", "sparse", "sparse_offsets"}:
                continue
            first, size = spec["offset"], spec["nbytes"]
            if spec["dtype"] != "<u4" or first < 0 or first + size > len(raw):
                raise ValueError(f"Source112 component has invalid extent: {name}.")
            components[name] = np.frombuffer(raw, dtype="<u4", count=size // 4, offset=first)
        if set(components) != {"dense", "dense_offsets", "sparse", "sparse_offsets"}:
            raise ValueError("Source112 record must contain all four native source components.")
        for name, ranks, minimum_length in (("dense", 17466, 1), ("sparse", 19398, 0)):
            offsets = components[name + "_offsets"]
            checkpoint_words = ranks if name == "dense" else ranks + 1
            if offsets.size != checkpoint_words + ranks * 8:
                raise ValueError(f"Source112 {name} seeks have incorrect extent.")
            # One checkpoint per 32 streams, followed by 32 byte lengths.
            lengths = offsets[checkpoint_words:].view(np.uint8).astype(np.uint64) + minimum_length
            prefix = np.concatenate([np.zeros(1, np.uint64), np.cumsum(lengths)])
            terminal = components[name].size if name == "dense" else int(components[name][0])
            if (not np.array_equal(prefix[:-1:32], offsets[:ranks])
                    or prefix[-1] != terminal or (name == "sparse" and offsets[ranks] != terminal)):
                raise ValueError(f"Source112 {name} checkpoints do not cover their complete source.")
        self._components = {name: cp.asarray(values) for name, values in components.items()}
        self._ids = cp.asarray(self.models[chunk // 4])
        self._chunk = chunk

    def decode_window(self, chunk: int, first_scan: int) -> np.ndarray:
        """Return exactly 512 native patterns in a bounded, explicitly selected window."""
        if not 0 <= chunk < 1056 or first_scan % 512 or not 0 <= first_scan < 16384:
            raise ValueError("Select a source chunk in [0,1056) and a 512-aligned scan window.")
        import cupy as cp

        with self.device:
            if chunk != self._chunk:
                self._record(chunk)
            parts = self._components
            raw = cp.zeros((512, 192, 192), cp.uint16)
            errors = cp.zeros(1, cp.uint32)
            self.module.get_function("decode_dense253")((137,), (128,), (
                parts["dense"], np.uint32(parts["dense"].size), parts["dense_offsets"],
                self.dense, self._ids, self.tables, np.uint32(81),
                np.int32(first_scan // 512), np.int32(1), raw, errors,
            ))
            self.module.get_function("decode_sparse253")((152,), (128,), (
                parts["sparse"], np.uint32(parts["sparse"].size), parts["sparse_offsets"],
                self.sparse, np.int32(first_scan // 512), np.int32(1), raw, errors,
            ))
            status = int(errors.get()[0])
            if status:
                raise ValueError(f"Retained source112 decode rejected malformed input (status {status}).")
            return raw.get()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self) -> None:
        """Release this reader's device-array references without touching other owners."""
        self._components = None
        self._chunk = None
        self._ids = None
        self.dense = self.sparse = self.tables = None


def _convert_source112_acquisition(path, output, *, acquisition: int, device: int = 0):
    """Convert one entire acquisition with bounded native buffers and no data reduction."""
    if not 0 <= acquisition < 66:
        raise ValueError("Select an acquisition index from 0 to 65.")
    reader = _Source112Archive(path, device=device)

    def blocks():
        for chunk in range(acquisition * 16, (acquisition + 1) * 16):
            for first in range(0, 16384, 512):
                window = reader.decode_window(chunk, first)
                yield window[:256]
                yield window[256:]

    metadata = {
        "legacy_format": "quantem-resident-source112-index180-v1",
        "legacy_archive": str(Path(path).resolve()),
        "legacy_acquisition": reader.manifest["original_acquisitions"]["acquisitions"][acquisition],
        "source_shape": [512, 512, 192, 192],
        "working_shape": [512, 512, 192, 192],
        "lossless_exact": True,
        "detector_mask_policy": "preserve-stored-counts",
    }
    try:
        return _write_ans_blocks(
            output, blocks(), shape=(512, 512, 192, 192), dtype=np.uint16,
            metadata=metadata, block_frames=256, scale=15,
        )
    finally:
        reader.close()
