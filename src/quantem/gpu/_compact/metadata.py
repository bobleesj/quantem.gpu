"""Validate format-owned metadata before any accelerator allocation."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from . import FORMAT, _matches_format
from .layout import validate_layout

# These identify the codec/layout of the first supported prepared format.
# Kernels execute only from the package, never from the selected dataset.
_KERNEL_HASHES = {
    "module": "1fce6910c7e03fabaa8be12b25027bb2287f55a16bf354051ca3103366c7b540",
    "sparse_module": "9ffbc8bbf686e1c3523d62b5ef6f0cf20895f5d5db47aa8455abf19fc9b452d6",
    "tile_module": "8e41a6803310b6b3161a1785612470016743e34ea605bcbb12547773dfc2a800",
    "coarse_module": "4ab6639e7b304ad6809b1caeb0da31fb75597161792de179f9180338fc4ac3ef",
    "pattern_module": "c937c503bf2f4d1149a6e5616f359bb2082631e181f9c9d43d7d9af94e8096e3",
}


def read_metadata(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    """Check completed layout, metadata integrity, and fixed codec compatibility."""
    path = Path(path)
    manifest = json.loads((path / "checkpoint.json").read_text())
    if (
        not _matches_format(manifest.get("format"), FORMAT)
        or manifest.get("complete") is not True
    ):
        raise ValueError(
            f"Load a completed {FORMAT} checkpoint; got {manifest.get('format')!r}."
        )
    validate_layout(manifest["layout"], require_complete=True)
    if manifest["layout"]["profile"] != "query-ready":
        raise NotImplementedError(
            "Prepare interaction indexes before loading this source-only archive."
        )
    if {
        key: value["sha256"] for key, value in manifest["kernels"].items()
    } != _KERNEL_HASHES:
        raise ValueError(
            "This checkpoint uses a different codec/index kernel layout; use its compatible loader."
        )
    state_file = Path(manifest["global_state_file"])
    if state_file.name != str(state_file):
        raise ValueError(
            "Global metadata must be a file within the prepared dataset folder."
        )
    state_path = path / state_file
    if (
        hashlib.sha256(state_path.read_bytes()).hexdigest()
        != manifest["global_state_file_sha256"]
    ):
        raise ValueError(
            "Global metadata checksum differs from the completed checkpoint; restore the complete file."
        )
    with np.load(state_path, allow_pickle=False) as saved:
        metadata = {name: saved[name] for name in saved.files}
    for name, array in metadata.items():
        spec = manifest["global_state"][name]
        if (
            list(array.shape) != spec["shape"]
            or array.dtype.str != spec["dtype"]
            or array.nbytes != spec["nbytes"]
            or hashlib.sha256(array.tobytes()).hexdigest() != spec["sha256"]
        ):
            raise ValueError(
                f"Global metadata {name!r} differs from the completed checkpoint."
            )
    ids = metadata["model_ids"]
    if (
        ids.shape != (264, 36864)
        or ids.dtype != np.uint8
        or np.any((ids >= 81) & ~metadata["planner__hardware"])
    ):
        raise ValueError(
            "Prepared model assignments must cover all 264 contexts with 81 models."
        )
    if (
        metadata["codec__decoding"].shape != (81, 1024)
        or metadata["codec__decoding"].dtype != np.uint32
    ):
        raise ValueError("Prepared decoding tables must be uint32[81,1024].")
    cache_map = metadata["planner__cache_map"]
    cached = metadata["cached"]
    if (
        cache_map.shape != (36864,)
        or cache_map.dtype != np.int32
        or cached.shape != (19398,)
    ):
        raise ValueError(
            "Prepared dense/sparse column ordering has an incompatible shape."
        )
    expected_map = np.full(36864, -1, np.int32)
    expected_map[cached] = np.arange(len(cached), dtype=np.int32)
    if not np.array_equal(cache_map, expected_map):
        raise ValueError("Prepared sparse positions disagree with their column map.")
    kernel = (Path(__file__).with_name("kernels") / "module.cu").read_text()
    ranks = np.fromstring(
        re.search(r"retained_rank\[36864\]=\{([^}]+)\}", kernel)[1],
        sep=",",
        dtype=np.uint16,
    )
    expected_ranks = np.full(36864, 65535, np.uint16)
    expected_ranks[cache_map < 0] = np.arange(17466, dtype=np.uint16)
    if not np.array_equal(ranks, expected_ranks):
        raise ValueError(
            "Prepared column order differs from the supported kernel format."
        )
    return manifest, metadata
