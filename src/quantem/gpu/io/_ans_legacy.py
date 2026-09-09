"""Explicit seven-tilt artifact admission to the canonical count-rANS contract.

Retained source112/index180 is tANS plus a sparse index, not byte-rANS. It must
remain with its original decoder until an integer-verified conversion exists.
"""

import hashlib
import math
from pathlib import Path

import numpy as np

from ._ans_contract import _validate_arrays


def _legacy_rans_arguments(record: dict) -> dict:
    """Map a seven-tilt build record without entropy decode or payload copies.

    The caller owns the returned payload memory map and must close its
    ``_mmap`` after a runtime has copied the arrays into owned storage. The
    payload digest is verified against the supplied, independently retained
    build record. Tables have structural validation only: their scientific
    identity still requires parity against the retained source products.
    """
    required = {"shape", "dtype", "scan_block", "scale_bits", "model_frames",
                "models", "artifacts", "payload_sha256"}
    if not required <= record.keys():
        raise ValueError(
            "Expected a seven-tilt detector-rANS build record. "
            "Retained source112/index180 requires its original tANS decoder; "
            "do not relabel it as count-ANS."
        )
    shape = tuple(record["shape"])
    if len(shape) != 4 or record["dtype"] != "uint16":
        raise ValueError("Legacy detector-rANS requires four-dimensional uint16 counts.")
    pixels = math.prod(shape[2:])
    frames = int(record["scan_block"])
    model_frames = int(record["model_frames"])
    if frames < 1 or model_frames < frames or model_frames % frames:
        raise ValueError("Legacy model boundaries must align with positive scan blocks.")
    blocks = (math.prod(shape[:2]) + frames - 1) // frames
    root = Path(record["artifacts"])
    payload_path = root / "payload.bin"
    digest = hashlib.sha256()
    with payload_path.open("rb") as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    if digest.hexdigest() != record["payload_sha256"]:
        raise ValueError("Legacy rANS payload differs from the retained build digest.")
    starts = np.load(root / "block_starts.npy", allow_pickle=False)
    local = np.load(root / "offsets.npy", allow_pickle=False)
    if (starts.dtype != np.uint64 or starts.shape != (blocks + 1,)
            or local.dtype != np.uint32 or local.shape != (blocks, pixels + 1)
            or starts[0] != 0 or starts[-1] != payload_path.stat().st_size
            or np.any(starts[1:] < starts[:-1])
            or np.any(local[:, 0] != 0)
            or np.any(local[:, -1] != np.diff(starts))
            or np.any(local[:, 1:] < local[:, :-1])):
        raise ValueError("Legacy rANS offsets do not partition the payload exactly.")
    offsets = np.concatenate([
        (local[:, :-1].astype(np.uint64) + starts[:-1, None]).reshape(-1),
        starts[-1:],
    ])
    tables = {name: [] for name in ("symbols", "cumulative", "frequencies", "literal")}
    context_parts = []
    entries = 0
    for model in record["models"]:
        with np.load(root / model["path"], allow_pickle=False) as saved:
            contexts = saved["context_offsets"]
            if contexts.dtype != np.uint32 or contexts.shape != (pixels + 1,):
                raise ValueError("Legacy model must declare one context per detector pixel.")
            if (contexts[0] != 0 or contexts[-1] != len(saved["symbols"])
                    or saved["literal"].shape != (pixels,)):
                raise ValueError("Legacy context offsets must cover their symbol table.")
            context_parts.append(contexts[:-1].astype(np.uint64) + entries)
            entries += len(saved["symbols"])
            for name in tables:
                tables[name].append(saved[name])
    if not context_parts or entries >= 2**32:
        raise ValueError("Legacy model tables must fit the canonical uint32 index range.")
    contexts = np.concatenate([*context_parts, np.asarray([entries], np.uint64)])
    selectors = (
        np.arange(blocks, dtype=np.uint64) * frames // model_frames
    )[:, None] * pixels + np.arange(pixels, dtype=np.uint64)[None, :]
    if selectors.max() >= len(record["models"]) * pixels or selectors.max() >= 2**32:
        raise ValueError("Legacy model selection extends beyond the retained tables.")
    payload = np.memmap(payload_path, mode="r", dtype=np.uint8)
    arguments = dict(
        shape=shape, block_frames=frames, scale=int(record["scale_bits"]),
        payload=payload, offsets=offsets, model_ids=selectors.astype(np.uint32).reshape(-1),
        context_offsets=contexts.astype(np.uint32),
        **{name: np.concatenate(parts) for name, parts in tables.items()},
    )
    try:
        _validate_arrays(**arguments)
    except BaseException:
        payload._mmap.close()
        raise
    return dict(dtype="uint16", **arguments)
