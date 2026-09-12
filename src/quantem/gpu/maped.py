"""Complete accelerator-resident MAPED merging."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    import torch

    from .io import FourDSTEMData


def merge(
    sources: Sequence[FourDSTEMData],
    real_space_shifts: torch.Tensor,
    diffraction_shifts: torch.Tensor,
    *,
    save_to: str | Path,
    dtype: str = "scaled_uint16",
    close_sources: bool = False,
    verbose: bool = False,
) -> FourDSTEMData:
    """Merge aligned resident acquisitions and reopen the bounded result.

    Parameters
    ----------
    sources
        Exact accelerator-resident 4D-STEM acquisitions.
    real_space_shifts, diffraction_shifts
        Float32 Torch tensors shaped ``(tilts, 2)`` in ``(row, column)`` order.
        Their CUDA or MPS device selects the backend automatically.
    save_to
        Destination for the merged 4D-STEM dataset. Saving is required because
        a complete dense float32 merge may exceed accelerator memory.
    dtype
        Output intensity representation. Currently ``"scaled_uint16"``.
    close_sources
        Close the input residents after saving and before reopening the result.
    verbose
        Print one concise precision report after saving.

    Returns
    -------
    FourDSTEMData
        The packed merged result resident on the selected accelerator.
    """
    import h5py
    import torch

    from . import io
    from ._maped import resident_merge

    if dtype != "scaled_uint16":
        raise ValueError("MAPED merge currently requires dtype='scaled_uint16'.")
    device = getattr(real_space_shifts, "device", None)
    backend = getattr(device, "type", None)
    if backend not in {"cuda", "mps"}:
        raise ValueError(
            "MAPED merge requires CUDA or MPS float32 shift tensors."
        )
    sources = list(sources)
    started = time.perf_counter()
    generated = resident_merge(
        sources,
        real_space_shifts,
        diffraction_shifts,
    )
    generated.release_sources_before_reopen = bool(close_sources)
    io.save(
        save_to,
        generated,
        dtype=dtype,
        backend=backend,
        verbose=verbose,
    )
    if close_sources:
        for source in sources:
            source.close()
        if backend == "cuda":
            torch.cuda.empty_cache()
        else:
            torch.mps.empty_cache()
    reopen_started = time.perf_counter()
    result = io.load(
        save_to,
        backend=backend,
        representation="packed",
        verbose=False,
    )
    record = dict(result.metadata.get("maped_merge", {}))
    record.update(
        released_sources_before_reopen=bool(close_sources),
        reopen_seconds=time.perf_counter() - reopen_started,
        total_seconds=time.perf_counter() - started,
    )
    with h5py.File(save_to, "r+") as handle:
        handle.attrs["quantem_maped_merge_v1"] = json.dumps(record)
    result.metadata["maped_merge"] = record
    return result


__all__ = ["merge"]
