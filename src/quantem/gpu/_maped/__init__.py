"""Backend dispatch for bounded MAPED merging from encoded residents."""

import json
import time


def resident_merge(sources, real_shifts, diffraction_shifts):
    """Build a private re-readable merge source for :func:`quantem.gpu.io.save`."""
    device = getattr(real_shifts, "device", None)
    backend = getattr(device, "type", None)
    if backend == "cuda":
        from .cuda import resident_merge as implementation
    elif backend == "mps":
        from .mps import resident_merge as implementation
    else:
        raise ValueError(
            "MAPED encoded merging requires CUDA or MPS float32 shift tensors."
        )
    return implementation(sources, real_shifts, diffraction_shifts)


def merge_to_scaled_h5(
    sources,
    real_shifts,
    diffraction_shifts,
    output_path,
    *,
    release_sources_before_reopen: bool = False,
    verbose: bool = False,
):
    """Compatibility wrapper around the generic ``io.save`` resident path."""
    import h5py
    import torch

    from quantem.gpu import io

    device = getattr(real_shifts, "device", None)
    backend = getattr(device, "type", None)
    if backend not in {"cuda", "mps"}:
        raise ValueError(
            "MAPED encoded merging requires CUDA or MPS float32 shift tensors."
        )
    started = time.perf_counter()
    generated = resident_merge(sources, real_shifts, diffraction_shifts)
    generated.release_sources_before_reopen = bool(release_sources_before_reopen)
    io.save(
        output_path,
        generated,
        dtype="scaled_uint16",
        backend=backend,
        verbose=verbose,
    )
    if release_sources_before_reopen:
        for source in sources:
            source.close()
        if backend == "cuda":
            torch.cuda.empty_cache()
        else:
            torch.mps.empty_cache()
    reopen_started = time.perf_counter()
    result = io.load(
        output_path,
        backend=backend,
        representation="packed",
        verbose=False,
    )
    record = dict(result.metadata.get("maped_merge", {}))
    record.update(
        released_sources_before_reopen=bool(release_sources_before_reopen),
        reopen_seconds=time.perf_counter() - reopen_started,
        total_seconds=time.perf_counter() - started,
    )
    with h5py.File(output_path, "r+") as handle:
        handle.attrs["quantem_maped_merge_v1"] = json.dumps(record)
    result.metadata["maped_merge"] = record
    return result
