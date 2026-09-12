"""Backend dispatch for bounded MAPED merging from encoded residents."""


def merge_to_scaled_h5(
    sources,
    real_shifts,
    diffraction_shifts,
    output_path,
    *,
    release_sources_before_reopen: bool = False,
    verbose: bool = False,
):
    """Merge aligned residents through their accelerator-native backend."""
    device = getattr(real_shifts, "device", None)
    backend = getattr(device, "type", None)
    if backend == "cuda":
        from .cuda import merge_to_scaled_h5 as implementation
    elif backend == "mps":
        from .mps import merge_to_scaled_h5 as implementation
    else:
        raise ValueError(
            "MAPED encoded merging requires CUDA or MPS float32 shift tensors."
        )
    return implementation(
        sources,
        real_shifts,
        diffraction_shifts,
        output_path,
        release_sources_before_reopen=release_sources_before_reopen,
        verbose=verbose,
    )
