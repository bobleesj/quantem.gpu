"""Private backend dispatch for the public :mod:`quantem.gpu.maped` API."""


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
