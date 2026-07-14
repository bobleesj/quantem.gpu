"""GPU-accelerated image processing utilities.

``filters`` imports cupy at module load. Importing this package (e.g. to reach
the pure-numpy ``imaging.canonical_obj_phase_crop``) must NOT pull cupy in, so
the Browse / view path works on a MacBook. Expose the cupy-backed helpers
lazily via PEP 562 ``__getattr__``.
"""

__all__ = ['gaussian_blur_fft', 'freq_grid_2d']


def __getattr__(name):
    if name in __all__:
        from quantem.gpu.ssb.image import filters
        return getattr(filters, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
