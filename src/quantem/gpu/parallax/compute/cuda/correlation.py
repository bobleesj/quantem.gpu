"""Compatibility imports; canonical implementation: ``quantem.gpu.parallax.backends.cuda.correlation``."""

from quantem.gpu.parallax.backends.cuda.correlation import (
    _align_images_fourier_cp as _align_images_fourier_cp,
    _upsampled_correlation_batch_cp as _upsampled_correlation_batch_cp,
    _upsampled_correlation_cp as _upsampled_correlation_cp,
    cross_correlation_shift_batch_cp as cross_correlation_shift_batch_cp,
    cross_correlation_shift_cp as cross_correlation_shift_cp,
    dft_upsample_cp as dft_upsample_cp,
)
