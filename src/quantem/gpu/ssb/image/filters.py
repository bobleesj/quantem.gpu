"""
GPU-accelerated image filtering utilities.

Provides convolution and filtering operations using FFT for efficiency.
"""

import cupy as cp


def freq_grid_2d(
    shape: tuple[int, int],
    sampling: tuple[float, float] = (1.0, 1.0),
) -> tuple[cp.ndarray, cp.ndarray]:
    """
    Create 2D frequency grids for FFT operations.
    
    Parameters
    ----------
    shape : tuple[int, int]
        Grid shape ``(n_row, n_col)``.
    sampling : tuple[float, float], optional
        Real-space sampling ``(drow, dcol)``. Default ``(1.0, 1.0)``.

    Returns
    -------
    f_row, f_col : tuple[cp.ndarray, cp.ndarray]
        2D frequency grids with shape ``(n_row, n_col)``, in cycles/unit.

    Examples
    --------
    >>> f_row, f_col = freq_grid_2d((256, 256))
    >>> f_row.shape
    (256, 256)
    """
    n_row, n_col = shape
    drow, dcol = sampling
    f_row_1d = cp.fft.fftfreq(n_row, d=drow).astype(cp.float32)
    f_col_1d = cp.fft.fftfreq(n_col, d=dcol).astype(cp.float32)
    f_row, f_col = cp.meshgrid(f_row_1d, f_col_1d, indexing='ij')
    return f_row, f_col


def gaussian_blur_fft(image: cp.ndarray, sigma: float) -> cp.ndarray:
    """
    Apply Gaussian blur using FFT convolution.

    Parameters
    ----------
    image : cp.ndarray
        Input image with shape ``(n_row, n_col)``.
    sigma : float
        Standard deviation of Gaussian kernel in pixels.

    Returns
    -------
    cp.ndarray
        Blurred image with same shape as input.

    Notes
    -----
    Uses FFT-based convolution which is O(N log N) and efficient for
    large kernels. The Gaussian is applied directly in frequency domain.
    """
    f_row, f_col = freq_grid_2d(image.shape)
    # Gaussian in frequency domain: exp(-2 * pi^2 * sigma^2 * f^2)
    gaussian_filter = cp.exp(-2 * (cp.pi * sigma) ** 2 * (f_row**2 + f_col**2))
    # Apply filter via FFT
    image_fft = cp.fft.fft2(image)
    filtered = cp.fft.ifft2(image_fft * gaussian_filter)
    return cp.real(filtered).astype(image.dtype)
