"""
Parallax reconstruction.

Reconstructs high-resolution phase contrast images from 4D-STEM data
by measuring and correcting for parallax shifts across tilted beam positions.

Key concepts
------------
- **Bright field disk**: Circle of illuminated pixels in diffraction pattern,
  radius determined by convergence semi-angle alpha.
- **Parallax shift**: Each BF position at angle (kx, ky) sees the sample
  from a slightly different angle, causing real-space shifts proportional to (defocus x angle).
- **Parallax reconstruction**: Align all shifted BF images and combine for
  enhanced resolution and phase contrast.

The reconstruction pipeline:
1. Sample BF disk at grid of k-space positions
2. Measure relative shifts via cross-correlation
3. Align and combine via bilinear KDE upsampling
4. Optionally fit aberration coefficients from shift pattern

References
----------
- Ophus et al., "Four-Dimensional Scanning Transmission Electron Microscopy (4D-STEM)"
- Yang et al., "Simultaneous atomic-resolution electron ptychography and Z-contrast imaging"
"""

import math
import time
import cupy as cp

from quantem.gpu.parallax_results import BFImage, ParallaxResult

__all__ = ["BFImage", "Parallax", "ParallaxResult", "parallax"]


def circular_mask(
    shape: tuple[int, int],
    center: tuple[float, float],
    radius: float,
) -> cp.ndarray:
    """Return a circular bright-field mask on the detector plane."""
    n_row, n_col = shape
    center_row, center_col = center
    rows = cp.arange(n_row, dtype=cp.float32)[:, None]
    cols = cp.arange(n_col, dtype=cp.float32)[None, :]
    return (rows - float(center_row)) ** 2 + (cols - float(center_col)) ** 2 <= float(radius) ** 2


def _validate_scan_shape(data: cp.ndarray, scan_shape: tuple[int, int]) -> None:
    scan_n = int(scan_shape[0]) * int(scan_shape[1])
    data_n = int(data.shape[0] * data.shape[1]) if data.ndim == 4 else int(data.shape[0])
    if scan_n != data_n:
        raise ValueError(
            f"scan_shape={scan_shape} has {scan_n} positions, but data has {data_n} frames"
        )


def parallax(
    data,
    scan_shape: tuple[int, int],
    *,
    center: tuple[int, int] | None = None,
    bf_radius: int | None = None,
    sampling_radius: int | None = None,
    voltage_kV: float = 300,
    scan_sampling: float | None = None,
    upsampling_factor: int = 2,
    fit_aberrations: bool = False,
    verbose: bool = False,
) -> ParallaxResult:
    """Run parallax reconstruction.

    This is the one-call public wrapper around :class:`Parallax`. It accepts either
    flattened ``(N, det_row, det_col)`` data or 4D
    ``(scan_row, scan_col, det_row, det_col)`` data and keeps the computation on
    CUDA via CuPy.
    """
    from quantem.gpu.detector import detect_bf_radius

    t0 = time.perf_counter()
    data = cp.asarray(data, dtype=cp.float32)
    _validate_scan_shape(data, scan_shape)
    if data.ndim == 4:
        data = data.reshape(-1, data.shape[-2], data.shape[-1])
    if center is None or bf_radius is None:
        detected_center, detected_radius = detect_bf_radius(data.mean(axis=0))
        if center is None:
            center = tuple(int(round(v)) for v in detected_center)
        if bf_radius is None:
            bf_radius = int(round(detected_radius))
        if verbose:
            print(f"[parallax] Auto-detected center={center}, bf_radius={bf_radius}")
    obj = Parallax(
        data,
        scan_shape,
        center=center,
        bf_radius=int(bf_radius),
        radius=sampling_radius,
        verbose=verbose,
    )
    full_result = obj.reconstruct(
        voltage_kV=voltage_kV,
        delta_r_A=scan_sampling,
        upsampling_factor=upsampling_factor,
        fit_aberrations=fit_aberrations,
        plot=False,
        verbose=verbose,
    )
    result = ParallaxResult(
        image=full_result.image,
        density=full_result.density,
        shifts=full_result.shifts,
        aberrations=full_result.aberrations,
        elapsed=time.perf_counter() - t0,
    )
    del full_result, obj, data
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    return result

# =========================================================================
#  Core class
# =========================================================================

class Parallax:
    """
    Parallax reconstruction from 4D-STEM data.

    Implements parallax reconstruction using all pixels inside the BF disk.

    Parameters
    ----------
    data : cp.ndarray
        Flattened 4D-STEM data ``(N, k_row, k_col)``.
    scan_shape : tuple[int, int]
        Scan grid shape ``(scan_row, scan_col)``.
    center : tuple[int, int]
        BF disk center ``(row, col)`` in detector pixels.
    bf_radius : int
        BF disk radius in pixels.
    radius : int | None
        Override bf_radius for sampling (e.g., for memory limits).
        If None, uses bf_radius.

    Examples
    --------
    >>> from quantem.gpu.parallax import Parallax
    >>>
    >>> parallax = Parallax(data, scan_shape=(256, 256),
    ...             center=(128, 128), bf_radius=30)
    >>> parallax.reconstruct()
    """

    def __init__(
        self,
        data: cp.ndarray,
        scan_shape: tuple[int, int],
        center: tuple[int, int],
        bf_radius: int,
        radius: int | None = None,
        verbose: bool = True,
    ) -> None:
        self.data = cp.asarray(data, dtype=cp.float32)
        self.scan_shape = scan_shape
        self.scan_row, self.scan_col = scan_shape
        self.N, self.n_k_row, self.n_k_col = self.data.shape
        self.center = center
        self._bf_radius = bf_radius
        self._radius_override = radius
        # Create mask and compute sampling positions
        self._dp_mask = circular_mask(
            (self.n_k_row, self.n_k_col), center, self.bf_radius,
        )
        self._compute_sampling_positions()
        # Cache for BF images
        self._bf_images: list[BFImage] | None = None
        self._bf_stack: cp.ndarray | None = None
        self._shifts: list[tuple[float, float]] | None = None
        if verbose:
            print(f"[Parallax] Initialized:")
            print(f"  Center: {center}")
            print(f"  Radius: {self.bf_radius} px")
            print(f"  BF positions: {self.n_positions:,}")

    @property
    def bf_radius(self) -> int:
        """BF radius (override or detected)."""
        return self._radius_override if self._radius_override is not None else self._bf_radius

    # =========================================================================
    #  Init / Setup
    # =========================================================================

    def _compute_sampling_positions(self) -> None:
        """Compute k-space positions from mask (all pixels inside BF disk)."""
        # Get all pixel indices where mask is True using CuPy
        xy_inds = cp.argwhere(self._dp_mask)  # (n_positions, 2) with [row, col] order
        # Convert to list of (k_row, k_col) tuples
        # We keep (row, col) ordering for indexing consistency
        xy_inds_cpu = cp.asnumpy(xy_inds)
        self.positions = [(int(idx[0]), int(idx[1])) for idx in xy_inds_cpu]
        self.n_positions = len(self.positions)
        # Store xy_inds for vectorized operations (GPU array)
        self._xy_inds = xy_inds.astype(cp.int32)
        # Store center index for reference (closest to center)
        center_row, center_col = self.center
        center_dist = float('inf')
        self.center_idx = 0
        for i, (ky, kx) in enumerate(self.positions):
            dist = math.sqrt((kx - center_col)**2 + (ky - center_row)**2)
            if dist < center_dist:
                center_dist = dist
                self.center_idx = i

    # =========================================================================
    #  BF extraction
    # =========================================================================

    def extract_bf_images(self) -> list[BFImage]:
        """
        Extract BF images at all mask positions (full BF disk).

        Each BF image is the intensity at a single pixel in the diffraction
        pattern. Uses the canonical positions set at construction time.

        Returns
        -------
        list[BFImage]
            BF images at each k-space position.
        """
        bf_stack, positions, center_idx = self._extract_at_positions(self._xy_inds)
        self._bf_stack = bf_stack
        self._bf_images = self._make_bf_list(bf_stack, positions)
        return self._bf_images

    def _extract_at_positions(
        self,
        xy_inds: cp.ndarray,
    ) -> tuple[cp.ndarray, list[tuple[int, int]], int]:
        """
        Extract BF images at specified positions. Pure function — no mutation.

        Parameters
        ----------
        xy_inds : cp.ndarray
            (n_positions, 2) array of [row, col] indices into the DP.

        Returns
        -------
        bf_stack : cp.ndarray
            (n_positions, scan_row, scan_col) BF images.
        positions : list[tuple[int, int]]
            (k_row, k_col) for each position.
        center_idx : int
            Index of position closest to BF center.
        """
        center_row, center_col = self.center
        n_pos = len(xy_inds)
        pix_row = xy_inds[:, 0].astype(cp.int64)
        pix_col = xy_inds[:, 1].astype(cp.int64)
        pix_idx = pix_row * self.n_k_col + pix_col
        data_flat = self.data.reshape(self.N, -1)
        mem_gb = self.N * n_pos * 4 / 1e9
        if mem_gb < 8:
            bf_values = data_flat[:, pix_idx]
            bf_stack = bf_values.T.reshape(n_pos, self.scan_row, self.scan_col).astype(cp.float32)
        else:
            chunk_size = max(1, int(8e9 / (self.N * 4)))
            bf_stack = cp.zeros((n_pos, self.scan_row, self.scan_col), dtype=cp.float32)
            for start in range(0, n_pos, chunk_size):
                end = min(start + chunk_size, n_pos)
                chunk_idx = pix_idx[start:end]
                bf_values = data_flat[:, chunk_idx]
                bf_stack[start:end] = bf_values.T.reshape(end - start, self.scan_row, self.scan_col)
        xy_inds_cpu = cp.asnumpy(xy_inds)
        positions = [(int(idx[0]), int(idx[1])) for idx in xy_inds_cpu]
        # Find center_idx (vectorized on GPU)
        dist_sq = (xy_inds[:, 1].astype(cp.float32) - center_col) ** 2 + (xy_inds[:, 0].astype(cp.float32) - center_row) ** 2
        center_idx = int(cp.argmin(dist_sq))
        return bf_stack, positions, center_idx

    def _filter_positions(self, aperture_radius: int) -> cp.ndarray:
        """Return xy_inds subset within aperture_radius of BF center."""
        center_row, center_col = self.center
        dx = self._xy_inds[:, 1].astype(cp.float32) - center_col
        dy = self._xy_inds[:, 0].astype(cp.float32) - center_row
        dist = cp.sqrt(dx**2 + dy**2)
        return self._xy_inds[dist <= aperture_radius]

    def _make_bf_list(
        self,
        bf_stack: cp.ndarray,
        positions: list[tuple[int, int]],
    ) -> list[BFImage]:
        """Create BFImage list from stack and positions."""
        center_row, center_col = self.center
        return [
            BFImage(data=bf_stack[i], k_col=k_col - center_col, k_row=k_row - center_row)
            for i, (k_row, k_col) in enumerate(positions)
        ]

    # =========================================================================
    #  Shift measurement
    # =========================================================================

    def measure_shifts(
        self,
        bf_images: list[BFImage] | None = None,
        upsample_factor: int = 10,
    ) -> list[tuple[float, float]]:
        """
        Measure shifts between BF images using cross-correlation (batched).

        Uses the center BF image as reference and measures how much
        each other BF image is shifted relative to it.

        Parameters
        ----------
        bf_images : list[BFImage] | None
            BF images to measure. If None, uses cached images from
            ``extract_bf_images()`` (or extracts them on first call).
        upsample_factor : int
            Subpixel precision factor (default: 10 → 0.1px precision)

        Returns
        -------
        list[tuple[float, float]]
            Shifts (drow, dcol) in pixels for each BF image
        """
        if bf_images is not None:
            stack = cp.stack([bf.data for bf in bf_images], axis=0)
        elif self._bf_stack is not None:
            stack = self._bf_stack
        else:
            self.extract_bf_images()
            stack = self._bf_stack
        return self._measure_shifts_batched(stack)

    def _measure_shifts_batched(
        self,
        stack: cp.ndarray,
    ) -> list[tuple[float, float]]:
        """Batched DFT upsample cross-correlation for a stack of BF images."""
        from quantem.gpu.image.dft_upsample import cross_correlation_shift_batch_cp
        ref = stack[self.center_idx]
        shifts_gpu = cross_correlation_shift_batch_cp(ref, stack, upsample_factor=4)
        shifts_np = cp.asnumpy(shifts_gpu)
        return [(float(shifts_np[i, 0]), float(shifts_np[i, 1])) for i in range(len(shifts_np))]

    def measure_shifts_iterative(
        self,
        n_iterations: int = 3,
        verbose: bool = True,
    ) -> list[tuple[float, float]]:
        """
        Measure shifts with coarse-to-fine multiscale binning refinement.

        Uses multiscale binning (matching quantem's ``align_vbf_stack_multiscale``)
        to progressively refine shifts from coarse to fine resolution.

        Parameters
        ----------
        n_iterations : int
            Number of refinement iterations (default: 3).
        verbose : bool
            Print progress (default: True).

        Returns
        -------
        list[tuple[float, float]]
            Refined shifts (drow, dcol) for each BF image.
        """
        from quantem.gpu.parallax_utils import align_vbf_stack_multiscale_cp
        if self._bf_stack is None:
            self.extract_bf_images()
        bf_mask = self._dp_mask
        inds_row, inds_col = cp.where(bf_mask)
        reference = self._bf_stack.mean(axis=0)
        # Map n_iterations to bin_factors (matching quantem default: (3, 2, 1))
        if n_iterations == 1:
            bin_factors = (1,)
        elif n_iterations == 2:
            bin_factors = (2, 1)
        elif n_iterations == 3:
            bin_factors = (3, 2, 1)
        else:
            max_bin = min(n_iterations, max(3, self.bf_radius // 4))
            factors = cp.asnumpy(cp.linspace(max_bin, 1, n_iterations)).astype(int)
            bin_factors = tuple(int(f) for f in factors)
        global_shifts, aligned_stack = align_vbf_stack_multiscale_cp(
            vbf_stack=self._bf_stack,
            bf_mask=bf_mask,
            inds_i=inds_row,
            inds_j=inds_col,
            bin_factors=bin_factors,
            upsample_factor=4,
            reference=reference,
            verbose=verbose,
        )
        self._bf_stack = aligned_stack
        self._bf_images = self._make_bf_list(aligned_stack, self.positions)
        shifts_np = cp.asnumpy(global_shifts)
        shifts = [(float(shifts_np[i, 0]), float(shifts_np[i, 1])) for i in range(len(shifts_np))]
        self._shifts = shifts
        if verbose:
            max_shift = float(cp.max(cp.sqrt(global_shifts[:, 0]**2 + global_shifts[:, 1]**2)))
            print(f"  Max shift: {max_shift:.2f} pixels")
        return shifts

    # =========================================================================
    #  Reconstruction
    # =========================================================================

    def upsample(
        self,
        bf_images: list[BFImage] | None,
        shifts: list[tuple[float, float]] | None,
        upsampling_factor: int = 2,
    ) -> tuple[cp.ndarray, cp.ndarray]:
        """
        Combine BF images via Fourier tiling upsampling.

        Parameters
        ----------
        bf_images : list[BFImage] | None
            BF images to combine. If None, uses cached stack from
            ``extract_bf_images()`` or ``measure_shifts_iterative()``.
        shifts : list[tuple[float, float]] | None
            Shifts (drow, dcol) to apply before combining. If None or all zero,
            no shift is applied (use when called after iterative alignment).
        upsampling_factor : int
            Resolution enhancement factor (default: 2)

        Returns
        -------
        tuple[cp.ndarray, cp.ndarray]
            (reconstructed image, density map)
        """
        if bf_images is not None:
            images = cp.stack([bf.data for bf in bf_images], axis=0).astype(cp.float32)
        elif self._bf_stack is not None:
            images = self._bf_stack.astype(cp.float32)
        else:
            raise ValueError("No BF images — call extract_bf_images() first")
        n_images, H, W = images.shape
        # Shift-and-sum in Fourier domain (one batched FFT, sum with phase ramps)
        # This avoids materializing N shifted images — just accumulates in freq space
        has_shifts = shifts is not None and any(abs(s[0]) > 1e-8 or abs(s[1]) > 1e-8 for s in shifts)
        images_fft = cp.fft.fft2(images, axes=(1, 2))  # (N, H, W) complex
        if has_shifts:
            shifts_arr = cp.array(shifts, dtype=cp.float32)
            f_row = cp.fft.fftfreq(H, d=1.0).astype(cp.float32).reshape(1, -1, 1)
            f_col = cp.fft.fftfreq(W, d=1.0).astype(cp.float32).reshape(1, 1, -1)
            drow = shifts_arr[:, 0].reshape(-1, 1, 1)
            dcol = shifts_arr[:, 1].reshape(-1, 1, 1)
            phase = cp.exp(-2j * cp.pi * (f_row * drow + f_col * dcol))
            images_fft *= phase
        summed_fft = images_fft.sum(axis=0)  # (H, W) — sum in Fourier domain
        if upsampling_factor > 1:
            tiled = cp.concatenate(
                [cp.concatenate([summed_fft] * upsampling_factor, axis=1)] * upsampling_factor,
                axis=0,
            )
            result = cp.fft.ifft2(tiled).real
        else:
            result = cp.fft.ifft2(summed_fft).real
        density = cp.full_like(result, float(n_images))
        return result, density

    # =========================================================================
    #  Aberration fitting
    # =========================================================================

    def fit_aberrations(
        self,
        shifts: list[tuple[float, float]],
        voltage_kV: float,
        delta_r_A: float,
        delta_k_A: float | None = None,
    ) -> dict[str, float]:
        """
        Fit aberration coefficients from measured shifts via SVD polar decomposition.

        Matches quantem's ``fit_aberrations_from_shifts`` algorithm.

        Parameters
        ----------
        shifts : list[tuple[float, float]]
            Measured shifts (drow, dcol) in pixels
        voltage_kV : float
            Accelerating voltage in kV
        delta_r_A : float
            Real-space pixel size in Å
        delta_k_A : float | None
            K-space pixel size in 1/Å. If None, computed from delta_r_A.

        Returns
        -------
        dict[str, float]
            Aberration coefficients in Ångströms
        """
        from quantem.gpu.ssb.optics.aberration_fitting import fit_aberrations_svd_polar
        from quantem.gpu.ssb.optics.physics import wavelength_A_from_kV
        import numpy as np
        wavelength = wavelength_A_from_kV(voltage_kV)
        shifts_np = np.array(shifts, dtype=np.float64) * delta_r_A
        mask_np = cp.asnumpy(self._dp_mask)
        gpts = (self.n_k_row, self.n_k_col)
        if delta_k_A is None:
            N = max(self.n_k_row, self.n_k_col)
            delta_k_A = 1.0 / (N * delta_r_A)
        sampling = (1.0 / (gpts[0] * delta_k_A), 1.0 / (gpts[1] * delta_k_A))
        return fit_aberrations_svd_polar(shifts_np, mask_np, wavelength, gpts, sampling)

    # =========================================================================
    #  High-level API
    # =========================================================================

    def reconstruct(
        self,
        voltage_kV: float = 300.0,
        delta_r_A: float | None = None,
        upsampling_factor: int = 2,
        fit_aberrations: bool = False,
        plot: bool = True,
        verbose: bool = True,
    ) -> ParallaxResult:
        """
        Run the complete parallax reconstruction pipeline.

        This is the recommended high-level API:
        1. Extract BF images across sampling grid
        2. Measure shifts via iterative cross-correlation
        3. Combine via bilinear KDE upsampling
        4. Optionally fit aberration coefficients
        5. Optionally generate diagnostic plots

        Parameters
        ----------
        voltage_kV : float
            Accelerating voltage in kV (default: 300)
        delta_r_A : float | None
            Real-space pixel size in Å. Required if fit_aberrations=True.
        upsampling_factor : int
            Resolution enhancement factor (default: 2)
        fit_aberrations : bool
            If True, fit aberration coefficients (default: False)
        plot : bool
            If True, generate diagnostic plots (default: True)
        verbose : bool
            If True, print progress (default: True)

        Returns
        -------
        ParallaxResult
            Container with reconstructed image, shifts, and optional aberrations

        Examples
        --------
        Basic reconstruction:

        >>> result = parallax.reconstruct()
        >>> plt.imshow(cp.asnumpy(result.image))

        With aberration fitting:

        >>> result = parallax.reconstruct(
        ...     voltage_kV=300,
        ...     delta_r_A=0.5,
        ...     fit_aberrations=True
        ... )
        >>> print(f"Defocus: {result.aberrations['C10']:.1f} Å")
        """
        if verbose:
            print("=" * 60)
            print("Parallax Reconstruction")
            print("=" * 60)
            print(f"Sampling: {self.n_positions} BF positions")
            print(f"BF radius: ~{self.bf_radius} pixels")
        # Step 1: Plot sampling grid (extract only if plotting)
        if plot:
            if verbose:
                print("\n[1/3] Extracting BF images...")
            self.extract_bf_images()
            self.plot_sampling_grid()
        elif verbose:
            print("\n[1/3] Preparing BF images...")
        # Step 2: Measure shifts (re-extracts BF images iteratively)
        if verbose:
            print("\n[2/3] Measuring shifts (iterative alignment)...")
        shifts = self.measure_shifts_iterative(verbose=verbose)
        bf_images = self._bf_images
        if plot:
            self.plot_shifts(shifts)
        # Step 3: Upsample (uses cached _bf_stack from measure_shifts_iterative)
        if verbose:
            print(f"\n[3/3] Upsampling ({upsampling_factor}×)...")
        image, density = self.upsample(
            None, shifts,
            upsampling_factor=upsampling_factor,
        )
        if verbose:
            print(f"  Output shape: {image.shape}")
            print(f"  Mean density: {float(cp.mean(density)):.1f} samples/pixel")
        # Optional: fit aberrations
        aberrations = None
        if fit_aberrations:
            if delta_r_A is None:
                raise ValueError("delta_r_A required for aberration fitting")
            if verbose:
                print("\n[Bonus] Fitting aberrations...")
            aberrations = self.fit_aberrations(shifts, voltage_kV, delta_r_A)
            if verbose:
                print(f"  C10 (defocus): {aberrations.get('C10', 0):.1f} Å")
                print(f"  C12 (astigmatism): {aberrations.get('C12_a', 0):.1f} Å")
        if plot:
            self.plot_reconstruction(image)
        if verbose:
            print("\n" + "=" * 60)
            print("Reconstruction complete.")
            print("=" * 60)
        return ParallaxResult(
            image=image,
            density=density,
            shifts=shifts,
            bf_images=bf_images,
            aberrations=aberrations
        )

    # =========================================================================
    #  Visualization
    # =========================================================================

    def plot_sampling_grid(self) -> None:
        """Plot the k-space sampling positions overlaid on mean DP."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        fig, ax = plt.subplots(figsize=(8, 8))
        # Show mean diffraction pattern
        dp = cp.asnumpy(self.data.mean(axis=0))
        ax.imshow(dp, cmap='gray', origin='lower')
        # Draw BF disk
        center_row, center_col = self.center
        circle = Circle((center_col, center_row), self.bf_radius,
                        fill=False, edgecolor='red', linewidth=2, label='BF disk')
        ax.add_patch(circle)
        # Plot sampling positions
        kx_pos = [ky for kx, ky in self.positions]  # Note: swap for imshow
        ky_pos = [kx for kx, ky in self.positions]
        ax.scatter(kx_pos, ky_pos, c='cyan', s=20, marker='o', label='Sampling points')
        # Mark center
        ax.scatter([center_col], [center_row], c='yellow', s=100, marker='x', linewidth=2, label='Center')
        ax.set_title(f'K-space sampling: {self.n_positions} positions')
        ax.legend(loc='upper right')
        ax.set_xlabel('qy (pixels)')
        ax.set_ylabel('qx (pixels)')
        plt.tight_layout()
        plt.show()

    def plot_shifts(self, shifts: list[tuple[float, float]]) -> None:
        """Plot shift vectors as quiver plot."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 10))
        # Extract positions and shifts
        center_row, center_col = self.center
        kx_rel = []
        ky_rel = []
        shift_row_vals = []
        shift_col_vals = []
        # self.positions are stored as (ky, kx) = (row, col)
        for (ky, kx), (shift_row, shift_col) in zip(self.positions, shifts):
            kx_rel.append(kx - center_col)
            ky_rel.append(ky - center_row)
            # Negate to show correction direction
            shift_row_vals.append(-shift_row)
            shift_col_vals.append(-shift_col)
        # Quiver plot
        ax.quiver(
            kx_rel, ky_rel,
            shift_col_vals, shift_row_vals,
            angles='xy', scale_units='xy', scale=1.0,
            width=0.006, color='blue', alpha=0.8,
            headwidth=4, headlength=5
        )
        # Mark center
        ax.scatter([0], [0], s=200, c='red', marker='x', linewidth=3, label='Center')
        ax.set_xlabel('kx (pixels from center)')
        ax.set_ylabel('ky (pixels from center)')
        ax.set_title('Shift vectors (pointing toward correction direction)')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax.axvline(0, color='k', linestyle='--', alpha=0.3)
        ax.set_aspect('equal')
        ax.legend()
        plt.tight_layout()
        plt.show()

    def plot_reconstruction(self, image: cp.ndarray) -> None:
        """Plot the reconstructed image."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 10))
        img_np = cp.asnumpy(image)
        ax.imshow(img_np, cmap='gray', origin='lower')
        ax.set_title('Parallax Reconstruction')
        ax.set_xlabel('x (pixels)')
        ax.set_ylabel('y (pixels)')
        plt.tight_layout()
        plt.show()
