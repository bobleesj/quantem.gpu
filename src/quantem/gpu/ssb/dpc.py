"""
Differential Phase Contrast (DPC) reconstruction for 4D-STEM data using CuPy.
"""

import time
from typing import Any

import cupy as cp
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from quantem.gpu.ssb.image import freq_grid_2d


# =============================================================================
#  Alignment helpers
# =============================================================================


def _show_rotation(
    angles_rad: cp.ndarray,
    curls: cp.ndarray,
    curls_t: cp.ndarray,
    rotation_angle_deg: float,
    min_curl: float,
    use_transpose: bool
) -> None:
    """Plot curl metric vs rotation angle for alignment optimization."""
    # Periodic extension (duplicate 0° point at 180°)
    angles_p = cp.concatenate([angles_rad, cp.array([cp.pi])])
    angles_p_deg = cp.asnumpy(angles_p * 180 / cp.pi)
    curl_p = cp.asnumpy(cp.concatenate([curls, curls[0:1]]))
    curl_p_t = cp.asnumpy(cp.concatenate([curls_t, curls_t[0:1]]))
    min_curl_val = float(min_curl) if hasattr(min_curl, '__float__') else cp.asnumpy(min_curl)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(angles_p_deg, curl_p, color=(1, 0, 0), linewidth=2, label='Normal')
    ax.plot(angles_p_deg, curl_p_t, color=(0, 0.7, 1), linewidth=2, label='Transposed')
    ax.plot(
        rotation_angle_deg,
        min_curl_val,
        'o',
        markerfacecolor=(0, 1, 0),
        markeredgecolor=(0, 0, 0),
        markersize=8,
        label=f'Best: {rotation_angle_deg:.1f}° ({"T" if use_transpose else "N"})'
    )
    ax.set_xlim(0, 180)
    ax.set_xlabel('Rotation angle (degrees)')
    ax.set_ylabel('Mean squared curl')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


def _rotate_vector_batch(
    v_row: cp.ndarray,
    v_col: cp.ndarray,
    angles_rad: cp.ndarray
) -> tuple[cp.ndarray, cp.ndarray]:
    """
    Apply 2D rotation to vector field for multiple angles at once.

    Parameters
    ----------
    v_row : cp.ndarray
        Input vector field row-component, shape (n_row, n_col)
    v_col : cp.ndarray
        Input vector field col-component, shape (n_row, n_col)
    angles_rad : cp.ndarray
        Rotation angles in radians, shape (n_angles,)

    Returns
    -------
    v_row_rot : cp.ndarray
        Rotated row-component, shape (n_angles, n_row, n_col)
    v_col_rot : cp.ndarray
        Rotated col-component, shape (n_angles, n_row, n_col)
    """
    cos_angles = cp.cos(angles_rad)
    sin_angles = cp.sin(angles_rad)
    v_row_rot = cos_angles[:, None, None] * v_row - sin_angles[:, None, None] * v_col
    v_col_rot = sin_angles[:, None, None] * v_row + cos_angles[:, None, None] * v_col
    return v_row_rot, v_col_rot


def _compute_curl_batch(
    v_row: cp.ndarray,
    v_col: cp.ndarray
) -> cp.ndarray | float:
    """
    Compute curl magnitude of 2D vector field using central differences.

    Parameters
    ----------
    v_row : cp.ndarray
        Vector field row-component, shape (n_fields, n_row, n_col) or (n_row, n_col)
    v_col : cp.ndarray
        Vector field col-component, shape (n_fields, n_row, n_col) or (n_row, n_col)

    Returns
    -------
    cp.ndarray | float
        Mean squared curl for each field
    """
    if v_row.ndim == 2:
        # curl = d(v_col)/d_row - d(v_row)/d_col
        dv_row_dcol = 0.5 * (v_row[1:-1, 2:] - v_row[1:-1, :-2])
        dv_col_drow = 0.5 * (v_col[2:, 1:-1] - v_col[:-2, 1:-1])
        curl = dv_col_drow - dv_row_dcol
        return float((curl**2).mean())
    else:
        dv_row_dcol = 0.5 * (v_row[:, 1:-1, 2:] - v_row[:, 1:-1, :-2])
        dv_col_drow = 0.5 * (v_col[:, 2:, 1:-1] - v_col[:, :-2, 1:-1])
        curl = dv_col_drow - dv_row_dcol
        return (curl**2).mean(axis=(1, 2))


def compute_center_of_mass(
    data: cp.ndarray,
    normalize_zero_mean: bool = True
) -> tuple[cp.ndarray, cp.ndarray]:
    """
    Compute center-of-mass at each scan position.

    Parameters
    ----------
    data : cp.ndarray
        4D-STEM data, 3D ``(N, k_row, k_col)`` or 4D
        ``(scan_r, scan_c, k_row, k_col)``. 4D input is flattened (view, no
        copy) so the CoM output is always 1D ``(N,)``.
    normalize_zero_mean : bool, optional
        Subtract mean from CoM fields (default: True)

    Returns
    -------
    com_krow : cp.ndarray
        Center-of-mass in detector k_row direction, shape (N,)
    com_kcol : cp.ndarray
        Center-of-mass in detector k_col direction, shape (N,)
    """
    if data.ndim == 4:
        data = data.reshape(-1, *data.shape[-2:])
    N, N_krow, N_kcol = data.shape
    I_sum = data.sum(axis=(1, 2), dtype=cp.float64)
    I_sum = cp.maximum(I_sum, 1e-10)  # prevent division by zero
    # CoM in k_row direction: sum over k_col first, then weighted sum over k_row
    krow_idx = cp.arange(N_krow, dtype=cp.float64)
    data_sum_kcol = data.sum(axis=2, dtype=cp.float64)  # (N, N_krow)
    com_krow = (data_sum_kcol * krow_idx[None, :]).sum(axis=1) / I_sum
    del data_sum_kcol
    # CoM in k_col direction: sum over k_row first, then weighted sum over k_col
    kcol_idx = cp.arange(N_kcol, dtype=cp.float64)
    data_sum_krow = data.sum(axis=1, dtype=cp.float64)  # (N, N_kcol)
    com_kcol = (data_sum_krow * kcol_idx[None, :]).sum(axis=1) / I_sum
    del data_sum_krow
    com_krow = com_krow.astype(cp.float32)
    com_kcol = com_kcol.astype(cp.float32)
    if normalize_zero_mean:
        com_krow = com_krow - com_krow.mean()
        com_kcol = com_kcol - com_kcol.mean()
    return com_krow, com_kcol


def find_optimal_rotation(
    v_krow: cp.ndarray,
    v_kcol: cp.ndarray,
    scan_shape: tuple[int, int],
    rotation_steps: int = 180,
    plot_optimization: bool = False
) -> tuple[cp.ndarray, cp.ndarray, float, bool, cp.ndarray, cp.ndarray, cp.ndarray]:
    """
    Find rotation that minimizes curl of the CoM vector field.

    Parameters
    ----------
    v_krow : cp.ndarray
        Vector field component in detector k_row direction (flattened), shape (N,)
    v_kcol : cp.ndarray
        Vector field component in detector k_col direction (flattened), shape (N,)
    scan_shape : tuple[int, int]
        Shape of scan grid (Rx, Ry)
    rotation_steps : int, optional
        Number of rotation angles to test (default: 180)
    plot_optimization : bool, optional
        Plot curl metric vs rotation angle (default: False)

    Returns
    -------
    v_krow_aligned : cp.ndarray
        Aligned first gradient component, shape (Rx, Ry)
    v_kcol_aligned : cp.ndarray
        Aligned second gradient component, shape (Rx, Ry)
    angle : float
        Optimal rotation angle in degrees
    use_transpose : bool
        Whether transposed orientation was selected
    """
    Rx, Ry = scan_shape
    v_krow_2d = v_krow.reshape(Rx, Ry)
    v_kcol_2d = v_kcol.reshape(Rx, Ry)
    angles_rad = cp.linspace(0, cp.pi, rotation_steps, dtype=cp.float32)
    # Compute all rotations for both orientations (vectorized)
    v_krow_rot, v_kcol_rot = _rotate_vector_batch(v_krow_2d, v_kcol_2d, angles_rad)
    v_krow_rot_t, v_kcol_rot_t = _rotate_vector_batch(v_kcol_2d, v_krow_2d, angles_rad)
    # Compute curl for all angles and both orientations
    curls = _compute_curl_batch(v_krow_rot, v_kcol_rot)
    curls_t = _compute_curl_batch(v_krow_rot_t, v_kcol_rot_t)
    # Stack both and find global minimum curl
    curls_stacked = cp.stack([curls, curls_t])
    min_idx_flat = int(curls_stacked.flatten().argmin())
    min_curl = curls_stacked.flatten()[min_idx_flat]
    # Decode which orientation and which angle
    use_transpose = min_idx_flat >= rotation_steps
    angle_idx = min_idx_flat % rotation_steps
    best_angle_rad = angles_rad[angle_idx]
    rotation_angle_deg = float(best_angle_rad) * 180 / cp.pi
    if use_transpose:
        v_krow_aligned = v_krow_rot_t[angle_idx]
        v_kcol_aligned = v_kcol_rot_t[angle_idx]
    else:
        v_krow_aligned = v_krow_rot[angle_idx]
        v_kcol_aligned = v_kcol_rot[angle_idx]
    if plot_optimization:
        _show_rotation(
            angles_rad, curls, curls_t,
            rotation_angle_deg, min_curl, use_transpose
        )
    # Return the curl landscape so callers (DPC class, api.dpc) can surface it
    # for QC plots and the screening dashboard's rotation card. Converting
    # angles_rad → degrees here keeps a single source of truth.
    curl_angles_deg = cp.rad2deg(angles_rad)
    return (v_krow_aligned, v_kcol_aligned, rotation_angle_deg, use_transpose,
            curl_angles_deg, curls, curls_t)


# =============================================================================
#  FFT reconstruction
# =============================================================================


def reconstruct_phase_from_gradient(
    grad_0: cp.ndarray,
    grad_1: cp.ndarray
) -> cp.ndarray:
    """
    Reconstruct phase from gradient field using FFT method.

    Solves nabla-phi = (grad_0, grad_1) in Fourier space using:
        phi_fft = (-i x 0.25)(k0 * grad_0_fft + k1 * grad_1_fft) / (k0^2 + k1^2)

    Parameters
    ----------
    grad_0 : cp.ndarray
        Gradient component along dimension 0, shape (dim0, dim1)
    grad_1 : cp.ndarray
        Gradient component along dimension 1, shape (dim0, dim1)

    Returns
    -------
    cp.ndarray
        Reconstructed phase field, shape (dim0, dim1)
    """
    # Ensure float32
    grad_0 = grad_0.astype(cp.float32)
    grad_1 = grad_1.astype(cp.float32)
    # Transform to Fourier space
    grad_0_fft = cp.fft.fft2(grad_0)
    grad_1_fft = cp.fft.fft2(grad_1)
    # Use shared freq_grid_2d from quantem.gpu.ssb.image
    k0, k1 = freq_grid_2d(grad_0.shape)
    # Compute k^2 and avoid division by zero
    k2 = k0**2 + k1**2
    k2[0, 0] = 1.0  # prevent division by zero at DC
    # Solve for phase in Fourier space
    # Fourier gradient integration factor (Poisson solver)
    phase_fft = (-1j * 0.25) * (k0 * grad_0_fft + k1 * grad_1_fft) / k2
    phase_fft[0, 0] = 0  # Set DC component to zero
    # Transform back to real space
    phase = cp.real(cp.fft.ifft2(phase_fft))
    return phase


# =============================================================================
#  DPC class
# =============================================================================


class DPC:
    """
    Differential Phase Contrast reconstruction with stateful workflow.

    Two-step workflow:
    1. ``preprocess()``: compute center of mass and find optimal rotation
    2. ``reconstruct()``: Fourier-integrate gradients to get phase

    Parameters
    ----------
    data : cp.ndarray
        4D-STEM data with shape ``(N, k_row, k_col)`` (flattened scan).
    scan_shape : tuple[int, int]
        Shape of scan grid ``(scan_row, scan_col)``.

    Attributes
    ----------
    com_k_row_aligned : cp.ndarray | None
        Aligned center-of-mass in the k_row direction.
    com_k_col_aligned : cp.ndarray | None
        Aligned center-of-mass in the k_col direction.
    rotation_angle_deg : float | None
        Optimal rotation angle in degrees.
    use_transpose : bool | None
        Whether transposed orientation was used.
    phase : cp.ndarray | None
        Reconstructed phase after ``reconstruct()``.
    elapsed : float | None
        Wall-clock time for ``reconstruct()`` in seconds.

    Examples
    --------
    >>> from quantem.gpu import load
    >>> from quantem.gpu.ssb.dpc import DPC
    >>> data = load('lamella_data.h5').data
    >>> dpc = DPC(data, scan_shape=(256, 256))
    >>> dpc.preprocess().reconstruct()
    >>> phase = dpc.phase
    """

    data: cp.ndarray
    scan_shape: tuple[int, int]
    com_k_row_aligned: cp.ndarray | None
    com_k_col_aligned: cp.ndarray | None
    rotation_angle_deg: float | None
    use_transpose: bool | None
    phase: cp.ndarray | None
    elapsed: float | None
    # Curl-vs-rotation landscape from the auto-fit pass. Populated by
    # preprocess() so api.dpc() / the dashboard can plot the optimization
    # curve without recomputing the sweep.
    curl_angles_deg: cp.ndarray | None
    curl_normal: cp.ndarray | None
    curl_transpose: cp.ndarray | None

    def __init__(
        self,
        data: cp.ndarray,
        scan_shape: tuple[int, int],
    ) -> None:
        self.data = data
        self.scan_shape = scan_shape
        self.com_k_row_aligned = None
        self.com_k_col_aligned = None
        self.rotation_angle_deg = None
        self.use_transpose = None
        self.phase = None
        self.elapsed = None
        self.curl_angles_deg = None
        self.curl_normal = None
        self.curl_transpose = None

    # =========================================================================
    #  Core workflow
    # =========================================================================

    def preprocess(
        self,
        rotation_steps: int = 180,
        normalize_com: bool = True,
        plot_rotation: bool = False,
        plot_com: bool = False,
        **kwargs: Any,
    ) -> "DPC":
        """
        Compute center of mass and find optimal rotation.

        Parameters
        ----------
        rotation_steps : int, optional
            Number of rotation angles to test (default: 180).
        normalize_com : bool, optional
            Subtract mean from CoM (default: True).
        plot_rotation : bool, optional
            Plot curl vs rotation angle (default: False).
        plot_com : bool, optional
            Plot aligned CoM fields (default: False).

        Returns
        -------
        DPC
            Self, for method chaining.
        """
        com_krow, com_kcol = compute_center_of_mass(
            self.data, normalize_zero_mean=normalize_com
        )
        (
            self.com_k_row_aligned,
            self.com_k_col_aligned,
            self.rotation_angle_deg,
            self.use_transpose,
            self.curl_angles_deg,
            self.curl_normal,
            self.curl_transpose,
        ) = find_optimal_rotation(
            com_krow, com_kcol,
            scan_shape=self.scan_shape,
            rotation_steps=rotation_steps,
            plot_optimization=plot_rotation,
        )
        if plot_com:
            self._plot_com(**kwargs)
        return self

    def reconstruct(
        self,
        zero_mean: bool = True,
        show_results: bool = False,
        **kwargs: Any,
    ) -> "DPC":
        """
        Reconstruct phase from aligned gradients using Fourier integration.

        Parameters
        ----------
        zero_mean : bool, optional
            Subtract mean from phase (default: True).
        show_results : bool, optional
            Display result with matplotlib (default: False).

        Returns
        -------
        DPC
            Self, for method chaining.
        """
        if self.com_k_row_aligned is None:
            raise RuntimeError("Must call preprocess() before reconstruct()")
        _t0 = time.perf_counter()
        # When transpose=True, alignment swapped (k_col, k_row) - swap back
        if self.use_transpose:
            com_row, com_col = self.com_k_col_aligned, self.com_k_row_aligned
        else:
            com_row, com_col = self.com_k_row_aligned, self.com_k_col_aligned
        phase_pos = reconstruct_phase_from_gradient(com_row, com_col)
        if zero_mean:
            phase_pos = phase_pos - phase_pos.mean()
        # Negative phase: atoms appear dark (standard STEM convention)
        self.phase = -phase_pos
        self.elapsed = time.perf_counter() - _t0
        if show_results:
            self._plot_phase(**kwargs)
        return self

    def show_sign_options(
        self,
        reference_phase: cp.ndarray | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Display both sign options for the phase.

        Reconstructs with positive and negative gradients, optionally
        computing correlation against a reference to identify the correct sign.

        Parameters
        ----------
        reference_phase : cp.ndarray | None, optional
            Reference phase for correlation comparison.

        Returns
        -------
        dict[str, Any]
            ``'phases'``: dict mapping sign name to phase array.
            ``'correlations'``: dict of correlation values (if reference given).
            ``'axs'``: matplotlib axes.
        """
        if self.com_k_row_aligned is None:
            raise RuntimeError("Must call preprocess() before show_sign_options()")
        if self.use_transpose:
            com_row, com_col = self.com_k_col_aligned, self.com_k_row_aligned
        else:
            com_row, com_col = self.com_k_row_aligned, self.com_k_col_aligned
        phase_pos = reconstruct_phase_from_gradient(com_row, com_col)
        phase_neg = -phase_pos
        phase_pos = phase_pos - phase_pos.mean()
        phase_neg = phase_neg - phase_neg.mean()
        phases = {"Positive sign": phase_pos, "Negative sign": phase_neg}
        correlations = {}
        if reference_phase is not None:
            for name, p in phases.items():
                corr = float(cp.corrcoef(p.flatten(), reference_phase.flatten())[0, 1])
                correlations[name] = corr
        axs = self._plot_sign_options(
            phase_pos, phase_neg,
            correlations=correlations if reference_phase else None,
            **kwargs,
        )
        return {
            "phases": phases,
            "correlations": correlations if reference_phase is not None else None,
            "axs": axs,
        }

    # =========================================================================
    #  Plotting
    # =========================================================================

    def _plot_phase(self, cmap: str = "gray", title: str = "DPC Phase", **kwargs: Any) -> Axes:
        """Display the reconstructed phase."""
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(cp.asnumpy(self.phase), cmap=cmap, **kwargs)
        ax.set_title(title, fontsize=14)
        ax.axis("off")
        plt.tight_layout()
        plt.show()
        return ax

    def _plot_com(self, cmap: str = "RdBu_r", **kwargs: Any) -> list[Axes]:
        """Display aligned center-of-mass fields."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, com, title in zip(
            axes,
            [self.com_k_row_aligned, self.com_k_col_aligned],
            ["CoM k_row (aligned)", "CoM k_col (aligned)"],
        ):
            im = ax.imshow(cp.asnumpy(com), cmap=cmap, **kwargs)
            ax.set_title(title, fontsize=14)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046)
        plt.tight_layout()
        plt.show()
        return list(axes)

    @staticmethod
    def _plot_sign_options(
        phase_pos: cp.ndarray,
        phase_neg: cp.ndarray,
        correlations: dict[str, float] | None = None,
        cmap: str = "gray",
        **kwargs: Any,
    ) -> list[Axes]:
        """Display both sign options side by side."""
        if correlations is not None:
            titles = [
                f"Positive gradients\nCorr: {correlations['Positive sign']:.3f}",
                f"Negative gradients\nCorr: {correlations['Negative sign']:.3f}",
            ]
        else:
            titles = ["Positive gradients", "Negative gradients"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        for ax, phase, title in zip(axes, [phase_pos, phase_neg], titles):
            im = ax.imshow(cp.asnumpy(phase), cmap=cmap, **kwargs)
            ax.set_title(title, fontsize=14)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle("DPC phase reconstruction - Sign ambiguity", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.show()
        return list(axes)

    # =========================================================================
    #  Display
    # =========================================================================

    def __repr__(self) -> str:
        """String representation."""
        Rx, Ry = self.scan_shape
        lines = [f"DPC(data shape: {self.data.shape}, scan: {Rx}x{Ry})"]
        if self.rotation_angle_deg is not None:
            lines.append(f"  [done] Preprocessed (angle: {self.rotation_angle_deg:.1f} deg, transpose: {self.use_transpose})")
        else:
            lines.append("  [    ] Not preprocessed")
        if self.phase is not None:
            lines.append(f"  [done] Phase reconstructed (range: [{float(self.phase.min()):.3f}, {float(self.phase.max()):.3f}])")
        else:
            lines.append("  [    ] Phase not reconstructed")
        return "\n".join(lines)
