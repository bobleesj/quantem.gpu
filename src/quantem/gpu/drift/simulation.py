"""Known image translations and causal piezo-feedback simulation on CUDA.

These functions operate on reconstructed images; they neither simulate detector
counts nor connect to hardware. Image displacement uses (row, col) coordinates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from quantem.gpu.device.cuda_runtime import cp
from quantem.gpu.drift.registration import register
from quantem.gpu.drift.series import shifted


def simulate_drift(
    reference: cp.ndarray,
    *,
    sampling_A: tuple[float, float],
    scans: int = 11,
    step_nm: tuple[float, float] = (0.02, -0.015),
) -> tuple[cp.ndarray, np.ndarray]:
    """Move a reconstructed phase by known translations for a registration control.

    Parameters
    ----------
    reference
        CuPy phase image. Fourier shifts wrap at the edges; inspect the interior.
    sampling_A
        Reconstructed pixel size in (row, col) Angstrom.
    scans, step_nm
        Number of synthetic scans and added (row, col) drift per scan in nm.

    Returns
    -------
    images, displacement_nm
        CuPy image stack and exact translations, with the first image unshifted.

    Examples
    --------
    >>> simulated, truth_nm = simulate_drift(phases[0], sampling_A=sampling_A)
    """
    displacement_nm = np.arange(scans)[:, None] * np.asarray(step_nm)
    images = cp.stack(
        [
            shifted(reference, shift * 10 / np.asarray(sampling_A))
            for shift in displacement_nm
        ]
    ).astype(cp.float32)
    return images, displacement_nm


def simulate_feedback(
    images: cp.ndarray,
    *,
    sampling_A: tuple[float, float],
    image_from_piezo: np.ndarray,
    gain: float = 0.8,
    max_step_nm: float = 0.2,
    min_confidence: float = 0.04,
    min_correlation: float = 0.5,
) -> tuple[cp.ndarray, pd.DataFrame]:
    """Simulate causal piezo correction after each complete scan, with one-scan delay.

    Parameters
    ----------
    images
        Uncorrected recorded or simulated phase images, on the GPU.
    sampling_A
        Reconstructed (row, col) pixel size in Angstrom.
    image_from_piezo
        Simulated 2x2 Jacobian: image displacement (row, col) nm = matrix @
        piezo displacement (axis 1, axis 2) nm. It is NOT detector rotation.
        Use a measured matrix and device-specific limits before hardware work.
    gain, max_step_nm, min_confidence
        Fraction of measured displacement to cancel, maximum vector step in nm,
        and minimum registration margin (not a probability). Ambiguous
        measurements hold position; a margin alone does not bound shift error.
    min_correlation
        Minimum mean normalized correlation across the registration's frequency
        bands. A distinct peak can still be a poor match; both this score and
        the margin must pass. These gates do not establish physical accuracy.

    Returns
    -------
    observed, report
        Reconstructed images seen by the feedback loop and per-scan commands.
        Each image is registered against the last accepted image; its accumulated
        displacement stays relative to scan zero. No future scans are used.

    Examples
    --------
    >>> observed, feedback = simulate_feedback(
    ...     phases, sampling_A=sampling_A, image_from_piezo=np.eye(2))
    """
    matrix = np.asarray(image_from_piezo, dtype=float)
    if (
        matrix.shape != (2, 2)
        or not np.isfinite(matrix).all()
        or np.linalg.cond(matrix) > 100
    ):
        raise ValueError("Use a finite, well-conditioned 2x2 image_from_piezo matrix.")
    if not 0 < gain <= 1 or not np.isfinite(max_step_nm) or max_step_nm <= 0:
        raise ValueError("Use 0 < gain <= 1 and a finite positive max_step_nm.")
    if (
        not np.isfinite(min_confidence)
        or min_confidence <= 0
        or not 0 < min_correlation <= 1
    ):
        raise ValueError(
            "Use a finite positive min_confidence and 0 < min_correlation <= 1."
        )
    command_nm = np.zeros(2)
    reference, reference_nm = images[0], np.zeros(2)
    observed, records = [], []
    for index, image in enumerate(
        tqdm(images, desc="Simulating feedback", unit="scan")
    ):
        applied_nm = command_nm.copy()
        shifted_nm = matrix @ applied_nm
        current = shifted(image, shifted_nm * 10 / np.asarray(sampling_A)).astype(
            cp.float32
        )
        observed.append(current)
        found = register(reference, current, sampling_A=sampling_A)
        measured_nm = (
            reference_nm + np.asarray(found.shift_px) * np.asarray(sampling_A) / 10
        )
        confidence = min(
            found.margin, found.parts["coarse"] - found.best["coarse_second"]
        )
        correlation = np.mean(
            [value for band, value in found.parts.items() if band != "coarse"]
        )
        accepted = bool(
            np.isfinite(measured_nm).all()
            and np.isfinite(confidence)
            and confidence >= min_confidence
            and np.isfinite(correlation)
            and correlation >= min_correlation
        )
        correction_nm = (
            -gain * np.linalg.solve(matrix, measured_nm) if accepted else np.zeros(2)
        )
        scale = max(1.0, np.linalg.norm(correction_nm) / max_step_nm)
        command_nm += correction_nm / scale
        if accepted:
            reference, reference_nm = current, measured_nm.copy()
        records.append(
            {
                "scan_index": index,
                "row_nm": measured_nm[0] if accepted else np.nan,
                "col_nm": measured_nm[1] if accepted else np.nan,
                "residual_nm": np.linalg.norm(measured_nm) if accepted else np.nan,
                "confidence": confidence,
                "correlation": correlation,
                "accepted": accepted,
                "limited": scale > 1,
                "applied_axis1_nm": applied_nm[0],
                "applied_axis2_nm": applied_nm[1],
                "next_axis1_nm": command_nm[0],
                "next_axis2_nm": command_nm[1],
            }
        )
    return cp.stack(observed), pd.DataFrame(records)
