"""Prepared weak-object amplitude/phase systems on the acquisition's GPU.

For each measured scan frequency, all supported aliases are fitted together.
The transfer is the same tilted, finite-depth probe model used by SSB fitting.
Regularization acts on the solution, never on a rendered image.
"""

from dataclasses import dataclass

import cupy as cp
import numpy as np

from quantem.gpu.ssb.cuda.weak_object import transfer_frequencies
from quantem.gpu.ssb.units import higher_order_arrays

_evaluate = cp.ElementwiseKernel(
    "raw complex128 basis, raw float64 eigen, float64 strength, int32 dim",
    "complex64 coefficient",
    """complex<double> value(0.0, 0.0);
       for (int j = 0; j < dim; ++j) {
           value += basis[i * dim + j] / (eigen[(i / dim) * dim + j] + strength);
       }
       coefficient = complex<float>(value.real(), value.imag());""",
    "direct_ptycho_regularization",
)


@dataclass
class _Group:
    rows: object
    cols: object
    gram: object
    rhs: object
    prior: object
    basis: object | None
    eigen: object | None


@dataclass
class PreparedDirect:
    """Sufficient statistics for one fixed data, optics and model selection."""

    shape: tuple[int, int]
    factor: int
    groups: list[_Group]
    complex_object: bool
    scale: object
    signal_power: object
    incident_scale: float

    @classmethod
    def prepare(
        cls,
        engine,
        optics,
        factor: int,
        *,
        complex_object: bool,
        smoothness_inv_A: float,
        detector_weighting: str,
    ):
        """Build normal equations once; strengths are applied at evaluation."""
        nbf, nrow, ncol = engine._bf_grid()
        if min(nrow, ncol) < 3:
            raise ValueError(
                "Direct inversion needs at least 3 scan positions per side; select a larger region."
            )
        out_row, out_col = factor * nrow, factor * ncol
        cache = engine._cache
        aperture2 = cache["aperture_k_1d"] ** 2
        dose = cp.sum(engine.G_qk[:, 0, 0].real, dtype=cp.float64)
        dose /= nrow * ncol * aperture2.sum()
        if not bool(cp.isfinite(dose) & (dose > 0)):
            raise ValueError(
                "No positive incident intensity estimate; check counts and bright-field calibration."
            )
        if detector_weighting == "uniform":
            weights = cp.ones(nbf, cp.float32)
        else:
            mean = cp.maximum(engine.G_qk[:, 0, 0].real / (nrow * ncol), 0)
            floor = cp.median(mean) * 0.05
            if float(floor) <= 0:
                raise ValueError(
                    "Variance weighting needs positive bright-field counts; use uniform weighting."
                )
            weights = 1 / cp.maximum(mean, floor)
            weights /= weights.mean()
        scale = cp.sum(weights * aperture2**2, dtype=cp.float64)
        rows, cols = cp.indices((nrow, ncol // 2 + 1), dtype=cp.int32)
        keep = (cols > 0) | ((rows > 0) & (rows <= (nrow - 1) // 2))
        if nrow % 2 == 0:
            keep &= rows != nrow // 2
        if ncol % 2 == 0:
            keep &= cols != ncol // 2
        rows, cols = rows[keep], cols[keep]
        alias_row, alias_col = cp.indices((factor, factor), dtype=cp.int32)
        output_rows = rows[:, None] + alias_row.ravel()[None, :] * nrow
        output_cols = cols[:, None] + alias_col.ravel()[None, :] * ncol
        qr = (
            cp.where(
                output_rows <= (out_row - 1) // 2, output_rows, output_rows - out_row
            ).astype(cp.float32)
            * cache["qx_1d"][1]
        )
        qc = (
            cp.where(
                output_cols <= (out_col - 1) // 2, output_cols, output_cols - out_col
            ).astype(cp.float32)
            * cache["qy_1d"][1]
        )
        cutoff = (
            2 * cache["semiangle_rad"] + max(cache["ang_y_rad"], cache["ang_x_rad"])
        ) / engine.wavelength
        active = qr * qr + qc * qc < cutoff**2
        counts = active.sum(axis=1)
        higher = higher_order_arrays(optics["aberrations"])
        if higher is None:
            magnitudes = np.zeros(14, np.float32)
            angles = np.zeros(14, np.float32)
            magnitudes[:2] = [
                optics["aberrations"].get("C10_nm", 0),
                optics["aberrations"].get("C12_nm", 0),
            ]
            angles[1] = np.deg2rad(optics["aberrations"].get("phi12_deg", 0))
        else:
            magnitudes, angles = higher
        groups = []
        power = cp.zeros((), cp.float64)
        for count in range(int(counts.max()) + 1):
            selected = cp.flatnonzero(counts == count)
            chunks = []
            for start in range(0, selected.size, 128):
                indices = selected[start : start + 128]
                measured = engine.G_qk[:, rows[indices], cols[indices]].T / dose
                power += cp.sum(
                    cp.abs(measured) ** 2 * weights[None, :], dtype=cp.float64
                )
                if not count:
                    continue
                group_index, alias_index = cp.nonzero(active[indices])
                group_index = indices[group_index]
                r = output_rows[group_index, alias_index].reshape(-1, count)
                c = output_cols[group_index, alias_index].reshape(-1, count)
                amplitude, phase = transfer_frequencies(
                    engine,
                    qr[group_index, alias_index].reshape(-1, count),
                    qc[group_index, alias_index].reshape(-1, count),
                    magnitudes * 10,
                    angles,
                    tuple(optics["sample_tilt_mrad"]),
                    optics["depth_spread_nm"] * 10,
                )
                transfer = (
                    cp.concatenate((amplitude, phase), axis=1)
                    if complex_object
                    else phase
                )
                transfer = transfer.astype(cp.complex128)
                weighted = transfer.conj() * weights[None, None, :]
                gram = weighted @ transfer.swapaxes(1, 2)
                rhs = (weighted @ measured[:, :, None].astype(cp.complex128))[..., 0]
                prior = cp.ones((indices.size, count), cp.float64)
                if smoothness_inv_A:
                    frequency2 = (
                        qr[group_index, alias_index] ** 2
                        + qc[group_index, alias_index] ** 2
                    ).reshape(-1, count)
                    prior = (1 + frequency2 / smoothness_inv_A**2) ** 2
                if complex_object:
                    prior = cp.concatenate((prior, prior), axis=1)
                    basis = eigen = None
                else:
                    inverse = 1 / cp.sqrt(prior.astype(cp.float64))
                    eigen, vectors = cp.linalg.eigh(
                        gram * inverse[:, :, None] * inverse[:, None, :] / scale
                    )
                    projected = (
                        vectors.conj().swapaxes(1, 2)
                        @ (rhs * inverse / scale)[:, :, None]
                    )[..., 0]
                    basis = inverse[:, :, None] * vectors * projected[:, None, :]
                chunks.append((r, c, gram, rhs, prior, basis, eigen))
            if chunks:
                arrays = [
                    None
                    if chunks[0][i] is None
                    else cp.ascontiguousarray(
                        cp.concatenate([chunk[i] for chunk in chunks])
                    )
                    for i in range(7)
                ]
                groups.append(_Group(*arrays))
        return cls(
            (out_row, out_col),
            factor,
            groups,
            complex_object,
            scale,
            power,
            float(dose),
        )

    def solve(
        self,
        phase_regularization: float,
        amplitude_regularization: float | None,
        *,
        compute_loss: bool,
    ):
        """Evaluate arbitrary strengths, returning fresh wave and diagnostics."""
        spectra = [
            cp.zeros(self.shape, cp.complex64),
            cp.zeros(self.shape, cp.complex64),
        ]
        residual = self.signal_power.copy()
        for group in self.groups:
            count = group.rows.shape[1]
            if self.complex_object:
                matrix = group.gram.copy()
                penalty = group.prior.astype(cp.float64) * self.scale
                penalty[:, :count] *= amplitude_regularization
                penalty[:, count:] *= phase_regularization
                diagonal = cp.arange(matrix.shape[-1])
                matrix[:, diagonal, diagonal] += penalty
                coefficients = cp.linalg.solve(matrix, group.rhs[..., None])[
                    ..., 0
                ].astype(cp.complex64)
            else:
                nfrequency, dim = group.eigen.shape
                coefficients = _evaluate(
                    group.basis,
                    group.eigen,
                    phase_regularization,
                    dim,
                    size=nfrequency * dim,
                ).reshape(nfrequency, dim)
            if compute_loss:
                precise = coefficients.astype(cp.complex128)
                prediction = (group.gram @ precise[..., None])[..., 0]
                residual += cp.sum(
                    (precise.conj() * prediction).real
                    - 2 * (precise.conj() * group.rhs).real
                )
            channels = [(spectra[1], coefficients[:, -count:])]
            if self.complex_object:
                channels.append((spectra[0], coefficients[:, :count]))
            for spectrum, values in channels:
                values = values * self.factor**2
                spectrum[group.rows, group.cols] = values
                spectrum[
                    (-group.rows) % self.shape[0], (-group.cols) % self.shape[1]
                ] = values.conj()
        amplitude, phase = [
            cp.fft.ifft2(spectrum).real.astype(cp.float32) for spectrum in spectra
        ]
        wave = cp.exp(amplitude + 1j * phase).astype(cp.complex64)
        if not bool(cp.isfinite(wave).all()):
            raise ValueError(
                'Direct inversion diverged; increase regularization or use method="phase".'
            )
        diagnostics = {
            "model": "T = exp(eta + i phi), linearized about unity",
            "normalization": "zero mean phase and log-amplitude; relative amplitude only",
            "frequency_policy": "supported aliases fitted jointly; DC family and even-grid Nyquist axes excluded",
            "incident_scale": self.incident_scale,
        }
        if compute_loss:
            if float(self.signal_power) > 0:
                diagnostics["training_relative_sse"] = float(
                    cp.maximum(residual, 0) / self.signal_power
                )
            else:
                diagnostics["training_relative_sse"] = None
                diagnostics["prediction_error_status"] = "undefined: no non-DC signal"
        return wave, diagnostics
