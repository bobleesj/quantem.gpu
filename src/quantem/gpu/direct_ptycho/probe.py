"""Calculate the SSB optical model on a calibrated GPU Fourier grid."""

import math

from quantem.gpu.optics.physics import wavelength_A_from_kV
from quantem.gpu.direct_ptycho.higher_order import ORDERS


def probe_grid(result):
    """Resolve the aperture and leave room around its defocused footprint."""
    if result.voltage_kV is None or result.semiangle_mrad is None:
        raise ValueError("Probe calibration is missing. Run ptycho.run() with voltage and convergence semiangle.")
    wavelength = wavelength_A_from_kV(result.voltage_kV)
    semiangle = result.semiangle_mrad * 1e-3
    sampling = wavelength / (4 * semiangle)
    # Four times the largest geometric radius plus a margin for diffraction tails.
    radius = sum(10 * abs(result.aberrations.get(f"C{n}{m}_nm", 0))
                 * semiangle ** n * (1 + m / (n + 1)) for n, m in ORDERS)
    field = 4 * radius + 16 * wavelength / semiangle
    size = 2 ** math.ceil(math.log2(max(256, field / sampling)))
    angular_step = wavelength / (size * sampling) * 1000
    return (size, size), (sampling, sampling), (angular_step, angular_step)


def model_probe(result, *, space: str):
    """Return a centered model wave using the result's accelerator."""
    shape, sampling_A, _ = probe_grid(result)
    if result.backend == "cuda":
        import cupy as xp

        context = xp.cuda.Device(result.object_wave.device.id)
    elif result.backend == "mps":
        import mlx.core as xp

        context = xp.stream(xp.default_stream(xp.gpu))
    else:
        raise NotImplementedError("Model probes require a Python CUDA or MPS result.")
    with context:
        wavelength = wavelength_A_from_kV(result.voltage_kV)
        rows, columns = shape
        # Center the optical axis; measured detector-center offsets are calibration,
        # not a fitted beam tilt. Aberration angles are already in the scan frame.
        angle_row = (xp.arange(rows, dtype=xp.float32) - rows // 2)[:, None]
        angle_column = (xp.arange(columns, dtype=xp.float32) - columns // 2)[None, :]
        step_row = wavelength / (rows * sampling_A[0])
        step_column = wavelength / (columns * sampling_A[1])
        angle_row = angle_row * step_row
        angle_column = angle_column * step_column
        alpha_squared = angle_row ** 2 + angle_column ** 2
        phi = xp.arctan2(angle_column, angle_row)
        edge_width = xp.sqrt((xp.cos(phi) * step_row) ** 2 + (xp.sin(phi) * step_column) ** 2)
        aperture = xp.clip((result.semiangle_mrad * 1e-3 - xp.sqrt(alpha_squared)) / edge_width + 0.5, 0, 1)
        # The same 14 Krivanek terms as the reconstruction's Cartesian shader.
        # Public lengths are nm; wavelength and engine coefficients are Å.
        chi = xp.zeros_like(alpha_squared)
        for n, m in ORDERS:
            magnitude = 10 * result.aberrations.get(f"C{n}{m}_nm", 0.0)
            if magnitude == 0:
                continue
            azimuth = math.radians(result.aberrations.get(f"phi{n}{m}_deg", 0.0))
            chi = chi + (2 * math.pi / wavelength) * magnitude / (n + 1) * (
                alpha_squared ** ((n + 1) / 2) * xp.cos(m * (phi - azimuth))
            )
        wave = aperture * xp.exp(-1j * chi)
        wave = wave / xp.sqrt(xp.sum(xp.abs(wave) ** 2))
        if space == "real":
            wave = xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(wave))) * math.sqrt(rows * columns)
        if result.backend == "mps":
            xp.eval(wave)
        return wave
