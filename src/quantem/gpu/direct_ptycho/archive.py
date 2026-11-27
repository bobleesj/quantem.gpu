"""Portable reviewed waves for display without the original scan or accelerator."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from quantem.core.io.serialize import AutoSerialize, load

from quantem.gpu.direct_ptycho.results import DirectPtychoResult, host_array


class SavedDirectPtychoResult(DirectPtychoResult, AutoSerialize):
    """Reviewed reconstruction with its measured sampling and stored model probe."""

    @property
    def phase_limits(self) -> tuple[float, float]:
        """Use the live phase viewer's 1–99% contrast for a reviewed wave."""
        if self.phase_clim is not None:
            return tuple(self.phase_clim)
        low, high = np.nanpercentile(self.phase, [1, 99])
        if low == high:
            return float(low) - 0.5, float(high) + 0.5
        return float(low), float(high)

    def probe(self, space="real"):
        if space == "real":
            return self.real_probe
        if space == "fourier":
            return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(self.real_probe), norm="ortho"))
        raise ValueError("Choose probe space 'real' or 'fourier'.")

    @property
    def probe_sampling_A(self):
        return self.real_probe_sampling_A

    @property
    def probe_sampling_mrad(self):
        return self.fourier_probe_sampling_mrad


def save_review(result: DirectPtychoResult, folder: Path) -> None:
    """Archive the reconstruction and portable optics without a live session."""
    snapshot = SavedDirectPtychoResult(
        object_wave=host_array(result.object_wave), backend=result.backend, aberrations=result.aberrations,
        sample_tilt_mrad=result.sample_tilt_mrad, depth_spread_nm=result.depth_spread_nm,
        voltage_kV=result.voltage_kV, semiangle_mrad=result.semiangle_mrad,
        scan_detector_rotation_deg=result.scan_detector_rotation_deg,
        scan_detector_rotation_flipped=result.scan_detector_rotation_flipped,
        upsample=result.upsample, object_sampling_A=result.object_sampling_A,
        reconstruction_method=result.reconstruction_method, reconstruction_options=result.reconstruction_options,
        amplitude_estimated=result.amplitude_estimated, phase_clim=result.phase_clim,
    )
    snapshot.real_probe = host_array(result.probe())
    snapshot.real_probe_sampling_A = result.probe_sampling_A
    snapshot.fourier_probe_sampling_mrad = result.probe_sampling_mrad
    with TemporaryDirectory(dir=folder) as temporary:
        path = Path(temporary) / "result.zip"
        AutoSerialize.save(snapshot, path)
        path.replace(folder / "result.zip")


def load_review(folder: Path) -> SavedDirectPtychoResult:
    """Read the reviewed result, including folders from the earlier NumPy export."""
    import json

    path = folder if folder.is_file() else folder / "result.zip"
    if path.is_file():
        result = load(path)
        if not isinstance(result, SavedDirectPtychoResult):
            raise ValueError(f"{path} does not contain a saved direct ptychography result.")
        return result
    if not (folder / "waves.npz").is_file():
        raise FileNotFoundError(f"No saved reconstruction in {folder}. Save it with ptycho.save(save_dir=...).")
    with (folder / "calibration.json").open() as stream:
        record = json.load(stream)
    with np.load(folder / "waves.npz", allow_pickle=False) as waves:
        result = SavedDirectPtychoResult(
            object_wave=waves["object_wave"], backend=record.get("backend"), aberrations=record["aberrations"],
            voltage_kV=record["voltage_kV"], semiangle_mrad=record["semiangle_mrad"],
            scan_detector_rotation_deg=record["scan_detector_rotation_deg"],
            scan_detector_rotation_flipped=record["scan_detector_rotation_flipped"],
            sample_tilt_mrad=record["sample_tilt_mrad"], depth_spread_nm=record["depth_spread_nm"],
            upsample=record["upsample"], reconstruction_method=record["reconstruction"]["method"],
            reconstruction_options=record["reconstruction"], object_sampling_A=tuple(waves["object_sampling_A"]),
            amplitude_estimated=record["reconstruction"]["method"] == "complex",
        )
        result.real_probe = waves["probe"]
        result.real_probe_sampling_A = tuple(waves["probe_sampling_A"])
        from quantem.gpu.optics.physics import wavelength_A_from_kV

        result.fourier_probe_sampling_mrad = tuple(
            1000 * wavelength_A_from_kV(result.voltage_kV) / (length * step)
            for length, step in zip(result.real_probe.shape, result.real_probe_sampling_A)
        )
        return result
