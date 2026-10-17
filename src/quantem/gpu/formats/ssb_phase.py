"""Portable SSB phase: ordinary NumPy arrays and readable JSON calibration.

Agent-side utility only: NumPy/h5py are not app dependencies. This exporter
selects the original ssb_phase.npy and its computed.ssb record, never a separately
calibrated, locked, picked or denoised phase with different provenance.
"""

import argparse
import hashlib
import io
import json
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np

from quantem.gpu.formats.hdf5.frames import detector_sources
from quantem.gpu.ssb.series import stored_fit

# Carried by every portable phase record under "schema"; another value is refused.
# 2: the shared vocabulary (scan_sampling_A, detector_sampling_mrad, bright_field_center_px, aberrations in
#    C10_nm / C12_nm / phi12_deg, scan_detector_rotation_deg with detector_transposed, snake_case keys). Version 1
#    was format "live.ssb" with schemaVersion 1 and camelCase keys; its rotationDegrees was the angle below 180.
SCHEMA = "quantem.ssb-phase/2"


def _file_generation(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


class AcquisitionVerification:
    """Hash one completed acquisition, then guard it until publication.

    This short-lived verification lets a pipeline overlap hashing with GPU
    work. It is never persisted or reused after a member changes. Creating it
    reads every source byte; ``identity`` rechecks all members before export.

    Parameters
    ----------
    master : Path
        Completed ARINA master with external detector members.

    Examples
    --------
    >>> verification = AcquisitionVerification(master)
    >>> export_result(run_folder, master, verification=verification)
    """

    def __init__(self, master: Path) -> None:
        self._master = Path(master).resolve()
        before_master = _file_generation(self._master)
        with h5py.File(self._master, "r") as handle:
            sources = detector_sources(handle)
        if not all(source.external for source in sources):
            raise ValueError("This export currently requires an ARINA master with external data members.")
        self._paths = (self._master, *(Path(source.path).resolve() for source in sources))
        self._generation = tuple(_file_generation(path) for path in self._paths)
        if self._generation[0] != before_master:
            raise ValueError("Acquisition changed during verification. Retry with a completed acquisition.")
        digest = hashlib.sha256(b"live4dstem.dataset/v0.1\0")
        for index, path in enumerate(self._paths):
            with path.open("rb") as stream:
                member = hashlib.file_digest(stream, "sha256").hexdigest()
            digest.update((b"\0" if index else b"") + member.encode())
        self._digest = digest.hexdigest()
        self.identity(master)

    def identity(self, master: Path) -> str:
        """Return the byte identity only while every member is unchanged.

        Parameters
        ----------
        master : Path
            The same source master that was verified before computation.

        Returns
        -------
        str
            SHA-256 identity of the complete acquisition.

        Raises
        ------
        ValueError
            A source member was modified or replaced after verification.

        Examples
        --------
        >>> verification.identity(master)
        """
        if (Path(master).resolve() != self._master
                or tuple(_file_generation(path) for path in self._paths) != self._generation):
            raise ValueError("Acquisition changed during verification. Retry with a completed acquisition.")
        return self._digest


def acquisition_identity(master: Path) -> str:
    """Hash unchanged acquisition bytes in logical HDF5 member order.

    Folder names, paths and timestamps do not enter the identity. Rewriting an
    HDF5 master or re-encoding its shards deliberately creates another identity.
    """
    return AcquisitionVerification(master).identity(master)


def read_result(manifest: Path, expected_identity: str | None = None) -> tuple[dict, np.ndarray]:
    """Validate a portable JSON/NPY pair before publishing its scientific values."""
    try:
        return _read_result(manifest, expected_identity)
    except (KeyError, TypeError, OverflowError, EOFError) as exc:
        raise ValueError("Malformed SSB result. Keep its original JSON and NumPy companion together.") from exc


def _read_result(manifest: Path, expected_identity: str | None) -> tuple[dict, np.ndarray]:
    """Check every manifest field, the companion path and the phase checksum.

    A missing or mistyped field raises ``KeyError`` or ``TypeError`` here;
    ``read_result`` reports those as one malformed-result error.
    """
    if manifest.stat().st_size > 32 << 20:
        raise ValueError("SSB JSON exceeds 32 MB.")
    record = json.loads(manifest.read_text())
    name = record.get("phase_file", "")
    if record.get("schema") != SCHEMA:
        raise ValueError(f"SSB result schema {record.get('schema')!r} is not read; this reader accepts {SCHEMA!r}. Export the result again.")
    if not name.endswith(".npy") or Path(name).name != name or name.startswith(".") or "\\" in name:
        raise ValueError("Invalid SSB manifest or companion filename.")
    path = manifest.parent / name
    if path.resolve().parent != manifest.parent.resolve():
        raise ValueError("The companion phase must be inside the result folder.")
    rows, cols = record["shape"]
    if (type(rows) is not int or type(cols) is not int
            or not (0 < rows <= 4096 and 0 < cols <= 4096)
            or path.stat().st_size > rows * cols * 4 + 65536):
        raise ValueError("Invalid SSB phase dimensions or file size.")
    with path.open("rb") as stream:
        if stream.read(8) != b"\x93NUMPY\x01\x00":
            raise ValueError("Expected NumPy v1 phase data.")
    array = np.load(path, allow_pickle=False)
    identity = record["source_identity"]
    if (array.dtype.str != "<f4" or array.shape != (rows, cols)
            or not array.flags.c_contiguous or not np.isfinite(array).all()
            or record.get("phase_unit") != "rad"
            or record.get("phase_encoding") != "float32-le-row-major"
            or len(identity) != 64 or any(character not in "0123456789abcdef" for character in identity)
            or (expected_identity is not None and identity != expected_identity)
            or hashlib.sha256(array.tobytes()).hexdigest() != record["phase_sha256"]):
        raise ValueError("SSB result has wrong source identity, dtype, shape or checksum.")
    calibration = record["calibration"]
    for key in ("voltage_kV", "semiangle_mrad"):
        if not np.isfinite(calibration[key]) or calibration[key] <= 0:
            raise ValueError(f"Invalid calibration: {key}.")
    for key in ("scan_sampling_A", "detector_sampling_mrad"):
        if len(calibration[key]) != 2 or not all(np.isfinite(value) and value > 0 for value in calibration[key]):
            raise ValueError(f"Invalid calibration: {key} is a positive [row, col] pair.")
    aberrations = record["aberrations"]
    if set(aberrations) != {"C10_nm", "C12_nm", "phi12_deg"} or not all(np.isfinite(value) for value in aberrations.values()):
        raise ValueError("SSB aberrations are finite C10_nm, C12_nm and phi12_deg.")
    if not np.isfinite(record["scan_detector_rotation_deg"]) or type(record["detector_transposed"]) is not bool:
        raise ValueError("SSB geometry needs a finite scan_detector_rotation_deg and a detector_transposed flag.")
    center = calibration["bright_field_center_px"]
    if len(center) != 2 or not all(np.isfinite(value) for value in center):
        raise ValueError("Non-finite detector center.")
    radius = calibration.get("bright_field_radius_px")
    if radius is not None and (not np.isfinite(radius) or radius <= 0):
        raise ValueError("Bright-field radius must be positive.")
    if not isinstance(record.get("provenance"), dict) or not record["provenance"]:
        raise ValueError("SSB result needs recorded provenance.")
    if not all(isinstance(value, str) for value in record["provenance"].values()):
        raise ValueError("Provenance values must be strings.")
    if any(type(pixel) is not int or pixel < 0 for pixel in calibration.get("excluded_detector_pixels", []) or []):
        raise ValueError("Excluded detector pixels must be nonnegative integer indices.")
    if not isinstance(record["run_metadata"], dict):
        raise ValueError("SSB run metadata must be a readable JSON object.")
    return record, array


def _publish(path: Path, data: bytes) -> None:
    """Publish without replacing another result; identical repeated exports are safe."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError("Another result uses this filename. Choose a new name.")
    finally:
        temporary.unlink()


def export_result(run_folder: Path, master: Path, output: Path | None = None,
                  expected_identity: str | None = None, *,
                  output_folder: Path | None = None,
                  verification: AcquisitionVerification | None = None) -> Path:
    """Preserve phase bits and physical settings in a JSON/NumPy result pair.

    ``output_folder`` selects the parent of numbered acquisition folders;
    otherwise they are placed in ``<source folder>/live/screen``. An explicit
    ``output`` filename and ``output_folder`` are mutually exclusive.

    Example: export_result(run_folder, master, Path('result.json'), digest).

    ``verification`` can carry a byte hash computed earlier in the same job.
    Every source member must still match its verified file generation.

    ``output_folder`` selects a result catalogue without writing beside the raw
    acquisition. Automatic names retain previous results on repeated exports.
    """
    if output is not None and output_folder is not None:
        raise ValueError("Choose either an output filename or output_folder, not both.")
    identity = acquisition_identity(master) if verification is None else verification.identity(master)
    if expected_identity is not None and identity != expected_identity:
        raise ValueError("Local acquisition does not match the verified remote source. No result was exported.")
    config = json.loads((run_folder / "config.json").read_text())
    # Catalogue links are transport state, not part of original run provenance.
    config.pop("ssb_result", None)
    config.pop("ssb_result_source_identity", None)
    # Only a fit of the current schema is exported: an older block can hold Angstrom under nm keys or the angle
    # below 180 without its reversal, and the exported numbers would describe another reconstruction.
    ssb = stored_fit(config)
    missing = [key for key in ("voltage_kV", "semiangle_mrad", "scan_sampling_A", "detector_sampling_mrad",
                               "bright_field_center_px", "aberrations", "scan_detector_rotation_deg") if key not in ssb]
    if missing:
        raise ValueError(f"{run_folder / 'config.json'} holds no current SSB fit with {missing}; fit it again.")
    phase = np.load(run_folder / "ssb_phase.npy", allow_pickle=False)
    if (phase.dtype != np.dtype("float32") or phase.ndim != 2
            or any(size < 1 or size > 4096 for size in phase.shape)
            or not np.isfinite(phase).all()):
        raise ValueError("Expected a finite 2D float32 SSB phase, at most 4096 pixels per axis; no implicit precision conversion is allowed.")
    # The checksum and the .npy companion describe little-endian row-major float32 bytes.
    phase = np.ascontiguousarray(phase, dtype="<f4")
    phase_bytes = phase.tobytes(order="C")
    calibration = dict(
        voltage_kV=float(ssb["voltage_kV"]), semiangle_mrad=float(ssb["semiangle_mrad"]),
        scan_sampling_A=[float(value) for value in ssb["scan_sampling_A"]],
        detector_sampling_mrad=[float(value) for value in ssb["detector_sampling_mrad"]],
        bright_field_center_px=[float(value) for value in ssb["bright_field_center_px"]],
    )
    if ssb.get("bright_field_radius_px") is not None:
        calibration["bright_field_radius_px"] = float(ssb["bright_field_radius_px"])
    result = dict(
        schema=SCHEMA, source_identity=identity, shape=list(phase.shape),
        phase_encoding="float32-le-row-major", phase_unit="rad",
        phase_sha256=hashlib.sha256(phase_bytes).hexdigest(),
        calibration=calibration,
        aberrations={key: float(ssb["aberrations"][key]) for key in ("C10_nm", "C12_nm", "phi12_deg")},
        scan_detector_rotation_deg=float(ssb["scan_detector_rotation_deg"]),
        detector_transposed=bool(ssb.get("detector_transposed", False)),
        provenance={
            "producer": "quantem.live original screen result",
            "run": str(ssb.get("ran_at", config.get("timestamp", "unrecorded"))),
            "source_binding": "Current acquisition bytes hashed and verified unchanged at export; this does not establish an unrecorded historical run digest",
            "detector_sampling": "The detector sampling the fit used, from the stored fit",
            "phase_variant": "original ssb_phase.npy; not picked, calibrated or denoised",
            "backend_revision": str(ssb.get("backend_revision", "not recorded by original run")),
        },
        run_metadata=config,
    )
    if output is None:
        stem = master.stem
        folder = Path(output_folder) if output_folder is not None else master.parent / "live" / "screen"
        # Stable numbering, including distinct calibrations of the same image.
        for number in range(1, 10000):
            name = stem if number == 1 else f"{stem}-{number:02d}"
            candidate = folder / name / "ssb.json"
            if not candidate.exists():
                output = candidate
                break
            existing = json.loads(candidate.read_text())
            comparison = dict(existing)
            comparison.pop("phase_file", None)
            if comparison == result:
                read_result(candidate, identity)
                return candidate
        if output is None:
            raise ValueError("Too many saved SSB results. Choose an explicit output name.")
    if output.suffix != ".json":
        raise ValueError("Choose a .json result filename; its .npy companion is written automatically.")
    result["phase_file"] = output.with_suffix(".npy").name
    buffer = io.BytesIO()
    np.save(buffer, phase, allow_pickle=False)
    if verification is not None:
        verification.identity(master)
    _publish(output.with_suffix(".npy"), buffer.getvalue())
    _publish(output, (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
    read_result(output, identity)
    print(json.dumps({"output": str(output), "source_identity": identity,
                      "phase_sha256": result["phase_sha256"], "bytes": output.stat().st_size}))
    return output


def main() -> None:
    """Export a verified existing screening run without rerunning SSB."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-folder", type=Path, required=True)
    parser.add_argument("--source-master", type=Path, required=True)
    parser.add_argument("--expected-source-identity", required=True)
    parser.add_argument("--output", type=Path, help="Default: <source folder>/live/screen/<source stem>/ssb.json.")
    args = parser.parse_args()
    export_result(args.run_folder, args.source_master, args.output, args.expected_source_identity)


if __name__ == "__main__":
    main()
