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
from pathlib import Path
import tempfile

import h5py
import numpy as np


def acquisition_identity(master: Path) -> str:
    """Hash unchanged acquisition bytes in logical HDF5 member order.

    Folder names, paths and timestamps do not enter the identity. Rewriting an
    HDF5 master or re-encoding its shards deliberately creates another identity.
    """
    with h5py.File(master, "r") as handle:
        group = handle["entry/data"]
        members = []
        for key in sorted(group):
            link = group.get(key, getlink=True)
            if not isinstance(link, h5py.ExternalLink):
                raise ValueError("This export currently requires an ARINA master with external data members.")
            members.append(master.parent / link.filename)
    digest = hashlib.sha256(b"live4dstem.dataset/v0.1\0")
    for index, path in enumerate([master] + members):
        before = path.stat()
        with path.open("rb") as stream:
            member = hashlib.file_digest(stream, "sha256").hexdigest()
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Acquisition changed during verification. Retry with a completed acquisition.")
        digest.update((b"\0" if index else b"") + member.encode())
    return digest.hexdigest()


def read_result(manifest: Path, expected_identity: str | None = None) -> tuple[dict, np.ndarray]:
    """Validate a portable JSON/NPY pair before publishing its scientific values."""
    try:
        return _read_result(manifest, expected_identity)
    except (KeyError, TypeError, OverflowError, EOFError) as exc:
        raise ValueError("Malformed SSB result. Keep its original JSON and NumPy companion together.") from exc


def _read_result(manifest: Path, expected_identity: str | None) -> tuple[dict, np.ndarray]:
    if manifest.stat().st_size > 32 << 20:
        raise ValueError("SSB JSON exceeds 32 MB.")
    record = json.loads(manifest.read_text())
    name = record.get("phaseFile", "")
    if (record.get("format") != "live.ssb" or record.get("schemaVersion") != 1
            or not name.endswith(".npy") or Path(name).name != name
            or name.startswith(".") or "\\" in name):
        raise ValueError("Invalid SSB manifest or companion filename.")
    path = manifest.parent / name
    if path.resolve().parent != manifest.parent.resolve():
        raise ValueError("The companion phase must be inside the result folder.")
    rows, cols = record["rows"], record["columns"]
    if (type(rows) is not int or type(cols) is not int
            or not (0 < rows <= 4096 and 0 < cols <= 4096)
            or path.stat().st_size > rows * cols * 4 + 65536):
        raise ValueError("Invalid SSB phase dimensions or file size.")
    with path.open("rb") as stream:
        if stream.read(8) != b"\x93NUMPY\x01\x00":
            raise ValueError("Expected NumPy v1 phase data.")
    array = np.load(path, allow_pickle=False)
    identity = record["sourceIdentity"]
    if (array.dtype.str != "<f4" or array.shape != (rows, cols)
            or not array.flags.c_contiguous or not np.isfinite(array).all()
            or record.get("phaseUnits") != "rad"
            or record.get("phaseEncoding") != "float32-le-row-major"
            or len(identity) != 64 or any(c not in "0123456789abcdef" for c in identity)
            or (expected_identity is not None and identity != expected_identity)
            or hashlib.sha256(array.tobytes()).hexdigest() != record["phaseSHA256"]):
        raise ValueError("SSB result has wrong source identity, dtype, shape or checksum.")
    cal = record["calibration"]
    for key in ("beamEnergyKeV", "semiangleMrad", "scanStepRowAngstroms",
                "scanStepColumnAngstroms", "detectorStepRowMrad", "detectorStepColumnMrad"):
        if not np.isfinite(cal[key]) or cal[key] <= 0:
            raise ValueError(f"Invalid calibration: {key}.")
    if not all(np.isfinite(record[key]) for key in
               ("c10Nanometers", "c12Nanometers", "phi12Radians", "rotationDegrees")):
        raise ValueError("Non-finite SSB aberrations.")
    if not all(np.isfinite(cal[key]) for key in ("centerRow", "centerColumn")):
        raise ValueError("Non-finite detector center.")
    radius = cal.get("brightfieldRadiusPixels")
    if radius is not None and (not np.isfinite(radius) or radius <= 0):
        raise ValueError("Bright-field radius must be positive.")
    if not isinstance(record.get("provenance"), dict) or not record["provenance"]:
        raise ValueError("SSB result needs recorded provenance.")
    if not all(isinstance(value, str) for value in record["provenance"].values()):
        raise ValueError("Provenance values must be strings.")
    if any(type(pixel) is not int or pixel < 0 for pixel in cal.get("excludedDetectorPixels", []) or []):
        raise ValueError("Excluded detector pixels must be nonnegative integer indices.")
    if not isinstance(record["runMetadata"], dict):
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
                  expected_identity: str | None = None) -> Path:
    """Preserve phase bits and physical settings in a JSON/NumPy result pair.

    Example: export_result(run_folder, master, Path('result.json'), digest).
    """
    identity = acquisition_identity(master)
    if expected_identity is not None and identity != expected_identity:
        raise ValueError("Local acquisition does not match the verified remote source. No result was exported.")
    config = json.loads((run_folder / "config.json").read_text())
    # Catalogue links are transport state, not part of original run provenance.
    config.pop("ssb_result", None)
    config.pop("ssb_result_source_identity", None)
    computed = config["computed"]
    ssb = computed["ssb"]
    phase = np.load(run_folder / "ssb_phase.npy", allow_pickle=False)
    if (phase.dtype != np.dtype("float32") or phase.ndim != 2
            or any(size < 1 or size > 4096 for size in phase.shape)
            or not np.isfinite(phase).all()):
        raise ValueError("Expected a finite 2D float32 SSB phase, at most 4096 pixels per axis; no implicit precision conversion is allowed.")
    phase = np.ascontiguousarray(phase, dtype="<f4")
    values = phase.astype("<f4", copy=False).tobytes(order="C")
    radius = float(computed["bf_radius"])
    semiangle = float(ssb["semiangle_mrad"])
    step = float(ssb["scan_sampling_A"])
    calibration = dict(
        beamEnergyKeV=float(ssb["voltage_kV"]), semiangleMrad=semiangle,
        scanStepRowAngstroms=step, scanStepColumnAngstroms=step,
        detectorStepRowMrad=semiangle / radius, detectorStepColumnMrad=semiangle / radius,
        centerRow=float(computed["bf_center"][0]), centerColumn=float(computed["bf_center"][1]),
        brightfieldRadiusPixels=radius,
    )
    aberrations = ssb["aberrations"]
    result = dict(
        format="live.ssb",
        schemaVersion=1, sourceIdentity=identity, rows=phase.shape[0], columns=phase.shape[1],
        phaseEncoding="float32-le-row-major", phaseUnits="rad",
        phaseSHA256=hashlib.sha256(values).hexdigest(),
        calibration=calibration, c10Nanometers=float(aberrations["C10"]),
        c12Nanometers=float(aberrations["C12"]), phi12Radians=float(aberrations["phi12"]),
        rotationDegrees=float(ssb["rotation_angle_deg"]),
        provenance={
            "producer": "quantem.live original screen result",
            "run": str(ssb.get("ran_at", config.get("timestamp", "unrecorded"))),
            "sourceBinding": "Current acquisition bytes hashed at export; this does not establish an unrecorded historical run digest",
            "detectorSampling": "Derived from recorded semi-angle / BF radius",
            "phaseVariant": "original ssb_phase.npy; not picked, calibrated or denoised",
            "backendRevision": str(ssb.get("backend_revision", "not recorded by original run")),
        },
        runMetadata=config,
    )
    if output is None:
        stem = master.stem
        folder = master.parent / "live" / "screen"
        # Stable numbering, including distinct calibrations of the same image.
        for number in range(1, 10000):
            name = stem if number == 1 else f"{stem}-{number:02d}"
            candidate = folder / name / "ssb.json"
            if not candidate.exists():
                output = candidate
                break
            existing = json.loads(candidate.read_text())
            comparison = dict(existing)
            comparison.pop("phaseFile", None)
            if comparison == result:
                read_result(candidate, identity)
                return candidate
        if output is None:
            raise ValueError("Too many saved SSB results. Choose an explicit output name.")
    if output.suffix != ".json":
        raise ValueError("Choose a .json result filename; its .npy companion is written automatically.")
    result["phaseFile"] = output.with_suffix(".npy").name
    buffer = io.BytesIO()
    np.save(buffer, phase, allow_pickle=False)
    _publish(output.with_suffix(".npy"), buffer.getvalue())
    _publish(output, (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
    read_result(output, identity)
    print(json.dumps({"output": str(output), "sourceIdentity": identity,
                      "phaseSHA256": result["phaseSHA256"], "bytes": output.stat().st_size}))
    return output


def main() -> None:
    """Export a verified existing screening run without rerunning SSB."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-folder", type=Path, required=True)
    parser.add_argument("--source-master", type=Path, required=True)
    parser.add_argument("--expected-source-id", required=True)
    parser.add_argument("--output", type=Path, help="Default: <source folder>/live/screen/<source stem>/ssb.json.")
    args = parser.parse_args()
    export_result(args.run_folder, args.source_master, args.output, args.expected_source_id)


if __name__ == "__main__":
    main()
