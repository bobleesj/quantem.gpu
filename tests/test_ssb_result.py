"""Portable phase validation is independent of renderer and file paths."""
import copy
import json
import math

import numpy as np
import pytest

from quantem.gpu.formats.ssb_phase import SCHEMA, read_result
from quantem.gpu.ssb.persistence import FIT_SCHEMA


def _fit(**changes):
    """A stored fit block of the current schema, as SSB.reconstruct_series writes it."""
    return {"schema": FIT_SCHEMA, "voltage_kV": 300, "semiangle_mrad": 30, "scan_sampling_A": [.25, .25],
            "detector_sampling_mrad": [0.55, 0.56], "bright_field_center_px": [3, 4], "bright_field_radius_px": 5,
            "scan_detector_rotation_deg": 192.0, "detector_transposed": False,
            "aberrations": {"C10_nm": 7, "C12_nm": 2, "phi12_deg": math.degrees(.1)}, **changes}


def test_export_to_selected_catalogue_preserves_prior_results(tmp_path):
    """Microscope screening exports beside processed results, not raw inputs."""
    import h5py
    from quantem.gpu.formats.ssb_phase import export_result

    raw = tmp_path / "raw"
    raw.mkdir()
    master = raw / "scan_master.h5"
    with h5py.File(raw / "scan_data.h5", "w") as handle:
        handle["data"] = np.ones((2, 2, 2), dtype=np.uint16)
    with h5py.File(master, "w") as handle:
        handle["entry/data/data_000001"] = h5py.ExternalLink("scan_data.h5", "/data")
    run = tmp_path / "run"
    run.mkdir()
    phase = np.array([[0, 1], [-1, 2]], dtype=np.float32)
    np.save(run / "ssb_phase.npy", phase)
    config = {"computed": {"ssb": _fit(voltage_kV=200, scan_sampling_A=[0.5, 0.5])}}
    (run / "config.json").write_text(json.dumps(config))
    catalogue = tmp_path / "processed"
    first = export_result(run, master, output_folder=catalogue)
    assert first.is_relative_to(catalogue)
    assert export_result(run, master, output_folder=catalogue) == first
    np.save(run / "ssb_phase.npy", phase + 1)
    second = export_result(run, master, output_folder=catalogue)
    assert first != second
    np.testing.assert_array_equal(read_result(first)[1], phase)
    np.testing.assert_array_equal(read_result(second)[1], phase + 1)
    assert not (raw / "live").exists()


def test_real_pair_validation(tmp_path):
    """Keep signed float bits and reject incomplete or misbound companions."""
    import hashlib
    phase = np.array([[-0.0, -1.25], [3.5, 1e-20]], dtype="<f4")
    np.save(tmp_path / "phase.npy", phase, allow_pickle=False)
    record = dict(schema=SCHEMA, phase_file="phase.npy", shape=[2, 2], phase_encoding="float32-le-row-major",
                  phase_unit="rad", source_identity="a" * 64, phase_sha256=hashlib.sha256(phase.tobytes()).hexdigest(),
                  calibration=dict(voltage_kV=300.0, semiangle_mrad=30.0, scan_sampling_A=[0.25, 0.25],
                                   detector_sampling_mrad=[0.55, 0.56], bright_field_center_px=[3.0, 4.0]),
                  aberrations={"C10_nm": 0.0, "C12_nm": 0.0, "phi12_deg": 0.0},
                  scan_detector_rotation_deg=192.0, detector_transposed=False,
                  provenance={"producer": "test fixture"},
                  run_metadata={"notes": "Exact phase, not a preview"})
    manifest = tmp_path / "sample-01.json"
    manifest.write_text(json.dumps(record))
    _, restored = read_result(manifest, "a" * 64)
    assert restored.tobytes() == phase.tobytes()
    with pytest.raises(ValueError):
        read_result(manifest, "b" * 64)
    for key, value in [("phase_file", "../phase.npy"), ("phase_file", "/phase.npy"),
                       ("phase_sha256", "0" * 64), ("phase_unit", "unknown"),
                       ("shape", [3, 2]), ("shape", [2.0, 2]), ("schema", "quantem.ssb-phase/1"),
                       ("provenance", {}), ("calibration", {}), ("detector_transposed", None),
                       ("aberrations", {"C10_nm": 0.0, "C12_nm": 0.0, "phi12": 0.0})]:
        changed = copy.deepcopy(record)
        changed[key] = value
        manifest.write_text(json.dumps(changed))
        with pytest.raises((ValueError, KeyError, TypeError)):
            read_result(manifest)
    manifest.write_text(json.dumps(record))
    (tmp_path / "phase.npy").unlink()
    with pytest.raises(FileNotFoundError):
        read_result(manifest)


def test_export_takes_only_a_current_fit_and_its_physical_rotation(tmp_path):
    """An exported record states the fit's own calibration, rotation and detector sampling.

    The version-1 export copied the angle below 180 as the rotation without the reversal flag (180 degrees off for a
    reversed scan), computed the detector sampling from another radius than the fit used, and copied aberrations from
    a block of any schema.
    """
    import h5py
    from quantem.gpu.formats.ssb_phase import export_result

    master = tmp_path / "scan_master.h5"
    with h5py.File(tmp_path / "scan_data.h5", "w") as handle:
        handle["data"] = np.ones((2, 2, 2), dtype=np.uint16)
    with h5py.File(master, "w") as handle:
        handle["entry/data/data_000001"] = h5py.ExternalLink("scan_data.h5", "/data")
    run = tmp_path / "run"
    run.mkdir()
    np.save(run / "ssb_phase.npy", np.zeros((2, 2), np.float32))
    earlier = {"voltage_kV": 300, "semiangle_mrad": 30, "scan_sampling_A": 0.25, "aberration_unit": "nm",
               "aberrations": {"C10": 7, "C12": 2, "phi12": .1}, "rotation_angle_deg": 12.0, "com_reversed": True}
    (run / "config.json").write_text(json.dumps({"computed": {"bf_radius": 5, "bf_center": [3, 4], "ssb": earlier}}))
    with pytest.raises(ValueError, match="no current SSB fit"):
        export_result(run, master, tmp_path / "out.json")
    (run / "config.json").write_text(json.dumps({"computed": {"ssb": _fit()}}))
    record, _ = read_result(export_result(run, master, tmp_path / "out.json"))
    assert record["scan_detector_rotation_deg"] == 192.0 and record["detector_transposed"] is False
    assert record["calibration"]["detector_sampling_mrad"] == [0.55, 0.56]
    assert record["aberrations"] == {"C10_nm": 7.0, "C12_nm": 2.0, "phi12_deg": math.degrees(.1)}


def test_numbered_exports_respect_selected_output_folder(tmp_path):
    """Repeated screening stays in its chosen output tree and preserves counts."""
    import h5py
    from quantem.gpu.formats.ssb_phase import export_result

    raw = tmp_path / "raw"
    raw.mkdir()
    master, member = raw / "gold_master.h5", raw / "gold_data.h5"
    with h5py.File(member, "w") as handle:
        handle["data"] = np.arange(16, dtype=np.uint16)
    with h5py.File(master, "w") as handle:
        handle["entry/data/data_000001"] = h5py.ExternalLink(member.name, "data")
    raw_bytes = {p.name: p.read_bytes() for p in raw.iterdir()}
    output_folder = tmp_path / "selected/live/screen"
    run = output_folder / "gold"
    run.mkdir(parents=True)
    phase = np.array([[-0., -1.25], [3.5, 1e-20]], dtype=np.float32)
    np.save(run / "ssb_phase.npy", phase)
    config = {"source_master": str(master), "computed": {"ssb": _fit()}}
    (run / "config.json").write_text(json.dumps(config))
    first = export_result(run, master, output_folder=output_folder)
    assert first == output_folder / "gold_master/ssb.json"
    assert export_result(run, master, output_folder=output_folder) == first
    assert read_result(first)[1].tobytes() == phase.tobytes()
    np.save(run / "ssb_phase.npy", phase + 1)
    second = export_result(run, master, output_folder=output_folder)
    assert second == output_folder / "gold_master-02/ssb.json"
    assert export_result(run, master, output_folder=output_folder) == second
    assert read_result(first)[1].tobytes() == phase.tobytes()
    assert read_result(second)[1].tobytes() == (phase + 1).tobytes()
    assert {p.name: p.read_bytes() for p in raw.iterdir()} == raw_bytes
    with pytest.raises(ValueError, match="either an output filename or output_folder"):
        export_result(run, master, tmp_path / "conflict.json", output_folder=output_folder)
    assert not (tmp_path / "conflict.json").exists()


@pytest.mark.parametrize("changed", ["master", "member", "replacement"])
def test_early_verification_rejects_changed_source(tmp_path, changed):
    """Hash once during computation; refuse replacement or modified inputs."""
    import hashlib
    import os
    import h5py
    from quantem.gpu.formats.ssb_phase import AcquisitionVerification, export_result

    member = tmp_path / "scan_data.h5"
    master = tmp_path / "scan_master.h5"
    with h5py.File(member, "w") as handle:
        handle["data"] = np.ones((2, 2, 2), dtype=np.uint16)
    with h5py.File(master, "w") as handle:
        handle["entry/data/data_000001"] = h5py.ExternalLink(member.name, "/data")
    hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in (master, member)]
    expected = hashlib.sha256(b"live4dstem.dataset/v0.1\0" + "\0".join(hashes).encode()).hexdigest()
    verified = AcquisitionVerification(master)
    assert verified.identity(master) == expected
    np.save(tmp_path / "ssb_phase.npy", np.ones((2, 2), dtype=np.float32))
    (tmp_path / "config.json").write_text(json.dumps({"computed": {"ssb": _fit()}}))
    record, _ = read_result(export_result(tmp_path, master, tmp_path / "result.json", verification=verified))
    assert record["source_identity"] == expected
    path = master if changed == "master" else member
    if changed == "replacement":
        original = path.stat()
        replacement = path.with_suffix(".replacement")
        replacement.write_bytes(path.read_bytes())
        os.utime(replacement, ns=(original.st_atime_ns, original.st_mtime_ns))
        os.replace(replacement, path)
    else:
        with h5py.File(path, "a") as handle:
            handle.attrs["updated"] = True
    with pytest.raises(ValueError, match="changed"):
        export_result(tmp_path, master, tmp_path / "refused.json", verification=verified)
    assert not (tmp_path / "refused.json").exists()


def test_qem_export_binds_its_own_bytes_and_preserves_phase(tmp_path):
    import hashlib
    from quantem.gpu.formats.qem.reference import save_array
    from quantem.gpu.formats.ssb_phase import AcquisitionVerification, export_result

    source = tmp_path / "scan.qem"
    save_array(source, np.arange(16, dtype=np.uint16).reshape(2, 2, 2, 2))
    verified = AcquisitionVerification(source)
    expected = hashlib.sha256(b"quantem.qem-source/v1\0" + hashlib.sha256(source.read_bytes()).hexdigest().encode()).hexdigest()
    assert verified.identity(source) == expected
    phase = np.array([[1, -1], [0.1, -0.0]], dtype=np.float32)
    np.save(tmp_path / "ssb_phase.npy", phase)
    (tmp_path / "config.json").write_text(json.dumps({"computed": {"ssb": _fit()}}))
    manifest = export_result(tmp_path, source, tmp_path / "result.json", verification=verified)
    record, recovered = read_result(manifest, expected)
    assert recovered.tobytes() == phase.tobytes()
    with source.open("r+b") as stream:
        stream.seek(-1, 2)
        stream.write(b"x")
    with pytest.raises(ValueError):
        verified.identity(source)
    with pytest.raises(ValueError):
        AcquisitionVerification(source)
