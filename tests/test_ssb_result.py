"""Portable phase validation is independent of renderer and file paths."""
import copy
import json

import numpy as np
import pytest

from quantem.gpu.io.ssb_result import read_result


def test_export_to_selected_catalogue_preserves_prior_results(tmp_path):
    """Microscope screening exports beside processed results, not raw inputs."""
    import h5py
    from quantem.gpu.io.ssb_result import export_result

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
    config = {"computed": {"bf_radius": 1, "bf_center": [1, 1], "ssb": {
        "semiangle_mrad": 20, "scan_sampling_A": 0.5, "voltage_kV": 200,
        "aberrations": {"C10": 0, "C12": 0, "phi12": 0},
        "rotation_angle_deg": 0,
    }}}
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
    record = dict(format="live.ssb", schemaVersion=1, phaseFile="phase.npy",
                  rows=2, columns=2, phaseEncoding="float32-le-row-major", phaseUnits="rad",
                  sourceIdentity="a" * 64, phaseSHA256=hashlib.sha256(phase.tobytes()).hexdigest(),
                  calibration={key: 1.0 for key in ("beamEnergyKeV", "semiangleMrad",
                      "scanStepRowAngstroms", "scanStepColumnAngstroms", "detectorStepRowMrad", "detectorStepColumnMrad", "centerRow", "centerColumn")},
                  c10Nanometers=0, c12Nanometers=0, phi12Radians=0, rotationDegrees=0,
                  provenance={"producer": "test fixture"},
                  runMetadata={"notes": "Exact phase, not a preview"})
    manifest = tmp_path / "sample-01.json"
    manifest.write_text(json.dumps(record))
    _, restored = read_result(manifest, "a" * 64)
    assert restored.tobytes() == phase.tobytes()
    with pytest.raises(ValueError):
        read_result(manifest, "b" * 64)
    for key, value in [("phaseFile", "../phase.npy"), ("phaseFile", "/phase.npy"),
                       ("phaseSHA256", "0" * 64), ("phaseUnits", "unknown"),
                       ("rows", 3), ("rows", 2.0), ("schemaVersion", 900),
                       ("provenance", {}), ("calibration", {})]:
        changed = copy.deepcopy(record)
        changed[key] = value
        manifest.write_text(json.dumps(changed))
        with pytest.raises(ValueError):
            read_result(manifest)
    manifest.write_text(json.dumps(record))
    (tmp_path / "phase.npy").unlink()
    with pytest.raises(FileNotFoundError):
        read_result(manifest)


def test_numbered_exports_respect_selected_output_folder(tmp_path):
    """Repeated screening stays in its chosen output tree and preserves counts."""
    import h5py
    from quantem.gpu.io.ssb_result import export_result

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
    config = {"source_master": str(master), "computed": {
        "bf_center": [3, 4], "bf_radius": 5,
        "ssb": {"voltage_kV": 300, "semiangle_mrad": 30, "scan_sampling_A": .25,
                "rotation_angle_deg": 12, "aberration_unit": "nm",
                "aberrations": {"C10": 7, "C12": 2, "phi12": .1}}}}
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
