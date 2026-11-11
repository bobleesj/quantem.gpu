"""scripts/rewrite_saved_records.py carries records saved under the earlier names to the current layouts once."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "rewrite_saved_records.py"


def rewrite(*arguments):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, arguments)], check=True, capture_output=True,
                          text=True).stdout


def test_a_saved_fit_is_renamed_and_read_again(tmp_path):
    """A /12 fit in ssb-fit/ssb-find_aberrations is the current direct-fit/direct-aberrations, which saved_fit reads."""
    from quantem.gpu.direct_ptycho.persistence import RESULT_SCHEMA
    from quantem.gpu.direct_ptycho.series import saved_fit

    folder = tmp_path / "ssb-fit"
    folder.mkdir()
    aberrations = {"C10_nm": -10.5, "C12_nm": 0.25, "phi12_deg": -0.5}
    earlier = {"schema": "quantem.ssb-result/12",
               "signature": {"schema": "quantem.ssb-result/12", "operation": "find_aberrations",
                             "settings": {"upsampling": 2}},
               "result": {"aberrations": aberrations, "scan_detector_rotation_deg": 210.0, "upsampling": 2,
                          "reconstruction_method": "ssb",
                          "reconstruction_options": {"method": "phase", "phase_damping": 0.3, "amplitude_damping": 0.1},
                          "reconstruction_diagnostics": {"nominal_upsampling_requirement": 3.0}}}
    text = json.dumps(earlier, indent=2) + "\n"
    (folder / "ssb-find_aberrations.json").write_text(text)
    np.savez(folder / "ssb-find_aberrations.npz", object_wave=np.ones((2, 2), np.complex64))
    assert saved_fit(tmp_path) is None
    assert "would rewrite" in rewrite("--dry-run", tmp_path)
    assert sorted(path.name for path in folder.iterdir()) == ["ssb-find_aberrations.json", "ssb-find_aberrations.npz"]
    rewrite(tmp_path)
    assert not folder.exists()
    folder = tmp_path / "direct-fit"
    assert sorted(path.name for path in folder.iterdir()) == [
        "direct-aberrations.json", "direct-aberrations.npz", "ssb-find_aberrations.json.before"]
    assert (folder / "ssb-find_aberrations.json.before").read_text() == text
    record = json.loads((folder / "direct-aberrations.json").read_text())
    assert record["schema"] == record["signature"]["schema"] == RESULT_SCHEMA
    assert record["signature"] == {"schema": RESULT_SCHEMA, "operation": "aberrations", "settings": {"upsample": 2}}
    assert record["result"]["upsample"] == 2 and record["result"]["reconstruction_method"] == "phase_of_mean"
    assert record["result"]["reconstruction_options"] == {"method": "phase", "phase_damp": 0.3, "amplitude_damp": 0.1}
    assert record["result"]["reconstruction_diagnostics"] == {"nominal_upsample_requirement": 3.0}
    assert saved_fit(tmp_path)["aberrations"] == aberrations
    # a second run finds nothing of the earlier schema
    assert rewrite(tmp_path).splitlines()[-1] == "0 file(s) rewritten, 0 skipped."


def test_a_version_13_fit_drops_repeated_fields(tmp_path):
    """A /13 fit loses phase_estimator and num_trials; its signature names the search budget trials."""
    from quantem.gpu.direct_ptycho.persistence import RESULT_SCHEMA
    from quantem.gpu.direct_ptycho.series import saved_fit

    folder = tmp_path / "ssb-fit"
    folder.mkdir()
    aberrations = {"C10_nm": -10.5, "C12_nm": 0.25, "phi12_deg": -0.5}
    earlier = {"schema": "quantem.ssb-result/13",
               "signature": {"schema": "quantem.ssb-result/13", "operation": "aberrations",
                             "settings": {"num_trials": 200, "seed": 7}},
               "result": {"aberrations": aberrations, "scan_detector_rotation_deg": 210.0, "num_trials": 200,
                          "phase_estimator": "phase_of_mean", "reconstruction_method": "phase_of_mean"}}
    text = json.dumps(earlier, indent=2) + "\n"
    (folder / "ssb-aberrations.json").write_text(text)
    np.savez(folder / "ssb-aberrations.npz", object_wave=np.ones((2, 2), np.complex64))
    assert saved_fit(tmp_path) is None
    rewrite(tmp_path)
    folder = tmp_path / "direct-fit"
    assert (folder / "ssb-aberrations.json.before").read_text() == text
    record = json.loads((folder / "direct-aberrations.json").read_text())
    assert record["schema"] == record["signature"]["schema"] == RESULT_SCHEMA
    assert record["signature"]["settings"] == {"trials": 200, "seed": 7}
    assert "num_trials" not in record["result"] and "phase_estimator" not in record["result"]
    assert saved_fit(tmp_path)["aberrations"] == aberrations
    assert rewrite(tmp_path).splitlines()[-1] == "0 file(s) rewritten, 0 skipped."


def test_stream_frames_keep_their_hash_chain(tmp_path):
    """Each /6 frame names the SHA-256 of the rewritten frame before it, as the stream computes it."""
    from quantem.gpu.direct_ptycho.stream import RECORD_SCHEMA

    first = {"schema": "quantem.ssb-stream-frame/4", "frame": "a",
             "identity": {"index": 0, "previous": None, "settings": {"upsampling": 2}}}
    second = {"schema": "quantem.ssb-stream-frame/4", "frame": "b",
              "identity": {"index": 1, "previous": "stale", "settings": {"upsampling": 2}}}
    for name, record in (("a", first), ("b", second)):
        (tmp_path / f"{name}.json").write_text(json.dumps(record))
    rewrite(tmp_path)
    first, second = (json.loads((tmp_path / f"{name}.json").read_text()) for name in ("a", "b"))
    assert first["schema"] == second["schema"] == RECORD_SCHEMA and first["identity"]["settings"] == {"upsample": 2}
    assert second["identity"]["previous"] == hashlib.sha256(
        json.dumps(first, sort_keys=True, allow_nan=False).encode()).hexdigest()


def test_a_phase_manifest_carries_its_screening_record(tmp_path):
    """An ssb.json with ssb.npy is direct.json with direct.npy; its screening record's fit block is
    computed.phase_of_mean and the tilt-corrected one computed.direct_tilt."""
    from quantem.gpu.direct_ptycho.persistence import FIT_SCHEMA, SCREEN_ACQUISITION_SCHEMA
    from quantem.gpu.formats.direct_phase import SCHEMA

    manifest = {"schema": "quantem.ssb-phase/2", "phase_file": "ssb.npy", "phase_sha256": "0" * 64,
                "run_metadata": {"schema": "quantem.screen-acquisition/2",
                                 "computed": {"ssb": {"loss": 0.1, "schema": "quantem.ssb-fit/7"},
                                              "ssb_tilt": {"loss": 0.2}}}}
    (tmp_path / "ssb.json").write_text(json.dumps(manifest))
    np.save(tmp_path / "ssb.npy", np.ones((2, 2), np.float32))
    rewrite(tmp_path)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["direct.json", "direct.npy", "ssb.json.before"]
    record = json.loads((tmp_path / "direct.json").read_text())
    assert record["schema"] == SCHEMA and record["phase_file"] == "direct.npy" and record["phase_sha256"] == "0" * 64
    assert record["run_metadata"] == {"schema": SCREEN_ACQUISITION_SCHEMA,
                                      "computed": {"phase_of_mean": {"loss": 0.1, "schema": FIT_SCHEMA},
                                                   "direct_tilt": {"loss": 0.2}}}
    assert rewrite(tmp_path).splitlines()[-1] == "0 file(s) rewritten, 0 skipped."


def test_a_neural_phase_checkpoint_names_its_factor_upsample(tmp_path):
    import torch

    from quantem.gpu.direct_ptycho.neural_phase import NeuralPhaseModel

    path = tmp_path / "model.pt"
    weights = {"unet": {"down.0.0.weight": torch.ones(2)}, "readout": torch.ones(18), "directions": [{"w": torch.ones(1)}]}
    torch.save({"schema": "quantem.phase-refinement/1", "upsampling": 2, "training_record": {"upsampling": 2},
                "voltage_kV": 300.0, **weights}, path)
    rewrite(path)
    checkpoint = torch.load(path, weights_only=True)
    assert checkpoint["schema"] == NeuralPhaseModel._SCHEMA and checkpoint["upsample"] == 2
    assert "upsampling" not in checkpoint and checkpoint["training_record"] == {"upsample": 2}
    assert torch.equal(checkpoint["readout"], weights["readout"]) and (tmp_path / "model.pt.before").is_file()


def test_a_two_key_calibration_holds_one_reconstruction_dict(tmp_path):
    """A /8 calibration with reconstruction_method beside reconstruction_options holds ShowPtycho's one dict, as /9."""
    from quantem.gpu.direct_ptycho.calibration import CALIBRATION_SCHEMA

    earlier = {"schema": "quantem.ssb-calibration/8", "upsample": 2, "reconstruction_method": "phase",
               "reconstruction_options": {"method": "phase", "phase_damp": 0.3}}
    (tmp_path / "calibration.json").write_text(json.dumps(earlier))
    rewrite(tmp_path)
    record = json.loads((tmp_path / "calibration.json").read_text())
    assert record == {"schema": CALIBRATION_SCHEMA, "upsample": 2,
                      "reconstruction": {"method": "phase", "phase_damp": 0.3}}
    assert rewrite(tmp_path).splitlines()[-1] == "0 file(s) rewritten, 0 skipped."


def test_a_current_ssb_result_takes_the_direct_names(tmp_path):
    """A /14 result in ssb-results is a direct-result/15 in direct-results, its keys and values said direct."""
    from quantem.gpu.direct_ptycho.persistence import RESULT_SCHEMA

    folder = tmp_path / "ssb-results"
    folder.mkdir()
    earlier = {"schema": "quantem.ssb-result/14",
               "signature": {"schema": "quantem.ssb-result/14", "operation": "run", "ssb_source_sha256": "a" * 64},
               "result": {"thickness_method": "ssb_depth", "has_ssb": True, "upsample": 2}}
    (folder / "ssb-run.json").write_text(json.dumps(earlier))
    np.savez(folder / "ssb-run.npz", object_wave=np.ones((2, 2), np.complex64))
    rewrite(tmp_path)
    folder = tmp_path / "direct-results"
    assert sorted(path.name for path in folder.iterdir()) == ["direct-run.json", "direct-run.npz", "ssb-run.json.before"]
    record = json.loads((folder / "direct-run.json").read_text())
    assert record == {"schema": RESULT_SCHEMA,
                      "signature": {"schema": RESULT_SCHEMA, "operation": "run", "direct_source_sha256": "a" * 64},
                      "result": {"thickness_method": "direct_depth", "has_direct": True, "upsample": 2}}
    assert rewrite(tmp_path).splitlines()[-1] == "0 file(s) rewritten, 0 skipped."
