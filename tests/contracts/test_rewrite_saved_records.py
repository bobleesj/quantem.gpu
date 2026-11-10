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
    """A /12 fit in ssb-fit/ssb-find_aberrations is /13 ssb-aberrations afterwards, which saved_fit reads."""
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
    assert sorted(path.name for path in folder.iterdir()) == [
        "ssb-aberrations.json", "ssb-aberrations.npz", "ssb-find_aberrations.json.before"]
    assert (folder / "ssb-find_aberrations.json.before").read_text() == text
    record = json.loads((folder / "ssb-aberrations.json").read_text())
    assert record["schema"] == record["signature"]["schema"] == RESULT_SCHEMA
    assert record["signature"] == {"schema": RESULT_SCHEMA, "operation": "aberrations", "settings": {"upsample": 2}}
    assert record["result"]["upsample"] == 2 and record["result"]["reconstruction_method"] == "phase_of_mean"
    assert record["result"]["reconstruction_options"] == {"method": "phase", "phase_damp": 0.3, "amplitude_damp": 0.1}
    assert record["result"]["reconstruction_diagnostics"] == {"nominal_upsample_requirement": 3.0}
    assert saved_fit(tmp_path)["aberrations"] == aberrations
    # a second run finds nothing of the earlier schema
    assert rewrite(tmp_path).splitlines()[-1] == "0 file(s) rewritten, 0 skipped."


def test_stream_frames_keep_their_hash_chain(tmp_path):
    """Each /5 frame names the SHA-256 of the rewritten frame before it, as the stream computes it."""
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
    """The screening record inside an ssb.json moves its fit block to computed.phase_of_mean."""
    from quantem.gpu.direct_ptycho.persistence import SCREEN_ACQUISITION_SCHEMA
    from quantem.gpu.formats.ssb_phase import SCHEMA

    manifest = {"schema": SCHEMA, "phase_sha256": "0" * 64,
                "run_metadata": {"schema": "quantem.screen-acquisition/2",
                                 "computed": {"ssb": {"loss": 0.1}, "ssb_tilt": {"loss": 0.2}}}}
    (tmp_path / "ssb.json").write_text(json.dumps(manifest))
    rewrite(tmp_path)
    record = json.loads((tmp_path / "ssb.json").read_text())
    assert record["schema"] == SCHEMA and record["phase_sha256"] == "0" * 64
    assert record["run_metadata"] == {"schema": SCREEN_ACQUISITION_SCHEMA,
                                      "computed": {"phase_of_mean": {"loss": 0.1}, "ssb_tilt": {"loss": 0.2}}}


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
