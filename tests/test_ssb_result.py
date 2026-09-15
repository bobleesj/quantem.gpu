"""Portable phase validation is independent of renderer and file paths."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from quantem.gpu.io.ssb_result import read_result


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
