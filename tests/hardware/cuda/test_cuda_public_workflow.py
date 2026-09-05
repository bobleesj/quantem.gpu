"""Opt-in real-source public Python and HTTP CUDA loading parity.

QGPU_CUDA_WORKFLOW_FIXTURES names a JSON list of independently qualified
sources, source/manifest seals, scan selections, and frozen product SHA-256s.
The fixtures are external; no large or private source files enter the package.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from quantem.gpu import detector, io


def _cases():
    path = os.environ.get("QGPU_CUDA_WORKFLOW_FIXTURES")
    return json.loads(Path(path).read_text()) if path else [None]


def _digest(values, dtype):
    return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes()).hexdigest()


@pytest.mark.parametrize("case", _cases())
def test_real_packed_source_through_public_load_and_http(case, tmp_path):
    if case is None:
        pytest.skip("Set QGPU_CUDA_WORKFLOW_FIXTURES for qualified real-source parity")
    cp = pytest.importorskip("cupy")
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from quantem.gpu.remote import (
        create_app,
        load_compact_browse_sources,
        prepare_browse_source,
    )

    integrity = io.SourceIntegrity.from_file(
        case["integrity_manifest"], expected_sha256=case["integrity_sha256"]
    )
    with io.load(
        case["source"],
        backend="cuda",
        source_integrity=integrity,
        expected_source_sha256=case["source_sha256"],
    ) as loaded:
        source = loaded.data
        assert loaded.representation is io.DataRepresentation.LOSSLESS_PACKED
        assert loaded.shape == tuple(case["shape"])
        assert loaded.dtype == np.dtype("uint16")
        assert loaded.resident_bytes == source.memory_pool_used_bytes
        assert loaded.logical_bytes == np.prod(case["shape"]) * 2
        assert loaded.lossless
        session = detector.prepare(loaded)
        calibration = source.metadata.detector_calibration
        center_row, center_column = calibration["detector_center_px"]
        radius = calibration["bright_field_radius_px"]
        rows, columns = np.ogrid[: loaded.shape[2], : loaded.shape[3]]
        distance = (rows - center_row) ** 2 + (columns - center_column) ** 2
        masks = {
            "BF": distance <= radius**2,
            "ABF": (distance >= (radius / 2) ** 2) & (distance <= radius**2),
            "ADF": (distance >= radius**2) & (distance <= (radius * 2) ** 2),
            "empty": np.zeros(loaded.shape[2:], dtype=bool),
        }
        for name in ["BF", "ABF", "ADF", "ABF", "empty", "BF", "empty", "empty"]:
            values = session.masked_sum_exact(masks[name])
            if name == "empty":
                assert not np.any(values)
            else:
                assert _digest(values, "<u4") == case["products"][name]
        row, column = session.center_of_mass()
        assert _digest(row, "<f4") == case["products"]["CoMy"]
        assert _digest(column, "<f4") == case["products"]["CoMx"]
        scan_row, scan_column = case["selected_scan"]
        diffraction = session.frame(scan_row * loaded.shape[1] + scan_column)
        assert _digest(diffraction, "<u4") == case["products"]["DP"]
        print(
            json.dumps(
                {
                    "shape": loaded.shape,
                    "resident_bytes": loaded.resident_bytes,
                    "logical_bytes": loaded.logical_bytes,
                    "public_products": "pass",
                    "integrity_mode": source.load_metrics.integrity_mode,
                }
            )
        )
    assert source.is_released
    with pytest.raises(RuntimeError, match="released"):
        source.extract_diffraction(0, 0)
    del session
    cp.get_default_memory_pool().free_all_blocks()

    master = Path(case["master"])
    registry = prepare_browse_source(
        master,
        case["source"],
        tmp_path / "deployment",
        expected_source_sha256=case["source_sha256"],
    )
    bindings = load_compact_browse_sources(registry, master.parent)
    app = create_app(master.parent, gpu=0, compact_sources=bindings)
    common = {"session": ".", "file": master.name, "det_bin": 1, "scan_bin": 1}
    with TestClient(app) as client:
        assert (
            client.get("/api/browse/residency", params=common).json()["resident"]
            is False
        )
        for name, inner, outer in [
            ("BF", 0, 1),
            ("ABF", 0.5, 1),
            ("ADF", 1, 2),
            ("CoMy", 0, 1),
            ("CoMx", 0, 1),
        ]:
            result = client.get(
                "/api/browse/realspace",
                params={
                    **common,
                    "mode": name,
                    "inner": inner,
                    "outer": outer,
                },
            )
            assert result.status_code == 200, result.text
            assert hashlib.sha256(result.content).hexdigest() == case["products"][name]
        result = client.get(
            "/api/browse/cbed",
            params={
                **common,
                "sx": case["selected_scan"][0],
                "sy": case["selected_scan"][1],
            },
        )
        assert result.status_code == 200, result.text
        assert hashlib.sha256(result.content).hexdigest() == case["products"]["DP"]
        receipt = client.get("/api/browse/residency", params=common).json()
        assert receipt["resident"] is True and receipt["stale"] is False
        assert receipt["representation"] == "lossless_packed"
        assert receipt["physical_resident_bytes"] > 0
        print(json.dumps({"shape": case["shape"], "http_products": "pass"}))
