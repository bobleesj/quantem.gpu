"""Retired packed SSB sources explain how to migrate to the canonical loader."""

import hashlib
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest

from quantem.gpu import SSB
from quantem.gpu.io._compact_h5 import prepare_compact_h5_metadata_copy


def test_packed_ssb_requires_reexport_from_original(tmp_path):
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("No CUDA device")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("No CUDA runtime")
    from quantem.gpu.ssb.backends.cuda.backend import CudaSSBBackend

    spec = spec_from_file_location(
        "compact_ssb_fixture", Path(__file__).parents[2] / "contracts/io/test_compact_h5.py"
    )
    fixture = module_from_spec(spec)
    spec.loader.exec_module(fixture)
    values = np.random.default_rng(41).integers(
        1, 240, size=(128, 128, 8, 8), dtype=np.uint16
    )
    values[:, :, 3, 3] = 0  # A zero-count term inside the calibrated disk.
    dense = cp.asarray(values)
    _, _, center = CudaSSBBackend._compute_bf_mask(dense, 0.0, 3)
    source = tmp_path / "packed.h5"
    fixture._write_v3_fixture(
        source, values.reshape(-1, 64), detector_shape=(8, 8), scan_shape=(128, 128)
    )
    prepared = tmp_path / "calibrated.h5"
    prepare_compact_h5_metadata_copy(
        source,
        prepared,
        expected_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        detector_calibration={
            "schema": "quantem.gpu.detector-calibration/v1",
            "detector_center_px": list(center),
            "bright_field_radius_px": 3.0,
            "dpc_rotation_degrees": 0.0,
            "dpc_component_order_exchanged": False,
            "method": "independent-dense-reference",
        },
    )
    options = {
        "backend": "cuda",
        "voltage_kV": 300,
        "semiangle_mrad": 25,
        "scan_sampling_A": 0.5,
        "det_sampling": 1.0,
    }
    with pytest.raises(NotImplementedError, match="Reopen the original acquisition"):
        SSB.open(
            str(prepared),
            expected_source_sha256=hashlib.sha256(prepared.read_bytes()).hexdigest(),
            **options,
        )
