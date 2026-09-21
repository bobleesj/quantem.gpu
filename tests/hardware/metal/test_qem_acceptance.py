"""Run native Swift/Metal QEM checks without implying application UI coverage."""

import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def native_gate():
    if os.environ.get("QEM_TEST_BACKEND") != "metal":
        pytest.skip("Select the native Metal acceptance gate explicitly.")


def _run(script, *paths):
    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / script), *map(str, paths)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS" in result.stdout, result.stdout


@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
def test_count_numpy_export_reopen(tmp_path, dtype):
    """Native users save count arrays and recover every DP and detector sum."""
    path = tmp_path / "counts.npy"
    values = np.arange(3 * 5 * 8 * 12).reshape(3, 5, 8, 12).astype(dtype)
    np.save(path, values)
    _run("check_npy_qem_roundtrip.sh", path)


def test_float_qem_bits(tmp_path):
    """Native decoding preserves signed zero, NaN payloads and infinity bits."""
    fixtures = ROOT / "tests/data/qem-v2"
    from quantem.gpu import io

    # The frozen NPY bit oracle remains valid. Its adjacent historical QEM uses
    # a retired codec, so export a current copy without rewriting that fixture.
    saved = tmp_path / "current.qem"
    with io.load(
        fixtures / "float32-special-bits.npy", backend="mps", verbose=False
    ) as loaded:
        io.save(saved, loaded)
    _run(
        "check_qem_float_reference.sh",
        fixtures / "float32-special-bits.npy",
        saved,
    )
