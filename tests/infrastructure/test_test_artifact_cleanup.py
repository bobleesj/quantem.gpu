"""Check real pytest sessions discard generated loading files and exports."""

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import pytest

POLICY = Path(__file__).resolve().parents[1] / "conftest.py"


def _project(root: Path, body: str) -> Path:
    root.mkdir()
    shutil.copy2(POLICY, root / "conftest.py")
    (root / "pytest.ini").write_text(
        "[pytest]\ntmp_path_retention_policy = none\ntmp_path_retention_count = 0\n"
    )
    receipt = root / "paths.json"
    (root / "test_loading.py").write_text(
        "import json, os, tempfile, time\nfrom pathlib import Path\n"
        "def test_export(tmp_path):\n"
        "    index = Path(tempfile.mkdtemp()) / 'loading.qh5idx'\n"
        "    index.write_bytes(b'index')\n"
        "    export = tmp_path / 'saved.qem'\n"
        "    export.write_bytes(b'export')\n"
        f"    Path({str(receipt)!r}).write_text(json.dumps([str(index), str(export)]))\n"
        f"    {body}\n"
    )
    return receipt


@pytest.mark.parametrize("exit_code", [0, 1])
def test_success_and_failure_discard_exports_and_loading_indexes(tmp_path, exit_code):
    project = tmp_path / "project"
    receipt = _project(project, f"assert {exit_code} == 0")
    process = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"], cwd=project,
        env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
        capture_output=True, text=True, timeout=20,
    )
    assert process.returncode == exit_code, process.stdout + process.stderr
    assert all(not Path(path).exists() for path in json.loads(receipt.read_text()))


@pytest.mark.skipif(os.name == "nt", reason="Windows terminate is not a catchable SIGTERM")
def test_cancellation_discards_loading_files(tmp_path):
    project = tmp_path / "project"
    receipt = _project(project, "time.sleep(60)")
    process = subprocess.Popen(
        [sys.executable, "-m", "pytest", "-q"], cwd=project,
        env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        deadline = time.monotonic() + 10
        while not receipt.exists():
            assert process.poll() is None
            assert time.monotonic() < deadline
            time.sleep(0.02)
        process.send_signal(signal.SIGTERM)
        output, errors = process.communicate(timeout=10)
        assert process.returncode == 2, output + errors
        assert all(not Path(path).exists() for path in json.loads(receipt.read_text()))
    finally:
        if process.poll() is None:
            process.terminate()
            process.communicate(timeout=10)
