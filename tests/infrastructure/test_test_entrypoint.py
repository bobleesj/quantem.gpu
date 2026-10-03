"""The test runner selects current suites and verifies checkout imports."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
import os
import subprocess
import sys

import quantem

ROOT = Path(__file__).resolve().parents[2]


def test_runner_preserves_current_node_ids_and_pytest_options(monkeypatch):
    spec = spec_from_file_location("qgpu_test_runner", ROOT / "scripts/run_tests.py")
    runner = module_from_spec(spec)
    spec.loader.exec_module(runner)
    calls = []

    def run(command, **options):
        calls.append((command, options))
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(runner.subprocess, "run", run)
    assert runner.main(["tests/contracts/io/test_load.py::test_name", "-q"]) == 7
    assert calls[0][0][-2:] == [
        "tests/contracts/io/test_load.py::test_name", "-q"
    ]
    assert calls[0][1]["cwd"] == ROOT


def test_runner_tests_checkout_with_stale_installed_gpu(tmp_path):
    """Refuse a stale installed GPU copy when running checkout tests."""
    installed = tmp_path / "installed" / "quantem"
    installed.mkdir(parents=True)
    native_package = Path(quantem.__file__).resolve().parent
    for child in native_package.iterdir():
        if child.name not in {"gpu", "__pycache__"}:
            (installed / child.name).symlink_to(child, target_is_directory=child.is_dir())
    (installed / "gpu").mkdir()
    (installed / "gpu" / "__init__.py").write_text(
        "raise RuntimeError('stale installed backend imported')\n"
    )
    check = tmp_path / "test_checkout.py"
    check.write_text(
        "def test_checkout():\n"
        "    from pathlib import Path\n"
        "    import quantem.gpu\n"
        "    from quantem.gpu.io.models import Dataset4dstemGPU\n"
        "    assert quantem.gpu.io.Dataset4dstemGPU is Dataset4dstemGPU\n"
        f"    assert Path(quantem.gpu.__file__).resolve().is_relative_to({str(ROOT / 'src')!r})\n"
    )
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_tests.py"), str(check), "-q"],
        env={**os.environ, "PYTHONPATH": str(installed.parent)},
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
