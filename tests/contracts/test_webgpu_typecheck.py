"""Type-check the bundled WebGPU sources with the compiler options quantem.widget uses."""

from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]


def test_bundled_sources_pass_the_widget_type_check():
    """Run ``npm run typecheck`` over every TypeScript file in ``webgpu/sources.json``.

    quantem.widget compiles these files with ``strict``, ``noUnusedLocals`` and
    ``noUnusedParameters``; checking them here with the same options means a source
    that would stop the widget's ``npm run typecheck`` fails in this repository first.
    """
    if shutil.which("node") is None or shutil.which("npm") is None:
        pytest.skip("Node.js is not installed; the type check runs the TypeScript compiler.")
    if not (ROOT / "node_modules" / ".bin" / "tsc").exists():
        pytest.fail("typescript is not installed: run `npm ci` in the repository root.")
    completed = subprocess.run(
        ["npm", "run", "--silent", "typecheck"], cwd=ROOT, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
