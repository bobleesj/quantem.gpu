"""Run the consumer import contract with the repository's JavaScript development tools."""

from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]


def test_webgpu_entrypoint_identity_and_cleanup(tmp_path):
    if shutil.which("node") is None or shutil.which("npx") is None:
        pytest.skip("Node.js is not installed; the entry-point contract runs the bundled TypeScript in Node.js.")
    dependency = subprocess.run(
        ["node", "-p", "require.resolve('jsfive')"],
        cwd=ROOT, capture_output=True, text=True,
    )
    if dependency.returncode or not (ROOT / "node_modules" / ".bin" / "esbuild").exists():
        pytest.fail("esbuild and jsfive are not installed: run `npm ci` in the repository root.")

    output = tmp_path / "webgpu-entrypoint.cjs"
    subprocess.run(
        [
            "npx", "--no-install", "esbuild",
            "tests/parity/webgpu/webgpu_entrypoint_contract.ts", "--bundle",
            "--platform=node", "--format=cjs", f"--outfile={output}",
        ],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    result = subprocess.run(
        ["node", str(output)],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    assert "entry-point identities and file-registry cleanup passed" in result.stdout
