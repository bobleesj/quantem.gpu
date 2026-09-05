"""Run the consumer import contract when the browser build tools are installed."""

from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]


def test_webgpu_entrypoint_identity_and_cleanup(tmp_path):
    if shutil.which("node") is None or shutil.which("npx") is None:
        pytest.skip("Install Node.js and esbuild to run the WebGPU entry-point contract")
    dependency = subprocess.run(
        ["node", "-p", "require.resolve('jsfive')"],
        cwd=ROOT, capture_output=True, text=True,
    )
    if dependency.returncode:
        pytest.skip("Install jsfive or set NODE_PATH to the consumer dependencies")

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
