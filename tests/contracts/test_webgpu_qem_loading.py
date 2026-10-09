"""Run the CPU-only WebGPU .qem loading contracts in Node.js."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("contract", ["qem-http", "qem-stream", "qem-resident", "rans-batch"])
def test_qem_loading_contract(tmp_path: Path, contract: str) -> None:
    """Bundle one tests/webgpu contract with esbuild and run it with node --test."""
    if shutil.which("node") is None or shutil.which("npx") is None:
        pytest.skip("Node.js is not installed; the loading contracts run the bundled TypeScript in Node.js.")
    if not (ROOT / "node_modules" / ".bin" / "esbuild").exists():
        pytest.fail("esbuild is not installed: run `npm ci` in the repository root.")
    bundle = tmp_path / f"{contract}.mjs"
    subprocess.run(
        [
            "npx", "--no-install", "esbuild", f"tests/webgpu/{contract}.ts",
            "--bundle", "--platform=node", "--format=esm", f"--outfile={bundle}",
        ],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    completed = subprocess.run(["node", "--test", str(bundle)], cwd=ROOT, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
