from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_webgpu_logical_pixel_hash_matches_node_crypto_oracle() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the executable WebGPU hash contract")

    version = subprocess.run(
        [node, "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    major = int(version.removeprefix("v").split(".", maxsplit=1)[0])
    if major < 22:
        pytest.skip("Node.js 22+ is required for direct erasable-TypeScript execution")

    completed = subprocess.run(
        [
            node,
            "--no-warnings",
            "--experimental-strip-types",
            str(ROOT / "tests" / "webgpu_logical_pixel_hash_contract.ts"),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)

    assert result["status"] == "passed"
    assert result["trustedOracle"] == "node:crypto SHA-256"
    assert result["knownVectors"] == 4
    assert result["correctionPixelWidths"] == [1, 2, 4]
