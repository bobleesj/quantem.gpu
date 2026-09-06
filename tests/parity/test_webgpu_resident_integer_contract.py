"""Host fixture/compilation checks, not WebGPU execution or qualification."""

import json
import subprocess
from pathlib import Path

import numpy as np

from quantem.gpu.io._compact_h5 import CompactH5Index, CompactH5ReferenceDecoder
from tests.contracts.io.test_compact_h5 import _write_fixture
from tests.parity.resident_integer_oracle import _fixture

ROOT = Path(__file__).resolve().parents[2]


def _write_contract_sources(directory: Path) -> tuple[Path, Path]:
    """Prepare raw compact-v1 test files from source values, never expectations."""
    case = _fixture()
    source = directory / "resident-integer-v1.h5"
    _write_fixture(
        source,
        np.asarray(case["raw_frames_u16"], dtype="<u2"),
        shape=tuple(case["shape"]),
        include_calibration=False,
        masked_pixels=tuple(case["excluded_detector_flat_indices"]),
    )
    large = case["large_sum"]
    values = np.full(large["shape"], large["fill"], dtype="<u2")
    for override in large["overrides"]:
        values.reshape(-1)[override["flat_index"]] = override["value"]
    large_source = directory / "resident-integer-large-v1.h5"
    _write_fixture(
        large_source,
        values.reshape(2, -1),
        shape=tuple(large["shape"]),
        include_calibration=False,
        masked_pixels=(),
    )
    return source, large_source


def test_frozen_sources_decode_exactly_before_device_testing(tmp_path: Path) -> None:
    case = _fixture()
    source, large_source = _write_contract_sources(tmp_path)
    decoder = CompactH5ReferenceDecoder(CompactH5Index.from_file(source))
    for scan, expected in enumerate(case["expected_working_frames_u16"]):
        row, column = divmod(scan, case["shape"][1])
        actual = [
            decoder.value(row, column, pixel // 4, pixel % 4) for pixel in range(12)
        ]
        np.testing.assert_array_equal(actual, expected)
    large_decoder = CompactH5ReferenceDecoder(CompactH5Index.from_file(large_source))
    for column, expected in enumerate(case["large_sum"]["expected_full_sum_u64"]):
        actual = sum(
            large_decoder.value(0, column, row, col)
            for row in range(17)
            for col in range(17)
        )
        assert actual == expected


def test_webgpu_adapter_parses_full_uint16_fixture_without_device(
    tmp_path: Path,
) -> None:
    source, large_source = _write_contract_sources(tmp_path)
    bundle = tmp_path / "resident-integer.mjs"
    subprocess.run(
        [
            "npx",
            "--no-install",
            "esbuild",
            str(ROOT / "tests/parity/webgpu/webgpu_resident_integer_contract.ts"),
            "--bundle",
            "--platform=browser",
            "--format=esm",
            f"--outfile={bundle}",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    result = subprocess.run(
        [
            "node",
            "--input-type=module",
            "-e",
            """
import fs from 'node:fs';
const adapter = await import(process.argv[1]);
const source = new File([fs.readFileSync(process.argv[2])], 'integer.h5');
const large = new File([fs.readFileSync(process.argv[3])], 'large.h5');
console.log(JSON.stringify(await adapter.inspectFixtureSources(source, large)));
""",
            bundle.as_uri(),
            str(source),
            str(large_source),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    observed = json.loads(result.stdout)
    assert observed["shape"] == _fixture()["shape"]
    assert observed["largeShape"] == _fixture()["large_sum"]["shape"]
    assert observed["workingDtype"] == "uint16"
    assert "float32" in observed["unsupportedAdapterCases"]["denseExactIntegerProduct"]
