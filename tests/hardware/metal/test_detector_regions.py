"""Exact original-file interactions with bounded optional detector-region sums.

The NumPy oracle reads generated counts, not packing descriptors or shader
helpers. No private acquisition or cached virtual image is used as evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin", reason="Requires a physical macOS Metal device"
)


@pytest.fixture(scope="module")
def detector_regions_executable(tmp_path_factory):
    if not shutil.which("swift"):
        pytest.skip("Install the macOS Command Line Tools for Metal parity")
    scratch = tmp_path_factory.mktemp("detector_regions-build")
    result = subprocess.run(
        [
            "swift",
            "build",
            "--disable-sandbox",
            "-c",
            "release",
            "--package-path",
            str(Path(__file__).parent / "swift_detector_regions"),
            "--scratch-path",
            str(scratch),
            "-Xswiftc",
            "-DQGPU_PACKING_DIAGNOSTICS",
        ],
        env={
            **os.environ,
            "QGPU_SOURCE_ROOT": str(Path(__file__).resolve().parents[3]),
        },
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return scratch / "release" / "DetectorRegionsParity"


def _masks(rows, columns):
    np = pytest.importorskip("numpy")
    row, col = np.indices((rows, columns))
    masks = [np.zeros((rows, columns), dtype=np.uint8)]
    # Every block appears independently, including high-bit interior sums.
    for first_row in range(0, rows, 16):
        for first_col in range(0, columns, 16):
            masks.append(
                (
                    (row >= first_row)
                    & (row < first_row + 16)
                    & (col >= first_col)
                    & (col < first_col + 16)
                ).astype(np.uint8)
            )
    for shift in (0, 1, 15, 16, -9, 0):
        radius2 = (row - rows / 2 - shift) ** 2 + (col - columns / 2 + shift) ** 2
        masks.extend(
            [
                (radius2 <= 25**2).astype(np.uint8),
                ((radius2 >= 17**2) & (radius2 <= 40**2)).astype(np.uint8),
            ]
        )
    full = np.ones((rows, columns), dtype=np.uint8)
    hole = full.copy()
    hole[15, 16] = 0
    masks.extend(
        [full, hole, ((row + col) % 2).astype(np.uint8), masks[1], full, masks[0]]
    )
    if rows == columns == 128:
        # Full -> 32 complete blocks requires signed removal through the
        # >=32-entry aggregate kernel. Its complement has the same size, so
        # complement/rebase heuristics cannot hide that delta path.
        checkerboard = ((row // 16 + col // 16) % 2 == 0).astype(np.uint8)
        mixed = 1 - checkerboard
        mixed[0, 16] = 0
        masks.extend([full, checkerboard, mixed])
    return np.stack(masks)


def _acquisition(folder, *, index=0, dtype="uint16", scans=8192, wide=False):
    np = pytest.importorskip("numpy")
    h5py = pytest.importorskip("h5py")
    plugin = pytest.importorskip("hdf5plugin")
    rows, columns = (128, 128) if wide else (64, 128 if dtype == "uint8" else 64)
    assert not wide or dtype == "uint16"
    row, col = np.indices((rows, columns))
    block = (row // 16 * (columns // 16) + col // 16).reshape(-1).astype(np.uint64)
    within = ((row % 16) * 16 + col % 16).reshape(-1).astype(np.uint64)
    masks = _masks(rows, columns)
    (folder / "masks.bin").write_bytes(masks.tobytes())
    hashes = [hashlib.sha256() for _ in masks]
    counts_hash, dpc_hash = hashlib.sha256(), hashlib.sha256()
    files = []
    for part, (start, stop) in enumerate(((0, 204), (204, scans))):
        if stop <= start:
            continue
        source = folder / f"sample-{index}_data_{part + 1:06d}.h5"
        files.append(source)
        with h5py.File(source, "x") as handle:
            data = handle.create_dataset(
                "entry/data/data",
                shape=(stop - start, rows, columns),
                dtype=dtype,
                chunks=(1, rows, columns),
                **plugin.Bitshuffle(nelems=0, cname="lz4"),
            )
            for first in range(start, stop, 32):
                last = min(first + 32, stop)
                scan = np.arange(first, last, dtype=np.uint64)[:, None]
                # Independent target sums span all widths through 24. Counts
                # preserve 65535, including at block and scan-tile boundaries.
                width = (scan // 32 + block[None, :] + index) % (
                    17 if dtype == "uint8" else 25
                )
                target = np.minimum(
                    np.left_shift(np.uint64(1), width) - 1, 256 * np.iinfo(dtype).max
                )
                values = (target // 256 + (within[None, :] < target % 256)).astype(
                    dtype
                )
                if wide:
                    # Every block in the first scan tile has the largest
                    # possible uint16 block sum: 16,776,960 (24 bits). This
                    # proves actual 24-bit data, not merely shader capability.
                    values[scan[:, 0] < 32] = np.iinfo(np.uint16).max
                    if first == 0:
                        block_sums = values.reshape(-1, 8, 16, 8, 16).sum(
                            axis=(2, 4), dtype=np.uint64
                        )
                        assert np.all(block_sums == 256 * 65535)
                data[first - start : last - start] = values.reshape(-1, rows, columns)
                counts_hash.update(values.astype("<u4").tobytes())
                basis = np.zeros((last - first, 4), dtype="<u8")
                basis[:, 0] = values.sum(axis=1, dtype=np.uint64)
                basis[:, 1] = (
                    values.astype(np.uint64) * row.reshape(-1).astype(np.uint64)
                ).sum(axis=1)
                basis[:, 2] = (
                    values.astype(np.uint64) * col.reshape(-1).astype(np.uint64)
                ).sum(axis=1)
                dpc_hash.update(basis.tobytes())
                for mask, digest in zip(masks, hashes):
                    expected = values[:, mask.reshape(-1) != 0].sum(
                        axis=1, dtype=np.uint64
                    )
                    assert np.all(expected <= np.iinfo(np.uint32).max)
                    digest.update(expected.astype("<u4").tobytes())
    master = folder / f"sample-{index}_master.h5"
    with h5py.File(master, "x") as handle:
        group = handle.create_group("entry/data")
        group.attrs["scan_shape"] = [scans // 64, 64]
        for part, source in enumerate(files, start=1):
            group[f"data_{part:06d}"] = h5py.ExternalLink(
                source.name, "/entry/data/data"
            )
        bad = np.zeros((rows, columns), dtype=np.uint32)
        bad[15, 16] = 1
        handle.create_dataset(
            "entry/instrument/detector/detectorSpecific/pixel_mask", data=bad
        )
    (folder / f"oracle-{index}.json").write_text(
        json.dumps(
            {
                "countsSHA256": counts_hash.hexdigest(),
                "detectorSHA256": [value.hexdigest() for value in hashes],
                "dpcSHA256": dpc_hash.hexdigest(),
                "wideBranchMaskSHA256": (
                    {
                        "signedRemoval": hashlib.sha256(
                            masks[-2].tobytes()
                        ).hexdigest(),
                        "mixedBoundary": hashlib.sha256(
                            masks[-1].tobytes()
                        ).hexdigest(),
                    }
                    if wide
                    else {}
                ),
            }
        )
    )
    return [master, *files]


def _run(executable, folder, mode, count=1):
    baseline_modes = {
        "baseline",
        "default",
        "plan-baseline",
        "tight-baseline",
    }
    env = {
        **os.environ,
        "QGPU_ORIGINAL_DETECTOR_REGIONS": "0" if mode in baseline_modes else "1",
        "QGPU_ORIGINAL_PROFILE": "1",
        "QGPU_ORIGINAL_CPU_PLAN": "1",
        "QGPU_ORIGINAL_DIRECT_BITSHUFFLE": "1",
        "QGPU_ORIGINAL_SCALAR_DECODE": "1",
        "COMPACT_UPDATE_HOST_PROFILE": "1",
        "COMPACT_UPDATE_PHASE_PROFILE": "0",
    }
    if mode == "default":
        env.pop("QGPU_ORIGINAL_DETECTOR_REGIONS")
    result = subprocess.run(
        [str(executable), str(folder), mode, str(count)],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "DETECTOR_REGIONS_EXACT_COUNTS_MASKS_DPC_SNAPSHOTS_PASS" in result.stdout
    records = [
        json.loads(line.split(" ", 1)[1])
        for line in result.stderr.splitlines()
        if line.startswith("ORIGINAL_DETECTOR_REGIONS ")
    ]
    stages = [
        json.loads(line)
        for line in result.stderr.splitlines()
        if line.startswith('{"') and '"detector_host_stages"' in line
    ]
    if mode in baseline_modes:
        assert records == []
        assert all(sum(stage["aggregate_entries"]) == 0 for stage in stages)
    elif mode == "budget":
        assert len(records) == count, records
        # Scratch from the finished decode phase is no longer live. Optional
        # sums may fit beneath that same peak, otherwise they must be skipped.
        assert all(
            record["status"] in {"built", "budgetSkipped"} for record in records
        ), records
        assert all(
            (record["resident_bytes"] == 0 and record["verified_values"] == 0)
            if record["status"] == "budgetSkipped"
            else record["resident_bytes"] > 0 and record["verified_values"] > 0
            for record in records
        )
    else:
        assert len(records) >= count and all(
            record["status"] == "built" for record in records
        ), records
        assert all(record["block_size"] == 8 for record in records)
        assert all(
            record["maximum_supported_sum_width"] == 22
            and record["verified_values"] > 0
            for record in records
        )
        assert max(sum(stage["aggregate_entries"]) for stage in stages) > 0
        assert all(stage["submission_count"] == 1 for stage in stages)
        oracle = json.loads((folder / "oracle-0.json").read_text())
        branches = oracle["wideBranchMaskSHA256"]
        if branches:
            signed = [
                stage
                for stage in stages
                if not stage["force_rebase"]
                and stage["mask_sha256"] == branches["signedRemoval"]
            ]
            assert any(
                stage["modes"] == ["delta"]
                and stage["raw_entries"] == [0]
                and stage["aggregate_entries"] == [128]
                for stage in signed
            ), signed
            mixed = [
                stage
                for stage in stages
                if not stage["force_rebase"]
                and stage["mask_sha256"] == branches["mixedBoundary"]
            ]
            assert any(
                stage["modes"] == ["rebase"]
                and stage["raw_entries"] == [63]
                and stage["aggregate_entries"] == [127]
                for stage in mixed
            ), mixed
    if mode in {"plan-baseline", "tight-baseline", "budget"}:
        profiles = [
            json.loads(line.split(" ", 1)[1])
            for line in result.stderr.splitlines()
            if line.startswith("ORIGINAL_PACK_PROFILE ")
        ]
        assert profiles, result.stderr
        assert profiles[-1]["packing_plan_status"] == "hit"
        assert profiles[-1]["direct_bitshuffle_windows"] == 2
        assert profiles[-1]["prepared_dpc_reused"] is True
        assert profiles[-1]["packing_plan_fallbacks"] == 0
    if mode == "cancel":
        assert "DETECTOR_REGIONS_CANCEL_DRAIN_RECOVERY" in result.stdout
        assert "DETECTOR_REGIONS_LATE_SOURCE_MUTATION_REJECTED" in result.stdout
    return result


@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
def test_original_detectors_and_budget_are_exact(
    detector_regions_executable, tmp_path, dtype
):
    """Whole blocks, curved boundaries and DP reads preserve original counts."""
    files = _acquisition(tmp_path, dtype=dtype, wide=dtype == "uint16")
    before = {file: hashlib.sha256(file.read_bytes()).hexdigest() for file in files}
    baseline = _run(detector_regions_executable, tmp_path, "baseline")
    default = _run(detector_regions_executable, tmp_path, "default")

    def load_lines(run):
        return [
            line
            for line in run.stdout.splitlines()
            if line.startswith("DETECTOR_REGIONS_LOAD ")
        ]

    assert load_lines(default) == load_lines(baseline)
    if dtype == "uint16":
        # Prove BOTH processes succeed under the same measured direct-reopen
        # budget. Exact optional sums may reuse memory from finished scratch.
        _run(detector_regions_executable, tmp_path, "plan-baseline")
        baseline = _run(detector_regions_executable, tmp_path, "tight-baseline")
        skipped = _run(detector_regions_executable, tmp_path, "budget")
        baseline_loads = [
            line
            for line in baseline.stdout.splitlines()
            if line.startswith("DETECTOR_REGIONS_LOAD ")
        ]
        skipped_loads = [
            line
            for line in skipped.stdout.splitlines()
            if line.startswith("DETECTOR_REGIONS_LOAD ")
        ]
        for line in baseline_loads + skipped_loads:
            fields = dict(field.split("=") for field in line.split()[1:])
            assert (
                int(fields["retained"])
                <= int(fields["planned"])
                <= int(fields["budget"])
            )
        assert len(baseline_loads) == len(skipped_loads) == 1
        assert (
            baseline_loads[0].split("budget=")[-1]
            == skipped_loads[0].split("budget=")[-1]
        )
    _run(detector_regions_executable, tmp_path, "regions")
    _run(detector_regions_executable, tmp_path, "cancel")
    assert {
        file: hashlib.sha256(file.read_bytes()).hexdigest() for file in files
    } == before


def test_seven_original_residents_share_one_exact_submission(
    detector_regions_executable, tmp_path
):
    """Distinct uint16 tilts retain exact images while all detectors update."""
    for index in range(7):
        _acquisition(tmp_path, index=index, scans=256)
    result = _run(detector_regions_executable, tmp_path, "series", count=7)
    assert "sources=7" in result.stdout
