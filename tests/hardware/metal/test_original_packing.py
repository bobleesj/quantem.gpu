"""Release-mode original Arina HDF5 parity on a real macOS Metal device.

These integration tests generate synthetic acquisitions; no private data is used.
They intentionally run optimized Swift code to catch ARC/staging regressions that
debug-only Metal tests can miss. Python is only a test oracle, not an app dependency.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin", reason="Requires a physical macOS Metal device"
)


@pytest.fixture(autouse=True)
def enable_loading_measurements(monkeypatch):
    """Record bounded loading phases without logging private source metadata."""
    monkeypatch.setenv("QGPU_ORIGINAL_PROFILE", "1")


@pytest.fixture(scope="module")
def original_packing_executable(tmp_path_factory):
    """Build the real public loader with release optimization enabled."""
    if not shutil.which("swift"):
        pytest.skip("Install the macOS Command Line Tools to run Metal parity")
    root = Path(__file__).resolve().parents[3]
    scratch = tmp_path_factory.mktemp("original-packing-build")
    result = subprocess.run(
        [
            "swift",
            "build",
            "-c",
            "release",
            "--package-path",
            str(Path(__file__).parent / "swift_original_packing"),
            "--scratch-path",
            str(scratch),
            "-Xswiftc",
            "-DQGPU_PACKING_DIAGNOSTICS",
        ],
        env={**os.environ, "QGPU_SOURCE_ROOT": str(root)},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return scratch / "release" / "OriginalPackingParity"


def test_production_decoder_alignment_and_malformed_streams(
    original_packing_executable,
):
    """All four shipped decoders preserve bytes and reject invalid LZ4 streams."""
    result = subprocess.run(
        [str(original_packing_executable.with_name("OriginalDecoderParity"))],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PRODUCTION_DECODER_5436_CASES_PASS" in result.stdout


def test_display_range_histogram_without_cpu_roundtrip(original_packing_executable):
    """DP display ranges and linear/log bins match the established GPU path."""
    result = subprocess.run(
        [str(original_packing_executable.with_name("DisplayRangeParity"))],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "DISPLAY_RANGE_HISTOGRAM_EXACT_PASS" in result.stdout


def _layout_headers(plan):
    """Independent bounded byte-LZ4 decoder for test-only plan mutation."""
    manifest = json.loads(plan[48 : 48 + struct.unpack_from("<I", plan, 8)[0]])
    binding = manifest["binding"]
    size = (
        binding["detectorRows"]
        * binding["detectorColumns"]
        * binding["headerWordsPerPixel"]
        * 4
    )
    blocks = (size + 8191) // 8192
    windows = []
    for record in manifest["records"]:
        start = record["offset"]
        compressed = plan[
            start + blocks * 8 : start + blocks * 8 + record["compressedBytes"]
        ]
        decoded = bytearray()
        for block in range(blocks):
            offset, count = struct.unpack_from("<II", plan, start + block * 8)
            source = compressed[offset : offset + count]
            output = bytearray()
            cursor = 0
            while cursor < len(source):
                token = source[cursor]
                cursor += 1
                literal = token >> 4
                if literal == 15:
                    while True:
                        value = source[cursor]
                        cursor += 1
                        literal += value
                        if value != 255:
                            break
                output.extend(source[cursor : cursor + literal])
                cursor += literal
                if cursor == len(source):
                    break
                distance = int.from_bytes(source[cursor : cursor + 2], "little")
                cursor += 2
                assert 0 < distance <= len(output)
                count = (token & 15) + 4
                if token & 15 == 15:
                    while True:
                        value = source[cursor]
                        cursor += 1
                        count += value
                        if value != 255:
                            break
                for _ in range(count):
                    output.append(output[-distance])
            assert len(output) == 8192
            decoded.extend(output)
        windows.append(decoded[:size])
    return manifest, windows


def _rewrite_layout(manifest, windows):
    """Recompute all checksums so faults must reach semantic validation."""
    size = len(windows[0])
    padded = (size + 8191) // 8192 * 8192
    records = []
    body = bytearray()
    for index, headers in enumerate(windows):
        metadata, compressed = bytearray(), bytearray()
        padded_headers = headers + bytes(padded - size)
        for first in range(0, padded, 8192):
            literal = (
                bytes([0xF0])
                + bytes([255]) * 32
                + bytes([17])
                + padded_headers[first : first + 8192]
            )
            metadata.extend(struct.pack("<II", len(compressed), len(literal)))
            compressed.extend(literal)
        words = manifest["records"][index]["payloadWordCount"]
        domain = b"packing-layout-window/v1\0" + struct.pack(
            "<IIII", index, size, padded, words
        )
        records.append(
            {
                "offset": 65584 + len(body),
                "compressedBytes": len(compressed),
                "payloadWordCount": words,
                "checksum": hashlib.sha256(domain + metadata + compressed).hexdigest(),
            }
        )
        body.extend(metadata + compressed)
    manifest = {**manifest, "records": records}
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    assert len(encoded) <= 65536
    return (
        b"QGLAYOU1"
        + struct.pack("<II", len(encoded), 1)
        + hashlib.sha256(encoded).digest()
        + encoded
        + bytes(65536 - len(encoded))
        + body
    )


def _corrupt_second_layout_record(plan, *, malformed_stream):
    """Leave window zero valid so the next-record failure occurs during overlap."""
    damaged = bytearray(plan)
    manifest = json.loads(plan[48 : 48 + struct.unpack_from("<I", plan, 8)[0]])
    binding = manifest["binding"]
    header_bytes = (
        binding["detectorRows"]
        * binding["detectorColumns"]
        * binding["headerWordsPerPixel"]
        * 4
    )
    padded = (header_bytes + 8191) // 8192 * 8192
    metadata_bytes = padded // 8192 * 8
    record = manifest["records"][1]
    offset = record["offset"]
    if malformed_stream:
        # Match distance zero: repaired digests must still reach checked decode.
        compressed = offset + metadata_bytes
        damaged[compressed : compressed + 3] = b"\0\0\0"
        domain = b"packing-layout-window/v1\0" + struct.pack(
            "<IIII", 1, header_bytes, padded, record["payloadWordCount"]
        )
        record["checksum"] = hashlib.sha256(
            domain + damaged[offset : compressed + record["compressedBytes"]]
        ).hexdigest()
        encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        assert len(encoded) <= 65536
        damaged[8:12] = struct.pack("<I", len(encoded))
        damaged[16:48] = hashlib.sha256(encoded).digest()
        damaged[48:65584] = encoded + bytes(65536 - len(encoded))
    else:
        damaged[offset] ^= 1  # normalized metadata offset no longer matches SHA
    return damaged


@pytest.mark.parametrize("dtype", ["uint8", "uint16_low", "uint16_high"])
def test_original_packing_plan_reopen_and_faults(
    original_packing_executable, tmp_path, dtype
):
    """Plan bytes never substitute for fresh full-count, sum or DPC validation."""
    h5py = pytest.importorskip("h5py")
    plugin = pytest.importorskip("hdf5plugin")
    np = pytest.importorskip("numpy")
    scans, rows, cols = 8192, 64, (128 if dtype == "uint8" else 64)
    pixels = rows * cols
    count_hash, dpc_hash = hashlib.sha256(), hashlib.sha256()
    detectors = [hashlib.sha256() for _ in range(3)]
    detector_sum = np.zeros(pixels, np.uint64)
    maximum = 0
    source = tmp_path / "plan_data_000001.h5"
    with h5py.File(source, "x") as handle:
        data = handle.create_dataset(
            "entry/data/data",
            shape=(scans, rows, cols),
            dtype="uint8" if dtype == "uint8" else "uint16",
            chunks=(1, rows, cols),
            **plugin.Bitshuffle(nelems=0, cname="lz4"),
        )
        for first in range(0, scans, 64):
            values = (
                (
                    np.arange(first, first + 64, dtype=np.uint32)[:, None] * 13
                    + np.arange(pixels, dtype=np.uint32)[None, :] * 7
                )
                % 251
            ).astype(data.dtype)
            if dtype == "uint16_high":
                values[:, 17] = 65535
                values[:, 33] = 32768
            data[first : first + 64] = values.reshape(64, rows, cols)
            count_hash.update(values.astype("<u4").tobytes())
            maximum = max(maximum, int(values.max()))
            detector_sum += values.sum(axis=0, dtype=np.uint64)
            for offset, digest in enumerate(detectors):
                digest.update(
                    values[:, (np.arange(pixels) + offset) % 3 == 0]
                    .sum(axis=1, dtype=np.uint64)
                    .astype("<u4")
                    .tobytes()
                )
            basis = np.zeros((64, 4), dtype="<u8")
            basis[:, 0] = values.sum(axis=1, dtype=np.uint64)
            basis[:, 1] = (values * (np.arange(pixels, dtype=np.uint64) // cols)).sum(
                axis=1, dtype=np.uint64
            )
            basis[:, 2] = (values * (np.arange(pixels, dtype=np.uint64) % cols)).sum(
                axis=1, dtype=np.uint64
            )
            dpc_hash.update(basis.tobytes())
    master = tmp_path / "plan_master.h5"
    with h5py.File(master, "x") as handle:
        group = handle.create_group("entry/data")
        group.attrs["scan_shape"] = [128, 64]
        group["data_000001"] = h5py.ExternalLink(source.name, "/entry/data/data")
    oracle = tmp_path / "oracle.json"
    oracle.write_text(
        json.dumps(
            {
                "countsSHA256": count_hash.hexdigest(),
                "detectorSHA256": [value.hexdigest() for value in detectors],
                "dpcSHA256": dpc_hash.hexdigest(),
                "sumSHA256": hashlib.sha256(
                    detector_sum.astype("<u8").tobytes()
                ).hexdigest(),
                "maximum": maximum,
            }
        )
    )
    plan = tmp_path / "layout.qgplan"
    before = {
        path: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
        for path in (source, master)
    }

    def run(mode, selected_plan=plan):
        result = subprocess.run(
            [
                str(original_packing_executable.with_name("PackingPlanParity")),
                str(master),
                str(selected_plan),
                str(oracle),
                mode,
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "PACKING_PLAN_EXACT_ALL_DP_DPC_SUMS_PASS" in result.stdout
        return [
            json.loads(line.split(" ", 1)[1])
            for line in result.stderr.splitlines()
            if line.startswith("ORIGINAL_PACK_PROFILE ")
        ]

    profiles = run("cycle")
    assert profiles[0]["packing_plan_status"] == "stored"
    assert profiles[0]["packing_plan_reused_windows"] == 0
    assert profiles[1]["packing_plan_status"] == "hit"
    assert profiles[1]["packing_plan_reused_windows"] == 2
    assert profiles[1]["source_read_bytes"] == profiles[0]["source_read_bytes"] > 0
    assert profiles[1]["decode_gpu_seconds"] > 0
    if os.environ.get("QGPU_ORIGINAL_CPU_PLAN") == "1":
        assert profiles[1]["packing_plan_decode_cpu_seconds"] > 0
        assert profiles[1]["packing_plan_decode_gpu_seconds"] == 0
    original_plan = plan.read_bytes()
    run("recover")
    manifest, windows = _layout_headers(original_plan)
    # A repaired record checksum must not allow an invalid checkpoint offset.
    struct.pack_into(
        "<I", windows[0], 4, struct.unpack_from("<I", windows[0], 4)[0] + 1
    )
    plan.write_bytes(_rewrite_layout(manifest, windows))
    profiles = run("audit")
    assert profiles[-1]["packing_plan_fallbacks"] == 1
    if dtype == "uint16_low":
        manifest, windows = _layout_headers(original_plan)
        stride, tiles = manifest["binding"]["headerWordsPerPixel"], 128
        for pixel in range(pixels):
            base = pixel * stride * 4
            for checkpoint in range(4):
                struct.pack_into(
                    "<I",
                    windows[0],
                    base + checkpoint * 4,
                    pixel * tiles * 16 if checkpoint == 0 else checkpoint * 32 * 16,
                )
            windows[0][base + 16 : base + stride * 4] = bytes([255]) * (
                (stride - 4) * 4
            )
        manifest["records"][0]["payloadWordCount"] = pixels * tiles * 16
        plan.write_bytes(_rewrite_layout(manifest, windows))
        profiles = run("audit")
        assert profiles[-1]["packing_plan_fallbacks"] == 1
        # Both windows now request oversized sixteen-bit layouts. Admission
        # uses actual staging, so this small fixture no longer exhausts the
        # budget. Count verification must still reject the nonminimal cached
        # layout, repair it, and permit an exact subsequent reopen. The large
        # direct-read fixture below separately proves hard-budget rejection.
        windows[1] = windows[0].copy()
        manifest["records"][1]["payloadWordCount"] = pixels * tiles * 16
        plan.write_bytes(_rewrite_layout(manifest, windows))
        profiles = run("payload-budget")
        assert profiles[-1]["packing_plan_fallbacks"] == 1
        assert profiles[-1]["packing_plan_status"] == "stored"
        profiles = run("audit")
        assert profiles[-1]["packing_plan_status"] == "hit"
        assert profiles[-1]["packing_plan_fallbacks"] == 0
        plan.write_bytes(original_plan)
        profiles = run("reserve")
        assert profiles[-1]["packing_plan_fallbacks"] == 1
        profiles = run("source-change")
        assert profiles[-1]["packing_plan_reused_windows"] == 0
        # Cache failure must not hide a concurrent source change on retry.
        manifest, windows = _layout_headers(plan.read_bytes())
        struct.pack_into(
            "<I", windows[1], 4, struct.unpack_from("<I", windows[1], 4)[0] + 1
        )
        plan.write_bytes(_rewrite_layout(manifest, windows))
        profiles = run("source-change")
        assert profiles[-1]["packing_plan_reused_windows"] == 0
        # Source aliases and unrelated existing files are not disposable plans.
        unrelated = tmp_path / "unrelated-notes.txt"
        unrelated.write_bytes(b"Important unrelated data, not a packing plan.")
        hardlink = tmp_path / "source-hardlink.h5"
        os.link(source, hardlink)
        for destination in (master, source, hardlink, unrelated):
            unchanged = destination.read_bytes()
            run("audit", destination)
            assert destination.read_bytes() == unchanged
    assert before == {
        path: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
        for path in (source, master)
    }


def test_direct_read_ahead_midstream_failure_recovers(
    original_packing_executable, tmp_path
):
    """Reject a changed source or exhausted budget without poisoning reopen."""
    h5py = pytest.importorskip("h5py")
    plugin = pytest.importorskip("hdf5plugin")
    np = pytest.importorskip("numpy")
    data = tmp_path / "regression_data_000001.h5"
    with h5py.File(data, "x") as handle:
        counts = handle.create_dataset(
            "entry/data/data",
            shape=(12288, 64, 64),
            dtype="uint16",
            chunks=(1, 64, 64),
            **plugin.Bitshuffle(nelems=0, cname="lz4"),
        )
        block = np.full((64, 64, 64), 65535, dtype=np.uint16)
        for first in range(0, 12288, 64):
            counts[first : first + 64] = block
    master = tmp_path / "regression_master.h5"
    with h5py.File(master, "x") as handle:
        group = handle.create_group("entry/data")
        group.attrs["scan_shape"] = [192, 64]
        group["data_000001"] = h5py.ExternalLink(data.name, "/entry/data/data")
    before = {
        p: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
        for p in (master, data)
    }
    result = subprocess.run(
        [
            str(original_packing_executable),
            str(master),
            str(tmp_path / "unused.qgix"),
            "direct-regression",
        ],
        env={
            **os.environ,
            "QGPU_ORIGINAL_DIRECT_READ": "1",
            "QGPU_ORIGINAL_READ_AHEAD": "1",
            "QGPU_ORIGINAL_SCALAR_DECODE": "0",
        },
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "DIRECT_MIDSTREAM_BUDGET_SOURCE_MUTATION_AND_RECOVERY_PASS" in result.stdout
    assert "FINAL_PUBLICATION_CANCELLATION_AND_EXACT_RECOVERY_PASS" in result.stdout
    assert before == {
        p: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
        for p in (master, data)
    }
    assert not (tmp_path / "unused.qgix").exists()


@pytest.mark.parametrize(
    "high_counts", [False, True], ids=["uint16-low", "uint16-high"]
)
@pytest.mark.parametrize(
    "simd_gather", [False, True], ids=["scalar-gather", "simd-gather"]
)
def test_direct_bitshuffle_original_reopen(
    original_packing_executable, tmp_path, high_counts, simd_gather
):
    """A split original acquisition reopens exactly without a dense decode window."""
    h5py = pytest.importorskip("h5py")
    plugin = pytest.importorskip("hdf5plugin")
    np = pytest.importorskip("numpy")
    scans, pixels, columns = 8192, 4096, 64
    count_hash, dpc_hash = hashlib.sha256(), hashlib.sha256()
    detector_hashes = [hashlib.sha256() for _ in range(3)]
    detector_sum = np.zeros(pixels, dtype=np.uint64)
    coordinates = np.arange(pixels, dtype=np.uint64)
    maximum = 0
    sources = []
    # The first 4096-frame window crosses an unaligned external-shard boundary.
    for ordinal, (start, stop) in enumerate([(0, 204), (204, scans)], start=1):
        source = tmp_path / f"split_data_{ordinal:06d}.h5"
        sources.append(source)
        with h5py.File(source, "x") as handle:
            data = handle.create_dataset(
                "entry/data/data",
                shape=(stop - start, 64, 64),
                dtype="uint16",
                chunks=(1, 64, 64),
                **plugin.Bitshuffle(nelems=0, cname="lz4"),
            )
            for first in range(start, stop, 32):
                last = min(first + 32, stop)
                frame = np.arange(first, last, dtype=np.uint32)[:, None]
                widths = coordinates % (17 if high_counts else 9)
                values = (
                    (frame * 13 + coordinates[None, :] * 7)
                    & ((1 << widths) - 1)[None, :]
                ).astype(np.uint16)
                values[:, 0] = 0
                if high_counts:
                    values[:, 17] = 65535
                    values[:, 33] = 32768
                    values[:, 49] = 256
                data[first - start : last - start] = values.reshape(
                    last - first, 64, 64
                )
                count_hash.update(values.astype("<u4").tobytes())
                maximum = max(maximum, int(values.max()))
                detector_sum += values.sum(axis=0, dtype=np.uint64)
                for offset, digest in enumerate(detector_hashes):
                    selected = (coordinates + offset) % 3 == 0
                    digest.update(
                        values[:, selected]
                        .sum(axis=1, dtype=np.uint64)
                        .astype("<u4")
                        .tobytes()
                    )
                basis = np.zeros((last - first, 4), dtype="<u8")
                basis[:, 0] = values.sum(axis=1, dtype=np.uint64)
                basis[:, 1] = (values * (coordinates // columns)).sum(
                    axis=1, dtype=np.uint64
                )
                basis[:, 2] = (values * (coordinates % columns)).sum(
                    axis=1, dtype=np.uint64
                )
                dpc_hash.update(basis.tobytes())
    master = tmp_path / "split_master.h5"
    with h5py.File(master, "x") as handle:
        group = handle.create_group("entry/data")
        group.attrs["scan_shape"] = [128, 64]
        for ordinal, source in enumerate(sources, start=1):
            group[f"data_{ordinal:06d}"] = h5py.ExternalLink(
                source.name, "/entry/data/data"
            )
        mask = np.zeros((64, 64), dtype=np.uint32)
        mask.flat[17] = 1
        handle.create_dataset(
            "entry/instrument/detector/detectorSpecific/pixel_mask", data=mask
        )
    oracle = tmp_path / "oracle.json"
    oracle.write_text(
        json.dumps(
            {
                "countsSHA256": count_hash.hexdigest(),
                "detectorSHA256": [digest.hexdigest() for digest in detector_hashes],
                "dpcSHA256": dpc_hash.hexdigest(),
                "sumSHA256": hashlib.sha256(
                    detector_sum.astype("<u8").tobytes()
                ).hexdigest(),
                "maximum": maximum,
            }
        )
    )
    plan = tmp_path / "layout.qgplan"
    before = {
        path: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
        for path in [master, *sources]
    }
    environment = {
        **os.environ,
        "QGPU_ORIGINAL_DIRECT_BITSHUFFLE": "1",
        "QGPU_ORIGINAL_SIMD_GATHER": "1" if simd_gather else "0",
        "QGPU_ORIGINAL_CPU_PLAN": "1",
        "QGPU_ORIGINAL_SCALAR_DECODE": "1",
        "QGPU_ORIGINAL_DIRECT_READ": "1",
        "QGPU_ORIGINAL_READ_AHEAD": "1",
        "QGPU_ORIGINAL_ALIGNED_FILL": "1",
        "QGPU_ORIGINAL_FUSED_PACK": "1",
        "QGPU_ORIGINAL_CHECKPOINT_PACK": "1",
    }

    def run(mode, *, overlap=False):
        # Inject the second window only after the first completes, without an
        # already submitted read-ahead racing that controlled fixture change.
        run_environment = {
            **environment,
            "QGPU_ORIGINAL_PLAN_OVERLAP": "1" if overlap else "0",
        }
        if mode in ("malformed", "overlap-cancel"):
            run_environment["QGPU_ORIGINAL_READ_AHEAD"] = "0"
        result = subprocess.run(
            [
                str(original_packing_executable.with_name("PackingPlanParity")),
                str(master),
                str(plan),
                str(oracle),
                "bitshuffle-" + mode,
            ],
            env=run_environment,
            capture_output=True,
            text=True,
            timeout=240,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "PACKING_PLAN_EXACT_ALL_DP_DPC_SUMS_PASS" in result.stdout
        profiles = [
            json.loads(line.split(" ", 1)[1])
            for line in result.stderr.splitlines()
            if line.startswith("ORIGINAL_PACK_PROFILE ")
        ]
        # Count the actual selected kernel, including any completed direct
        # command whose interval is retained through a metadata-plan fallback.
        for profile in profiles:
            expected = profile["direct_bitshuffle_windows"] if simd_gather else 0
            assert profile["direct_bitshuffle_simd_gather_windows"] == expected, profile
        return profiles, result.stdout

    profiles, _ = run("cycle")
    assert profiles[0]["packing_plan_status"] == "stored"
    assert profiles[0]["direct_bitshuffle_windows"] == 0
    direct = profiles[-1]
    assert direct["packing_plan_status"] == "hit"
    assert direct["prepared_dpc_reused"]
    assert direct["direct_bitshuffle_windows"] == 2
    assert direct["direct_bitshuffle_short_slices"] == 1
    assert direct["direct_bitshuffle_dense_bytes"] == 0
    assert direct["direct_bitshuffle_gpu_seconds"] > 0
    assert direct["source_read_bytes"] == profiles[0]["source_read_bytes"] > 0
    original_plan = plan.read_bytes()
    profiles, _ = run("audit", overlap=True)
    assert profiles[-1]["packing_plan_overlap_windows"] == 1
    assert profiles[-1]["packing_plan_overlap_header_bytes"] == pixels * 20 * 4 + 1
    assert profiles[-1]["direct_bitshuffle_windows"] == 2
    assert profiles[-1]["source_read_bytes"] == direct["source_read_bytes"]
    for malformed_stream in (False, True):
        plan.write_bytes(
            _corrupt_second_layout_record(
                original_plan, malformed_stream=malformed_stream
            )
        )
        profiles, output = run("overlap-fault", overlap=True)
        assert profiles[-1]["packing_plan_fallbacks"] == 1
        assert profiles[-1]["direct_bitshuffle_windows"] == 1
        assert profiles[-1]["direct_bitshuffle_gpu_seconds"] > 0
        assert "PACKING_PLAN_OVERLAP_FALLBACK_COMBINED_INTERVAL_PASS" in output
    plan.write_bytes(original_plan)
    profiles, output = run("overlap-cancel", overlap=True)
    assert "PACKING_PLAN_OVERLAP_PREFETCH_CANCEL_DRAIN_RECOVERY_PASS" in output
    assert sum(value["packing_plan_overlap_windows"] == 1 for value in profiles) == 3
    manifest, headers = _layout_headers(original_plan)
    struct.pack_into(
        "<I", headers[0], 4, struct.unpack_from("<I", headers[0], 4)[0] + 1
    )
    plan.write_bytes(_rewrite_layout(manifest, headers))
    profiles, _ = run("audit")
    assert profiles[-1]["packing_plan_fallbacks"] == 1
    if not high_counts:
        manifest, headers = _layout_headers(original_plan)
        stride = manifest["binding"]["headerWordsPerPixel"]
        for window in headers:
            for pixel in range(pixels):
                base = pixel * stride * 4
                for checkpoint in range(4):
                    struct.pack_into(
                        "<I",
                        window,
                        base + checkpoint * 4,
                        pixel * 128 * 16 if checkpoint == 0 else checkpoint * 32 * 16,
                    )
                window[base + 16 : base + stride * 4] = bytes([255]) * (
                    (stride - 4) * 4
                )
        for record in manifest["records"]:
            record["payloadWordCount"] = pixels * 128 * 16
        plan.write_bytes(_rewrite_layout(manifest, headers))
        profiles, _ = run("audit")
        assert profiles[-1]["packing_plan_fallbacks"] == 1
    plan.write_bytes(original_plan)
    profiles, output = run("recover")
    assert any(value["direct_bitshuffle_windows"] == 2 for value in profiles)
    assert "DIRECT_BITSHUFFLE_CANCEL_SOURCE_FRESHNESS_RECOVERY_PASS" in output
    profiles, output = run("malformed")
    assert profiles[1]["direct_bitshuffle_windows"] == 2
    assert profiles[1]["packing_plan_status"] == "hit"
    assert "DIRECT_BITSHUFFLE_MIDSTREAM_SOURCE_MUTATION_RECOVERY_PASS" in output
    assert before == {
        path: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
        for path in [master, *sources]
    }


@pytest.mark.parametrize(
    "kind",
    [
        "u8",
        "u16_low",
        "u16_high",
        "u16_late_high",
        "u16_multiwindow",
        "zeros",
        "u16_saturated",
        "u16_widths",
    ],
)
@pytest.mark.parametrize("mode", ["file", "direct", "direct-cache"])
def test_original_hdf5_exact_packed_resident(
    kind, mode, original_packing_executable, tmp_path
):
    """Preserve counts across external shards, narrowing, masks and cancellation."""
    h5py = pytest.importorskip("h5py")
    plugin = pytest.importorskip("hdf5plugin")
    np = pytest.importorskip("numpy")
    direct = mode != "file"
    dtype = "uint8" if kind == "u8" else "uint16"
    multiwindow = kind == "u16_multiwindow"
    scan_rows, scan_cols = (192, 64) if multiwindow else (8, 8)
    scans = scan_rows * scan_cols
    detector = 64 if multiwindow else 192
    values = (
        np.arange(scans * detector * detector, dtype=np.uint32).reshape(
            scans, detector, detector
        )
        * 13
        % 251
    ).astype(dtype)
    if kind == "zeros":
        values.fill(0)
    elif kind == "u16_saturated":
        # DPC row/column products exceed uint32; SIMD reductions must retain
        # carry bits even when every source pixel is at the uint16 maximum.
        values.fill(65535)
    elif kind == "u16_widths":
        # Exercise every packing width, including word straddles and 15→16.
        widths = np.arange(detector * detector).reshape(detector, detector) % 17
        values[:] = ((1 << widths) - 1).astype(np.uint16)
    elif kind == "u16_high":
        # Source metadata marks this pixel as bad. Packing must retain it.
        values[0, 0, 0] = 65535
        values[32, 3, 17] = 32768
        values[63, 63, 63] = 256
    elif kind == "u16_late_high":
        values[-1, -1, -1] = 256
    elif multiwindow:
        # Three windows reuse the first buffer slot. A late high count must
        # invalidate narrowing even after earlier windows have been written.
        values[4096, 0, 0] = 65535
        values[-1, -1, -1] = 256
    sources = []
    for index, block in enumerate(np.array_split(values, 2), start=1):
        source = tmp_path / f"sample_data_{index:06d}.h5"
        with h5py.File(source, "x") as handle:
            handle.create_dataset(
                "entry/data/data",
                data=block,
                chunks=(1, detector, detector),
                **plugin.Bitshuffle(nelems=0, cname="lz4"),
            )
        sources.append(source)
    master = tmp_path / "sample_master.h5"
    with h5py.File(master, "x") as handle:
        group = handle.create_group("entry/data")
        group.attrs["scan_shape"] = [scan_rows, scan_cols]
        for index, source in enumerate(sources, start=1):
            group[f"data_{index:06d}"] = h5py.ExternalLink(
                source.name, "/entry/data/data"
            )
        mask = np.zeros((detector, detector), np.uint32)
        mask[0, 0] = 1
        handle.create_dataset(
            "entry/instrument/detector/detectorSpecific/pixel_mask", data=mask
        )
    sources.append(master)
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    oracle = tmp_path / "oracle.u16"
    values.astype("<u2").tofile(oracle)
    result = subprocess.run(
        [
            str(original_packing_executable),
            str(master),
            str(tmp_path / "packed.qgix"),
            str(oracle),
        ]
        + ([mode] if direct else []),
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    profiles = [
        json.loads(line.split(" ", 1)[1])
        for line in result.stderr.splitlines()
        if line.startswith("ORIGINAL_PACK_PROFILE ")
    ]
    if multiwindow and os.environ.get("QGPU_ORIGINAL_SCALAR_DECODE") == "1":
        assert profiles and profiles[0]["scalar_decode_slices"] > 0, result.stderr
    if os.environ.get("QGPU_ORIGINAL_FUSED_PACK") == "1":
        assert profiles and profiles[0]["fused_packing_windows"] > 0, result.stderr
        if os.environ.get("QGPU_ORIGINAL_CHECKPOINT_PACK") == "1":
            assert profiles[0]["checkpoint_packing_windows"] > 0, result.stderr
    if multiwindow and os.environ.get("QGPU_ORIGINAL_FUSED_DPC") == "1":
        assert profiles and profiles[0]["fused_dpc_windows"] > 0, result.stderr
    if mode == "direct-cache":
        assert profiles[1]["prepared_dpc_reused"], result.stderr
        assert profiles[1]["fused_dpc_windows"] == 0, result.stderr
        assert "CACHED_DPC_REUSE_AND_REJECTION_PASS" in result.stdout
        if multiwindow and os.environ.get("QGPU_ORIGINAL_TRANSPOSE_UNSHUFFLE") == "1":
            assert profiles[1]["transpose_unshuffle_slices"] > 0, result.stderr
    if os.environ.get("QGPU_ORIGINAL_PRIVATE_DENSE") == "1":
        assert profiles and profiles[0]["private_dense_window"] == direct, result.stderr
        if mode == "direct-cache":
            assert profiles[1]["private_dense_window"], result.stderr
    if os.environ.get("QGPU_ORIGINAL_FUSE_DECODE_HEADERS") == "1":
        assert profiles and profiles[0]["fused_decode_header_windows"] == 0, (
            result.stderr
        )
        expected_fusion = (
            mode == "direct-cache"
            and os.environ.get("QGPU_ORIGINAL_PROFILE_KERNELS") != "1"
        )
        if mode == "direct-cache":
            assert (
                profiles[1]["fused_decode_header_windows"] > 0
            ) == expected_fusion, result.stderr
        public_metrics = [
            json.loads(line.split(" ", 1)[1])
            for line in result.stdout.splitlines()
            if line.startswith("PUBLIC_LOAD_METRICS ")
        ]
        assert len(public_metrics) == 1, result.stdout
        assert (public_metrics[0]["gpu_decode_and_header_ms"] > 0) == expected_fusion
    if os.environ.get("QGPU_ORIGINAL_READ_AHEAD") == "1":
        expected = direct and os.environ.get("QGPU_ORIGINAL_DIRECT_READ") == "1"
        assert profiles[0]["compressed_read_ahead"] == expected, result.stderr
        if expected:
            assert profiles[0]["maximum_concurrent_compressed_input_bytes"] > 0
            assert profiles[0]["additional_compressed_read_reserve_bytes"] > 0
    assert f"EXACT_PARITY_PASS {values.size}" in result.stdout
    assert {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources} == hashes
    if direct:
        assert "DIRECT_CANCELLATION_BUDGET_AND_NO_CACHE_PASS" in result.stdout
        assert not (tmp_path / "packed.qgix").exists()
        return
    assert "CANCELLATION_AND_IMMUTABILITY_PASS" in result.stdout
    # The same artifact must also satisfy the backend-independent reader.
    from quantem.gpu.io._compact_h5 import CompactH5Index, CompactH5ReferenceDecoder

    index = CompactH5Index.from_file(tmp_path / "packed.qgix")
    reference = CompactH5ReferenceDecoder(index)
    assert index.shape == (scan_rows, scan_cols, detector, detector)
    assert index.manifest["source_dtype"] == dtype
    assert (
        index.manifest["source_raw_logical_sha256"]
        == hashlib.sha256(values.tobytes()).hexdigest()
    )
    for frame in [0, 31, scans // 2, scans - 1]:
        np.testing.assert_array_equal(
            reference.raw_diffraction(frame // scan_cols, frame % scan_cols),
            values[frame],
        )


@pytest.mark.parametrize(
    "scans,rows,columns,cacheable",
    [(96, 960, 960, False), (224, 512, 960, True)],
)
def test_wide_detector_short_scan_reopens_without_partial_packing_tiles(
    original_packing_executable, tmp_path, scans, rows, columns, cacheable
):
    """A wide detector keeps every scan when its decode window must shrink."""
    h5py = pytest.importorskip("h5py")
    plugin = pytest.importorskip("hdf5plugin")
    np = pytest.importorskip("numpy")
    pixels = rows * columns
    coordinates = np.arange(pixels, dtype=np.uint64)
    count_hash, dpc_hash = hashlib.sha256(), hashlib.sha256()
    detector_hashes = [hashlib.sha256() for _ in range(3)]
    detector_sum = np.zeros(pixels, dtype=np.uint64)
    source = tmp_path / "wide_data_000001.h5"
    with h5py.File(source, "x") as handle:
        data = handle.create_dataset(
            "entry/data/data",
            shape=(scans, rows, columns),
            dtype="uint16",
            chunks=(1, rows, columns),
            **plugin.Bitshuffle(nelems=0, cname="lz4"),
        )
        for frame in range(scans):
            values = ((coordinates * 7 + frame * 13) % 251).astype(np.uint16)
            values[17] = 65535
            data[frame] = values.reshape(rows, columns)
            count_hash.update(values.astype("<u4").tobytes())
            detector_sum += values
            for offset, digest in enumerate(detector_hashes):
                selected = (coordinates + offset) % 3 == 0
                digest.update(np.asarray(values[selected].sum(), dtype="<u4").tobytes())
            basis = np.asarray(
                [
                    values.sum(dtype=np.uint64),
                    (values * (coordinates // columns)).sum(dtype=np.uint64),
                    (values * (coordinates % columns)).sum(dtype=np.uint64),
                    0,
                ],
                dtype="<u8",
            )
            dpc_hash.update(basis.tobytes())
    master = tmp_path / "wide_master.h5"
    with h5py.File(master, "x") as handle:
        group = handle.create_group("entry/data")
        group.attrs["scan_shape"] = [scans // 8, 8]
        group["data_000001"] = h5py.ExternalLink(source.name, "/entry/data/data")
    oracle = tmp_path / "oracle.json"
    oracle.write_text(
        json.dumps(
            {
                "countsSHA256": count_hash.hexdigest(),
                "detectorSHA256": [digest.hexdigest() for digest in detector_hashes],
                "dpcSHA256": dpc_hash.hexdigest(),
                "sumSHA256": hashlib.sha256(
                    detector_sum.astype("<u8").tobytes()
                ).hexdigest(),
                "maximum": 65535,
            }
        )
    )
    before = {
        path: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
        for path in (master, source)
    }
    profiles = []
    for _ in range(2):
        result = subprocess.run(
            [
                str(original_packing_executable.with_name("PackingPlanParity")),
                str(master),
                str(tmp_path / "wide.qgplan"),
                str(oracle),
                "audit",
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert result.stdout.count("PACKING_PLAN_EXACT_ALL_DP_DPC_SUMS_PASS") == 1
        profiles.extend(
            json.loads(line.split(" ", 1)[1])
            for line in result.stderr.splitlines()
            if line.startswith("ORIGINAL_PACK_PROFILE ")
        )
    assert len(profiles) == 2
    assert all(profile["decode_window_frames"] == 32 for profile in profiles)
    # Large metadata can exceed the existing 4 MiB layout-cache record limit.
    # That is an uncached reread, not permission to omit scans or change counts.
    assert profiles[1]["packing_plan_reused_windows"] == (
        scans // 32 if cacheable else 0
    )
    assert (tmp_path / "wide.qgplan").exists() is cacheable
    assert before == {
        path: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
        for path in (master, source)
    }


def test_direct_read_ahead_tight_budget_preserves_full_counts(
    original_packing_executable, tmp_path
):
    """Admit reported bounded-window staging, reject a true later-window excess."""
    h5py = pytest.importorskip("h5py")
    plugin = pytest.importorskip("hdf5plugin")
    np = pytest.importorskip("numpy")
    data = tmp_path / "budget_data_000001.h5"
    with h5py.File(data, "x") as handle:
        counts = handle.create_dataset(
            "entry/data/data",
            shape=(12288, 192, 192),
            dtype="uint16",
            chunks=(1, 192, 192),
            **plugin.Bitshuffle(nelems=0, cname="lz4"),
        )
        # A bounded 4.5 MiB fixture buffer, not a dense acquisition-sized copy.
        block = np.full((64, 192, 192), 65535, dtype=np.uint16)
        for first in range(0, 12288, 64):
            counts[first : first + 64] = block
    master = tmp_path / "budget_master.h5"
    with h5py.File(master, "x") as handle:
        group = handle.create_group("entry/data")
        group.attrs["scan_shape"] = [192, 64]
        group["data_000001"] = h5py.ExternalLink(data.name, "/entry/data/data")
    before = {
        path: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
        for path in (master, data)
    }
    result = subprocess.run(
        [
            str(original_packing_executable),
            str(master),
            str(tmp_path / "unused.qgix"),
            "direct-budget",
        ],
        env={
            **os.environ,
            "QGPU_ORIGINAL_DIRECT_READ": "1",
            "QGPU_ORIGINAL_READ_AHEAD": "1",
            "QGPU_ORIGINAL_SCALAR_DECODE": "1",
        },
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "DIRECT_TIGHT_BUDGET_EXACT_RECOVERY_PASS" in result.stdout
    profiles = [
        json.loads(line.split(" ", 1)[1])
        for line in result.stderr.splitlines()
        if line.startswith("ORIGINAL_PACK_PROFILE ")
    ]
    assert len(profiles) == 3
    assert all(profile["compressed_read_ahead"] for profile in profiles)
    assert all(profile["scalar_decode_slices"] == 6 for profile in profiles)
    assert all(profile["packing_plan_status"] == "notRequested" for profile in profiles)
    assert before == {
        path: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
        for path in (master, data)
    }
    assert not (tmp_path / "unused.qgix").exists()
