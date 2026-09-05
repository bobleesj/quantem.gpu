from __future__ import annotations

import hashlib
import json
import struct
import subprocess
import zlib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/quantem/gpu/io/backends/webgpu/compact-h5.ts"


@pytest.fixture(scope="session")
def compact_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the browser module once for Node-hosted metadata tests."""
    output = tmp_path_factory.mktemp("webgpu-v3") / "compact-h5.mjs"
    subprocess.run(
        [
            "npx",
            "--no-install",
            "esbuild",
            str(SOURCE),
            "--bundle",
            "--platform=browser",
            "--format=esm",
            f"--outfile={output}",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return output


def _write_v3(
    path: Path,
    *,
    masked_pixels: tuple[int, ...] = (2,),
    masked_pixel_sha256: str | None = None,
    raw_values: list[object] | None = None,
    include_pixel_sha256: bool = True,
    include_raw_values: bool = True,
    reserved: int = 0,
    scan_tile: int = 32,
    header_encoding: int = 1,
    include_json_shards: bool = False,
    include_prepared_dpc_moments: bool = False,
    prepared_dpc_overrides: dict[str, object] | None = None,
) -> str:
    scans = np.arange(64, dtype="<u2")
    values = np.stack(
        [
            scans % 2,
            (scans * 13 + 17) % 256,
            np.full_like(scans, 65535),
            (scans * 3) % 127,
            np.zeros_like(scans),
            (scans * 5 + 11) % 64,
        ],
        axis=1,
    ).astype("<u2", copy=False)
    working = values.copy()
    for pixel in masked_pixels:
        working[:, pixel] = 0
    tile_count = 2
    detector_pixels = values.shape[1]
    widths = np.zeros((detector_pixels, tile_count), dtype=np.uint8)
    payload_words: list[int] = []
    pixel_bases: list[int] = []
    for pixel in range(detector_pixels):
        pixel_bases.append(len(payload_words))
        for tile in range(tile_count):
            column = working[tile * 32 : (tile + 1) * 32, pixel]
            width = int(np.bitwise_or.reduce(column)).bit_length()
            widths[pixel, tile] = width
            words = [0] * width
            for scan, item in enumerate(column):
                if width == 0:
                    break
                bit = scan * width
                word, shift = divmod(bit, 32)
                words[word] |= int(item) << shift
                if shift + width > 32:
                    words[word + 1] |= int(item) >> (32 - shift)
            payload_words.extend(word & 0xFFFFFFFF for word in words)
    payload = struct.pack(f"<{len(payload_words)}I", *payload_words)
    headers = np.zeros((detector_pixels, 2), dtype="<u4")
    headers[:, 0] = pixel_bases
    headers[:, 1] = widths[:, 0].astype(np.uint32) | (
        widths[:, 1].astype(np.uint32) << 4
    )
    header_payload = headers.tobytes()
    shape = (1, 64, 2, 3)
    source_identity = hashlib.sha256(values.tobytes()).digest()
    mask_bytes = struct.pack(f"<{len(masked_pixels)}I", *masked_pixels)
    payload_offset = 65536
    headers_offset = payload_offset + len(payload)
    prepared_moments = np.zeros((shape[0] * shape[1], 8), dtype="<u4")
    working_u64 = working.astype(np.uint64)
    detector_rows = np.arange(shape[2], dtype=np.uint64).repeat(shape[3])
    detector_columns = np.tile(np.arange(shape[3], dtype=np.uint64), shape[2])
    totals = working_u64.sum(axis=1, dtype=np.uint64)
    row_moments = (working_u64 * detector_rows).sum(axis=1, dtype=np.uint64)
    column_moments = (working_u64 * detector_columns).sum(axis=1, dtype=np.uint64)
    for pair, values_u64 in enumerate((totals, row_moments, column_moments)):
        prepared_moments[:, pair * 2] = (values_u64 & np.uint64(0xFFFFFFFF)).astype(np.uint32)
        prepared_moments[:, pair * 2 + 1] = (values_u64 >> np.uint64(32)).astype(np.uint32)
    prepared_bytes = prepared_moments.tobytes()
    prepared_offset = headers_offset + len(header_payload)
    manifest: dict[str, object] = {
        "schema": "quantem.gpu.packed-detector-h5/v3",
        "status": "complete",
        "payload_codec": "direct-bitpacked-u32",
        "source_identity_sha256": source_identity.hex(),
        "source_raw_logical_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        "source_shape": list(shape),
        "source_dtype": "uint16",
        "working_dtype": "uint8",
        "working_value_definition": (
            "all admitted source counts exactly; authenticated dead pixels set to zero"
        ),
        "prepared_uint8_sha256": hashlib.sha256(
            working.astype(np.uint8).tobytes()
        ).hexdigest(),
        "detector_mask_sha256": hashlib.sha256(mask_bytes).hexdigest(),
        "masked_detector_pixels": list(masked_pixels),
        "scan_bin": 1,
        "detector_bin": 1,
        "crop": None,
        "scan_tile": 32,
        "shard_count": 1,
    }
    if include_json_shards:
        manifest["shards"] = []
    if include_pixel_sha256:
        manifest["masked_detector_pixels_sha256"] = (
            masked_pixel_sha256
            if masked_pixel_sha256 is not None
            else hashlib.sha256(mask_bytes).hexdigest()
        )
    if include_raw_values:
        manifest["masked_detector_raw_values"] = (
            raw_values
            if raw_values is not None
            else [int(values[0, pixel]) for pixel in masked_pixels]
        )
    if include_prepared_dpc_moments:
        selected = detector_pixels - len(masked_pixels)
        included = [pixel for pixel in range(detector_pixels) if pixel not in masked_pixels]
        total_bound = selected * 255
        row_bound = 255 * sum(pixel // shape[3] for pixel in included)
        column_bound = 255 * sum(pixel % shape[3] for pixel in included)
        prepared_manifest: dict[str, object] = {
            "schema": "quantem.gpu.prepared-dpc-moments/v1",
            "source_identity_sha256": source_identity.hex(),
            "working_uint8_sha256": manifest["prepared_uint8_sha256"],
            "detector_mask_sha256": manifest["detector_mask_sha256"],
            "detector_selection": "all-nonexcluded-v1",
            "scan_count": shape[0] * shape[1],
            "selected_detector_pixels": selected,
            "detector_columns": shape[3],
            "dtype": "little-endian-u32",
            "word_order": "little-endian-u32-pairs",
            "words_per_scan": 8,
            "layout": [
                "total_lo", "total_hi", "row_lo", "row_hi",
                "column_lo", "column_hi", "padding_0", "padding_1",
            ],
            "file_offset": prepared_offset,
            "file_bytes": len(prepared_bytes),
            "sha256": hashlib.sha256(prepared_bytes).hexdigest(),
            "total_bound": str(total_bound),
            "row_moment_bound": str(row_bound),
            "column_moment_bound": str(column_bound),
            "narrow_integer": total_bound <= np.iinfo(np.uint32).max,
            "narrow_products": max(row_bound, column_bound)
            <= np.iinfo(np.uint32).max,
        }
        prepared_manifest.update(prepared_dpc_overrides or {})
        manifest["prepared_dpc_moments"] = prepared_manifest
    header = json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode()
    binary = bytearray(
        struct.pack(
            "<8sIIIIIIIII",
            b"QGIX\0\0\0\3",
            1,
            reserved,
            *shape,
            64,
            scan_tile,
            header_encoding,
        )
    )
    binary.extend(struct.pack("<I", len(masked_pixels)))
    binary.extend(mask_bytes)
    binary.extend(source_identity)
    binary.extend(
        struct.pack(
            "<QQQQQQQII32s",
            payload_offset,
            len(payload),
            0,
            0,
            headers_offset,
            len(header_payload),
            len(payload),
            headers.size,
            0,
            hashlib.sha256(payload).digest(),
        )
    )
    binary_offset = (24 + len(header) + 7) & ~7
    user_block = bytearray(65536)
    user_block[:24] = struct.pack(
        "<8sIIII",
        b"QGPUH5\0\1",
        len(header),
        zlib.crc32(header),
        binary_offset,
        len(binary),
    )
    user_block[24 : 24 + len(header)] = header
    user_block[binary_offset : binary_offset + len(binary)] = binary
    path.write_bytes(
        user_block
        + payload
        + header_payload
        + (prepared_bytes if include_prepared_dpc_moments else b"")
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse(bundle: Path, source: Path) -> dict[str, object]:
    script = """
import fs from 'node:fs';
import path from 'node:path';
const module = await import(process.argv[1]);
const bytes = fs.readFileSync(process.argv[2]);
const file = new File([bytes], path.basename(process.argv[2]));
try {
  const index = await module.parseCompactH5Index(file);
  console.log(JSON.stringify({ok:true,schemaVersion:index.schemaVersion,scanTile:index.scanTile,
    headerEncoding:index.headerEncoding,payloadCodec:index.payloadCodec,
    rawReconstructionAvailable:index.rawReconstructionAvailable,
    maskedDetectorPixelsSha256:index.maskedDetectorPixelsSha256,
    maskedDetectorRawValues:index.maskedDetectorRawValues ? [...index.maskedDetectorRawValues] : null,
    residentBytes:index.residentBytes,shape:index.shape,
    preparedDpcMoments:index.preparedDpcMoments ? {
      fileOffset:index.preparedDpcMoments.fileOffset,
      fileBytes:index.preparedDpcMoments.fileBytes,
      sha256:index.preparedDpcMoments.sha256,
      selectedDetectorPixels:index.preparedDpcMoments.selectedDetectorPixels,
    } : undefined}));
} catch (error) {
  console.log(JSON.stringify({ok:false,error:String(error)}));
}
"""
    completed = subprocess.run(
        ["node", "--input-type=module", "-e", script, bundle.as_uri(), str(source)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_session_qualification_cache_is_bound_to_one_immutable_file_object(
    compact_bundle: Path,
) -> None:
    """Equal-looking files cannot inherit another File object's proof."""
    script = """
const module = await import(process.argv[1]);
const cache = new module.CompactH5SessionQualificationCache();
const seal = 'a'.repeat(64);
const first = new File(['same bytes'], 'same.h5');
const second = new File(['same bytes'], 'same.h5');
const remote = {size:first.size,name:first.name,readRange:async () => new Uint8Array()};
cache.record(first, seal);
cache.record(remote, seal);
console.log(JSON.stringify({
  first:cache.has(first, seal),
  wrongSeal:cache.has(first, 'b'.repeat(64)),
  second:cache.has(second, seal),
  remote:cache.has(remote, seal),
}));
"""
    completed = subprocess.run(
        ["node", "--input-type=module", "-e", script, compact_bundle.as_uri()],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == {
        "first": True,
        "wrongSeal": False,
        "second": False,
        "remote": False,
    }


def test_webgpu_v3_parser_accepts_enriched_direct_contract(
    compact_bundle: Path, tmp_path: Path
) -> None:
    source = tmp_path / "enriched-v3.h5"
    _write_v3(source)

    result = _parse(compact_bundle, source)

    assert result == {
        "ok": True,
        "schemaVersion": 3,
        "scanTile": 32,
        "headerEncoding": 1,
        "payloadCodec": "direct-bitpacked-u32",
        "rawReconstructionAvailable": True,
        "maskedDetectorPixelsSha256": hashlib.sha256(
            struct.pack("<I", 2)
        ).hexdigest(),
        "maskedDetectorRawValues": [65535],
        "residentBytes": 224,
        "shape": [1, 64, 2, 3],
    }


def test_webgpu_v3_parser_accepts_source_bound_prepared_dpc_moments(
    compact_bundle: Path, tmp_path: Path
) -> None:
    baseline = tmp_path / "baseline-v3.h5"
    prepared = tmp_path / "prepared-dpc-v3.h5"
    _write_v3(baseline)
    _write_v3(prepared, include_prepared_dpc_moments=True)

    baseline_result = _parse(compact_bundle, baseline)
    prepared_result = _parse(compact_bundle, prepared)

    assert baseline_result["ok"] is True
    assert prepared_result["ok"] is True
    assert prepared_result["residentBytes"] == baseline_result["residentBytes"] + 64 * 8 * 4
    assert prepared_result["preparedDpcMoments"] == {
        "fileOffset": prepared.stat().st_size - 64 * 8 * 4,
        "fileBytes": 64 * 8 * 4,
        "sha256": hashlib.sha256(
            prepared.read_bytes()[-64 * 8 * 4 :]
        ).hexdigest(),
        "selectedDetectorPixels": 5,
    }


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"selected_detector_pixels": 4}, "selected_detector_pixels"),
        ({"working_uint8_sha256": "0" * 64}, "working_uint8_sha256"),
        ({"detector_mask_sha256": "0" * 64}, "detector_mask_sha256"),
        ({"file_bytes": 4}, "byte range"),
        ({"file_offset": 65536}, "overlaps shard 0 payload"),
        ({"sha256": "not-a-digest"}, "SHA-256"),
        ({"layout": ["total_lo"]}, "word layout"),
    ],
)
def test_webgpu_v3_parser_rejects_mismatched_prepared_dpc_moments(
    compact_bundle: Path,
    tmp_path: Path,
    overrides: dict[str, object],
    message: str,
) -> None:
    source = tmp_path / "mismatched-prepared-dpc-v3.h5"
    _write_v3(
        source,
        include_prepared_dpc_moments=True,
        prepared_dpc_overrides=overrides,
    )

    result = _parse(compact_bundle, source)

    assert result["ok"] is False
    assert message in str(result["error"])


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"reserved": 1}, "reserved=0"),
        ({"scan_tile": 16}, "scan_tile=32"),
        ({"header_encoding": 2}, "header encoding 1"),
        ({"masked_pixels": (2, 1), "raw_values": [65535, 17]}, "ordered row-major"),
        ({"masked_pixel_sha256": "0" * 64}, "ordered binary-index"),
        ({"raw_values": []}, "aligned one-to-one"),
        ({"raw_values": [65536]}, "exact uint16"),
        ({"raw_values": [True]}, "exact uint16"),
        ({"include_json_shards": True}, "unsupported JSON shard table"),
    ],
)
def test_webgpu_v3_parser_rejects_malformed_contract(
    compact_bundle: Path,
    tmp_path: Path,
    options: dict[str, object],
    message: str,
) -> None:
    source = tmp_path / "malformed-v3.h5"
    _write_v3(source, **options)

    result = _parse(compact_bundle, source)

    assert result["ok"] is False
    assert message in str(result["error"])


@pytest.mark.parametrize(
    ("include_pixel_sha256", "include_raw_values"),
    [(False, False), (True, False), (False, True)],
)
def test_webgpu_v3_legacy_files_remain_mask_applied_only(
    compact_bundle: Path,
    tmp_path: Path,
    include_pixel_sha256: bool,
    include_raw_values: bool,
) -> None:
    source = tmp_path / "legacy-v3.h5"
    _write_v3(
        source,
        include_pixel_sha256=include_pixel_sha256,
        include_raw_values=include_raw_values,
    )

    result = _parse(compact_bundle, source)

    assert result["ok"] is True
    assert result["rawReconstructionAvailable"] is False
