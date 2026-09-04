#!/usr/bin/env python3
"""Validate the exact Apple resident consumer contract on real fixtures.

Each fixture argument is a JSON file that binds a prepared QGIX v3 source to
an independently sealed exact-product manifest. Parity readback is deliberately
kept outside the measured resident interaction boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import resource
import subprocess
import threading
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Self

import numpy as np

from quantem.gpu.io.backends.mps.compact_v3 import load_compact_v3_mps
from quantem.gpu.io.backends.mps.consumer import (
    MPSPublicationCounters,
    MPSPublicationMilestone,
    MPSPublicationRecorder,
    MPSResidentRepresentation,
)
from quantem.gpu.io.backends.mps.resident_dpc import (
    MPSDPCConfiguration,
    MPSDPCProcessor,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixture",
        action="append",
        required=True,
        type=Path,
        help="JSON fixture binding. Pass exactly two for the A-B-A gate.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reopen-repeats", type=int, default=5)
    options = parser.parse_args()
    if len(options.fixture) != 2:
        parser.error("pass exactly two --fixture bindings")
    if options.reopen_repeats < 1:
        parser.error("--reopen-repeats must be positive")
    return options


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(values: np.ndarray, dtype: str) -> str:
    return hashlib.sha256(values.astype(dtype, copy=False).tobytes()).hexdigest()


def _command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            command, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _distribution(samples: list[float]) -> dict[str, float | int]:
    ordered = sorted(samples)

    def percentile(probability: float) -> float:
        rank = max(1, math.ceil(probability * len(ordered)))
        return ordered[min(rank - 1, len(ordered) - 1)]

    return {
        "n": len(ordered),
        "p50_ms": percentile(0.50),
        "p95_ms": percentile(0.95),
        "max_ms": ordered[-1],
    }


def _rss_bytes() -> int:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except ImportError:
        return 0


class _PeakSampler:
    def __init__(self, device) -> None:
        self._device = device
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self.peak_rss_bytes = _rss_bytes()
        self.peak_device_allocated_bytes = int(device.currentAllocatedSize())

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(self, *unused: object) -> None:
        self._stop.set()
        self._thread.join()
        self._capture()

    def _capture(self) -> None:
        self.peak_rss_bytes = max(self.peak_rss_bytes, _rss_bytes())
        self.peak_device_allocated_bytes = max(
            self.peak_device_allocated_bytes,
            int(self._device.currentAllocatedSize()),
        )

    def _sample(self) -> None:
        while not self._stop.wait(0.01):
            self._capture()


def _load_binding(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    value = json.loads(raw)
    required = {
        "label",
        "prepared_source",
        "exact_manifest",
        "prepared_product_receipt",
        "original_master",
        "original_master_sha256",
    }
    if not isinstance(value, dict) or not required <= value.keys():
        raise ValueError(f"fixture binding is incomplete: {path}")
    value["binding_path"] = str(path.resolve())
    value["binding_sha256"] = hashlib.sha256(raw).hexdigest()
    return value


def _artifact(root: dict[str, Any], name: str, manifest_path: Path) -> Path:
    relative = root["exact_product_cache"]["artifacts"][name]["path"]
    return manifest_path.parent / relative


def _exact_comparison(actual: np.ndarray, expected_path: Path, dtype: str) -> dict:
    expected = np.fromfile(expected_path, dtype=dtype)
    if actual.size != expected.size:
        raise ValueError(
            f"oracle size mismatch for {expected_path}: {actual.size} != {expected.size}"
        )
    normalized = actual.astype(dtype, copy=False).reshape(-1)
    mismatch_count = int(np.count_nonzero(normalized != expected))
    return {
        "status": "pass" if mismatch_count == 0 else "fail",
        "compared_values": int(expected.size),
        "mismatch_count": mismatch_count,
        "actual_sha256": hashlib.sha256(normalized.tobytes()).hexdigest(),
        "oracle_path": str(expected_path),
        "oracle_sha256": _sha256_file(expected_path),
    }


def _float_comparison(
    actual: np.ndarray,
    expected_path: Path,
    *,
    atol: float,
) -> dict:
    expected = np.fromfile(expected_path, dtype="<f4")
    normalized = actual.astype("<f4", copy=False).reshape(-1)
    if normalized.size != expected.size:
        raise ValueError(
            f"oracle size mismatch for {expected_path}: "
            f"{normalized.size} != {expected.size}"
        )
    delta = np.abs(normalized.astype(np.float64) - expected.astype(np.float64))
    violations = int(np.count_nonzero(delta > atol))
    return {
        "status": "pass" if violations == 0 else "fail",
        "compared_values": int(expected.size),
        "atol": atol,
        "rtol": 0.0,
        "violation_count": violations,
        "maximum_absolute_error": float(delta.max(initial=0.0)),
        "rms_error": float(np.sqrt(np.mean(np.square(delta), dtype=np.float64))),
        "actual_sha256": hashlib.sha256(normalized.tobytes()).hexdigest(),
        "oracle_path": str(expected_path),
        "oracle_sha256": _sha256_file(expected_path),
    }


def _validate_binding(binding: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    compact = Path(binding["prepared_source"]).resolve(strict=True)
    manifest_path = Path(binding["exact_manifest"]).resolve(strict=True)
    original = Path(binding["original_master"]).resolve(strict=True)
    product_receipt_path = Path(binding["prepared_product_receipt"]).resolve(
        strict=True
    )
    manifest_raw = manifest_path.read_bytes()
    product_receipt_raw = product_receipt_path.read_bytes()
    root = json.loads(manifest_raw)
    product_receipt = json.loads(product_receipt_raw)
    cache = root.get("exact_product_cache")
    product_schema = product_receipt.get("schema")
    if product_schema == "quantem.gpu.original-hdf5-calibrated-detector-products/v1":
        product_result = next(
            (
                result
                for result in product_receipt.get("results", [])
                if result.get("label") == binding["label"]
            ),
            None,
        )
    elif product_schema == "quantem.gpu.partner-calibration-correction/v1":
        product_result = {
            "source": product_receipt.get("source"),
            "products": product_receipt.get("products"),
        }
    else:
        product_result = None
    if (
        root.get("schema") != "quantem-gpu-android-qh5-index-manifest-v2"
        or not isinstance(cache, dict)
        or cache.get("status") != "PASS"
        or root.get("source_shape") != [512, 512, 192, 192]
        or root.get("source_dtype") != "uint16"
        or cache.get("working_dtype") != "uint8"
        or cache["contract"]["radial_prefix"]["exactness_bound"][
            "valid_maximum"
        ]
        != 29
        or cache["contract"].get("dpc_component_order_exchanged") is not False
        or product_schema
        not in {
            "quantem.gpu.original-hdf5-calibrated-detector-products/v1",
            "quantem.gpu.partner-calibration-correction/v1",
        }
        or product_receipt.get("status") != "succeeded"
        or not isinstance(product_result, dict)
        or product_result["source"].get("master_sha256")
        != binding["original_master_sha256"]
        or product_result["source"].get("source_identity_sha256")
        != cache["source_identity_sha256"]
        or product_result["source"].get("dtype") != "uint16"
        or product_result["source"].get("shape") != [512, 512, 192, 192]
        or _sha256_file(original) != binding["original_master_sha256"]
    ):
        raise ValueError(f"fixture provenance failed closed: {binding['label']}")
    product_paths = {}
    for name, artifact in product_result["products"].items():
        local_path = product_receipt_path.parent / Path(artifact["path"]).name
        if _sha256_file(local_path) != artifact["sha256"]:
            raise ValueError(
                f"prepared {name} oracle failed authentication: {binding['label']}"
            )
        product_paths[name] = local_path
    return compact, {
        "root": root,
        "manifest_path": manifest_path,
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "prepared_product_receipt_path": product_receipt_path,
        "prepared_product_receipt_sha256": hashlib.sha256(
            product_receipt_raw
        ).hexdigest(),
        "prepared_product_paths": product_paths,
        "compact_sha256": _sha256_file(compact),
        "original_master_sha256": binding["original_master_sha256"],
    }


def _validate_source(source, binding: dict[str, Any], oracle: dict[str, Any], processor):
    root = oracle["root"]
    cache = root["exact_product_cache"]
    manifest_path = oracle["manifest_path"]
    identity = cache["source_identity_sha256"]
    exact_external_mask_binding = (
        source.index.raw_access_mode == "mask_applied_only_legacy"
        and source.index.working_logical_sha256 == cache["working_logical_sha256"]
        and source.index.detector_mask_sha256 == cache["detector_mask_sha256"]
        and source.index.source_raw_logical_sha256
        == cache["raw_logical_source_sha256"]
        and len(source.index.excluded_detector_pixels) == 4
    )
    if (
        source.index.source_identity_sha256 != identity
        or list(source.index.shape) != [512, 512, 192, 192]
        or source.index.manifest.get("source_dtype") != "uint16"
        or source.index.manifest.get("working_dtype") != "uint8"
        or source.index.manifest.get("scan_bin") != 1
        or source.index.manifest.get("detector_bin") != 1
        or source.index.manifest.get("crop") is not None
        or (
            source.index.raw_access_mode != "exact_exclusion_constants"
            and not exact_external_mask_binding
        )
    ):
        raise ValueError(f"resident source provenance failed: {binding['label']}")

    selected_coordinate = cache["contract"]["selected_scan_coordinate"]
    started = time.perf_counter()
    selected = source.extract_diffraction(*selected_coordinate)
    selected_ms = (time.perf_counter() - started) * 1_000.0
    comparisons: dict[str, Any] = {
        "selected_diffraction": {
            **_exact_comparison(
                selected,
                _artifact(root, "selected_center_masked", manifest_path),
                "|u1",
            ),
            "wall_ms": selected_ms,
            "scan_coordinate": selected_coordinate,
        }
    }

    mean_started = time.perf_counter()
    mean = source.mean_diffraction_pattern()
    mean_ms = (time.perf_counter() - mean_started) * 1_000.0
    comparisons["diffraction_sum"] = _exact_comparison(
        mean.detector_sum,
        _artifact(root, "diffraction_sum", manifest_path),
        "<u8",
    )
    comparisons["mean_diffraction"] = {
        **_exact_comparison(
            mean.mean,
            _artifact(root, "mean_diffraction", manifest_path),
            "<f4",
        ),
        "wall_ms": mean_ms,
        "gpu_ms": mean.gpu_ms,
        "dispatch_count": mean.dispatch_count,
        "audit_readback_bytes": mean.readback_bytes,
    }

    detector_metrics = {}
    for product in ("bf", "abf", "adf"):
        metrics = source.activate_prepared_detector_product(product)
        comparison = _exact_comparison(
            source.virtual_detector_values(),
            oracle["prepared_product_paths"][product],
            "<u4",
        )
        detector_metrics[product] = {
            **asdict(metrics),
            "fft_dispatch_count": 0,
            "comparison": comparison,
        }
    comparisons["prepared_detectors"] = detector_metrics

    moments = source.prepared_dpc_moment_values()
    maps = source.prepared_dpc_values()
    row_buffer = source.prepared_dpc_display_buffer("row")
    column_buffer = source.prepared_dpc_display_buffer("column")
    if moments is None or maps is None or row_buffer is None or column_buffer is None:
        raise ValueError(f"prepared DPC is incomplete: {binding['label']}")
    comparisons["dpc_moments"] = {
        "total": _exact_comparison(
            moments.total, _artifact(root, "total_intensity", manifest_path), "<u8"
        ),
        "detector_row": _exact_comparison(
            moments.detector_row_moment,
            _artifact(root, "detector_row_moment", manifest_path),
            "<u8",
        ),
        "detector_column": _exact_comparison(
            moments.detector_column_moment,
            _artifact(root, "detector_column_moment", manifest_path),
            "<u8",
        ),
    }
    comparisons["centered_dpc"] = {
        "row": _exact_comparison(
            maps[0], _artifact(root, "com_row", manifest_path), "<f4"
        ),
        "column": _exact_comparison(
            maps[1], _artifact(root, "com_column", manifest_path), "<f4"
        ),
    }

    configuration = MPSDPCConfiguration(
        scan_rows=512,
        scan_columns=512,
        rotation_degrees=float(cache["contract"]["dpc_rotation_degrees"]),
        transpose_components=False,
    )
    result = processor.process_buffers(row_buffer, column_buffer, configuration)
    try:
        phase_bytes = configuration.count * np.dtype(np.float32).itemsize
        phase = np.frombuffer(
            result.phase_buffer.contents().as_buffer(phase_bytes), dtype="<f4"
        ).copy()
        comparisons["idpc"] = {
            **_float_comparison(
                phase,
                _artifact(root, "idpc", manifest_path),
                atol=2e-5,
            ),
            "metrics": asdict(result.metrics),
            "parity_audit_readback_bytes": phase_bytes,
            "interaction_path_readback_bytes": result.metrics.readback_bytes,
            "gradient_fft_storage_mode": int(result.gradient_fft_buffer.storageMode()),
            "phase_fft_storage_mode": int(result.phase_fft_buffer.storageMode()),
        }
    finally:
        result.release()

    statuses: list[str] = []

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            if "status" in value and value["status"] in {"pass", "fail"}:
                statuses.append(value["status"])
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(comparisons)
    return {
        "status": (
            "pass"
            if statuses and all(status == "pass" for status in statuses)
            else "fail"
        ),
        "comparisons": comparisons,
    }


def _load_sample(path: Path, device) -> tuple[Any, dict[str, Any]]:
    before_rss = _rss_bytes()
    before_device = int(device.currentAllocatedSize())
    started = time.perf_counter()
    with _PeakSampler(device) as peak:
        source = load_compact_v3_mps(path)
    wall_ms = (time.perf_counter() - started) * 1_000.0
    return source, {
        "wall_ms": wall_ms,
        "loader_resident_ready_ms": source.load_metrics.resident_ready_ms,
        "rss_bytes_before": before_rss,
        "rss_bytes_after": _rss_bytes(),
        "peak_rss_bytes": peak.peak_rss_bytes,
        "device_allocated_bytes_before": before_device,
        "device_allocated_bytes_after": int(device.currentAllocatedSize()),
        "peak_device_allocated_bytes": peak.peak_device_allocated_bytes,
        "storage_read_bytes": source.load_metrics.mapped_authentication_bytes,
        "resident_bytes": source.load_metrics.total_resident_bytes,
        "source_read_ms": source.load_metrics.source_read_ms,
        "authentication_ms": source.load_metrics.authentication_ms,
        "prepared_dpc_read_ms": source.load_metrics.prepared_dpc_read_ms,
        "prepared_detector_product_read_ms": (
            source.load_metrics.prepared_detector_product_read_ms
        ),
    }


def _reopen_series(path: Path, device, repeats: int) -> dict[str, Any]:
    samples = []
    for sample_index in range(repeats):
        source, sample = _load_sample(path, device)
        sample["sample_index"] = sample_index
        samples.append(sample)
        source.release()
    return {
        "boundary": "prepared-reopen-to-resident-ready",
        "cache_state": "uncontrolled",
        "cold_claim": False,
        "samples": samples,
        "resident_ready": _distribution([item["wall_ms"] for item in samples]),
        "peak_host_rss_bytes": max(item["peak_rss_bytes"] for item in samples),
        "peak_device_allocated_bytes": max(
            item["peak_device_allocated_bytes"] for item in samples
        ),
    }


def _aba(sources: list[Any], bindings: list[dict[str, Any]]) -> dict[str, Any]:
    recorder = MPSPublicationRecorder()
    samples = []
    for generation, source_index in enumerate((0, 1, 0), start=1):
        source = sources[source_index]
        binding = bindings[source_index]
        started = time.perf_counter()
        accepted = recorder.begin(
            generation,
            source.index.source_identity_sha256,
            MPSResidentRepresentation.COMPACT_QGIX_V3_UINT8,
        )
        if not accepted:
            raise RuntimeError("new A-B-A generation was unexpectedly rejected")
        _ = source.extract_diffraction(256, 256)
        detector = source.activate_prepared_detector_product("adf")
        counters = MPSPublicationCounters(
            source_bytes=source.index.file_bytes,
            resident_bytes=source.load_metrics.total_resident_bytes,
            process_rss_bytes=_rss_bytes(),
            device_allocated_bytes=int(source._device.currentAllocatedSize()),
            storage_read_bytes=0,
            upload_bytes=0,
            readback_bytes=0,
            synchronization_count=2,
        )
        recorder.record(
            generation,
            MPSPublicationMilestone.RESIDENT_READY,
            counters,
            "already-resident exact DP and prepared ADF ready; no presentation claim",
        )
        samples.append(
            {
                "generation": generation,
                "label": binding["label"],
                "source_identity_sha256": source.index.source_identity_sha256,
                "wall_ms": (time.perf_counter() - started) * 1_000.0,
                "storage_read_bytes": 0,
                "upload_bytes": 0,
                "readback_bytes": 0,
                "detector_mode": detector.mode,
                "actual_present": False,
            }
        )
    stale_accepted = recorder.record(
        1,
        MPSPublicationMilestone.FIRST_RESIDENT_PRESENT,
        detail="intentional stale-generation rejection probe",
    )
    if stale_accepted:
        raise RuntimeError("stale A generation unexpectedly published")
    return {
        "boundary": "exact-switch-to-resident-ready",
        "samples": samples,
        "distribution": _distribution([sample["wall_ms"] for sample in samples]),
        "source_storage_reads_during_switch": 0,
        "complete_sources_retained": 2,
        "actual_present": False,
        "events": [event.to_dict() for event in recorder.events()],
    }


def _run(options: argparse.Namespace) -> dict[str, Any]:
    bindings = [_load_binding(path) for path in options.fixture]
    validated = [_validate_binding(binding) for binding in bindings]
    try:
        import Metal
    except ImportError as error:
        raise RuntimeError("PyObjC Metal is required") from error
    device = Metal.MTLCreateSystemDefaultDevice()
    if device is None:
        raise RuntimeError("no physical Metal device is available")

    processor = MPSDPCProcessor(device)
    reopens = [
        _reopen_series(compact, device, options.reopen_repeats)
        for compact, _ in validated
    ]
    resident_sources = []
    retained_samples = []
    parity = []
    with _PeakSampler(device) as peak:
        for binding, (compact, oracle) in zip(bindings, validated, strict=True):
            source, sample = _load_sample(compact, device)
            resident_sources.append(source)
            retained_samples.append(sample)
            parity.append(_validate_source(source, binding, oracle, processor))
        aba = _aba(resident_sources, bindings)
    try:
        status = (
            "pass"
            if all(result["status"] == "pass" for result in parity)
            else "fail"
        )
        return {
            "schema": "quantem.gpu.apple-resident-consumer-acceptance/v1",
            "status": status,
            "timing_grant": (
                "exclusive Apple M5 GPU lane clean timing start "
                "2026-09-04T09:27:54-07:00"
            ),
            "hardware": {
                "device_name": str(device.name()),
                "registry_id": int(device.registryID()),
                "unified_memory": bool(device.hasUnifiedMemory()),
                "recommended_working_set_bytes": int(
                    device.recommendedMaxWorkingSetSize()
                ),
                "operating_system": platform.platform(),
                "python": platform.python_version(),
                "backend": "Python-hosted Metal",
            },
            "fixtures": [
                {
                    "binding": binding,
                    "compact_path": str(compact),
                    "compact_bytes": compact.stat().st_size,
                    "compact_sha256": oracle["compact_sha256"],
                    "exact_manifest_path": str(oracle["manifest_path"]),
                    "exact_manifest_sha256": oracle["manifest_sha256"],
                    "prepared_product_receipt_path": str(
                        oracle["prepared_product_receipt_path"]
                    ),
                    "prepared_product_receipt_sha256": oracle[
                        "prepared_product_receipt_sha256"
                    ],
                    "source_identity_sha256": oracle["root"]["exact_product_cache"]
                    ["source_identity_sha256"],
                    "shape": [512, 512, 192, 192],
                    "source_dtype": "uint16",
                    "working_dtype": "uint8",
                    "scan_bin": 1,
                    "detector_bin": 1,
                    "crop": None,
                    "valid_maximum": 29,
                }
                for binding, (compact, oracle) in zip(
                    bindings, validated, strict=True
                )
            ],
            "prepared_reopen": {
                binding["label"]: reopen
                for binding, reopen in zip(bindings, reopens, strict=True)
            },
            "retained_load_samples": retained_samples,
            "parity": {
                binding["label"]: result
                for binding, result in zip(bindings, parity, strict=True)
            },
            "already_resident_aba": aba,
            "memory": {
                "peak_host_rss_bytes": peak.peak_rss_bytes,
                "peak_device_allocated_bytes": peak.peak_device_allocated_bytes,
                "ru_maxrss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                "swap": _command_output(["sysctl", "-n", "vm.swapusage"]),
            },
            "presentation": {
                "actual_present_measured": False,
                "reason": "UI-free library has no compositor; consumer AppKit gate is separate",
                "first_resident_present": None,
            },
            "cold_original": {
                "measured": False,
                "reason": "this gate measures authenticated prepared reopen only",
            },
            "prepared_creation": {
                "measured": False,
                "reason": "existing sealed products were reused without regeneration",
            },
        }
    finally:
        for source in resident_sources:
            source.release()


def main() -> None:
    options = _arguments()
    options.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        receipt = _run(options)
        exit_code = 0 if receipt["status"] == "pass" else 2
    except Exception as error:  # noqa: BLE001 - the failure receipt is mandatory
        receipt = {
            "schema": "quantem.gpu.apple-resident-consumer-acceptance/v1",
            "status": "fail",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        exit_code = 2
    options.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
