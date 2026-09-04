#!/usr/bin/env python3
"""Run the native exact uint16 packer and seal its machine-readable manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(16 << 20):
            digest.update(block)
    return digest.hexdigest()


def _sha256_value(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be one lowercase SHA-256 digest")
    return value


def run_packed_detector_uint16(
    packer_path: Path,
    qh5_manifest_path: Path,
    output_directory: Path,
    *,
    expected_packer_sha256: str,
) -> dict[str, Any]:
    """Run one preserved native packer after validating all indexed inputs."""
    packer_path = packer_path.resolve()
    qh5_manifest_path = qh5_manifest_path.resolve()
    output_directory = output_directory.resolve()
    expected_packer_sha256 = _sha256_value(
        expected_packer_sha256,
        "expected_packer_sha256",
    )
    observed_packer_sha256 = _sha256_file(packer_path)
    if observed_packer_sha256 != expected_packer_sha256:
        raise ValueError(
            f"native packer SHA-256 is {observed_packer_sha256}, "
            f"expected {expected_packer_sha256}"
        )
    qh5 = json.loads(qh5_manifest_path.read_text())
    if qh5.get("source_shape") != [512, 512, 192, 192]:
        raise ValueError("QH5 manifest does not describe the exact tilt shape")
    if qh5.get("source_dtype") != "uint16":
        raise ValueError("QH5 manifest does not describe a uint16 source")
    raw_logical_sha256 = _sha256_value(
        qh5.get("logical_source_sha256"),
        "logical_source_sha256",
    )
    files = qh5.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("QH5 manifest contains no indexed source members")
    command = [str(packer_path), str(output_directory), raw_logical_sha256]
    for ordinal, record in enumerate(files):
        if record.get("ordinal") != ordinal:
            raise ValueError("QH5 member ordinals are not contiguous")
        source = Path(record["source_path"])
        index = Path(record["index_path"])
        if source.stat().st_size != int(record["source_bytes"]):
            raise ValueError(f"source member {ordinal} changed size")
        if _sha256_file(index) != record["index_sha256"]:
            raise ValueError(f"QH5 index {ordinal} changed after preparation")
        command.extend((str(source), str(index)))

    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(
            f"native packer exited {completed.returncode}: {completed.stderr.strip()}"
        )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("native packer did not return one JSON object") from error
    if (
        result.get("raw_logical_sha256") != raw_logical_sha256
        or result.get("recovered_logical_sha256") != raw_logical_sha256
    ):
        raise ValueError("native packer did not preserve the audited logical hash")
    shards = result.get("shards")
    if not isinstance(shards, list) or len(shards) != 64:
        raise ValueError("native packer did not return 64 complete scan shards")
    for ordinal, shard in enumerate(shards):
        if shard.get("ordinal") != ordinal or shard.get("scan_count") != 4096:
            raise ValueError("native packer returned incomplete shard coverage")
        for label in ("descriptors", "payload"):
            artifact = shard[label]
            artifact_path = output_directory / artifact["path"]
            if artifact_path.stat().st_size != int(artifact["bytes"]):
                raise ValueError(f"packed shard {ordinal} {label} size changed")
            if _sha256_file(artifact_path) != artifact["sha256"]:
                raise ValueError(f"packed shard {ordinal} {label} hash changed")

    manifest = {
        **result,
        "schema": "quantem.gpu.packed-detector-uint16/v1",
        "status": "complete",
        "source_shape": qh5["source_shape"],
        "source_dtype": "uint16",
        "source_identity_sha256": qh5.get("source_identity_sha256"),
        "qh5_manifest_path": str(qh5_manifest_path),
        "qh5_manifest_sha256": _sha256_file(qh5_manifest_path),
        "native_packer_path": str(packer_path),
        "native_packer_sha256": observed_packer_sha256,
        "native_packer_stderr": completed.stderr.splitlines(),
        "gpu_executed": False,
    }
    manifest_path = output_directory / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to replace existing {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "shard_count": len(shards),
        "raw_logical_sha256": raw_logical_sha256,
        "total_artifact_bytes": result["total_artifact_bytes"],
        "host_prepare_and_roundtrip_seconds": result[
            "host_prepare_and_roundtrip_seconds"
        ],
        "host_peak_rss_bytes": result["host_peak_rss_bytes"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packer", required=True, type=Path)
    parser.add_argument("--qh5-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-packer-sha256", required=True)
    arguments = parser.parse_args()
    receipt = run_packed_detector_uint16(
        arguments.packer,
        arguments.qh5_manifest,
        arguments.output_dir,
        expected_packer_sha256=arguments.expected_packer_sha256,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
