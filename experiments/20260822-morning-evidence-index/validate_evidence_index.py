"""Validate the source-only overnight evidence index.

This checker verifies references and provenance only. It never launches a
scientific runtime, opens a device, or rewrites an evidence artifact.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

INDEX_PATH = Path(__file__).with_name("evidence-index.json")
CAMPAIGN_ROOT = Path(__file__).resolve().parents[4]
AUDIT_WORKTREE = CAMPAIGN_ROOT / "work" / "quantem-gpu-morning-evidence-index-20260822"
ALLOWED_DISPOSITIONS = {"accepted", "refuted", "pending"}
PRIVATE_PATH_MARKERS = ("/Users/", "/home/", "private-fixture://")


def sha256_bytes(data: bytes) -> str:
    """Return the lowercase SHA-256 digest of ``data``."""

    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest of one immutable artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_git(worktree: Path, *args: str) -> bytes:
    """Run one read-only Git query and return exact stdout bytes."""

    return subprocess.check_output(["git", "-C", str(worktree), *args])


def resolve_campaign_path(relative: str) -> Path:
    """Resolve a campaign-relative path and reject path traversal."""

    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise AssertionError(f"unsafe campaign path: {relative}")
    resolved = (CAMPAIGN_ROOT / path).resolve()
    resolved.relative_to(CAMPAIGN_ROOT.resolve())
    return resolved


def require_unique_ids(records: list[dict[str, Any]], label: str) -> None:
    """Require non-empty unique identifiers in a registry section."""

    ids = [record.get("id") for record in records]
    assert all(isinstance(value, str) and value for value in ids), f"missing {label} id"
    assert len(ids) == len(set(ids)), f"duplicate {label} id"


def verify_artifact_file(record: dict[str, Any], path_key: str, hash_key: str) -> None:
    """Verify one referenced artifact and its sealed digest."""

    relative = record.get(path_key)
    expected = record.get(hash_key)
    if relative is None and expected is None:
        return
    assert isinstance(relative, str) and relative, f"{record['id']}: missing {path_key}"
    assert isinstance(expected, str) and len(expected) == 64, f"{record['id']}: bad {hash_key}"
    if relative.startswith("git-object://"):
        object_reference = relative.removeprefix("git-object://")
        revision, separator, repository_path = object_reference.partition("/")
        assert separator and len(revision) == 40, f"{record['id']}: bad Git object path"
        assert all(character in "0123456789abcdef" for character in revision), (
            f"{record['id']}: bad Git object revision"
        )
        path = Path(repository_path)
        assert repository_path and not path.is_absolute() and ".." not in path.parts, (
            f"{record['id']}: unsafe Git object path"
        )
        content = run_git(AUDIT_WORKTREE, "show", f"{revision}:{repository_path}")
        actual = sha256_bytes(content)
    else:
        path = resolve_campaign_path(relative)
        assert path.is_file(), f"{record['id']}: missing {relative}"
        actual = sha256_file(path)
    assert actual == expected, f"{record['id']}: {hash_key} mismatch: {actual}"


def verify_worktree(record: dict[str, Any]) -> None:
    """Verify the observed source HEAD and dirty-state fingerprint."""

    relative = record.get("source_worktree")
    if relative is None:
        return
    worktree = resolve_campaign_path(relative)
    assert worktree.is_dir(), f"{record['id']}: source worktree is missing"
    head = run_git(worktree, "rev-parse", "HEAD").decode().strip()
    assert head == record["observed_head"], f"{record['id']}: HEAD moved to {head}"

    status = run_git(worktree, "status", "--porcelain=v1")
    observed_clean = not status
    assert observed_clean is record["observed_clean"], (
        f"{record['id']}: clean state changed to {observed_clean}"
    )
    if observed_clean:
        return

    tracked_diff = run_git(worktree, "diff", "--binary")
    untracked = run_git(worktree, "ls-files", "--others", "--exclude-standard")
    expected_diff = record.get("observed_tracked_diff_sha256")
    expected_untracked = record.get("observed_untracked_inventory_sha256")
    expected_status = record.get("observed_status_sha256")
    if expected_diff:
        assert sha256_bytes(tracked_diff) == expected_diff, f"{record['id']}: tracked diff moved"
    if expected_untracked:
        assert sha256_bytes(untracked) == expected_untracked, (
            f"{record['id']}: untracked inventory moved"
        )
    if expected_status:
        assert sha256_bytes(status) == expected_status, f"{record['id']}: status moved"
    expected_candidate = record.get("expected_tracked_diff_sha256")
    if expected_candidate:
        assert sha256_bytes(tracked_diff) == expected_candidate, (
            f"{record['id']}: current diff no longer matches the sealed candidate"
        )


def verify_revision(record: dict[str, Any], key: str) -> None:
    """Require every recorded revision to resolve in the shared repository."""

    revision = record.get(key)
    if revision is None:
        return
    assert isinstance(revision, str) and len(revision) == 40, f"{record['id']}: bad {key}"
    subprocess.run(
        ["git", "-C", str(AUDIT_WORKTREE), "cat-file", "-e", f"{revision}^{{commit}}"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def verify_measurement(
    measurement: dict[str, Any], artifacts: dict[str, dict[str, Any]], fixtures: dict[str, dict[str, Any]]
) -> None:
    """Verify one atomic measurement row and its comparison boundary."""

    required = (
        "artifact_id",
        "disposition",
        "functional_area",
        "platform",
        "computer",
        "device",
        "date",
        "fixture_id",
        "fixture_sha256",
        "source_scan_rows",
        "source_scan_columns",
        "source_detector_rows",
        "source_detector_columns",
        "working_detector_rows",
        "working_detector_columns",
        "source_dtype",
        "working_dtype",
        "scan_bin",
        "detector_bin",
        "crop",
        "cache_state",
        "cold_claim",
        "timing_boundary",
        "sample_count",
        "parity",
        "comparability_group",
    )
    for key in required:
        assert key in measurement and measurement[key] not in ("", []), (
            f"{measurement['id']}: missing {key}"
        )

    artifact = artifacts[measurement["artifact_id"]]
    assert measurement["disposition"] == "accepted", f"{measurement['id']}: non-accepted metric"
    assert artifact["disposition"] == "accepted", f"{measurement['id']}: source artifact not accepted"
    fixture = fixtures[measurement["fixture_id"]]
    assert measurement["fixture_sha256"] == fixture["sha256"], f"{measurement['id']}: fixture mismatch"
    assert measurement["source_scan_rows"] == fixture["scan_rows"]
    assert measurement["source_scan_columns"] == fixture["scan_columns"]
    assert measurement["source_detector_rows"] == fixture["detector_rows"]
    assert measurement["source_detector_columns"] == fixture["detector_columns"]
    assert measurement["source_dtype"] == fixture["dtype"]
    assert measurement["scan_bin"] == 1
    assert measurement["crop"] == "none"
    assert measurement["cold_claim"] is False, f"{measurement['id']}: unsupported cold claim"
    if (
        measurement["platform"] == "cuda"
        and measurement["functional_area"] == "screening"
    ):
        assert measurement["working_dtype"] == "not-applicable-streamed-summaries"
        assert measurement["accumulation_dtype"] == "uint64"
        assert measurement["first_usable_p50_seconds"] == measurement[
            "exact_complete_p50_seconds"
        ]
    if measurement["functional_area"] == "selective-loading":
        assert measurement["platform"] == "webgpu"
        selected_rows = (
            measurement["selected_scan_row_stop"]
            - measurement["selected_scan_row_start"]
        )
        selected_columns = (
            measurement["selected_scan_column_stop"]
            - measurement["selected_scan_column_start"]
        )
        assert 0 < selected_rows <= measurement["source_scan_rows"]
        assert 0 < selected_columns <= measurement["source_scan_columns"]
        expected_bytes = (
            selected_rows
            * selected_columns
            * measurement["working_detector_rows"]
            * measurement["working_detector_columns"]
            * 2
        )
        assert measurement["logical_resident_bytes"] == expected_bytes
        assert 0 < measurement["source_shards_read"] < measurement["source_shards_total"]
        assert measurement["source_storage_bytes_read"] > 0
        assert measurement["accelerator_peak_bytes"] is None
        assert measurement["browser_tree_peak_rss_bytes"] > 0
    if measurement["functional_area"] == "screening-cache-reopen":
        assert measurement["working_dtype"] == "not-applicable-cached-results"
        assert measurement["cache_bytes"] == measurement["comparison_baseline_cache_bytes"]
        assert measurement["cache_bytes"] > measurement["legacy_cache_bytes"]
        assert measurement["retained_phase_cache_size_delta_bytes"] == (
            measurement["cache_bytes"] - measurement["legacy_cache_bytes"]
        )
        assert measurement["wall_p50_seconds"] < 0.006
        assert measurement["comparison_baseline_wall_p50_seconds"] > 0.09

    count = measurement["sample_count"]
    assert isinstance(count, int) and count >= 1
    if count == 1:
        assert measurement["single_wall_seconds"] is not None
        assert measurement["wall_p50_seconds"] is None
        assert measurement["wall_p95_seconds"] is None
        assert measurement["wall_max_seconds"] is None
    else:
        assert measurement["single_wall_seconds"] is None
        p50 = measurement["wall_p50_seconds"]
        p95 = measurement["wall_p95_seconds"]
        maximum = measurement["wall_max_seconds"]
        assert 0 < p50 <= p95 <= maximum

    for key in (
        "logical_resident_bytes",
        "accelerator_peak_bytes",
        "process_peak_rss_bytes",
        "process_peak_footprint_bytes",
    ):
        value = measurement[key]
        assert value is None or (isinstance(value, int) and value >= 0), (
            f"{measurement['id']}: invalid {key}"
        )


def verify_external_evidence(artifact: dict[str, Any]) -> None:
    """Validate a sealed external URI without claiming local resolution."""

    for record in artifact.get("external_evidence", []):
        assert record["uri"].startswith("local-evidence://")
        assert len(record["sha256_manifest"]) == 64
        assert record["resolved_in_phil_campaign"] is False


def verify_refuted_diagnostic(
    diagnostic: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
    fixtures: dict[str, dict[str, Any]],
) -> None:
    """Require refuted timing to stay bound to failed scientific parity."""

    artifact = artifacts[diagnostic["artifact_id"]]
    fixture = fixtures[diagnostic["fixture_id"]]
    assert diagnostic["disposition"] == "refuted"
    assert artifact["disposition"] == "refuted"
    assert diagnostic["promotion_allowed"] is False
    assert diagnostic["fixture_sha256"] == fixture["sha256"]
    assert diagnostic["phase_parity_pass"] is False
    assert diagnostic["loss_parity_pass"] is True
    assert diagnostic["phase_wrapped_max_vs_cuda_radians"] > diagnostic[
        "phase_wrapped_max_tolerance_radians"
    ]
    assert diagnostic["phase_wrapped_max_vs_mps_radians"] > diagnostic[
        "phase_wrapped_max_tolerance_radians"
    ]
    assert diagnostic["cache_state"]
    assert diagnostic["repeated_sample_count"] >= 2
    assert 0 < diagnostic["repeated_wall_p50_seconds"] <= diagnostic[
        "repeated_wall_p95_seconds"
    ] <= diagnostic["repeated_wall_max_seconds"]


def verify_pending_observation(
    observation: dict[str, Any], artifacts: dict[str, dict[str, Any]]
) -> None:
    """Keep provisional sub-results outside accepted measurement rows."""

    assert observation["artifact_id"] in artifacts
    assert observation["observation"]
    assert observation["cache_state"]
    assert observation["sample_count"] >= 1


def verify_source_placeholder(placeholder: dict[str, Any]) -> None:
    """Validate unpromoted source revisions and any observed working state."""

    assert placeholder["status"] in {"pending-evidence", "awaiting-final-evidence-pin"}
    assert placeholder["expected_evidence_revision"] is None
    assert placeholder["claim_scope"]
    for key in (
        "implementation_revision",
        "parent_evidence_revision",
        "parent_revision",
        "expected_evidence_revision",
    ):
        verify_revision(placeholder, key)
    if placeholder.get("source_worktree"):
        verify_worktree(placeholder)


def verify_no_private_paths(index: dict[str, Any]) -> None:
    """Ensure the public-facing index contains no raw private filesystem path."""

    rendered = json.dumps(index, sort_keys=True)
    for marker in PRIVATE_PATH_MARKERS:
        assert marker not in rendered, f"private path marker leaked: {marker}"


def main() -> int:
    """Validate the complete evidence index and print a compact result."""

    index = json.loads(INDEX_PATH.read_text())
    assert index["schema"] == "quantem.gpu.morning-evidence-index/v1"
    assert index["reconciled_at"] >= index["generated_at"]
    assert index["coordinator_base"] == "23d25619cfe22d5e89761fda2d2796a7c82ba090"
    assert index["runtime_or_hardware_execution"] is False
    assert set(index["disposition_vocabulary"]) == ALLOWED_DISPOSITIONS
    verify_no_private_paths(index)

    fixtures_list = index["fixtures"]
    artifacts_list = index["artifacts"]
    measurements = index["measurements"]
    require_unique_ids(fixtures_list, "fixture")
    require_unique_ids(artifacts_list, "artifact")
    require_unique_ids(measurements, "measurement")
    fixtures = {record["id"]: record for record in fixtures_list}
    artifacts = {record["id"]: record for record in artifacts_list}

    for fixture in fixtures_list:
        assert len(fixture["sha256"]) == 64
        assert fixture["path_disclosed"] is False

    for artifact in artifacts_list:
        assert artifact["disposition"] in ALLOWED_DISPOSITIONS
        if artifact["consumer_eligible"]:
            assert artifact["disposition"] == "accepted"
        if artifact["disposition"] != "accepted":
            assert artifact["consumer_eligible"] is False
        verify_artifact_file(artifact, "path", "sha256")
        verify_artifact_file(artifact, "primary_evidence_path", "primary_evidence_sha256")
        verify_artifact_file(artifact, "secondary_evidence_path", "secondary_evidence_sha256")
        verify_artifact_file(artifact, "seal_path", "seal_sha256")
        verify_worktree(artifact)
        verify_external_evidence(artifact)
        for key in ("observed_head", "implementation_revision", "test_revision", "evidence_revision"):
            verify_revision(artifact, key)

    for measurement in measurements:
        verify_measurement(measurement, artifacts, fixtures)
    for diagnostic in index.get("refuted_diagnostics", []):
        verify_refuted_diagnostic(diagnostic, artifacts, fixtures)
    for observation in index.get("pending_observations", []):
        verify_pending_observation(observation, artifacts)
    for placeholder in index.get("source_placeholders", []):
        verify_source_placeholder(placeholder)

    assert len(index["unresolved_metric_cells"]) >= 1
    assert len(index["incomparable_or_stale"]) >= 1
    counts = {status: 0 for status in sorted(ALLOWED_DISPOSITIONS)}
    for artifact in artifacts_list:
        counts[artifact["disposition"]] += 1
    print(
        "evidence index valid: "
        f"{len(artifacts_list)} artifacts "
        f"({counts['accepted']} accepted, {counts['refuted']} refuted, {counts['pending']} pending), "
        f"{len(measurements)} accepted atomic metrics, "
        f"{len(index.get('refuted_diagnostics', []))} refuted diagnostic, "
        f"{len(index.get('source_placeholders', []))} source placeholders, "
        f"{len(index['unresolved_metric_cells'])} unresolved cells"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
