"""Versioned SSB request/result boundary for native private-loopback clients."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import resource
import sys
import threading
import time
from collections.abc import Callable
from contextlib import nullcontext
from importlib.metadata import version
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np

logger = logging.getLogger("quantem.gpu.remote.ssb")

CONTRACT_VERSION = "live4dstem.ssb/v0.2"
ALGORITHM_VERSION = "quantem.gpu.SSB/v0.1"
JOBS_CONTRACT_VERSION = "live4dstem.ssb.jobs/v0.1"
PREPARE_CONTRACT_V0_1 = "live4dstem.ssb.prepare/v0.1"
PREPARE_CONTRACT_VERSION = "live4dstem.ssb.prepare/v0.2"
SOURCE_RESOLUTION_CONTRACT_VERSION = "live4dstem.ssb.source-resolution/v0.1"
INTERACTIVE_CONTRACT_V0_1 = "live4dstem.ssb.interactive/v0.1"
INTERACTIVE_CONTRACT_VERSION = "live4dstem.ssb.interactive/v0.2"
INTERACTIVE_EVALUATION_PHASE = "exact_full_bf_object_phase"
INTERACTIVE_EVALUATION_PHASE_AND_LOSS = (
    "exact_full_bf_object_phase_and_loss"
)
FIT_CONTRACT_VERSION = "live4dstem.ssb.fit/v0.1"
FIT_EVIDENCE_VERSION = "live4dstem.ssb.fit.evidence/v0.2"
_TERMINAL_JOB_STATES = {
    "completed",
    "cancelled",
    "failed",
    "expired",
    "superseded",
}


class SSBProtocolError(ValueError):
    """One actionable request or result contract failure."""


class SSBPayloadNotReady(SSBProtocolError):
    """A validated job exists, but it has not published a phase payload."""


class SSBPayloadUnavailable(SSBProtocolError):
    """A terminal job has no phase payload."""


def _fit_history_counts(
    *, backend_kind: str, optimizer_trials: int, recorded_history_count: int
) -> tuple[int, int, int]:
    """Validate and label the public backend's recorded fit history."""

    baseline_history_count = 0 if backend_kind == "remote_cuda" else 1
    expected_recorded_count = optimizer_trials + baseline_history_count
    if optimizer_trials != 200 or recorded_history_count != expected_recorded_count:
        raise SSBProtocolError(
            "Initial SSB fit history does not match the backend's exact "
            f"200-trial semantics: backend={backend_kind}, "
            f"optimizer={optimizer_trials}, baseline={baseline_history_count}, "
            f"recorded={recorded_history_count}."
        )
    return optimizer_trials, baseline_history_count, recorded_history_count


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_identity_sha256(master_hash: str, member_hashes: list[str]) -> str:
    """Return the Live4DSTEM dataset-v0.1 ordered source digest.

    The byte stream is the UTF-8 dataset schema, a NUL separator, the lowercase
    hexadecimal master digest, then one NUL plus lowercase hexadecimal digest
    for every member in acquisition order. Paths and JSON serialization never
    participate in the identity.
    """
    digest = hashlib.sha256()
    digest.update(b"live4dstem.dataset/v0.1\0")
    digest.update(master_hash.lower().encode())
    for value in member_hashes:
        digest.update(b"\0")
        digest.update(value.lower().encode())
    return digest.hexdigest()


def count_audit_sha256(
    source_identity_sha256: str,
    native_dtype: str,
    audited_element_count: int,
    maximum_count: int,
    counts_above_working_maximum: int,
) -> str:
    """Return the canonical digest for a complete detector count-range audit."""

    fields = (
        "live4dstem.count-audit/v0.1",
        source_identity_sha256.lower(),
        np.dtype(native_dtype).name,
        str(int(audited_element_count)),
        str(int(maximum_count)),
        "255",
        str(int(counts_above_working_maximum)),
    )
    return hashlib.sha256("\0".join(fields).encode()).hexdigest()


def selection_descriptor_sha256(descriptor: dict[str, Any]) -> str:
    """Return the canonical digest for one prepared scientific SSB selection."""

    canonical = dict(descriptor)
    canonical.pop("selectionSHA256", None)
    payload = json.dumps(canonical, separators=(",", ":"), sort_keys=True).encode()
    digest = hashlib.sha256()
    digest.update(str(descriptor.get("contractVersion", "")).encode())
    digest.update(b"\0")
    digest.update(payload)
    return digest.hexdigest()


def _calibration_binding(
    calibration: dict[str, Any], source_identity_sha256: str
) -> dict[str, str]:
    resolution = calibration.get("resolution") or {}
    if resolution.get("state") != "valid":
        raise SSBProtocolError("SSB calibration is not resolved.")
    if calibration.get("sourceIdentitySHA256") != source_identity_sha256:
        raise SSBProtocolError(
            "SSB calibration belongs to a different source identity."
        )
    candidate = resolution.get("calibration") or {}
    candidate_id = str(candidate.get("id", "")).strip()
    evidence = str(candidate.get("evidenceSHA256", "")).lower()
    try:
        evidence_is_digest = len(evidence) == 64 and int(evidence, 16) >= 0
    except ValueError:
        evidence_is_digest = False
    if not candidate_id or not evidence_is_digest or not candidate.get("calibration"):
        raise SSBProtocolError(
            "SSB requires a named calibration candidate with immutable evidence."
        )
    calibration_payload = json.dumps(
        candidate, separators=(",", ":"), sort_keys=True
    ).encode()
    return {
        "sourceIdentitySHA256": source_identity_sha256,
        "candidateID": candidate_id,
        "evidenceSHA256": evidence,
        "calibrationSHA256": hashlib.sha256(calibration_payload).hexdigest(),
    }


def _phase_bytes(value: object) -> bytes:
    try:
        import cupy as cp

        if isinstance(value, cp.ndarray):
            value = cp.asnumpy(value)
    except ImportError:
        pass
    phase = np.ascontiguousarray(value, dtype="<f4")
    if phase.ndim != 2:
        raise SSBProtocolError(f"SSB phase must be 2-D; got {phase.shape}.")
    return phase.tobytes()


def _calibrated_detector_sampling_mrad(request: dict[str, Any]) -> tuple[float, float]:
    """Return required detector angular sampling in public ``(row, col)`` order."""

    values = request["calibration"]["resolution"]["calibration"]["calibration"]
    sampling = []
    for name in (
        "detectorSamplingRowMilliradiansPerPixel",
        "detectorSamplingColumnMilliradiansPerPixel",
    ):
        field = values.get(name) or {}
        try:
            value = float(field.get("value", float("nan")))
        except (TypeError, ValueError) as exc:
            raise SSBProtocolError(
                "SSB requires numeric calibrated detector sampling in mrad/pixel."
            ) from exc
        if (
            field.get("unit") != "mrad/pixel"
            or not math.isfinite(value)
            or value <= 0.0
        ):
            raise SSBProtocolError(
                "SSB requires finite positive calibrated detector sampling "
                "for both row and column in mrad/pixel."
            )
        sampling.append(value)
    return sampling[0], sampling[1]


def _validated_precision(request: dict[str, Any]) -> dict[str, Any]:
    """Validate the native storage and proven-lossless SSB working precision."""

    precision = request.get("precision") or {}
    if (
        precision.get("realDType") != "float32"
        or precision.get("complexDType") != "complex64"
    ):
        raise SSBProtocolError("SSB arithmetic precision must be float32/complex64.")
    try:
        native = np.dtype(str(precision["nativeSourceDType"])).name
        working = np.dtype(str(precision["workingSourceDType"])).name
    except (KeyError, TypeError) as exc:
        raise SSBProtocolError(
            "SSB precision must name nativeSourceDType and workingSourceDType."
        ) from exc
    if native not in {"uint8", "uint16", "uint32"} or working != "uint8":
        raise SSBProtocolError(
            "SSB v0.2 requires unsigned native detector counts and uint8 working precision."
        )
    audit = precision.get("losslessWorkingDTypeAudit") or {}
    try:
        maximum = int(audit["maximumCount"])
        above = int(audit["countsAboveWorkingMaximum"])
        audited_count = int(audit["auditedElementCount"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SSBProtocolError(
            "SSB uint8 working precision requires an exact count-range audit."
        ) from exc
    if (
        audit.get("scope") != "complete_native_detector_source"
        or audit.get("workingMaximum") != 255
        or maximum < 0
        or maximum > 255
        or above != 0
        or audited_count <= 0
    ):
        raise SSBProtocolError(
            "SSB uint8 working precision is not proven lossless for all unmasked detector counts."
        )
    source_identity = str((request.get("source") or {}).get("sourceIdentitySHA256", ""))
    if audit.get("sourceIdentitySHA256") != source_identity:
        raise SSBProtocolError(
            "SSB count audit belongs to a different source identity."
        )
    expected_count = math.prod(
        int(request[shape][dimension])
        for shape in ("scanShape", "detectorShape")
        for dimension in ("rows", "columns")
    )
    if audited_count != expected_count:
        raise SSBProtocolError(
            "SSB count audit does not cover the complete native detector source."
        )
    expected_evidence = count_audit_sha256(
        source_identity, native, audited_count, maximum, above
    )
    if audit.get("evidenceSHA256") != expected_evidence:
        raise SSBProtocolError("SSB count audit evidence digest is invalid.")
    return {
        "nativeSourceDType": native,
        "workingSourceDType": working,
        "losslessWorkingDTypeAudit": {
            "scope": "complete_native_detector_source",
            "sourceIdentitySHA256": source_identity,
            "auditedElementCount": audited_count,
            "maximumCount": maximum,
            "workingMaximum": 255,
            "countsAboveWorkingMaximum": 0,
            "evidenceSHA256": expected_evidence,
        },
        "realDType": "float32",
        "complexDType": "complex64",
    }


def _executed_precision(
    request: dict[str, Any], *, working_dtype: object
) -> dict[str, Any]:
    """Bind the opened session's working dtype to the accepted exact audit."""

    requested = _validated_precision(request)
    try:
        working = np.dtype(working_dtype).name
    except (TypeError, ValueError) as exc:
        raise SSBProtocolError(
            "The SSB backend did not report its opened working dtype."
        ) from exc
    if working != requested["workingSourceDType"]:
        raise SSBProtocolError(
            "The SSB backend working dtype does not match the bound lossless audit."
        )
    return requested


class SSBProtocolService:
    """Validate SSB jobs and retain only validated phase payloads in memory."""

    def __init__(
        self,
        data_folder: str | Path,
        *,
        available_gpus: Callable[[], list[int]],
        device_name: Callable[[int | None], str],
        runner: Callable[[Path, int | None, dict[str, Any]], dict[str, Any]]
        | None = None,
        preparer: Callable[[Path, int | None, dict[str, Any]], dict[str, Any]]
        | None = None,
        source_inspector: Callable[..., Any] | None = None,
        session_opener: Callable[[Path, int | None, dict[str, Any]], tuple[Any, Any]]
        | None = None,
        session_device_context: Callable[[int | None], Any] | None = None,
        runtime_diagnostics: Callable[[], dict[str, Any]] | None = None,
        clock: Callable[[], float] = time.time,
        session_lease_seconds: float = 300.0,
        backend_kind: str = "remote_cuda",
        implementation_revision: str = "unrecorded",
    ) -> None:
        if backend_kind not in {"remote_cuda", "local_mps"}:
            raise ValueError("backend_kind must be remote_cuda or local_mps")
        self.data_folder = Path(data_folder).expanduser().resolve()
        self._available_gpus = available_gpus
        self._device_name = device_name
        self.backend_kind = backend_kind
        self.implementation_revision = implementation_revision.strip() or "unrecorded"
        self._runner = runner or (
            self._run_cuda if backend_kind == "remote_cuda" else self._run_mps
        )
        self._preparer = preparer or (
            self._prepare_cuda if backend_kind == "remote_cuda" else self._prepare_mps
        )
        if source_inspector is None:
            from quantem.gpu.io import inspect

            source_inspector = inspect
        self._source_inspector = source_inspector
        self._session_opener = session_opener or (
            self._open_cuda_session
            if backend_kind == "remote_cuda"
            else self._open_mps_session
        )
        self._session_device_context = session_device_context or (
            self._cuda_device_context
            if backend_kind == "remote_cuda"
            else lambda _gpu: nullcontext()
        )
        self._runtime_diagnostics = runtime_diagnostics or self._default_runtime_diagnostics
        self._clock = clock
        self._session_lease_seconds = float(session_lease_seconds)
        if self._session_lease_seconds <= 0.0:
            raise ValueError("session_lease_seconds must be positive")
        self._lock = threading.Lock()
        self._payloads: dict[tuple[str, int], tuple[bytes, dict[str, Any]]] = {}
        self._jobs: dict[tuple[str, int], dict[str, Any]] = {}
        self._sync_keys: set[tuple[str, int]] = set()
        self._device_locks: dict[tuple[str, int | None], threading.Lock] = {}
        self._retained_sessions: dict[str, dict[str, Any]] = {}
        self._device_sessions: dict[tuple[str, int | None], str] = {}
        self._interactive_jobs: dict[tuple[str, int], dict[str, Any]] = {}
        self._interactive_payloads: dict[
            tuple[str, int], tuple[bytes, dict[str, Any]]
        ] = {}

    @staticmethod
    def advertised_capability(
        backend_kind: str = "remote_cuda",
        implementation_revision: str = "unrecorded",
        device_name: str | None = None,
        gpu_index: int | None = None,
        unavailable_reason: str | None = None,
    ) -> dict[str, Any]:
        ready = (
            implementation_revision != "unrecorded"
            and device_name is not None
            and (backend_kind == "local_mps" or gpu_index is not None)
            and unavailable_reason is None
        )
        return {
            "name": "ssb",
            "contractVersion": CONTRACT_VERSION,
            "algorithmVersion": ALGORITHM_VERSION,
            "backendKind": backend_kind,
            "implicitFallback": False,
            "ready": ready,
            "unavailableReason": unavailable_reason,
            "implementationRevision": implementation_revision,
            "device": {
                "backend": "cuda" if backend_kind == "remote_cuda" else "mps",
                "deviceName": device_name,
                "gpuIndex": gpu_index,
            },
            "resultPayload": "job_generation_endpoint",
            "preparation": {
                "contractVersion": PREPARE_CONTRACT_VERSION,
                "compatibleContractVersions": [PREPARE_CONTRACT_V0_1],
                "endpoint": "/api/ssb/prepare",
                "method": "POST",
                "productStatus": "beta",
                "workflow": "direct_ptychography",
                "selectionPolicy": "full_active",
                "detectorBin": 1,
                "scanCropSupported": False,
                "workingPrecision": "lossless_uint8_complete_source_audit",
                "requiresExplicitBackend": True,
                "implicitFallback": False,
            },
            "sourceResolution": {
                "contractVersion": SOURCE_RESOLUTION_CONTRACT_VERSION,
                "mode": "configured_root_exact_identity",
                "remoteClientPathAccepted": False,
                "missingIsError": True,
                "ambiguousIsError": True,
            },
            "precisionContract": {
                "nativeSourceDTypes": ["uint8", "uint16", "uint32"],
                "workingSourceDTypes": ["uint8"],
                "uint8RequiresCompleteSourceAudit": True,
            },
            "requiredCalibration": {
                "detectorSamplingOrder": ["row", "column"],
                "detectorSamplingUnit": "mrad/pixel",
                "implicitDetectorSampling": False,
            },
            "fixedReconstructionControls": {
                "aberrations": [
                    {"name": "C10", "unit": "nm"},
                    {"name": "C12", "unit": "nm"},
                    {"name": "phi12", "unit": "radian"},
                ],
                "scanRotation": {"name": "scanRotation", "unit": "degree"},
                "higherOrderAberrations": False,
                "preparedSessionWarmReconstruct": True,
                "contractVersion": INTERACTIVE_CONTRACT_VERSION,
                "compatibleRequestVersions": [INTERACTIVE_CONTRACT_V0_1],
                "sessionEndpoint": "/api/ssb/interactive/sessions",
                "jobEndpoint": "/api/ssb/interactive/jobs",
                "evaluations": [
                    INTERACTIVE_EVALUATION_PHASE,
                    INTERACTIVE_EVALUATION_PHASE_AND_LOSS,
                ],
                "settleBinding": "exact_prior_preview_control_generation",
                "saveEvidenceRequiresLossState": "settled",
                "sessionLeaseSeconds": 300,
                "cancellationMode": "stage_boundary",
                "reconnectScope": "same_server_process",
                "serverRestartResume": False,
            },
            "initialAberrationFit": {
                "supported": True,
                "contractVersion": FIT_CONTRACT_VERSION,
                "evidenceVersion": FIT_EVIDENCE_VERSION,
                "endpoint": "/api/ssb/interactive/fits",
                "optimizer": "optuna_tpe",
                "optimizerTrials": 200,
                "objective": "exact_full_active_bf_phase_variance_float32",
                "searchRanges": {
                    "C10Nanometers": {"minimum": -400.0, "maximum": 400.0},
                    "C12Nanometers": {"minimum": 0.0, "maximum": 100.0},
                    "phi12Radians": {
                        "minimum": -math.pi / 2.0,
                        "maximum": math.pi / 2.0,
                    },
                },
                "seedRequired": True,
                "scanRotation": "fixed_to_retained_session",
                "refinement": None,
                "higherOrderAberrations": False,
                "candidateBatchSize": 4 if backend_kind == "remote_cuda" else 2,
                "optimizerTrialHistoryCount": 200,
                "baselineHistoryCount": 0 if backend_kind == "remote_cuda" else 1,
                "recordedHistoryCount": 200 if backend_kind == "remote_cuda" else 201,
                "totalObjectiveEvaluations": "not_exposed_by_public_backend",
                "retainedSessionBehavior": (
                    "reuses_prepared_accelerator"
                    if backend_kind == "remote_cuda"
                    else "reprepares_from_retained_source_object"
                ),
                "sourceReopen": False,
                "resultPersistence": "operator_acceptance_required",
                "cancellationMode": "stage_boundary",
                "reconnectScope": "same_server_process",
                "implicitFallback": False,
            },
            "jobLifecycle": {
                "contractVersion": JOBS_CONTRACT_VERSION,
                "cancellationMode": "stage_boundary",
                "reconnectScope": "same_server_process",
                "progress": "stage_only_indeterminate",
                "serverRestartResume": False,
                "resultRetention": "same_server_process",
            },
            "stageTimingAvailability": {
                "sourceLoad": True,
                "firstReconstruct": True,
                "warmReconstruct": True,
                "gQKConstruction": False,
                "kernel": False,
            },
        }

    def capability(self) -> dict[str, Any]:
        """Advertise this exact backend, device, and implementation revision."""

        gpu = None
        device = None
        unavailable_reason = None
        try:
            if self.backend_kind == "remote_cuda":
                available = self._available_gpus()
                if not available:
                    raise RuntimeError("no CUDA device is available")
                gpu = available[0]
            device = self._device_name(gpu)
            if not device:
                raise RuntimeError("device identity is unavailable")
        except (ImportError, RuntimeError, ValueError) as exc:
            unavailable_reason = f"{type(exc).__name__}: {exc}"
        if self.implementation_revision == "unrecorded":
            unavailable_reason = (
                "The exact quantem.gpu implementation revision is not recorded."
            )
        capability = self.advertised_capability(
            self.backend_kind,
            self.implementation_revision,
            device,
            gpu,
            unavailable_reason,
        )
        capability["fixedReconstructionControls"]["sessionLeaseSeconds"] = (
            self._session_lease_seconds
        )
        return capability

    @staticmethod
    def request_sha256(request: dict[str, Any]) -> str:
        """Hash one canonical scientific request for idempotent submission."""

        canonical = json.dumps(request, separators=(",", ":"), sort_keys=True).encode()
        return hashlib.sha256(canonical).hexdigest()

    def submit(self, request: dict[str, Any]) -> dict[str, Any]:
        """Accept one idempotent background job and return its current snapshot."""

        request = json.loads(json.dumps(request, separators=(",", ":"), sort_keys=True))
        job_id = str(UUID(str(request["jobID"])))
        generation = int(request["datasetGeneration"])
        key = (job_id, generation)
        request_sha = self.request_sha256(request)
        now = time.time()
        with self._lock:
            if key in self._sync_keys:
                raise SSBProtocolError(
                    "The SSB job ID is active on the synchronous compatibility endpoint."
                )
            existing = self._jobs.get(key)
            if existing is not None:
                if existing["requestSHA256"] != request_sha:
                    raise SSBProtocolError(
                        "The SSB job ID already belongs to a different request digest."
                    )
                return self._public_snapshot(existing)
            job = {
                "jobID": job_id,
                "datasetGeneration": generation,
                "requestSHA256": request_sha,
                "sourceIdentitySHA256": request["source"]["sourceIdentitySHA256"],
                "selection": request["selection"],
                "selectionSHA256": (request.get("preparedSelection") or {}).get(
                    "selectionSHA256"
                ),
                "requestedBackend": request["backend"],
                "sequence": 0,
                "state": "accepted",
                "progress": {"stage": "accepted", "determinate": False},
                "acceptedAt": now,
                "updatedAt": now,
                "result": None,
                "error": None,
                "cancelRequested": False,
            }
            self._jobs[key] = job
            snapshot = self._public_snapshot(job)
        threading.Thread(
            target=self._run_submitted_job,
            args=(key, request),
            name=f"ssb-{job_id}",
            daemon=True,
        ).start()
        return snapshot

    def job_snapshot(self, job_id: str, generation: int) -> dict[str, Any]:
        """Return the latest same-process snapshot for reconnect polling."""

        key = (str(UUID(job_id)), int(generation))
        with self._lock:
            job = self._jobs.get(key)
            if job is None:
                raise SSBProtocolError("No SSB job exists for this ID and generation.")
            return self._public_snapshot(job)

    def cancel_job(self, job_id: str, generation: int) -> dict[str, Any]:
        """Request cancellation at the next honest worker stage boundary."""

        key = (str(UUID(job_id)), int(generation))
        with self._lock:
            job = self._jobs.get(key)
            if job is None:
                raise SSBProtocolError("No SSB job exists for this ID and generation.")
            if job["state"] not in _TERMINAL_JOB_STATES | {"cancel_requested"}:
                job["cancelRequested"] = True
                self._advance_locked(job, "cancel_requested")
            return self._public_snapshot(job)

    def _run_submitted_job(self, key: tuple[str, int], request: dict[str, Any]) -> None:
        try:
            if self._cancelled_at_boundary(key):
                return
            gpu = self._requested_device(request)
            with self._device_lock(gpu):
                if not self._begin_stage(key, "validating"):
                    return
                source, validated_gpu = self._validate_request(request)
                if not self._begin_stage(key, "reconstructing_first"):
                    return
                payload, descriptor, result = self._execute_validated(
                    source, validated_gpu, request
                )
                with self._lock:
                    job = self._jobs[key]
                    if job["cancelRequested"]:
                        self._payloads.pop(key, None)
                        self._advance_locked(job, "cancelled")
                    else:
                        self._payloads[key] = (payload, descriptor)
                        job["result"] = result
                        self._advance_locked(job, "completed")
        except Exception as exc:  # noqa: BLE001 - worker errors become typed snapshots
            with self._lock:
                job = self._jobs[key]
                if job["cancelRequested"]:
                    self._payloads.pop(key, None)
                    self._advance_locked(job, "cancelled")
                else:
                    job["error"] = {
                        "message": str(exc),
                        "recovery": "Verify the source, calibration, backend, and retry.",
                    }
                    self._advance_locked(job, "failed")

    def _cancelled_at_boundary(self, key: tuple[str, int]) -> bool:
        with self._lock:
            job = self._jobs[key]
            if not job["cancelRequested"]:
                return False
            self._payloads.pop(key, None)
            self._advance_locked(job, "cancelled")
            return True

    def _begin_stage(self, key: tuple[str, int], state: str) -> bool:
        """Atomically cancel at a boundary or enter the next opaque stage."""

        with self._lock:
            job = self._jobs[key]
            if job["cancelRequested"]:
                self._payloads.pop(key, None)
                self._advance_locked(job, "cancelled")
                return False
            self._advance_locked(job, state)
            return True

    def _device_lock(self, gpu: int | None) -> threading.Lock:
        key = (self.backend_kind, gpu)
        with self._lock:
            return self._device_locks.setdefault(key, threading.Lock())

    @staticmethod
    def _advance_locked(job: dict[str, Any], state: str) -> None:
        job["sequence"] += 1
        job["state"] = state
        job["progress"] = {"stage": state, "determinate": False}
        job["updatedAt"] = time.time()

    @staticmethod
    def _public_snapshot(job: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in job.items() if key != "cancelRequested"}

    def _default_runtime_diagnostics(self) -> dict[str, Any]:
        """Return bounded process/allocator evidence without synchronizing compute."""

        peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform != "darwin":
            peak_rss *= 1024
        diagnostics: dict[str, Any] = {
            "processID": os.getpid(),
            "processPeakRSSBytes": peak_rss,
            "mlxActiveBytes": None,
            "mlxPeakBytes": None,
            "mlxCacheBytes": None,
        }
        if self.backend_kind != "local_mps":
            return diagnostics
        try:
            import mlx.core as mx

            for field, name in (
                ("mlxActiveBytes", "get_active_memory"),
                ("mlxPeakBytes", "get_peak_memory"),
                ("mlxCacheBytes", "get_cache_memory"),
            ):
                getter = getattr(mx, name, None)
                if callable(getter):
                    diagnostics[field] = int(getter())
        except Exception:
            logger.exception("Could not sample retained MPS allocator state")
        return diagnostics

    def _interactive_binding(
        self, request: dict[str, Any], gpu: int | None
    ) -> dict[str, Any]:
        prepared = request["preparedSelection"]
        binding = {
            "contractVersion": INTERACTIVE_CONTRACT_VERSION,
            "sourceIdentitySHA256": request["source"]["sourceIdentitySHA256"],
            "datasetGeneration": int(request["datasetGeneration"]),
            "calibrationBinding": prepared["calibrationBinding"],
            "selectionSHA256": prepared["selectionSHA256"],
            "precision": request["precision"],
            "backend": request["backend"],
            "device": {
                "backend": "cuda" if self.backend_kind == "remote_cuda" else "mps",
                "deviceName": self._device_name(gpu),
                "gpuIndex": gpu,
            },
            "implementationRevision": prepared["implementationRevision"],
        }
        binding["sessionBindingSHA256"] = hashlib.sha256(
            json.dumps(binding, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest()
        return binding

    @staticmethod
    def _interactive_controls(request: dict[str, Any]) -> dict[str, float]:
        controls = request.get("controls") or {}
        expected = {"C10", "C12", "phi12", "scanRotation"}
        if set(controls) != expected:
            raise SSBProtocolError(
                "Interactive SSB controls must contain exactly C10, C12, phi12, "
                "and scanRotation."
            )
        result = {name: float(controls[name]) for name in expected}
        if not all(math.isfinite(value) for value in result.values()):
            raise SSBProtocolError("Interactive SSB controls must be finite.")
        return result

    @staticmethod
    def _interactive_evaluation(
        request: dict[str, Any],
    ) -> tuple[str, int | None]:
        """Resolve explicit v0.2 evaluation or the tested v0.1 compatibility form."""
        version = request.get("contractVersion")
        if version == INTERACTIVE_CONTRACT_V0_1:
            if "evaluation" in request or "settlesControlGeneration" in request:
                raise SSBProtocolError(
                    "Interactive v0.1 requests use computeLoss, not v0.2 evaluation fields."
                )
            evaluation = (
                INTERACTIVE_EVALUATION_PHASE_AND_LOSS
                if bool(request.get("computeLoss", True))
                else INTERACTIVE_EVALUATION_PHASE
            )
            return evaluation, None
        if version != INTERACTIVE_CONTRACT_VERSION:
            raise SSBProtocolError("Unsupported interactive SSB job contract.")
        if "computeLoss" in request:
            raise SSBProtocolError(
                "Interactive v0.2 replaces computeLoss with the evaluation enum."
            )
        evaluation = request.get("evaluation")
        if evaluation not in {
            INTERACTIVE_EVALUATION_PHASE,
            INTERACTIVE_EVALUATION_PHASE_AND_LOSS,
        }:
            raise SSBProtocolError("Interactive v0.2 requires a supported evaluation.")
        settles = request.get("settlesControlGeneration")
        if evaluation == INTERACTIVE_EVALUATION_PHASE:
            if settles is not None:
                raise SSBProtocolError(
                    "Phase-only preview settlesControlGeneration must be null."
                )
            return evaluation, None
        if settles is None:
            raise SSBProtocolError(
                "Phase-and-loss settle requires settlesControlGeneration."
            )
        settles = int(settles)
        control_generation = int(request["controlGeneration"])
        if settles < 0 or settles >= control_generation:
            raise SSBProtocolError(
                "settlesControlGeneration must identify an earlier preview generation."
            )
        return evaluation, settles

    @staticmethod
    def _initial_controls(request: dict[str, Any]) -> dict[str, float]:
        values = request["calibration"]["resolution"]["calibration"]["calibration"]
        return {
            "C10": float(values["c10Nanometers"]["value"]),
            "C12": float(values["c12Nanometers"]["value"]),
            "phi12": float(values["phi12Radians"]["value"]),
            "scanRotation": float(values["scanRotationDegrees"]["value"]),
        }

    @staticmethod
    def _fit_specification(
        request: dict[str, Any], retained: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate one deterministic, fixed-rotation 200-trial fit request."""

        if request.get("contractVersion") != FIT_CONTRACT_VERSION:
            raise SSBProtocolError("Unsupported initial SSB fit contract.")
        if int(request.get("optimizerTrials", 0)) != 200:
            raise SSBProtocolError("Initial SSB fit requires exactly 200 optimizer trials.")
        if request.get("objective") != "exact_full_active_bf_phase_variance_float32":
            raise SSBProtocolError("Initial SSB fit requires the exact full-active-BF objective.")
        if request.get("refinement") is not None:
            raise SSBProtocolError(
                "Initial SSB fit refinement must be null so the budget remains 200 trials."
            )
        seed = int(request["seed"])
        ranges = request.get("searchRanges") or {}
        expected = {"C10Nanometers", "C12Nanometers", "phi12Radians"}
        if set(ranges) != expected:
            raise SSBProtocolError(
                "Initial SSB fit ranges must contain exactly C10, C12, and phi12; "
                "higher-order aberrations are unavailable."
            )
        normalized: dict[str, dict[str, float]] = {}
        for name in sorted(expected):
            value = ranges[name]
            if set(value) != {"minimum", "maximum"}:
                raise SSBProtocolError(f"{name} requires minimum and maximum.")
            minimum = float(value["minimum"])
            maximum = float(value["maximum"])
            if not math.isfinite(minimum) or not math.isfinite(maximum) or minimum >= maximum:
                raise SSBProtocolError(f"{name} requires a finite increasing range.")
            normalized[name] = {"minimum": minimum, "maximum": maximum}
        fixed_rotation = float(request["fixedScanRotationDegrees"])
        session_rotation = SSBProtocolService._initial_controls(
            retained["baseRequest"]
        )["scanRotation"]
        if not math.isclose(fixed_rotation, session_rotation, rel_tol=0.0, abs_tol=1e-12):
            raise SSBProtocolError(
                "Initial SSB fit scan rotation must stay fixed to the retained session."
            )
        return {
            "optimizer": "optuna_tpe",
            "optimizerTrials": 200,
            "seed": seed,
            "objective": "exact_full_active_bf_phase_variance_float32",
            "searchRanges": normalized,
            "fixedScanRotationDegrees": fixed_rotation,
            "refinement": None,
        }

    def _session_fit_outcome(
        self,
        session: Any,
        gpu: int | None,
        request: dict[str, Any],
        specification: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run one public backend fit without reopening the retained source."""

        ranges = specification["searchRanges"]
        backend_ranges = {
            "C10_nm": (
                ranges["C10Nanometers"]["minimum"],
                ranges["C10Nanometers"]["maximum"],
            ),
            "C12_nm": (
                ranges["C12Nanometers"]["minimum"],
                ranges["C12Nanometers"]["maximum"],
            ),
            "phi12_deg": tuple(
                math.degrees(ranges["phi12Radians"][bound])
                for bound in ("minimum", "maximum")
            ),
        }
        with self._session_device_context(gpu):
            started = time.perf_counter()
            result = session.fit(
                trials=200,
                refinement=None,
                search_ranges=backend_ranges,
                seed=specification["seed"],
                force=True,
                verbose=False,
            )
            fit_seconds = time.perf_counter() - started
            state = session.browser_state()
            encode_started = time.perf_counter()
            phase_value = result.phase
            transfer = getattr(phase_value, "get", None)
            phase = np.asarray(
                transfer() if callable(transfer) else phase_value,
                dtype=np.float32,
            )
            encode_seconds = time.perf_counter() - encode_started
        fitted = {
            "C10": float(result.aberrations["C10"]),
            "C12": float(result.aberrations["C12"]),
            "phi12": float(result.aberrations["phi12"]),
            "scanRotation": specification["fixedScanRotationDegrees"],
        }
        trial_history = list(result.optuna_trials or ())
        optimizer_history_count, baseline_history_count, recorded_history_count = (
            _fit_history_counts(
                backend_kind=self.backend_kind,
                optimizer_trials=int(result.n_trials),
                recorded_history_count=len(trial_history),
            )
        )
        fit_evidence = {
            "evidenceVersion": FIT_EVIDENCE_VERSION,
            "specification": specification,
            "fittedControls": fitted,
            "sliderSeed": fitted,
            "loss": None if result.loss is None else float(result.loss),
            "optimizerTrialsCompleted": int(result.n_trials),
            "optimizerTrialHistoryCount": optimizer_history_count,
            "baselineHistoryCount": baseline_history_count,
            "recordedHistoryCount": recorded_history_count,
            "totalObjectiveEvaluationCount": None,
            "totalObjectiveEvaluationCountReason": (
                "Public CUDA and MPS fit APIs perform backend-specific baseline, "
                "warm-up, and final-loss evaluations outside the 200 Optuna trials."
            ),
            "candidateBatchSize": 4 if self.backend_kind == "remote_cuda" else 2,
            "operatorAcceptanceRequired": True,
            "persistedAsDatasetCalibration": False,
            "backendTimings": dict(result.timings or {}),
        }
        outcome = {
            "phase": phase,
            "logicalBrightFieldCount": state.num_bf,
            "activeBrightFieldCount": state.active_num_bf,
            "detectorSamplingMilliradians": tuple(
                float(value) * 1e3 for value in state.angular_sampling_rad
            ),
            "precision": _executed_precision(
                request, working_dtype=session.source_dtype
            ),
            "loss": result.loss,
            "driverVersion": None,
            "runtimeVersion": None,
            "timings": {
                "sourceReadSeconds": None,
                "sourceDecodeSeconds": None,
                "gQKConstructionSeconds": None,
                "kernelSeconds": None,
                "firstReconstructSeconds": None,
                "warmReconstructSeconds": None,
                "resultEncodeSeconds": encode_seconds,
                "sshRequestToFirstByteSeconds": None,
                "transferSeconds": None,
                "clientDecodeSeconds": None,
                "paintSeconds": None,
                "inputToPaintSeconds": None,
                "sessionOpenSeconds": None,
                "initialAberrationFitSeconds": fit_seconds,
            },
        }
        return outcome, fit_evidence

    def _session_outcome(
        self,
        session: Any,
        gpu: int | None,
        request: dict[str, Any],
        controls: dict[str, float],
        *,
        evaluation: str,
        initial: bool = False,
    ) -> dict[str, Any]:
        compute_loss = evaluation == INTERACTIVE_EVALUATION_PHASE_AND_LOSS
        with self._session_device_context(gpu):
            started = time.perf_counter()
            session.set_rotation(controls["scanRotation"])
            result = session.reconstruct(
                {name: controls[name] for name in ("C10", "C12", "phi12")},
                compute_loss=compute_loss,
                force=True,
            )
            reconstruct_seconds = time.perf_counter() - started
            state = session.browser_state()
            encode_started = time.perf_counter()
            transfer = getattr(result.phase, "get", None)
            phase = np.asarray(
                transfer() if callable(transfer) else result.phase,
                dtype=np.float32,
            )
            encode_seconds = time.perf_counter() - encode_started
        if compute_loss:
            if result.loss is None or not math.isfinite(float(result.loss)):
                raise SSBProtocolError("Settled interactive SSB loss must be finite.")
        elif result.loss is not None:
            raise SSBProtocolError("Phase-only interactive SSB must not compute loss.")
        return {
            "phase": phase,
            "logicalBrightFieldCount": state.num_bf,
            "activeBrightFieldCount": state.active_num_bf,
            "detectorSamplingMilliradians": tuple(
                float(value) * 1e3 for value in state.angular_sampling_rad
            ),
            "precision": _executed_precision(
                request, working_dtype=session.source_dtype
            ),
            "loss": result.loss,
            "driverVersion": None,
            "runtimeVersion": None,
            "timings": {
                "sourceReadSeconds": None,
                "sourceDecodeSeconds": None,
                "gQKConstructionSeconds": None,
                "kernelSeconds": None,
                "firstReconstructSeconds": reconstruct_seconds if initial else None,
                "warmReconstructSeconds": None if initial else reconstruct_seconds,
                "resultEncodeSeconds": encode_seconds,
                "sshRequestToFirstByteSeconds": None,
                "transferSeconds": None,
                "clientDecodeSeconds": None,
                "paintSeconds": None,
                "inputToPaintSeconds": None,
                "sessionOpenSeconds": None,
            },
        }

    def open_interactive_session(self, request: dict[str, Any]) -> dict[str, Any]:
        """Open one retained source-bound session and run its initial reconstruction."""

        if request.get("contractVersion") not in {
            INTERACTIVE_CONTRACT_VERSION,
            INTERACTIVE_CONTRACT_V0_1,
        }:
            raise SSBProtocolError("Unsupported interactive SSB session contract.")
        session_id = str(UUID(str(request["sessionID"])))
        initial = json.loads(json.dumps(request.get("initialRequest") or {}))
        source, gpu = self._validate_request(initial)
        binding = self._interactive_binding(initial, gpu)
        device_key = (self.backend_kind, gpu)
        self._expire_interactive_sessions()
        with self._device_lock(gpu):
            with self._lock:
                existing = self._retained_sessions.get(session_id)
                if existing is not None:
                    if existing["binding"] != binding:
                        raise SSBProtocolError(
                            "The retained SSB session ID belongs to a different binding."
                        )
                    return {
                        "contractVersion": INTERACTIVE_CONTRACT_VERSION,
                        "session": self._session_snapshot(existing),
                        "initialResult": existing["initialResult"],
                    }
                if device_key in self._device_sessions:
                    raise SSBProtocolError(
                        "The selected device already has a retained SSB session."
                    )
            context = None
            try:
                open_started = time.perf_counter()
                context, session = self._session_opener(source, gpu, initial)
                open_seconds = time.perf_counter() - open_started
                controls = self._initial_controls(initial)
                outcome = self._session_outcome(
                    session,
                    gpu,
                    initial,
                    controls,
                    evaluation=INTERACTIVE_EVALUATION_PHASE_AND_LOSS,
                    initial=True,
                )
                outcome["timings"]["sessionOpenSeconds"] = open_seconds
                outcome["timings"]["sourceReadSeconds"] = getattr(
                    session, "source_load_seconds", None
                )
                job_id = str(UUID(str(initial["jobID"])))
                generation = int(initial["datasetGeneration"])
                payload_path = (
                    f"/api/ssb/interactive/jobs/{job_id}/phase"
                    f"?generation={generation}"
                )
                payload, descriptor, result = self._validated_result(
                    outcome, gpu, initial, payload_path=payload_path
                )
                now = self._clock()
                retained = {
                    "sessionID": session_id,
                    "binding": binding,
                    "backend": initial["backend"],
                    "gpu": gpu,
                    "context": context,
                    "session": session,
                    "baseRequest": initial,
                    "latestControlGeneration": 0,
                    "createdAt": now,
                    "expiresAt": now + self._session_lease_seconds,
                    "initialResult": None,
                }
                key = (job_id, generation)
                result["interactiveSession"] = self._session_snapshot(retained)
                result["interactiveControls"] = controls
                result["evaluation"] = INTERACTIVE_EVALUATION_PHASE_AND_LOSS
                result["settlesControlGeneration"] = None
                result["lossState"] = "settled"
                result["saveEvidenceEligible"] = True
                result["runtimeMemory"] = self._runtime_diagnostics()
                self._refresh_result_provenance(result)
                retained["initialResult"] = result
                initial_job = {
                    "contractVersion": INTERACTIVE_CONTRACT_VERSION,
                    "sessionID": session_id,
                    "jobID": job_id,
                    "datasetGeneration": generation,
                    "controlGeneration": 0,
                    "sessionBindingSHA256": binding["sessionBindingSHA256"],
                    "sourceIdentitySHA256": binding["sourceIdentitySHA256"],
                    "selectionSHA256": binding["selectionSHA256"],
                    "requestedBackend": initial["backend"],
                    "controls": controls,
                    "evaluation": INTERACTIVE_EVALUATION_PHASE_AND_LOSS,
                    "settlesControlGeneration": None,
                    "lossState": "settled",
                    "sequence": 1,
                    "state": "completed",
                    "progress": {"stage": "completed", "determinate": False},
                    "acceptedAt": now,
                    "updatedAt": now,
                    "result": result,
                    "error": None,
                    "cancelRequested": False,
                }
                with self._lock:
                    if key in self._interactive_jobs:
                        raise SSBProtocolError(
                            "The initial interactive SSB job ID is already in use."
                        )
                    self._retained_sessions[session_id] = retained
                    self._device_sessions[device_key] = session_id
                    self._interactive_jobs[key] = initial_job
                    self._interactive_payloads[key] = (payload, descriptor)
                return {
                    "contractVersion": INTERACTIVE_CONTRACT_VERSION,
                    "session": self._session_snapshot(retained),
                    "initialResult": result,
                }
            except Exception:
                if context is not None:
                    with self._session_device_context(gpu):
                        context.__exit__(None, None, None)
                raise

    def interactive_session_snapshot(self, session_id: str) -> dict[str, Any]:
        """Return one same-process retained-session snapshot for reconnect."""

        self._expire_interactive_sessions()
        session_id = str(UUID(session_id))
        with self._lock:
            retained = self._retained_sessions.get(session_id)
            if retained is None:
                raise SSBProtocolError("No retained SSB session exists for this ID.")
            return self._session_snapshot(retained)

    @staticmethod
    def _session_snapshot(retained: dict[str, Any]) -> dict[str, Any]:
        return {
            "sessionID": retained["sessionID"],
            "binding": retained["binding"],
            "backend": retained["backend"],
            "latestControlGeneration": retained["latestControlGeneration"],
            "createdAt": retained["createdAt"],
            "expiresAt": retained["expiresAt"],
            "restartResumable": False,
        }

    def _expire_interactive_sessions(self) -> None:
        now = self._clock()
        with self._lock:
            expired = [
                session_id
                for session_id, retained in self._retained_sessions.items()
                if retained["expiresAt"] <= now
            ]
        for session_id in expired:
            self.close_interactive_session(session_id, expired=True)

    def close_interactive_session(
        self, session_id: str, *, expired: bool = False
    ) -> dict[str, Any]:
        """Close one retained session after any opaque device stage finishes."""

        with self._lock:
            retained = self._retained_sessions.get(session_id)
            if retained is None:
                raise SSBProtocolError("No retained SSB session exists for this ID.")
            for job in self._interactive_jobs.values():
                if job["sessionID"] == session_id and job["state"] not in (
                    _TERMINAL_JOB_STATES | {"cancel_requested"}
                ):
                    job["cancelRequested"] = True
                    self._advance_locked(job, "cancel_requested")
            gpu = retained["gpu"]
        with self._device_lock(gpu):
            with self._lock:
                retained = self._retained_sessions.pop(session_id, None)
                if retained is None:
                    raise SSBProtocolError("The retained SSB session is already closed.")
                self._device_sessions.pop((self.backend_kind, gpu), None)
            with self._session_device_context(gpu):
                retained["context"].__exit__(None, None, None)
        return {
            "contractVersion": INTERACTIVE_CONTRACT_VERSION,
            "sessionID": session_id,
            "state": "expired" if expired else "closed",
            "restartResumable": False,
        }

    def close(self) -> None:
        """Release every retained session when the service process shuts down."""

        with self._lock:
            session_ids = list(self._retained_sessions)
        for session_id in session_ids:
            try:
                self.close_interactive_session(session_id)
            except SSBProtocolError:
                pass

    def submit_interactive(self, request: dict[str, Any]) -> dict[str, Any]:
        """Submit one latest-wins warm reconstruction against a retained session."""

        evaluation, settles_control_generation = self._interactive_evaluation(request)
        self._expire_interactive_sessions()
        session_id = str(UUID(str(request["sessionID"])))
        job_id = str(UUID(str(request["jobID"])))
        generation = int(request["datasetGeneration"])
        control_generation = int(request["controlGeneration"])
        controls = self._interactive_controls(request)
        key = (job_id, generation)
        with self._lock:
            retained = self._retained_sessions.get(session_id)
            if retained is None:
                raise SSBProtocolError("The retained SSB session is missing or expired.")
            if request.get("sessionBindingSHA256") != retained["binding"][
                "sessionBindingSHA256"
            ]:
                raise SSBProtocolError("The interactive SSB session binding changed.")
            if request.get("sourceIdentitySHA256") != retained["binding"][
                "sourceIdentitySHA256"
            ] or request.get("selectionSHA256") != retained["binding"][
                "selectionSHA256"
            ]:
                raise SSBProtocolError("The interactive SSB source or selection changed.")
            if request.get("backend") != retained["backend"]:
                raise SSBProtocolError("The interactive SSB backend or device changed.")
            if generation != int(retained["baseRequest"]["datasetGeneration"]):
                raise SSBProtocolError("The interactive SSB dataset generation changed.")
            if control_generation <= retained["latestControlGeneration"]:
                raise SSBProtocolError(
                    "Interactive SSB controlGeneration must increase monotonically."
                )
            if (
                request.get("contractVersion") == INTERACTIVE_CONTRACT_VERSION
                and evaluation == INTERACTIVE_EVALUATION_PHASE_AND_LOSS
            ):
                if settles_control_generation != retained["latestControlGeneration"]:
                    raise SSBProtocolError(
                        "Interactive settle targets a stale preview control generation."
                    )
                preview = next(
                    (
                        candidate
                        for candidate in self._interactive_jobs.values()
                        if candidate.get("sessionID") == session_id
                        and candidate.get("controlGeneration")
                        == settles_control_generation
                        and candidate.get("evaluation")
                        == INTERACTIVE_EVALUATION_PHASE
                        and candidate.get("state") == "completed"
                    ),
                    None,
                )
                if preview is None or preview.get("controls") != controls:
                    raise SSBProtocolError(
                        "Interactive settle requires the exact completed preview controls."
                    )
            if key in self._interactive_jobs:
                raise SSBProtocolError("The interactive SSB job ID is already in use.")
            retained["latestControlGeneration"] = control_generation
            retained["expiresAt"] = self._clock() + self._session_lease_seconds
            now = self._clock()
            job = {
                "contractVersion": request["contractVersion"],
                "sessionID": session_id,
                "jobID": job_id,
                "datasetGeneration": generation,
                "controlGeneration": control_generation,
                "sessionBindingSHA256": request["sessionBindingSHA256"],
                "sourceIdentitySHA256": request["sourceIdentitySHA256"],
                "selectionSHA256": request["selectionSHA256"],
                "requestedBackend": request["backend"],
                "controls": controls,
                "evaluation": evaluation,
                "settlesControlGeneration": settles_control_generation,
                "lossState": (
                    "settled"
                    if evaluation == INTERACTIVE_EVALUATION_PHASE_AND_LOSS
                    else "pending_exact_phase_variance"
                ),
                "sequence": 0,
                "state": "accepted",
                "progress": {"stage": "accepted", "determinate": False},
                "acceptedAt": now,
                "updatedAt": now,
                "result": None,
                "error": None,
                "cancelRequested": False,
            }
            self._interactive_jobs[key] = job
            snapshot = self._public_snapshot(job)
        threading.Thread(
            target=self._run_interactive_job,
            args=(key, request),
            name=f"ssb-interactive-{job_id}",
            daemon=True,
        ).start()
        return snapshot

    def _run_interactive_job(
        self, key: tuple[str, int], request: dict[str, Any]
    ) -> None:
        session_id = str(UUID(str(request["sessionID"])))
        try:
            with self._lock:
                retained = self._retained_sessions.get(session_id)
                if retained is None:
                    raise SSBProtocolError("The retained SSB session expired.")
                gpu = retained["gpu"]
            with self._device_lock(gpu):
                with self._lock:
                    job = self._interactive_jobs[key]
                    retained = self._retained_sessions.get(session_id)
                    if retained is None or job["cancelRequested"]:
                        self._advance_locked(job, "cancelled")
                        return
                    if job["controlGeneration"] != retained["latestControlGeneration"]:
                        self._advance_locked(job, "superseded")
                        return
                    self._advance_locked(job, "reconstructing_warm")
                    base_request = json.loads(json.dumps(retained["baseRequest"]))
                    session = retained["session"]
                base_request["jobID"] = key[0]
                base_request["computeLoss"] = (
                    job["evaluation"] == INTERACTIVE_EVALUATION_PHASE_AND_LOSS
                )
                outcome = self._session_outcome(
                    session,
                    gpu,
                    base_request,
                    job["controls"],
                    evaluation=job["evaluation"],
                )
                payload_path = (
                    f"/api/ssb/interactive/jobs/{key[0]}/phase"
                    f"?generation={key[1]}"
                )
                payload, descriptor, result = self._validated_result(
                    outcome, gpu, base_request, payload_path=payload_path
                )
                with self._lock:
                    job = self._interactive_jobs[key]
                    retained = self._retained_sessions.get(session_id)
                    if job["cancelRequested"]:
                        self._advance_locked(job, "cancelled")
                    elif retained is None:
                        self._advance_locked(job, "expired")
                    elif job["controlGeneration"] != retained["latestControlGeneration"]:
                        self._advance_locked(job, "superseded")
                    else:
                        result["interactiveSession"] = self._session_snapshot(retained)
                        result["interactiveControls"] = job["controls"]
                        result["controlGeneration"] = job["controlGeneration"]
                        result["evaluation"] = job["evaluation"]
                        result["settlesControlGeneration"] = job[
                            "settlesControlGeneration"
                        ]
                        result["lossState"] = job["lossState"]
                        result["saveEvidenceEligible"] = job["lossState"] == "settled"
                        result["runtimeMemory"] = self._runtime_diagnostics()
                        self._refresh_result_provenance(result)
                        self._interactive_payloads[key] = (payload, descriptor)
                        job["result"] = result
                        self._advance_locked(job, "completed")
        except BaseException as exc:
            logger.exception(
                "Retained SSB reconstruction failed for job=%s generation=%s",
                key[0],
                key[1],
            )
            with self._lock:
                job = self._interactive_jobs[key]
                if job["cancelRequested"]:
                    self._advance_locked(job, "cancelled")
                else:
                    job["error"] = {"message": str(exc)}
                    self._advance_locked(job, "failed")

    def submit_interactive_fit(self, request: dict[str, Any]) -> dict[str, Any]:
        """Submit a fixed-rotation initial fit against one retained session."""

        self._expire_interactive_sessions()
        session_id = str(UUID(str(request["sessionID"])))
        job_id = str(UUID(str(request["jobID"])))
        generation = int(request["datasetGeneration"])
        control_generation = int(request["controlGeneration"])
        key = (job_id, generation)
        with self._lock:
            retained = self._retained_sessions.get(session_id)
            if retained is None:
                raise SSBProtocolError("The retained SSB session is missing or expired.")
            if request.get("sessionBindingSHA256") != retained["binding"][
                "sessionBindingSHA256"
            ]:
                raise SSBProtocolError("The initial SSB fit session binding changed.")
            if request.get("sourceIdentitySHA256") != retained["binding"][
                "sourceIdentitySHA256"
            ] or request.get("selectionSHA256") != retained["binding"][
                "selectionSHA256"
            ]:
                raise SSBProtocolError("The initial SSB fit source or selection changed.")
            if request.get("backend") != retained["backend"]:
                raise SSBProtocolError("The initial SSB fit backend or device changed.")
            if generation != int(retained["baseRequest"]["datasetGeneration"]):
                raise SSBProtocolError("The initial SSB fit dataset generation changed.")
            if control_generation <= retained["latestControlGeneration"]:
                raise SSBProtocolError(
                    "Initial SSB fit controlGeneration must increase monotonically."
                )
            if key in self._interactive_jobs:
                raise SSBProtocolError("The initial SSB fit job ID is already in use.")
            specification = self._fit_specification(request, retained)
            retained["latestControlGeneration"] = control_generation
            retained["expiresAt"] = self._clock() + self._session_lease_seconds
            now = self._clock()
            job = {
                "contractVersion": FIT_CONTRACT_VERSION,
                "operation": "initial_aberration_fit",
                "sessionID": session_id,
                "jobID": job_id,
                "datasetGeneration": generation,
                "controlGeneration": control_generation,
                "sessionBindingSHA256": request["sessionBindingSHA256"],
                "sourceIdentitySHA256": request["sourceIdentitySHA256"],
                "selectionSHA256": request["selectionSHA256"],
                "requestedBackend": request["backend"],
                "fitSpecification": specification,
                "sequence": 0,
                "state": "accepted",
                "progress": {"stage": "accepted", "determinate": False},
                "acceptedAt": now,
                "updatedAt": now,
                "result": None,
                "error": None,
                "cancelRequested": False,
            }
            self._interactive_jobs[key] = job
            snapshot = self._public_snapshot(job)
        threading.Thread(
            target=self._run_interactive_fit,
            args=(key,),
            name=f"ssb-fit-{job_id}",
            daemon=True,
        ).start()
        return snapshot

    def _run_interactive_fit(self, key: tuple[str, int]) -> None:
        """Execute one serialized, stage-boundary-cancellable retained fit."""

        try:
            with self._lock:
                job = self._interactive_jobs[key]
                session_id = job["sessionID"]
                retained = self._retained_sessions.get(session_id)
                if retained is None:
                    raise SSBProtocolError("The retained SSB session expired.")
                gpu = retained["gpu"]
            with self._device_lock(gpu):
                with self._lock:
                    job = self._interactive_jobs[key]
                    retained = self._retained_sessions.get(session_id)
                    if retained is None or job["cancelRequested"]:
                        self._advance_locked(job, "cancelled")
                        return
                    if job["controlGeneration"] != retained["latestControlGeneration"]:
                        self._advance_locked(job, "superseded")
                        return
                    self._advance_locked(job, "fitting_initial_aberrations")
                    base_request = json.loads(json.dumps(retained["baseRequest"]))
                    session = retained["session"]
                    specification = job["fitSpecification"]
                base_request["jobID"] = key[0]
                outcome, fit_evidence = self._session_fit_outcome(
                    session, gpu, base_request, specification
                )
                payload_path = (
                    f"/api/ssb/interactive/jobs/{key[0]}/phase"
                    f"?generation={key[1]}"
                )
                payload, descriptor, result = self._validated_result(
                    outcome, gpu, base_request, payload_path=payload_path
                )
                with self._lock:
                    job = self._interactive_jobs[key]
                    retained = self._retained_sessions.get(session_id)
                    if job["cancelRequested"]:
                        self._advance_locked(job, "cancelled")
                    elif retained is None:
                        self._advance_locked(job, "expired")
                    elif job["controlGeneration"] != retained["latestControlGeneration"]:
                        self._advance_locked(job, "superseded")
                    else:
                        result["interactiveSession"] = self._session_snapshot(retained)
                        result["controlGeneration"] = job["controlGeneration"]
                        result["initialAberrationFit"] = fit_evidence
                        result["runtimeMemory"] = self._runtime_diagnostics()
                        self._refresh_result_provenance(result)
                        self._interactive_payloads[key] = (payload, descriptor)
                        job["result"] = result
                        self._advance_locked(job, "completed")
        except BaseException as exc:
            logger.exception(
                "Retained SSB fit failed for job=%s generation=%s",
                key[0],
                key[1],
            )
            with self._lock:
                job = self._interactive_jobs[key]
                if job["cancelRequested"]:
                    self._advance_locked(job, "cancelled")
                else:
                    job["error"] = {"message": str(exc)}
                    self._advance_locked(job, "failed")

    def interactive_job_snapshot(self, job_id: str, generation: int) -> dict[str, Any]:
        self._expire_interactive_sessions()
        key = (str(UUID(job_id)), int(generation))
        with self._lock:
            job = self._interactive_jobs.get(key)
            if job is None:
                raise SSBProtocolError("No interactive SSB job exists for this ID.")
            return self._public_snapshot(job)

    def cancel_interactive_job(self, job_id: str, generation: int) -> dict[str, Any]:
        key = (str(UUID(job_id)), int(generation))
        with self._lock:
            job = self._interactive_jobs.get(key)
            if job is None:
                raise SSBProtocolError("No interactive SSB job exists for this ID.")
            if job["state"] not in _TERMINAL_JOB_STATES | {"cancel_requested"}:
                job["cancelRequested"] = True
                self._advance_locked(job, "cancel_requested")
            return self._public_snapshot(job)

    def interactive_payload(
        self, job_id: str, generation: int
    ) -> tuple[bytes, dict[str, Any]]:
        self._expire_interactive_sessions()
        key = (str(UUID(job_id)), int(generation))
        with self._lock:
            job = self._interactive_jobs.get(key)
            stored = self._interactive_payloads.get(key)
            if job is None:
                raise SSBProtocolError("No interactive SSB job exists for this ID.")
            if job["state"] == "completed" and stored is not None:
                return stored
            if job["state"] not in _TERMINAL_JOB_STATES:
                raise SSBPayloadNotReady("The interactive SSB job is not complete.")
            raise SSBPayloadUnavailable(
                f"Interactive SSB job state {job['state']} has no phase payload."
            )

    @staticmethod
    def _refresh_result_provenance(result: dict[str, Any]) -> None:
        result.pop("provenanceSHA256", None)
        result["provenanceSHA256"] = hashlib.sha256(
            json.dumps(result, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest()

    def source_identity(self, master_path: str) -> dict[str, Any]:
        master = self._resolve_master(master_path)
        stem = master.name.removesuffix("_master.h5")
        members = sorted(master.parent.glob(f"{stem}_data_*.h5"))
        master_hash = _sha256(master)
        member_hashes = [_sha256(path) for path in members]
        return {
            "datasetSchema": "live4dstem.dataset/v0.1",
            "masterPath": str(master),
            "masterSHA256": master_hash,
            "orderedMemberSHA256": member_hashes,
            "sourceIdentitySHA256": _source_identity_sha256(master_hash, member_hashes),
        }

    @staticmethod
    def _expected_source_identity(value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise SSBProtocolError("SSB source locator requires an immutable expected identity.")
        expected = {
            "datasetSchema": value.get("datasetSchema"),
            "masterSHA256": value.get("masterSHA256"),
            "orderedMemberSHA256": value.get("orderedMemberSHA256"),
            "sourceIdentitySHA256": value.get("sourceIdentitySHA256"),
        }
        if expected["datasetSchema"] != "live4dstem.dataset/v0.1":
            raise SSBProtocolError("Unsupported SSB source identity schema.")
        member_hashes = expected["orderedMemberSHA256"]
        if not isinstance(member_hashes, list):
            raise SSBProtocolError("SSB expected source identity requires ordered member hashes.")
        digests = [expected["masterSHA256"], *member_hashes, expected["sourceIdentitySHA256"]]
        if any(
            not isinstance(digest, str)
            or len(digest) != 64
            or digest != digest.lower()
            or any(character not in "0123456789abcdef" for character in digest)
            for digest in digests
        ):
            raise SSBProtocolError("SSB expected source identity contains an invalid SHA-256 digest.")
        if _source_identity_sha256(expected["masterSHA256"], member_hashes) != expected[
            "sourceIdentitySHA256"
        ]:
            raise SSBProtocolError("SSB expected source identity digest is inconsistent.")
        return expected

    def _resolve_configured_root_identity(
        self, expected: dict[str, Any]
    ) -> tuple[Path, dict[str, Any]]:
        matches: list[tuple[Path, dict[str, Any]]] = []
        for candidate in sorted(self.data_folder.rglob("*_master.h5")):
            resolved = candidate.resolve()
            try:
                resolved.relative_to(self.data_folder)
            except ValueError:
                continue
            identity = self.source_identity(str(resolved))
            if all(identity[key] == expected[key] for key in expected):
                matches.append((resolved, identity))
        if not matches:
            raise SSBProtocolError(
                "No configured-root SSB source matches the complete immutable identity."
            )
        if len(matches) != 1:
            raise SSBProtocolError("Configured-root SSB source identity is ambiguous.")
        return matches[0]

    def _prepare_source(
        self, request: dict[str, Any]
    ) -> tuple[Path, dict[str, Any], str]:
        version = request.get("contractVersion")
        if version == PREPARE_CONTRACT_V0_1:
            master = self._resolve_master(str(request.get("masterPath", "")))
            identity = self.source_identity(str(master))
            expected_digest = request.get("expectedSourceIdentitySHA256")
            if expected_digest is not None and str(expected_digest).lower() != identity[
                "sourceIdentitySHA256"
            ]:
                raise SSBProtocolError("SSB source identity changed before preparation.")
            return master, identity, PREPARE_CONTRACT_V0_1
        if version != PREPARE_CONTRACT_VERSION:
            raise SSBProtocolError("Unsupported SSB prepare contract version.")
        locator = request.get("sourceLocator") or {}
        expected = self._expected_source_identity(locator.get("expectedIdentity"))
        kind = locator.get("kind")
        if self.backend_kind == "remote_cuda":
            if kind != "configured_root_identity" or "masterPath" in locator:
                raise SSBProtocolError(
                    "remote_cuda requires configured_root_identity and rejects client paths."
                )
            master, identity = self._resolve_configured_root_identity(expected)
            return master, identity, PREPARE_CONTRACT_VERSION
        if kind != "local_path":
            raise SSBProtocolError("local_mps requires an explicit local_path source locator.")
        master = self._resolve_master(str(locator.get("masterPath", "")))
        identity = self.source_identity(str(master))
        if any(identity[key] != expected[key] for key in expected):
            raise SSBProtocolError(
                "Local SSB source does not match the complete immutable identity."
            )
        return master, identity, PREPARE_CONTRACT_VERSION

    def prepare(self, request: dict[str, Any]) -> dict[str, Any]:
        """Prepare one source-bound SSB selection without reconstructing a phase."""

        if self.implementation_revision == "unrecorded":
            raise SSBProtocolError(
                "The SSB service implementation revision is not recorded; restart it with an exact revision."
            )
        gpu = self._requested_device(request)
        master, identity, prepare_version = self._prepare_source(request)

        calibration = request.get("calibration") or {}
        calibration_binding = _calibration_binding(
            calibration, identity["sourceIdentitySHA256"]
        )
        requested_selection = request.get("selection") or {}
        if (
            requested_selection.get("policy") != "full_active"
            or int(requested_selection.get("detectorBin", 0)) != 1
            or requested_selection.get("scanCrop") is not None
        ):
            raise SSBProtocolError(
                "Native SSB preparation requires full_active BF, no crop, and detector bin 1."
            )

        inspection = self._source_inspector(str(master))
        if not inspection.ready:
            raise SSBProtocolError(
                f"SSB source is not ready: {inspection.reason} {inspection.action}".strip()
            )
        if inspection.scan_shape is None or inspection.detector_shape is None:
            raise SSBProtocolError(
                "SSB preparation requires inspected native scan and detector shapes."
            )
        scan_shape = {
            "rows": int(inspection.scan_shape[0]),
            "columns": int(inspection.scan_shape[1]),
        }
        detector_shape = {
            "rows": int(inspection.detector_shape[0]),
            "columns": int(inspection.detector_shape[1]),
        }
        execution_request = {
            **request,
            "source": identity,
            "scanShape": scan_shape,
            "detectorShape": detector_shape,
        }
        precision = _validated_precision(execution_request)
        if np.dtype(inspection.dtype).name != precision["nativeSourceDType"]:
            raise SSBProtocolError(
                "SSB native source dtype mismatch: prepare request declares "
                f"{precision['nativeSourceDType']}, source is {inspection.dtype}."
            )
        detector_sampling = _calibrated_detector_sampling_mrad(execution_request)

        with self._device_lock(gpu):
            outcome = self._preparer(master, gpu, execution_request)
        executed_precision = outcome.get("precision")
        if executed_precision is None:
            executed_precision = _executed_precision(
                execution_request,
                working_dtype=outcome.get("workingSourceDType"),
            )
        if executed_precision != precision:
            raise SSBProtocolError(
                "SSB prepared precision or lossless count audit differs from the request."
            )
        executed_sampling = tuple(
            float(value) for value in outcome["detectorSamplingMilliradians"]
        )
        if len(executed_sampling) != 2 or not np.allclose(
            executed_sampling, detector_sampling, rtol=0.0, atol=1e-9
        ):
            raise SSBProtocolError(
                "SSB prepared detector sampling differs from the resolved calibration."
            )
        logical_count = int(outcome["logicalBrightFieldCount"])
        active_count = int(outcome["activeBrightFieldCount"])
        if logical_count <= 0 or active_count <= 0 or active_count > logical_count:
            raise SSBProtocolError(
                "SSB preparation returned invalid logical or aperture-active BF counts."
            )

        descriptor = {
            "contractVersion": prepare_version,
            "productStatus": "beta",
            "workflow": "direct_ptychography",
            "algorithmVersion": ALGORITHM_VERSION,
            "implementationRevision": self.implementation_revision,
            "source": identity,
            "scanShape": scan_shape,
            "detectorShape": detector_shape,
            "precision": precision,
            "calibrationBinding": calibration_binding,
            "detectorSamplingMilliradians": {
                "row": detector_sampling[0],
                "column": detector_sampling[1],
            },
            "selection": {
                "policy": "full_active",
                "logicalBrightFieldCount": logical_count,
                "activeBrightFieldCount": active_count,
                "detectorBin": 1,
                "scanCrop": None,
            },
        }
        descriptor["selectionSHA256"] = selection_descriptor_sha256(descriptor)
        return descriptor

    def _validate_prepared_selection(
        self,
        request: dict[str, Any],
        identity: dict[str, Any],
        precision: dict[str, Any],
    ) -> None:
        descriptor = request.get("preparedSelection")
        if not isinstance(descriptor, dict):
            raise SSBProtocolError(
                "SSB reconstruction requires a server-prepared selection descriptor."
            )
        if descriptor.get("contractVersion") not in {
            PREPARE_CONTRACT_V0_1,
            PREPARE_CONTRACT_VERSION,
        }:
            raise SSBProtocolError("Unsupported SSB prepared selection version.")
        if descriptor.get("productStatus") != "beta" or descriptor.get("workflow") != (
            "direct_ptychography"
        ):
            raise SSBProtocolError("SSB prepared selection workflow is unsupported.")
        if descriptor.get("algorithmVersion") != ALGORITHM_VERSION:
            raise SSBProtocolError("SSB prepared selection algorithm is stale.")
        if descriptor.get("implementationRevision") != self.implementation_revision:
            raise SSBProtocolError("SSB prepared selection implementation is stale.")
        if descriptor.get("selectionSHA256") != selection_descriptor_sha256(descriptor):
            raise SSBProtocolError("SSB prepared selection digest is invalid.")

        descriptor_source = descriptor.get("source") or {}
        for key in (
            "masterPath",
            "masterSHA256",
            "orderedMemberSHA256",
            "sourceIdentitySHA256",
        ):
            if descriptor_source.get(key) != identity[key]:
                raise SSBProtocolError(
                    f"SSB prepared selection is stale for source field {key}."
                )
        if descriptor.get("scanShape") != request.get("scanShape"):
            raise SSBProtocolError("SSB prepared selection scan shape changed.")
        if descriptor.get("detectorShape") != request.get("detectorShape"):
            raise SSBProtocolError("SSB prepared selection detector shape changed.")
        if descriptor.get("precision") != precision:
            raise SSBProtocolError("SSB prepared selection precision changed.")
        expected_binding = _calibration_binding(
            request.get("calibration") or {}, identity["sourceIdentitySHA256"]
        )
        if descriptor.get("calibrationBinding") != expected_binding:
            raise SSBProtocolError("SSB prepared selection calibration changed.")
        sampling = _calibrated_detector_sampling_mrad(request)
        if descriptor.get("detectorSamplingMilliradians") != {
            "row": sampling[0],
            "column": sampling[1],
        }:
            raise SSBProtocolError("SSB prepared detector sampling changed.")
        if descriptor.get("selection") != request.get("selection"):
            raise SSBProtocolError("SSB prepared scientific selection changed.")

    def reconstruct(self, request: dict[str, Any]) -> dict[str, Any]:
        job_id = str(UUID(str(request["jobID"])))
        generation = int(request["datasetGeneration"])
        key = (job_id, generation)
        gpu = self._requested_device(request)
        with self._lock:
            if key in self._jobs:
                raise SSBProtocolError(
                    "The SSB job ID belongs to the lifecycle endpoint; poll its snapshot."
                )
            if key in self._sync_keys:
                raise SSBProtocolError("The synchronous SSB job ID is already active.")
            self._sync_keys.add(key)
        try:
            with self._device_lock(gpu):
                source, validated_gpu = self._validate_request(request)
                payload, descriptor, result = self._execute_validated(
                    source, validated_gpu, request
                )
                with self._lock:
                    self._payloads[key] = (payload, descriptor)
                return result
        finally:
            with self._lock:
                self._sync_keys.discard(key)

    def _execute_validated(
        self, source: Path, gpu: int | None, request: dict[str, Any]
    ) -> tuple[bytes, dict[str, Any], dict[str, Any]]:
        outcome = self._runner(source, gpu, request)
        return self._validated_result(outcome, gpu, request)

    def _validated_result(
        self,
        outcome: dict[str, Any],
        gpu: int | None,
        request: dict[str, Any],
        *,
        payload_path: str | None = None,
    ) -> tuple[bytes, dict[str, Any], dict[str, Any]]:
        """Validate one backend outcome and bind its complete result identity."""

        phase = _phase_bytes(outcome["phase"])
        shape = [
            int(request["scanShape"]["rows"]),
            int(request["scanShape"]["columns"]),
        ]
        if len(phase) != shape[0] * shape[1] * 4:
            raise SSBProtocolError("SSB phase byte count does not match scan shape.")
        requested_logical = int(request["selection"]["logicalBrightFieldCount"])
        executed_logical = int(outcome["logicalBrightFieldCount"])
        if executed_logical != requested_logical:
            raise SSBProtocolError(
                "SSB logical BF count changed: requested "
                f"{requested_logical}, executed {executed_logical}."
            )
        requested_active = int(request["selection"]["activeBrightFieldCount"])
        executed_active = int(outcome["activeBrightFieldCount"])
        if executed_active != requested_active:
            raise SSBProtocolError(
                "SSB active BF count changed: requested "
                f"{requested_active}, executed {executed_active}."
            )
        requested_sampling = _calibrated_detector_sampling_mrad(request)
        executed_sampling = tuple(
            float(value) for value in outcome["detectorSamplingMilliradians"]
        )
        if len(executed_sampling) != 2 or not np.allclose(
            executed_sampling, requested_sampling, rtol=0.0, atol=1e-9
        ):
            raise SSBProtocolError(
                "SSB detector sampling changed: requested "
                f"{requested_sampling}, executed {executed_sampling}."
            )
        requested_precision = _validated_precision(request)
        executed_precision = outcome["precision"]
        if executed_precision != requested_precision:
            raise SSBProtocolError(
                "SSB executed precision or lossless count audit differs from the request."
            )

        job_id = str(UUID(str(request["jobID"])))
        generation = int(request["datasetGeneration"])
        phase_hash = hashlib.sha256(phase).hexdigest()
        descriptor = {
            "shape": request["scanShape"],
            "dtype": "float32",
            "byteCount": len(phase),
            "sha256": phase_hash,
        }
        if payload_path is None:
            payload_path = f"/api/ssb/jobs/{job_id}/phase"
        calibration = request["calibration"]["resolution"]["calibration"]
        result = {
            "contractVersion": CONTRACT_VERSION,
            "productStatus": "beta",
            "workflow": "direct_ptychography",
            "jobID": job_id,
            "datasetGeneration": generation,
            "source": request["source"],
            "calibration": calibration,
            "selection": request["selection"],
            "preparedSelection": request["preparedSelection"],
            "executedDetectorSamplingMilliradians": {
                "row": executed_sampling[0],
                "column": executed_sampling[1],
            },
            "executedPrecision": executed_precision,
            "executedBrightFieldCounts": {
                "logical": executed_logical,
                "active": executed_active,
            },
            "requestedBackend": request["backend"],
            "executedDevice": {
                "backend": "cuda" if self.backend_kind == "remote_cuda" else "mps",
                "deviceName": self._device_name(gpu),
                "gpuIndex": gpu,
                "driverVersion": outcome.get("driverVersion"),
                "runtimeVersion": outcome.get("runtimeVersion"),
                "implementationRevision": self.implementation_revision,
            },
            "phase": descriptor,
            "phasePayload": {
                "jobID": job_id,
                "datasetGeneration": generation,
                "path": payload_path,
                "descriptor": descriptor,
            },
            "amplitude": None,
            "loss": outcome.get("loss"),
            "timings": outcome["timings"],
        }
        result["provenanceSHA256"] = hashlib.sha256(
            json.dumps(result, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest()
        return phase, descriptor, result

    def payload(self, job_id: str, generation: int) -> tuple[bytes, dict[str, Any]]:
        key = (str(UUID(job_id)), int(generation))
        with self._lock:
            stored = self._payloads.get(key)
            job = self._jobs.get(key)
            if job is not None:
                if job["state"] == "completed" and stored is not None:
                    return stored
                if job["state"] not in _TERMINAL_JOB_STATES:
                    raise SSBPayloadNotReady(
                        "The SSB job has not completed a validated phase payload."
                    )
                raise SSBPayloadUnavailable(
                    f"The terminal SSB job state {job['state']} has no phase payload."
                )
            if stored is not None:
                return stored
            if key in self._sync_keys:
                raise SSBPayloadNotReady(
                    "The synchronous SSB job has not published a phase payload."
                )
        raise SSBProtocolError(
            "No validated SSB phase exists for this job and dataset generation."
        )

    def _resolve_master(self, master_path: str) -> Path:
        master = Path(master_path).expanduser().resolve()
        try:
            master.relative_to(self.data_folder)
        except ValueError as exc:
            raise SSBProtocolError(
                "SSB source is outside the configured data folder."
            ) from exc
        if not master.is_file() or not master.name.endswith("_master.h5"):
            raise SSBProtocolError(f"SSB master file was not found: {master}")
        return master

    def _validate_request(self, request: dict[str, Any]) -> tuple[Path, int | None]:
        if self.implementation_revision == "unrecorded":
            raise SSBProtocolError(
                "The SSB service implementation revision is not recorded; restart it with an exact revision."
            )
        if request.get("contractVersion") != CONTRACT_VERSION:
            raise SSBProtocolError("Unsupported SSB contract version.")
        if request.get("algorithmVersion") != ALGORITHM_VERSION:
            raise SSBProtocolError("Unsupported SSB algorithm version.")
        gpu = self._requested_device(request)
        calibration = request.get("calibration") or {}
        resolution = calibration.get("resolution") or {}
        if resolution.get("state") != "valid":
            raise SSBProtocolError("SSB calibration is not resolved.")
        selection = request.get("selection") or {}
        if (
            selection.get("policy") != "full_active"
            or int(selection.get("detectorBin", 0)) != 1
            or selection.get("scanCrop") is not None
        ):
            raise SSBProtocolError(
                "Native SSB requires full_active BF, no crop, and detector bin 1."
            )
        precision = _validated_precision(request)
        _calibrated_detector_sampling_mrad(request)
        source = request.get("source") or {}
        master = self._resolve_master(str(source.get("masterPath", "")))
        identity = self.source_identity(str(master))
        for key in ("masterSHA256", "orderedMemberSHA256", "sourceIdentitySHA256"):
            if source.get(key) != identity[key]:
                raise SSBProtocolError(f"SSB source identity mismatch for {key}.")
        if calibration.get("sourceIdentitySHA256") != identity["sourceIdentitySHA256"]:
            raise SSBProtocolError(
                "SSB calibration belongs to a different source identity."
            )
        inspection = self._source_inspector(
            str(master),
            scan_shape=(
                int(request["scanShape"]["rows"]),
                int(request["scanShape"]["columns"]),
            ),
        )
        if not inspection.ready:
            raise SSBProtocolError(
                f"SSB source is not ready: {inspection.reason} {inspection.action}".strip()
            )
        if inspection.dtype != precision["nativeSourceDType"]:
            raise SSBProtocolError(
                "SSB native source dtype mismatch: request declares "
                f"{precision['nativeSourceDType']}, source is {inspection.dtype}."
            )
        detector_shape = (
            int(request["detectorShape"]["rows"]),
            int(request["detectorShape"]["columns"]),
        )
        if inspection.detector_shape != detector_shape:
            raise SSBProtocolError(
                "SSB detector shape does not match the inspected native source."
            )
        self._validate_prepared_selection(request, identity, precision)
        return master, gpu

    def _requested_device(self, request: dict[str, Any]) -> int | None:
        backend = request.get("backend") or {}
        if backend.get("kind") != self.backend_kind:
            raise SSBProtocolError(
                f"This endpoint requires explicit {self.backend_kind} selection."
            )
        gpu = backend.get("gpu_index")
        if self.backend_kind == "remote_cuda":
            if gpu is None or int(gpu) not in self._available_gpus():
                raise SSBProtocolError(
                    "The explicitly selected CUDA GPU is unavailable."
                )
            return int(gpu)
        if gpu is not None:
            raise SSBProtocolError("local_mps does not accept a CUDA GPU index.")
        return None

    @staticmethod
    def _session_open_kwargs(request: dict[str, Any]) -> dict[str, Any]:
        values = request["calibration"]["resolution"]["calibration"]["calibration"]
        controls = SSBProtocolService._initial_controls(request)
        return {
            "dtype": request["precision"]["workingSourceDType"],
            "voltage_kV": float(values["accelerationVoltageKilovolts"]["value"]),
            "semiangle_mrad": float(
                values["convergenceSemiangleMilliradians"]["value"]
            ),
            "scan_sampling_A": (
                float(values["scanSamplingRowAngstrom"]["value"]),
                float(values["scanSamplingColumnAngstrom"]["value"]),
            ),
            "scan_shape": (
                int(request["scanShape"]["rows"]),
                int(request["scanShape"]["columns"]),
            ),
            "det_sampling": _calibrated_detector_sampling_mrad(request),
            "aberrations": {
                name: controls[name] for name in ("C10", "C12", "phi12")
            },
            "rotation_angle_deg": controls["scanRotation"],
        }

    @staticmethod
    def _cuda_device_context(gpu: int | None):
        import cupy as cp

        if gpu is None:
            raise SSBProtocolError("remote_cuda requires an explicit GPU index.")
        return cp.cuda.Device(gpu)

    @staticmethod
    def _open_cuda_session(
        source: Path, gpu: int | None, request: dict[str, Any]
    ) -> tuple[Any, Any]:
        import cupy as cp

        from quantem.gpu import SSB

        if gpu is None:
            raise SSBProtocolError("remote_cuda requires an explicit GPU index.")
        with cp.cuda.Device(gpu):
            context = SSB.open(
                str(source), backend="cuda", **SSBProtocolService._session_open_kwargs(request)
            )
            try:
                return context, context.__enter__()
            except Exception:
                context.__exit__(None, None, None)
                raise

    @staticmethod
    def _open_mps_session(
        source: Path, gpu: int | None, request: dict[str, Any]
    ) -> tuple[Any, Any]:
        from quantem.gpu import SSB

        if gpu is not None:
            raise SSBProtocolError("local_mps cannot execute on a CUDA GPU index.")
        context = SSB.open(
            str(source), backend="mps", **SSBProtocolService._session_open_kwargs(request)
        )
        try:
            return context, context.__enter__()
        except Exception:
            context.__exit__(None, None, None)
            raise

    @staticmethod
    def _prepare_cuda(
        source: Path, gpu: int | None, request: dict[str, Any]
    ) -> dict[str, Any]:
        import cupy as cp

        from quantem.gpu import SSB

        if gpu is None:
            raise SSBProtocolError("remote_cuda requires an explicit GPU index.")
        values = request["calibration"]["resolution"]["calibration"]["calibration"]
        detector_sampling_mrad = _calibrated_detector_sampling_mrad(request)
        aberrations = {
            "C10": float(values["c10Nanometers"]["value"]),
            "C12": float(values["c12Nanometers"]["value"]),
            "phi12": float(values["phi12Radians"]["value"]),
        }
        with cp.cuda.Device(gpu):
            session_context = SSB.open(
                str(source),
                backend="cuda",
                dtype=request["precision"]["workingSourceDType"],
                voltage_kV=float(values["accelerationVoltageKilovolts"]["value"]),
                semiangle_mrad=float(
                    values["convergenceSemiangleMilliradians"]["value"]
                ),
                scan_sampling_A=(
                    float(values["scanSamplingRowAngstrom"]["value"]),
                    float(values["scanSamplingColumnAngstrom"]["value"]),
                ),
                scan_shape=(
                    int(request["scanShape"]["rows"]),
                    int(request["scanShape"]["columns"]),
                ),
                det_sampling=detector_sampling_mrad,
                aberrations=aberrations,
                rotation_angle_deg=float(values["scanRotationDegrees"]["value"]),
            )
            with session_context as session:
                state = session.browser_state()
                precision = _executed_precision(
                    request, working_dtype=session.source_dtype
                )
        return {
            "logicalBrightFieldCount": state.num_bf,
            "activeBrightFieldCount": state.active_num_bf,
            "detectorSamplingMilliradians": tuple(
                float(value) * 1e3 for value in state.angular_sampling_rad
            ),
            "precision": precision,
        }

    @staticmethod
    def _prepare_mps(
        source: Path, gpu: int | None, request: dict[str, Any]
    ) -> dict[str, Any]:
        from quantem.gpu import SSB

        if gpu is not None:
            raise SSBProtocolError("local_mps cannot execute on a CUDA GPU index.")
        values = request["calibration"]["resolution"]["calibration"]["calibration"]
        detector_sampling_mrad = _calibrated_detector_sampling_mrad(request)
        aberrations = {
            "C10": float(values["c10Nanometers"]["value"]),
            "C12": float(values["c12Nanometers"]["value"]),
            "phi12": float(values["phi12Radians"]["value"]),
        }
        session_context = SSB.open(
            str(source),
            backend="mps",
            dtype=request["precision"]["workingSourceDType"],
            voltage_kV=float(values["accelerationVoltageKilovolts"]["value"]),
            semiangle_mrad=float(values["convergenceSemiangleMilliradians"]["value"]),
            scan_sampling_A=(
                float(values["scanSamplingRowAngstrom"]["value"]),
                float(values["scanSamplingColumnAngstrom"]["value"]),
            ),
            scan_shape=(
                int(request["scanShape"]["rows"]),
                int(request["scanShape"]["columns"]),
            ),
            det_sampling=detector_sampling_mrad,
            aberrations=aberrations,
            rotation_angle_deg=float(values["scanRotationDegrees"]["value"]),
        )
        with session_context as session:
            state = session.browser_state()
            precision = _executed_precision(request, working_dtype=session.source_dtype)
        return {
            "logicalBrightFieldCount": state.num_bf,
            "activeBrightFieldCount": state.active_num_bf,
            "detectorSamplingMilliradians": tuple(
                float(value) * 1e3 for value in state.angular_sampling_rad
            ),
            "precision": precision,
        }

    @staticmethod
    def _run_cuda(
        source: Path, gpu: int | None, request: dict[str, Any]
    ) -> dict[str, Any]:
        import cupy as cp

        from quantem.gpu import SSB

        if gpu is None:
            raise SSBProtocolError("remote_cuda requires an explicit GPU index.")

        values = request["calibration"]["resolution"]["calibration"]["calibration"]
        detector_sampling_mrad = _calibrated_detector_sampling_mrad(request)
        aberrations = {
            "C10": float(values["c10Nanometers"]["value"]),
            "C12": float(values["c12Nanometers"]["value"]),
            "phi12": float(values["phi12Radians"]["value"]),
        }
        with cp.cuda.Device(gpu):
            opened = time.perf_counter()
            session_context = SSB.open(
                str(source),
                backend="cuda",
                dtype=request["precision"]["workingSourceDType"],
                voltage_kV=float(values["accelerationVoltageKilovolts"]["value"]),
                semiangle_mrad=float(
                    values["convergenceSemiangleMilliradians"]["value"]
                ),
                scan_sampling_A=(
                    float(values["scanSamplingRowAngstrom"]["value"]),
                    float(values["scanSamplingColumnAngstrom"]["value"]),
                ),
                scan_shape=(
                    int(request["scanShape"]["rows"]),
                    int(request["scanShape"]["columns"]),
                ),
                det_sampling=detector_sampling_mrad,
                aberrations=aberrations,
                rotation_angle_deg=float(values["scanRotationDegrees"]["value"]),
            )
            with session_context as session:
                open_seconds = time.perf_counter() - opened
                started = time.perf_counter()
                result = session.reconstruct(
                    aberrations, compute_loss=bool(request["computeLoss"])
                )
                first_seconds = time.perf_counter() - started
                warm_seconds = None
                if request.get("measureWarm", False):
                    started = time.perf_counter()
                    result = session.reconstruct(
                        aberrations,
                        compute_loss=bool(request["computeLoss"]),
                        force=True,
                    )
                    warm_seconds = time.perf_counter() - started
                encoded = time.perf_counter()
                phase = cp.asnumpy(result.phase).astype(np.float32, copy=False)
                brightfield_state = session.browser_state()
                precision = _executed_precision(
                    request,
                    working_dtype=session.source_dtype,
                )
                encode_seconds = time.perf_counter() - encoded
                source_load_seconds = session.source_load_seconds
            runtime = cp.cuda.runtime.runtimeGetVersion()
            driver = cp.cuda.runtime.driverGetVersion()
        return {
            "phase": phase,
            "logicalBrightFieldCount": brightfield_state.num_bf,
            "activeBrightFieldCount": brightfield_state.active_num_bf,
            "detectorSamplingMilliradians": tuple(
                float(value) * 1e3 for value in brightfield_state.angular_sampling_rad
            ),
            "precision": precision,
            "loss": result.loss,
            "driverVersion": str(driver),
            "runtimeVersion": str(runtime),
            "implementationRevision": "live4dstem-ssb-protocol-v0.1",
            "timings": {
                "sourceReadSeconds": source_load_seconds,
                "sourceDecodeSeconds": None,
                "gQKConstructionSeconds": None,
                "kernelSeconds": None,
                "firstReconstructSeconds": first_seconds,
                "warmReconstructSeconds": warm_seconds,
                "resultEncodeSeconds": encode_seconds,
                "sshRequestToFirstByteSeconds": None,
                "transferSeconds": None,
                "clientDecodeSeconds": None,
                "paintSeconds": None,
                "inputToPaintSeconds": None,
                "sessionOpenSeconds": open_seconds,
            },
        }

    @staticmethod
    def _run_mps(
        source: Path, gpu: int | None, request: dict[str, Any]
    ) -> dict[str, Any]:
        import platform

        import numpy as np

        from quantem.gpu import SSB

        if gpu is not None:
            raise SSBProtocolError("local_mps cannot execute on a CUDA GPU index.")
        values = request["calibration"]["resolution"]["calibration"]["calibration"]
        detector_sampling_mrad = _calibrated_detector_sampling_mrad(request)
        aberrations = {
            "C10": float(values["c10Nanometers"]["value"]),
            "C12": float(values["c12Nanometers"]["value"]),
            "phi12": float(values["phi12Radians"]["value"]),
        }
        opened = time.perf_counter()
        session_context = SSB.open(
            str(source),
            backend="mps",
            dtype=request["precision"]["workingSourceDType"],
            voltage_kV=float(values["accelerationVoltageKilovolts"]["value"]),
            semiangle_mrad=float(values["convergenceSemiangleMilliradians"]["value"]),
            scan_sampling_A=(
                float(values["scanSamplingRowAngstrom"]["value"]),
                float(values["scanSamplingColumnAngstrom"]["value"]),
            ),
            scan_shape=(
                int(request["scanShape"]["rows"]),
                int(request["scanShape"]["columns"]),
            ),
            det_sampling=detector_sampling_mrad,
            aberrations=aberrations,
            rotation_angle_deg=float(values["scanRotationDegrees"]["value"]),
        )
        with session_context as session:
            open_seconds = time.perf_counter() - opened
            started = time.perf_counter()
            result = session.reconstruct(
                aberrations, compute_loss=bool(request["computeLoss"])
            )
            first_seconds = time.perf_counter() - started
            warm_seconds = None
            if request.get("measureWarm", False):
                started = time.perf_counter()
                result = session.reconstruct(
                    aberrations,
                    compute_loss=bool(request["computeLoss"]),
                    force=True,
                )
                warm_seconds = time.perf_counter() - started
            encoded = time.perf_counter()
            phase = np.asarray(result.phase, dtype=np.float32)
            brightfield_state = session.browser_state()
            precision = _executed_precision(
                request,
                working_dtype=session.source_dtype,
            )
            encode_seconds = time.perf_counter() - encoded
            source_load_seconds = session.source_load_seconds
        return {
            "phase": phase,
            "logicalBrightFieldCount": brightfield_state.num_bf,
            "activeBrightFieldCount": brightfield_state.active_num_bf,
            "detectorSamplingMilliradians": tuple(
                float(value) * 1e3 for value in brightfield_state.angular_sampling_rad
            ),
            "precision": precision,
            "loss": result.loss,
            "driverVersion": platform.mac_ver()[0],
            "runtimeVersion": version("mlx"),
            "implementationRevision": "live4dstem-ssb-protocol-v0.1",
            "timings": {
                "sourceReadSeconds": source_load_seconds,
                "sourceDecodeSeconds": None,
                "gQKConstructionSeconds": None,
                "kernelSeconds": None,
                "firstReconstructSeconds": first_seconds,
                "warmReconstructSeconds": warm_seconds,
                "resultEncodeSeconds": encode_seconds,
                "sshRequestToFirstByteSeconds": None,
                "transferSeconds": None,
                "clientDecodeSeconds": None,
                "paintSeconds": None,
                "inputToPaintSeconds": None,
                "sessionOpenSeconds": open_seconds,
            },
        }
