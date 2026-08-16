"""Versioned SSB request/result boundary for native private-loopback clients."""

from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import threading
import time
from typing import Any, Callable
from uuid import UUID

import numpy as np


CONTRACT_VERSION = "live4dstem.ssb/v0.1"
ALGORITHM_VERSION = "quantem.gpu.SSB/v0.1"
JOBS_CONTRACT_VERSION = "live4dstem.ssb.jobs/v0.1"
_TERMINAL_JOB_STATES = {"completed", "cancelled", "failed", "expired"}


class SSBProtocolError(ValueError):
    """One actionable request or result contract failure."""


class SSBPayloadNotReady(SSBProtocolError):
    """A validated job exists, but it has not published a phase payload."""


class SSBPayloadUnavailable(SSBProtocolError):
    """A terminal job has no phase payload."""


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


class SSBProtocolService:
    """Validate SSB jobs and retain only validated phase payloads in memory."""

    def __init__(
        self,
        data_folder: str | Path,
        *,
        available_gpus: Callable[[], list[int]],
        device_name: Callable[[int | None], str],
        runner: Callable[[Path, int | None, dict[str, Any]], dict[str, Any]] | None = None,
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
        self._lock = threading.Lock()
        self._payloads: dict[tuple[str, int], tuple[bytes, dict[str, Any]]] = {}
        self._jobs: dict[tuple[str, int], dict[str, Any]] = {}
        self._sync_keys: set[tuple[str, int]] = set()
        self._device_locks: dict[tuple[str, int | None], threading.Lock] = {}

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
        return self.advertised_capability(
            self.backend_kind,
            self.implementation_revision,
            device,
            gpu,
            unavailable_reason,
        )

    @staticmethod
    def request_sha256(request: dict[str, Any]) -> str:
        """Hash one canonical scientific request for idempotent submission."""

        canonical = json.dumps(request, separators=(",", ":"), sort_keys=True).encode()
        return hashlib.sha256(canonical).hexdigest()

    def submit(self, request: dict[str, Any]) -> dict[str, Any]:
        """Accept one idempotent background job and return its current snapshot."""

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

    def _run_submitted_job(
        self, key: tuple[str, int], request: dict[str, Any]
    ) -> None:
        try:
            if self._cancelled_at_boundary(key):
                return
            gpu = self._requested_device(request)
            with self._device_lock(gpu):
                if self._cancelled_at_boundary(key):
                    return
                self._advance(key, "validating")
                source, validated_gpu = self._validate_request(request)
                if self._cancelled_at_boundary(key):
                    return
                self._advance(key, "reconstructing_first")
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

    def _advance(self, key: tuple[str, int], state: str) -> None:
        with self._lock:
            self._advance_locked(self._jobs[key], state)

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

    def source_identity(self, master_path: str) -> dict[str, Any]:
        master = self._resolve_master(master_path)
        stem = master.name.removesuffix("_master.h5")
        members = sorted(master.parent.glob(f"{stem}_data_*.h5"))
        master_hash = _sha256(master)
        member_hashes = [_sha256(path) for path in members]
        return {
            "masterPath": str(master),
            "masterSHA256": master_hash,
            "orderedMemberSHA256": member_hashes,
            "sourceIdentitySHA256": _source_identity_sha256(master_hash, member_hashes),
        }

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

        job_id = str(UUID(str(request["jobID"])))
        generation = int(request["datasetGeneration"])
        phase_hash = hashlib.sha256(phase).hexdigest()
        descriptor = {
            "shape": request["scanShape"],
            "dtype": "float32",
            "byteCount": len(phase),
            "sha256": phase_hash,
        }
        payload_path = f"/api/ssb/jobs/{job_id}/phase"
        calibration = request["calibration"]["resolution"]["calibration"]
        result = {
            "contractVersion": CONTRACT_VERSION,
            "jobID": job_id,
            "datasetGeneration": generation,
            "source": request["source"],
            "calibration": calibration,
            "selection": request["selection"],
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
            raise SSBProtocolError("SSB source is outside the configured data folder.") from exc
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
            raise SSBProtocolError("Native SSB requires full_active BF, no crop, and detector bin 1.")
        precision = request.get("precision") or {}
        if precision != {
            "sourceDType": "uint8",
            "realDType": "float32",
            "complexDType": "complex64",
        }:
            raise SSBProtocolError("Unsupported SSB source or arithmetic precision.")
        source = request.get("source") or {}
        master = self._resolve_master(str(source.get("masterPath", "")))
        identity = self.source_identity(str(master))
        for key in ("masterSHA256", "orderedMemberSHA256", "sourceIdentitySHA256"):
            if source.get(key) != identity[key]:
                raise SSBProtocolError(f"SSB source identity mismatch for {key}.")
        if calibration.get("sourceIdentitySHA256") != identity["sourceIdentitySHA256"]:
            raise SSBProtocolError("SSB calibration belongs to a different source identity.")
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
                raise SSBProtocolError("The explicitly selected CUDA GPU is unavailable.")
            return int(gpu)
        if gpu is not None:
            raise SSBProtocolError("local_mps does not accept a CUDA GPU index.")
        return None

    @staticmethod
    def _run_cuda(source: Path, gpu: int | None, request: dict[str, Any]) -> dict[str, Any]:
        import cupy as cp
        from quantem.gpu import SSB

        if gpu is None:
            raise SSBProtocolError("remote_cuda requires an explicit GPU index.")

        values = request["calibration"]["resolution"]["calibration"]["calibration"]
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
                dtype="auto",
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
                aberrations=aberrations,
                rotation_angle_deg=float(values["scanRotationDegrees"]["value"]),
            )
            with session_context as session:
                open_seconds = time.perf_counter() - opened
                if session.source_dtype not in {"uint8", "u1"}:
                    raise SSBProtocolError(
                        "The requested uint8 SSB path was not proven lossless by quantem.gpu."
                    )
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
                encode_seconds = time.perf_counter() - encoded
                source_load_seconds = session.source_load_seconds
            runtime = cp.cuda.runtime.runtimeGetVersion()
            driver = cp.cuda.runtime.driverGetVersion()
        return {
            "phase": phase,
            "logicalBrightFieldCount": brightfield_state.num_bf,
            "activeBrightFieldCount": brightfield_state.active_num_bf,
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
    def _run_mps(source: Path, gpu: int | None, request: dict[str, Any]) -> dict[str, Any]:
        import platform

        import numpy as np

        from quantem.gpu import SSB

        if gpu is not None:
            raise SSBProtocolError("local_mps cannot execute on a CUDA GPU index.")
        values = request["calibration"]["resolution"]["calibration"]["calibration"]
        aberrations = {
            "C10": float(values["c10Nanometers"]["value"]),
            "C12": float(values["c12Nanometers"]["value"]),
            "phi12": float(values["phi12Radians"]["value"]),
        }
        opened = time.perf_counter()
        session_context = SSB.open(
            str(source),
            backend="mps",
            dtype="auto",
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
            aberrations=aberrations,
            rotation_angle_deg=float(values["scanRotationDegrees"]["value"]),
        )
        with session_context as session:
            open_seconds = time.perf_counter() - opened
            if session.source_dtype not in {"uint8", "u1"}:
                raise SSBProtocolError(
                    "The requested uint8 SSB path was not proven lossless by quantem.gpu."
                )
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
            encode_seconds = time.perf_counter() - encoded
            source_load_seconds = session.source_load_seconds
        return {
            "phase": phase,
            "logicalBrightFieldCount": brightfield_state.num_bf,
            "activeBrightFieldCount": brightfield_state.active_num_bf,
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
