"""Versioned SSB request/result boundary for native private-loopback clients."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import threading
import time
from typing import Any, Callable
from uuid import UUID

import numpy as np


CONTRACT_VERSION = "live4dstem.ssb/v0.1"
ALGORITHM_VERSION = "quantem.gpu.SSB/v0.1"


class SSBProtocolError(ValueError):
    """One actionable request or result contract failure."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
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
        device_name: Callable[[int], str],
        runner: Callable[[Path, int, dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.data_folder = Path(data_folder).expanduser().resolve()
        self._available_gpus = available_gpus
        self._device_name = device_name
        self._runner = runner or self._run_cuda
        self._lock = threading.Lock()
        self._payloads: dict[tuple[str, int], tuple[bytes, dict[str, Any]]] = {}

    @staticmethod
    def advertised_capability() -> dict[str, Any]:
        return {
            "name": "ssb",
            "contractVersion": CONTRACT_VERSION,
            "algorithmVersion": ALGORITHM_VERSION,
            "resultPayload": "job_generation_endpoint",
            "stageTimingAvailability": {
                "sourceLoad": True,
                "firstReconstruct": True,
                "warmReconstruct": True,
                "gQKConstruction": False,
                "kernel": False,
            },
        }

    def source_identity(self, master_path: str) -> dict[str, Any]:
        master = self._resolve_master(master_path)
        stem = master.name.removesuffix("_master.h5")
        members = sorted(master.parent.glob(f"{stem}_data_*.h5"))
        master_hash = _sha256(master)
        member_hashes = [_sha256(path) for path in members]
        canonical = json.dumps(
            {
                "master": master_hash,
                "members": member_hashes,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        return {
            "masterPath": str(master),
            "masterSHA256": master_hash,
            "orderedMemberSHA256": member_hashes,
            "sourceIdentitySHA256": hashlib.sha256(canonical).hexdigest(),
        }

    def reconstruct(self, request: dict[str, Any]) -> dict[str, Any]:
        source, gpu = self._validate_request(request)
        outcome = self._runner(source, gpu, request)
        phase = _phase_bytes(outcome["phase"])
        shape = [
            int(request["scanShape"]["rows"]),
            int(request["scanShape"]["columns"]),
        ]
        if len(phase) != shape[0] * shape[1] * 4:
            raise SSBProtocolError("SSB phase byte count does not match scan shape.")
        requested_active = int(request["selection"]["activeBrightFieldCount"])
        if int(outcome["activeBrightFieldCount"]) != requested_active:
            raise SSBProtocolError(
                "SSB active BF count changed: requested "
                f"{requested_active}, executed {outcome['activeBrightFieldCount']}."
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
        with self._lock:
            self._payloads[(job_id, generation)] = (phase, descriptor)

        calibration = request["calibration"]["resolution"]["calibration"]
        result = {
            "contractVersion": CONTRACT_VERSION,
            "jobID": job_id,
            "datasetGeneration": generation,
            "source": request["source"],
            "calibration": calibration,
            "selection": request["selection"],
            "requestedBackend": request["backend"],
            "executedDevice": {
                "backend": "cuda",
                "deviceName": self._device_name(gpu),
                "gpuIndex": gpu,
                "driverVersion": outcome.get("driverVersion"),
                "runtimeVersion": outcome.get("runtimeVersion"),
                "implementationRevision": outcome["implementationRevision"],
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
        return result

    def payload(self, job_id: str, generation: int) -> tuple[bytes, dict[str, Any]]:
        key = (str(UUID(job_id)), int(generation))
        with self._lock:
            stored = self._payloads.get(key)
        if stored is None:
            raise SSBProtocolError(
                "No validated SSB phase exists for this job and dataset generation."
            )
        return stored

    def _resolve_master(self, master_path: str) -> Path:
        master = Path(master_path).expanduser().resolve()
        try:
            master.relative_to(self.data_folder)
        except ValueError as exc:
            raise SSBProtocolError("SSB source is outside the configured data folder.") from exc
        if not master.is_file() or not master.name.endswith("_master.h5"):
            raise SSBProtocolError(f"SSB master file was not found: {master}")
        return master

    def _validate_request(self, request: dict[str, Any]) -> tuple[Path, int]:
        if request.get("contractVersion") != CONTRACT_VERSION:
            raise SSBProtocolError("Unsupported SSB contract version.")
        if request.get("algorithmVersion") != ALGORITHM_VERSION:
            raise SSBProtocolError("Unsupported SSB algorithm version.")
        backend = request.get("backend") or {}
        if backend.get("kind") != "remote_cuda":
            raise SSBProtocolError("This endpoint requires explicit remote_cuda selection.")
        gpu = backend.get("gpu_index")
        if gpu is None or int(gpu) not in self._available_gpus():
            raise SSBProtocolError("The explicitly selected CUDA GPU is unavailable.")
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
        return master, int(gpu)

    @staticmethod
    def _run_cuda(source: Path, gpu: int, request: dict[str, Any]) -> dict[str, Any]:
        import cupy as cp
        from quantem.gpu import SSB

        values = request["calibration"]["resolution"]["calibration"]["calibration"]
        aberrations = {
            "C10": float(values["c10Nanometers"]["value"]),
            "C12": float(values["c12Nanometers"]["value"]),
            "phi12": float(values["phi12Radians"]["value"]),
        }
        with cp.cuda.Device(gpu):
            opened = time.perf_counter()
            session = SSB.open(
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
            open_seconds = time.perf_counter() - opened
            if session.source_dtype not in {"uint8", "u1"}:
                raise SSBProtocolError(
                    "The requested uint8 SSB path was not proven lossless by quantem.gpu."
                )
            started = time.perf_counter()
            result = session.reconstruct(aberrations, compute_loss=bool(request["computeLoss"]))
            first_seconds = time.perf_counter() - started
            warm_seconds = None
            if request.get("measureWarm", False):
                started = time.perf_counter()
                result = session.reconstruct(
                    aberrations, compute_loss=bool(request["computeLoss"]), force=True
                )
                warm_seconds = time.perf_counter() - started
            encoded = time.perf_counter()
            phase = cp.asnumpy(result.phase).astype(np.float32, copy=False)
            encode_seconds = time.perf_counter() - encoded
            runtime = cp.cuda.runtime.runtimeGetVersion()
            driver = cp.cuda.runtime.driverGetVersion()
        return {
            "phase": phase,
            "activeBrightFieldCount": result.num_bf,
            "loss": result.loss,
            "driverVersion": str(driver),
            "runtimeVersion": str(runtime),
            "implementationRevision": "live4dstem-ssb-protocol-v0.1",
            "timings": {
                "sourceReadSeconds": session.source_load_seconds,
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
