"""Versioned remote MAPED service for native Live4DSTEM clients.

The service owns protocol validation, source identity, small scientific-array
payloads, cache provenance, and worker lifecycle.  The MAPED algorithm remains
in :mod:`quantem.diffraction`; production runs import that implementation only
inside an isolated CUDA worker process.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import threading
import time
from typing import Any
from uuid import UUID, uuid4

import numpy as np

from .ssb_api import _source_identity_sha256


CONTRACT_VERSION = "live4dstem.maped.request/v0.1"
CACHE_VERSION = "live4dstem.maped.cache/v0.1"
ALGORITHM_VERSION = "quantem.gpu.MAPED/v0.1"
QUANTEM_ALGORITHM_REVISION = "58eb7dad1067a9e76b00212c8669616d13a17c59"
SUPPORTED_TILT_COUNTS = (2, 3, 5, 7)
VALIDATED_TILT_COUNTS = (7,)

_TILT_PATTERN = re.compile(
    r"(?:^|_)(-?\d+(?:\.\d+)?)x_(-?\d+(?:\.\d+)?)y(?:_|$)"
)
_CANONICAL_TILT_ORDER = (
    (-17.0, 0.0),
    (-8.5, -14.72),
    (-8.5, 14.72),
    (0.0, 0.0),
    (17.0, 0.0),
    (8.5, -14.72),
    (8.5, 14.72),
)
_VALIDATION_KINDS = {"reference_parity", "integrity_only"}
_RUN_STAGES = {"inventory", "load", "registration", "merge", "save", "ready"}
_MAX_PAYLOADS = 64


class MAPEDProtocolError(ValueError):
    """An actionable MAPED request, execution, or cache failure."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "invalidCollection",
        status_code: int = 409,
        recovery: str = "Correct the request and try again.",
        stage: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.recovery = recovery
        self.stage = stage


class _RunCancelled(Exception):
    pass


@dataclass
class _RunRecord:
    run_id: str
    request: dict[str, Any]
    working_directory: Path
    events: list[dict[str, Any]] = field(default_factory=list)
    condition: threading.Condition = field(default_factory=threading.Condition)
    cancel_requested: threading.Event = field(default_factory=threading.Event)
    terminal: bool = False
    thread: threading.Thread | None = None


@dataclass
class _DeviceState:
    condition: threading.Condition = field(default_factory=threading.Condition)
    active_owner: str | None = None
    waiting_owners: list[str] = field(default_factory=list)


class _DeviceArbiter:
    """Serialize MAPED CUDA ownership independently for each device."""

    def __init__(self) -> None:
        self._states_lock = threading.Lock()
        self._states: dict[int, _DeviceState] = {}

    def acquire_run(
        self,
        gpu: int,
        owner: str,
        cancel_requested: threading.Event,
        on_queued: Callable[[], None],
    ) -> bool:
        state = self._state(gpu)
        with state.condition:
            state.waiting_owners.append(owner)
            on_queued()
            while state.active_owner is not None or state.waiting_owners[0] != owner:
                if cancel_requested.is_set():
                    state.waiting_owners.remove(owner)
                    state.condition.notify_all()
                    return False
                state.condition.wait()
            if cancel_requested.is_set():
                state.waiting_owners.remove(owner)
                state.condition.notify_all()
                return False
            state.waiting_owners.pop(0)
            state.active_owner = owner
            return True

    def try_acquire_interactive(self, gpu: int, owner: str) -> bool:
        state = self._state(gpu)
        with state.condition:
            if state.active_owner is not None or state.waiting_owners:
                return False
            state.active_owner = owner
            return True

    def release(self, gpu: int, owner: str) -> None:
        state = self._state(gpu)
        with state.condition:
            if state.active_owner != owner:
                raise RuntimeError(
                    f"MAPED CUDA GPU {gpu} is not owned by {owner}."
                )
            state.active_owner = None
            state.condition.notify_all()

    def wake(self, gpu: int) -> None:
        state = self._state(gpu)
        with state.condition:
            state.condition.notify_all()

    def _state(self, gpu: int) -> _DeviceState:
        with self._states_lock:
            return self._states.setdefault(gpu, _DeviceState())


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _identity_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _runtime_implementation_revision() -> str:
    override = os.environ.get("QUANTEM_GPU_IMPLEMENTATION_REVISION")
    if override:
        return override
    revision = _git_revision(Path(__file__).resolve().parent)
    return revision or "quantem.gpu-package"


def _tilt_coordinate(name: str) -> dict[str, float] | None:
    match = _TILT_PATTERN.search(name)
    if match is None:
        return None
    return {"xDegrees": float(match.group(1)), "yDegrees": float(match.group(2))}


def _tilt_key(coordinate: dict[str, Any]) -> tuple[int, int, int]:
    pair = (round(float(coordinate["xDegrees"]), 2), round(float(coordinate["yDegrees"]), 2))
    try:
        rank = _CANONICAL_TILT_ORDER.index(pair)
    except ValueError:
        rank = len(_CANONICAL_TILT_ORDER)
    return rank, int(round(pair[0] * 100)), int(round(pair[1] * 100))


def _source_paths(master: Path) -> list[Path]:
    """Return the master and actual HDF5 external members in stable order."""
    members: list[Path] = []
    try:
        import h5py

        with h5py.File(master, "r") as handle:
            group = handle.get("entry/data")
            if group is not None:
                for name in sorted(group):
                    link = group.get(name, getlink=True)
                    if isinstance(link, h5py.ExternalLink):
                        member = Path(link.filename)
                        if not member.is_absolute():
                            member = master.parent / member
                        members.append(member.resolve())
    except (OSError, KeyError, TypeError, ValueError):
        stem = master.name.removesuffix("_master.h5")
        members.extend(sorted(master.parent.glob(f"{stem}_data_*.h5")))
    return [master.resolve(), *dict.fromkeys(members)]


def _source_stat(master: Path) -> dict[str, Any]:
    records = []
    for path in _source_paths(master):
        try:
            stat = path.stat()
            records.append(
                {
                    "path": str(path),
                    "byteCount": int(stat.st_size),
                    "modificationNanoseconds": int(stat.st_mtime_ns),
                }
            )
        except OSError:
            records.append({"path": str(path), "missing": True})
    return {
        "url": str(master.resolve()),
        "masterPath": str(master.resolve()),
        "files": records,
        "byteCount": sum(int(item.get("byteCount", 0)) for item in records),
        "modificationNanoseconds": max(
            (int(item.get("modificationNanoseconds", 0)) for item in records),
            default=0,
        ),
        "fileSetFingerprint": _identity_hash(records),
    }


def _calibration_envelope(
    metadata: dict[str, Any], source_identity_sha256: str
) -> dict[str, Any]:
    del metadata
    return {
        "schemaVersion": "live4dstem.dataset/v0.1",
        "sourceIdentitySHA256": source_identity_sha256,
        "resolution": {
            "state": "missing",
            "reason": (
                "No complete physical calibration was supplied by the source. "
                "Enter or resolve calibration in dataset metadata before a "
                "calibrated reconstruction."
            ),
        },
    }


def _validate_calibration_identity(
    calibration: dict[str, Any] | None, source_identity_sha256: str
) -> None:
    if calibration is None:
        return
    if not isinstance(calibration, dict):
        raise MAPEDProtocolError("MAPED calibration envelope must be an object or null.")
    if calibration.get("schemaVersion") != "live4dstem.dataset/v0.1":
        raise MAPEDProtocolError("Unsupported MAPED calibration schema version.")
    if calibration.get("sourceIdentitySHA256") != source_identity_sha256:
        raise MAPEDProtocolError("MAPED calibration belongs to a different source identity.")
    resolution = calibration.get("resolution")
    if not isinstance(resolution, dict):
        raise MAPEDProtocolError("MAPED calibration requires a resolution envelope.")
    state = resolution.get("state")
    if state not in {"missing", "invalid", "unresolved", "valid"}:
        raise MAPEDProtocolError(
            "MAPED calibration state must be missing, invalid, unresolved, or valid."
        )
    if state in {"missing", "invalid"}:
        if not str(resolution.get("reason", "")).strip():
            raise MAPEDProtocolError(
                f"MAPED {state} calibration requires a concrete reason."
            )
        return
    if state == "unresolved":
        candidates = resolution.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise MAPEDProtocolError(
                "Unresolved MAPED calibration requires candidate evidence."
            )
    else:
        candidates = [resolution.get("calibration")]
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise MAPEDProtocolError("MAPED calibration candidate is invalid.")
        if not str(candidate.get("id", "")).strip() or not _is_sha256(
            candidate.get("evidenceSHA256")
        ):
            raise MAPEDProtocolError(
                "MAPED calibration candidates require an ID and evidence SHA-256."
            )
        if not isinstance(candidate.get("calibration"), dict):
            raise MAPEDProtocolError(
                "MAPED calibration candidate requires physical calibration values."
            )


def _array_payload(array: object) -> tuple[bytes, dict[str, Any]]:
    image = np.ascontiguousarray(array, dtype="<f4")
    if image.ndim != 2:
        raise MAPEDProtocolError(
            f"MAPED scientific image payload must be 2-D; got {image.shape}."
        )
    if not np.isfinite(image).all():
        raise MAPEDProtocolError("MAPED scientific image payload contains non-finite values.")
    payload = image.tobytes()
    return payload, {
        "shape": {"rows": int(image.shape[0]), "columns": int(image.shape[1])},
        "dtype": "float32",
        "byteCount": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _difference(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    if candidate.shape != reference.shape:
        raise MAPEDProtocolError(
            "MAPED reference parity array shape does not match the candidate.",
            code="parityFailed",
        )
    absolute = np.abs(candidate.astype(np.float64) - reference.astype(np.float64))
    return {
        "meanAbsoluteError": float(np.mean(absolute)),
        "p99AbsoluteError": float(np.percentile(absolute, 99)),
        "maximumAbsoluteError": float(np.max(absolute)),
    }


def _validate_validation(value: dict[str, Any]) -> None:
    kind = value.get("kind")
    if kind not in _VALIDATION_KINDS:
        raise MAPEDProtocolError("MAPED validation kind is not recognized.")
    parity = value.get("parity")
    if value.get("passed") is not True:
        raise MAPEDProtocolError(
            "MAPED output validation did not pass.",
            code="parityFailed",
            recovery="Inspect the validation evidence and run MAPED again.",
        )
    if kind == "reference_parity":
        if not isinstance(parity, dict) or parity.get("passed") is not True:
            raise MAPEDProtocolError("Reference parity requires passing parity metrics.")
        if int(parity.get("sampleCount", 0)) <= 0:
            raise MAPEDProtocolError("Reference parity requires sampled comparisons.")
        metric_names = (
            "sampleMeanAbsoluteError",
            "sampleMaximumAbsoluteError",
            "brightFieldMeanAbsoluteError",
            "brightFieldMaximumAbsoluteError",
            "meanDiffractionMeanAbsoluteError",
            "meanDiffractionMaximumAbsoluteError",
        )
        try:
            metrics = [float(parity[name]) for name in metric_names]
        except (KeyError, TypeError, ValueError) as exc:
            raise MAPEDProtocolError(
                "Reference parity metrics are incomplete."
            ) from exc
        if not all(math.isfinite(metric) and metric >= 0 for metric in metrics):
            raise MAPEDProtocolError(
                "Reference parity metrics must be finite and nonnegative."
            )
    elif parity is not None:
        raise MAPEDProtocolError(
            "Integrity-only validation must not carry parity metrics."
        )


def _validate_automatic_shifts(
    alignment: dict[str, Any], coordinates: Sequence[dict[str, Any]]
) -> None:
    for name in ("realSpaceShifts", "diffractionShifts"):
        shifts = alignment.get(name)
        if not isinstance(shifts, list) or len(shifts) != len(coordinates):
            raise MAPEDProtocolError(
                "Completed MAPED output requires one ordered real-space and "
                "diffraction shift per tilt.",
                code="invalidProvenance",
            )
        for shift, coordinate in zip(shifts, coordinates, strict=True):
            if shift.get("tilt") != coordinate:
                raise MAPEDProtocolError(
                    "MAPED shift order does not match the requested tilt order.",
                    code="invalidProvenance",
                )
            try:
                values = float(shift["rowPixels"]), float(shift["columnPixels"])
            except (KeyError, TypeError, ValueError) as exc:
                raise MAPEDProtocolError(
                    "MAPED shifts require finite row and column pixels.",
                    code="invalidProvenance",
                ) from exc
            if not all(math.isfinite(value) for value in values):
                raise MAPEDProtocolError(
                    "MAPED shifts require finite row and column pixels.",
                    code="invalidProvenance",
                )


class MAPEDProtocolService:
    """Serve MAPED inventory, inspection, execution, and validated caches."""

    def __init__(
        self,
        data_folder: str | Path,
        *,
        available_gpus: Callable[[], list[int]],
        device_name: Callable[[int], str],
        inspector: Callable[[Path], Any] | None = None,
        previewer: Callable[[Path, int], tuple[object, object]] | None = None,
        diffraction_reader: (
            Callable[[Path, int, int, int, int, int], object] | None
        ) = None,
        runner: (
            Callable[
                [
                    dict[str, Any],
                    Callable[[dict[str, Any]], None],
                    threading.Event,
                    Path,
                ],
                dict[str, Any],
            ]
            | None
        ) = None,
        implementation_revision: str | None = None,
    ) -> None:
        self.data_folder = Path(data_folder).expanduser().resolve()
        self._available_gpus = available_gpus
        self._device_name = device_name
        self._inspector = inspector or self._inspect_source
        self._previewer = previewer or self._preview_cuda
        self._diffraction_reader = diffraction_reader or self._read_diffraction_cuda
        self._runner = runner or self._run_cuda_process
        self.implementation_revision = (
            implementation_revision or _runtime_implementation_revision()
        )
        self._hash_lock = threading.Lock()
        self._hash_cache: dict[tuple[str, int, int], str] = {}
        self._inventory_lock = threading.Lock()
        self._collections: dict[str, dict[str, Any]] = {}
        self._payload_lock = threading.Lock()
        self._payloads: OrderedDict[str, tuple[bytes, dict[str, Any]]] = OrderedDict()
        self._payload_generation = 0
        self._latest_lock = threading.Lock()
        self._latest_diffraction_request: dict[str, int] = {}
        self._runs_lock = threading.Lock()
        self._runs: dict[str, _RunRecord] = {}
        self._device_arbiter = _DeviceArbiter()

    def advertised_capability(self) -> dict[str, Any]:
        return {
            "name": "maped",
            "contractVersion": CONTRACT_VERSION,
            "cacheVersion": CACHE_VERSION,
            "algorithmVersion": ALGORITHM_VERSION,
            "algorithmRevision": QUANTEM_ALGORITHM_REVISION,
            "implementationRevision": self.implementation_revision,
            "supportedTiltCounts": list(SUPPORTED_TILT_COUNTS),
            "validatedTiltCounts": list(VALIDATED_TILT_COUNTS),
            "previewPayload": "scientific_float32_array",
            "runEvents": "sse",
            "cancellation": "delete_run",
            "validationKinds": sorted(_VALIDATION_KINDS),
            "devices": [
                {"gpuIndex": gpu, "deviceName": self._device_name(gpu)}
                for gpu in self._available_gpus()
            ],
        }

    def _resolve_folder(self, value: str | Path) -> Path:
        folder = Path(value).expanduser()
        if not folder.is_absolute():
            folder = self.data_folder / folder
        folder = folder.resolve()
        try:
            folder.relative_to(self.data_folder)
        except ValueError as exc:
            raise MAPEDProtocolError(
                "MAPED folder is outside the configured data root.", status_code=403
            ) from exc
        if not folder.is_dir():
            raise MAPEDProtocolError(
                f"MAPED folder does not exist: {folder}",
                code="missingInput",
                status_code=404,
                recovery="Choose a folder containing complete *_master.h5 tilts.",
            )
        return folder

    @staticmethod
    def _inspect_source(master: Path) -> Any:
        from quantem.gpu.io import inspect

        return inspect(master)

    def _file_hash(self, path: Path) -> str:
        stat = path.stat()
        key = (str(path.resolve()), int(stat.st_size), int(stat.st_mtime_ns))
        with self._hash_lock:
            cached = self._hash_cache.get(key)
        if cached is not None:
            return cached
        value = _sha256(path)
        with self._hash_lock:
            self._hash_cache[key] = value
        return value

    def _source_identity(self, master: Path) -> dict[str, Any]:
        paths = _source_paths(master)
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise MAPEDProtocolError(
                f"MAPED source members are missing: {', '.join(missing)}",
                code="missingInput",
                recovery="Wait for acquisition writing to finish or restore the linked files.",
            )
        master_hash = self._file_hash(paths[0])
        member_hashes = [self._file_hash(path) for path in paths[1:]]
        source_hash = _source_identity_sha256(master_hash, member_hashes)
        return {
            "datasetID": source_hash,
            "datasetSchema": "live4dstem.dataset/v0.1",
            "sourceIdentitySHA256": source_hash,
            "masterPath": str(paths[0]),
            "masterSHA256": master_hash,
            "orderedMemberSHA256": member_hashes,
        }

    def folder_snapshot(self, folder_path: str | Path) -> dict[str, Any]:
        folder = self._resolve_folder(folder_path)
        members = [
            _source_stat(path)
            for path in sorted(folder.glob("*_master.h5"))
            if not path.name.startswith("._")
        ]
        return {
            "contractVersion": CONTRACT_VERSION,
            "folderPath": str(folder),
            "members": members,
            "snapshotToken": _identity_hash(members),
        }

    def snapshot(self, request: dict[str, Any]) -> dict[str, Any]:
        self._validate_contract(request)
        return self.folder_snapshot(str(request.get("folderPath", "")))

    def inventory(self, request: dict[str, Any]) -> dict[str, Any]:
        self._validate_contract(request)
        folder = self._resolve_folder(str(request.get("folderPath", "")))
        rejected: list[dict[str, str]] = []
        candidates: list[dict[str, Any]] = []
        for master in sorted(folder.glob("*_master.h5")):
            if master.name.startswith("._"):
                continue
            coordinate = _tilt_coordinate(master.name)
            if coordinate is None:
                rejected.append(
                    {
                        "url": str(master),
                        "path": str(master),
                        "reason": "The master filename does not contain x/y tilt coordinates.",
                    }
                )
                continue
            try:
                inspection = self._inspector(master)
                if not bool(inspection.ready):
                    rejected.append(
                        {
                            "url": str(master),
                            "path": str(master),
                            "reason": str(inspection.reason),
                        }
                    )
                    continue
                if inspection.scan_shape is None or inspection.detector_shape is None:
                    raise MAPEDProtocolError(
                        "The source does not expose a complete scan and detector shape."
                    )
                source = self._source_identity(master)
                metadata = dict(inspection.metadata or {})
                calibration = _calibration_envelope(
                    metadata, source["sourceIdentitySHA256"]
                )
                acquisition_date = metadata.get(
                    "entry/instrument/detector/detectorSpecific/data_collection_date"
                )
                source_stat = _source_stat(master)
                candidates.append(
                    {
                        "id": source["datasetID"],
                        "datasetID": source["datasetID"],
                        "label": master.name,
                        "masterURL": str(master.resolve()),
                        "masterPath": str(master.resolve()),
                        "tilt": coordinate,
                        "scanShape": {
                            "rows": int(inspection.scan_shape[0]),
                            "columns": int(inspection.scan_shape[1]),
                        },
                        "detectorShape": {
                            "rows": int(inspection.detector_shape[0]),
                            "columns": int(inspection.detector_shape[1]),
                        },
                        "sourceDtype": str(inspection.dtype),
                        "sourceBytes": source_stat["byteCount"],
                        "acquisitionDate": (
                            None if acquisition_date is None else str(acquisition_date)
                        ),
                        "metadata": {
                            str(key): str(value) for key, value in metadata.items()
                        },
                        "sourceIdentity": source,
                        "calibration": calibration,
                    }
                )
            except (MAPEDProtocolError, OSError, KeyError, TypeError, ValueError) as exc:
                rejected.append(
                    {"url": str(master), "path": str(master), "reason": str(exc)}
                )

        candidates.sort(key=lambda item: _tilt_key(item["tilt"]))
        accepted: list[dict[str, Any]] = []
        seen: set[tuple[int, int]] = set()
        reference = next(
            (
                item
                for item in candidates
                if abs(item["tilt"]["xDegrees"]) < 0.005
                and abs(item["tilt"]["yDegrees"]) < 0.005
            ),
            candidates[0] if candidates else None,
        )
        for candidate in candidates:
            coordinate_key = (
                int(round(candidate["tilt"]["xDegrees"] * 100)),
                int(round(candidate["tilt"]["yDegrees"] * 100)),
            )
            if coordinate_key in seen:
                rejected.append(
                    {
                        "url": candidate["masterPath"],
                        "path": candidate["masterPath"],
                        "reason": "Duplicate tilt coordinate.",
                    }
                )
                continue
            seen.add(coordinate_key)
            if reference is not None and (
                candidate["scanShape"] != reference["scanShape"]
                or candidate["detectorShape"] != reference["detectorShape"]
                or candidate["sourceDtype"] != reference["sourceDtype"]
            ):
                rejected.append(
                    {
                        "url": candidate["masterPath"],
                        "path": candidate["masterPath"],
                        "reason": "Native shape or dtype does not match the tilt collection.",
                    }
                )
                continue
            accepted.append(candidate)

        problems = []
        if not accepted:
            problems.append(
                {
                    "code": "noUsableTilts",
                    "message": "No usable MAPED tilts were found.",
                    "blocksRun": True,
                }
            )
        if len(accepted) not in SUPPORTED_TILT_COUNTS:
            problems.append(
                {
                    "code": "unsupportedTiltCount",
                    "message": (
                        "MAPED needs a compatible 2, 3, 5, or 7 tilt collection; "
                        f"found {len(accepted)}."
                    ),
                    "blocksRun": True,
                }
            )
        if accepted and not any(
            abs(item["tilt"]["xDegrees"]) < 0.005
            and abs(item["tilt"]["yDegrees"]) < 0.005
            for item in accepted
        ):
            problems.append(
                {
                    "code": "missingCenterTilt",
                    "message": "No 0°, 0° center tilt was found; confirm the intended reference.",
                    "blocksRun": False,
                }
            )

        identity_body = {
            "orderedSources": [item["sourceIdentity"] for item in accepted],
            "orderedTilts": [item["tilt"] for item in accepted],
            "nativeShapes": [
                [
                    item["scanShape"]["rows"],
                    item["scanShape"]["columns"],
                    item["detectorShape"]["rows"],
                    item["detectorShape"]["columns"],
                ]
                for item in accepted
            ],
            "sourceDtypes": [item["sourceDtype"] for item in accepted],
        }
        collection_hash = _identity_hash(identity_body)
        response = {
            "contractVersion": CONTRACT_VERSION,
            "folderPath": str(folder),
            "collectionIdentitySHA256": collection_hash,
            "tilts": accepted,
            "rejectedInputs": sorted(rejected, key=lambda item: item["path"]),
            "problems": problems,
            "isRunnable": not any(item["blocksRun"] for item in problems),
            "snapshot": self.folder_snapshot(folder),
        }
        with self._inventory_lock:
            self._collections[collection_hash] = response
        return response

    def previews(self, request: dict[str, Any]) -> dict[str, Any]:
        self._validate_contract(request)
        collection = self._collection(str(request.get("collectionIdentitySHA256", "")))
        gpu = self._validate_backend(request.get("backend") or {})
        requested = request.get("datasetIDs")
        dataset_ids = (
            [str(value) for value in requested]
            if isinstance(requested, list)
            else [item["datasetID"] for item in collection["tilts"]]
        )
        by_id = {item["datasetID"]: item for item in collection["tilts"]}
        unknown = [value for value in dataset_ids if value not in by_id]
        if unknown:
            raise MAPEDProtocolError(f"Unknown MAPED preview datasets: {unknown}")
        owner = self._acquire_interactive(gpu, "preview")
        try:
            items = []
            for dataset_id in dataset_ids:
                dataset = by_id[dataset_id]
                bright_field, diffraction = self._previewer(
                    Path(dataset["masterPath"]), gpu
                )
                items.append(
                    {
                        "datasetID": dataset_id,
                        "brightField": self._store_payload(
                            bright_field,
                            kind="mean_bf",
                            dataset_id=dataset_id,
                            source_identity=dataset["sourceIdentity"],
                        ),
                        "meanDiffraction": self._store_payload(
                            diffraction,
                            kind="mean_dp",
                            dataset_id=dataset_id,
                            source_identity=dataset["sourceIdentity"],
                        ),
                    }
                )
        finally:
            self._device_arbiter.release(gpu, owner)
        return {
            "contractVersion": CONTRACT_VERSION,
            "collectionIdentitySHA256": collection["collectionIdentitySHA256"],
            "items": items,
        }

    def selected_diffraction(self, request: dict[str, Any]) -> dict[str, Any]:
        self._validate_contract(request)
        client_id = str(request.get("clientID", ""))
        if not client_id:
            raise MAPEDProtocolError("Selected diffraction requires a clientID.")
        try:
            request_id = int(request["requestID"])
        except (KeyError, TypeError, ValueError) as exc:
            raise MAPEDProtocolError(
                "Selected diffraction requires an integer requestID."
            ) from exc
        collection_hash = str(request.get("collectionIdentitySHA256", ""))
        collection = self._collection(collection_hash)
        dataset_id = str(request.get("datasetID", ""))
        dataset = next(
            (
                item
                for item in collection["tilts"]
                if item["datasetID"] == dataset_id
            ),
            None,
        )
        if dataset is None:
            raise MAPEDProtocolError(f"Unknown MAPED dataset: {dataset_id}")
        gpu = self._validate_backend(request.get("backend") or {})
        scan = request.get("scan") or {}
        averaging = request.get("averaging") or {}
        try:
            row = int(scan["row"])
            column = int(scan["column"])
            width = int(averaging.get("width", 1))
        except (KeyError, TypeError, ValueError) as exc:
            raise MAPEDProtocolError(
                "Selected diffraction requires integer row, column, and averaging width."
            ) from exc
        if width < 1 or width > 15 or width % 2 == 0:
            raise MAPEDProtocolError("Averaging width must be an odd integer from 1 to 15.")
        if averaging.get("aggregation", "mean") != "mean":
            raise MAPEDProtocolError("Selected diffraction supports mean averaging only.")
        rows = int(dataset["scanShape"]["rows"])
        columns = int(dataset["scanShape"]["columns"])
        if row < 0 or row >= rows or column < 0 or column >= columns:
            raise MAPEDProtocolError(
                f"Scan position ({row}, {column}) is outside {rows}×{columns}."
            )
        radius = width // 2
        row_start = max(0, row - radius)
        row_stop = min(rows, row + radius + 1)
        column_start = max(0, column - radius)
        column_stop = min(columns, column + radius + 1)
        owner = self._acquire_interactive(gpu, "selected-diffraction")
        try:
            with self._latest_lock:
                previous = self._latest_diffraction_request.get(client_id, -1)
                if request_id <= previous:
                    raise MAPEDProtocolError(
                        "Selected diffraction request was superseded by a newer request.",
                        code="cancelled",
                    )
                self._latest_diffraction_request[client_id] = request_id
            image = self._diffraction_reader(
                Path(dataset["masterPath"]),
                row_start,
                row_stop,
                column_start,
                column_stop,
                gpu,
            )
            with self._latest_lock:
                if self._latest_diffraction_request.get(client_id) != request_id:
                    raise MAPEDProtocolError(
                        "Selected diffraction request was superseded by a newer request.",
                        code="cancelled",
                    )
        finally:
            self._device_arbiter.release(gpu, owner)
        return {
            "contractVersion": CONTRACT_VERSION,
            "requestID": request_id,
            "clientID": client_id,
            "datasetID": dataset_id,
            "scan": {"row": row, "column": column},
            "sampleBounds": {
                "rowStart": row_start,
                "rowStop": row_stop,
                "columnStart": column_start,
                "columnStop": column_stop,
            },
            "averaging": {"width": width, "aggregation": "mean"},
            "diffraction": self._store_payload(
                image,
                kind="selected_diffraction",
                dataset_id=dataset_id,
                source_identity=dataset["sourceIdentity"],
            ),
        }

    def _acquire_interactive(self, gpu: int, operation: str) -> str:
        owner = f"{operation}:{uuid4()}"
        if not self._device_arbiter.try_acquire_interactive(gpu, owner):
            raise MAPEDProtocolError(
                f"CUDA GPU {gpu} is busy with another MAPED operation.",
                code="deviceBusy",
                recovery=(
                    "Wait for the active MAPED operation to finish or choose "
                    "another CUDA GPU."
                ),
                stage="load",
            )
        return owner

    def payload(self, payload_id: str) -> tuple[bytes, dict[str, Any]]:
        with self._payload_lock:
            stored = self._payloads.get(payload_id)
            if stored is not None:
                self._payloads.move_to_end(payload_id)
        if stored is None:
            raise MAPEDProtocolError(
                f"MAPED payload was not found: {payload_id}", status_code=404
            )
        return stored

    def _store_payload(
        self,
        array: object,
        *,
        kind: str,
        dataset_id: str,
        source_identity: dict[str, Any],
    ) -> dict[str, Any]:
        payload, descriptor = _array_payload(array)
        with self._payload_lock:
            self._payload_generation += 1
            generation = self._payload_generation
            payload_id = _identity_hash(
                {
                    "datasetID": dataset_id,
                    "generation": generation,
                    "kind": kind,
                    "sha256": descriptor["sha256"],
                }
            )
            complete = {
                "payloadID": payload_id,
                "path": f"/api/maped/payloads/{payload_id}",
                "kind": kind,
                "datasetID": dataset_id,
                "generation": generation,
                "sourceIdentity": source_identity,
                **descriptor,
            }
            self._payloads[payload_id] = (payload, complete)
            self._payloads.move_to_end(payload_id)
            while len(self._payloads) > _MAX_PAYLOADS:
                self._payloads.popitem(last=False)
        return complete

    def _result_products(
        self,
        cache_directory: Path,
        identity: dict[str, Any],
        integrity: dict[str, Any],
    ) -> dict[str, Any]:
        product_identity = integrity.get("products") or {}
        product_path = (
            cache_directory / str(product_identity.get("path", ""))
        ).resolve()
        try:
            product_path.relative_to(cache_directory.resolve())
            product_stat = product_path.stat()
        except (ValueError, OSError) as exc:
            raise MAPEDProtocolError(
                "Cached MAPED display products are missing or outside the cache."
            ) from exc
        if (
            int(product_identity.get("byteCount", -1)) != product_stat.st_size
            or not _is_sha256(product_identity.get("sha256"))
            or _sha256(product_path) != product_identity["sha256"]
        ):
            raise MAPEDProtocolError(
                "Cached MAPED display product identity changed."
            )
        cache_hash = _identity_hash(identity)
        source_identity = {
            "kind": "maped_result",
            "sourceIdentitySHA256": cache_hash,
            "orderedSourceIdentitySHA256": [
                source["sourceIdentitySHA256"]
                for source in identity["orderedSources"]
            ],
        }
        with np.load(product_path, allow_pickle=False) as products:
            try:
                bright_field = products["bright_field"]
                mean_diffraction = products["mean_diffraction"]
            except KeyError as exc:
                raise MAPEDProtocolError(
                    "Cached MAPED display products are incomplete."
                ) from exc
            return {
                "brightField": self._store_payload(
                    bright_field,
                    kind="maped_bf",
                    dataset_id=f"maped:{cache_hash}",
                    source_identity=source_identity,
                ),
                "meanDiffraction": self._store_payload(
                    mean_diffraction,
                    kind="maped_mean_dp",
                    dataset_id=f"maped:{cache_hash}",
                    source_identity=source_identity,
                ),
            }

    def cache_identity(self, request: dict[str, Any]) -> dict[str, Any]:
        self._validate_contract(request)
        collection = self._collection(str(request.get("collectionIdentitySHA256", "")))
        tilts = self._included_tilts(collection, request.get("includedDatasetIDs"))
        parameters = self._validate_parameters(request.get("parameters") or {})
        backend = request.get("backend") or {}
        self._validate_backend(backend)
        calibrations = request.get("orderedCalibrations")
        if not isinstance(calibrations, list) or len(calibrations) != len(tilts):
            raise MAPEDProtocolError(
                "MAPED run requires one ordered calibration envelope or null per tilt."
            )
        for calibration, tilt in zip(calibrations, tilts, strict=True):
            _validate_calibration_identity(
                calibration, tilt["sourceIdentity"]["sourceIdentitySHA256"]
            )
        if request.get("mode") != "automaticAlignment":
            raise MAPEDProtocolError("MAPED v0.1 supports automaticAlignment only.")
        if request.get("algorithmVersion", ALGORITHM_VERSION) != ALGORITHM_VERSION:
            raise MAPEDProtocolError("Unsupported MAPED algorithm version.")
        if request.get("implementationRevision") != self.implementation_revision:
            raise MAPEDProtocolError(
                "MAPED implementation revision does not match this service."
            )
        self._validate_validation_request(request.get("validation") or {})
        identity = {
            "schemaVersion": CACHE_VERSION,
            "algorithmVersion": ALGORITHM_VERSION,
            "orderedSources": [item["sourceIdentity"] for item in tilts],
            "orderedTilts": [item["tilt"] for item in tilts],
            "orderedCalibrations": calibrations,
            "parameters": parameters,
            "mode": "automaticAlignment",
            "backend": backend,
            "implementationRevision": self.implementation_revision,
        }
        self._validate_cache_identity(identity)
        return identity

    def validate_cache(self, request: dict[str, Any]) -> dict[str, Any]:
        identity = request.get("cacheIdentity")
        if not isinstance(identity, dict):
            identity = self.cache_identity(request)
        self._validate_cache_identity(identity)
        expected_hash = _identity_hash(identity)
        cache_hash = request.get("cacheIdentitySHA256", expected_hash)
        if cache_hash != expected_hash:
            raise MAPEDProtocolError("MAPED cache identity SHA-256 is invalid.")
        validation_request = request.get("validation") or {"kind": "integrity_only"}
        self._validate_validation_request(validation_request)
        cache_directory = self._cache_root(identity) / cache_hash
        manifest_path = cache_directory / "manifest.json"
        if not manifest_path.is_file():
            return {
                "contractVersion": CONTRACT_VERSION,
                "state": "miss",
                "reason": "No completed cache manifest exists for this identity.",
            }
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            return {
                "contractVersion": CONTRACT_VERSION,
                "state": "miss",
                "reason": f"The cache manifest is unreadable: {exc}",
            }
        receipt = manifest.get("receipt") or {}
        cache = receipt.get("cache") or {}
        if cache.get("identity") != identity or cache.get("state") != "complete":
            return {
                "contractVersion": CONTRACT_VERSION,
                "state": "miss",
                "reason": "The cache manifest identity or state does not match.",
            }
        output = Path(str(cache.get("outputPath", ""))).resolve()
        integrity = manifest.get("integrity") or {}
        try:
            output.relative_to(cache_directory.resolve())
            stat = output.stat()
        except (ValueError, OSError):
            return {
                "contractVersion": CONTRACT_VERSION,
                "state": "miss",
                "reason": "The cached MAPED output is missing or outside its cache directory.",
            }
        if (
            int(integrity.get("outputByteCount", -1)) != int(stat.st_size)
            or int(integrity.get("outputModificationNanoseconds", -1))
            != int(stat.st_mtime_ns)
            or not _is_sha256(cache.get("outputSHA256"))
        ):
            return {
                "contractVersion": CONTRACT_VERSION,
                "state": "miss",
                "reason": "The cached MAPED output file identity changed.",
            }
        try:
            _validate_automatic_shifts(
                cache.get("automaticAlignment") or {}, identity["orderedTilts"]
            )
            _validate_validation(cache.get("validation") or {})
            result_products = self._result_products(
                cache_directory, identity, integrity
            )
        except MAPEDProtocolError as exc:
            return {
                "contractVersion": CONTRACT_VERSION,
                "state": "miss",
                "reason": str(exc),
            }
        if validation_request["kind"] == "reference_parity":
            reference = manifest.get("referenceIdentity") or {}
            if (
                cache["validation"].get("kind") != "reference_parity"
                or reference.get("sha256") != validation_request["referenceSHA256"]
            ):
                return {
                    "contractVersion": CONTRACT_VERSION,
                    "state": "miss",
                    "reason": (
                        "The completed cache does not carry parity evidence for "
                        "the requested reference identity."
                    ),
                }
        response_receipt = json.loads(json.dumps(receipt))
        response_receipt["resultProducts"] = result_products
        return {
            "contractVersion": CONTRACT_VERSION,
            "state": "hit",
            "cacheIdentitySHA256": cache_hash,
            "receipt": response_receipt,
        }

    def start_run(self, request: dict[str, Any]) -> dict[str, Any]:
        collection = self._collection(str(request.get("collectionIdentitySHA256", "")))
        tilts = self._included_tilts(collection, request.get("includedDatasetIDs"))
        identity = self.cache_identity(request)
        cache_hash = _identity_hash(identity)
        request = dict(request)
        request["cacheIdentity"] = identity
        request["cacheIdentitySHA256"] = cache_hash
        request["orderedTiltInputs"] = tilts
        try:
            run_id = str(UUID(str(request["runID"])))
        except (KeyError, TypeError, ValueError) as exc:
            raise MAPEDProtocolError("MAPED run requires a UUID runID.") from exc
        output_root = self._cache_root(identity)
        working_directory = output_root / f".{run_id}.incomplete"
        record = _RunRecord(
            run_id=run_id,
            request=request,
            working_directory=working_directory,
        )
        with self._runs_lock:
            if run_id in self._runs:
                raise MAPEDProtocolError(f"MAPED run already exists: {run_id}")
            if any(
                active.request.get("cacheIdentitySHA256") == cache_hash
                and not active.terminal
                for active in self._runs.values()
            ):
                raise MAPEDProtocolError(
                    "A MAPED run for this exact cache identity is already active."
                )
            self._runs[run_id] = record
        thread = threading.Thread(
            target=self._execute_run,
            args=(record,),
            name=f"maped-{run_id}",
            daemon=True,
        )
        record.thread = thread
        thread.start()
        return {
            "contractVersion": CONTRACT_VERSION,
            "runID": run_id,
            "state": "accepted",
            "eventPath": f"/api/maped/runs/{run_id}/events",
            "cancelPath": f"/api/maped/runs/{run_id}",
        }

    def cancel_run(self, run_id: str) -> dict[str, Any]:
        record = self._run_record(run_id)
        with record.condition:
            already_terminal = record.terminal
        if already_terminal:
            return {
                "contractVersion": CONTRACT_VERSION,
                "runID": record.run_id,
                "state": "already_terminal",
            }
        record.cancel_requested.set()
        gpu = int(record.request["backend"]["gpu_index"])
        self._device_arbiter.wake(gpu)
        return {
            "contractVersion": CONTRACT_VERSION,
            "runID": record.run_id,
            "state": "cancellation_requested",
        }

    def run_events(
        self, run_id: str, *, after_sequence: int = 0
    ) -> Generator[dict[str, Any], None, None]:
        record = self._run_record(run_id)
        next_index = max(0, int(after_sequence))
        while True:
            with record.condition:
                while len(record.events) <= next_index and not record.terminal:
                    record.condition.wait(timeout=0.5)
                available = list(record.events[next_index:])
                terminal = record.terminal
            for event in available:
                next_index += 1
                yield event
            if terminal and next_index >= len(record.events):
                return

    def run_event_snapshot(self, run_id: str) -> list[dict[str, Any]]:
        record = self._run_record(run_id)
        with record.condition:
            return list(record.events)

    def _run_record(self, run_id: str) -> _RunRecord:
        try:
            canonical = str(UUID(str(run_id)))
        except (TypeError, ValueError) as exc:
            raise MAPEDProtocolError("MAPED run ID is not a UUID.", status_code=404) from exc
        with self._runs_lock:
            record = self._runs.get(canonical)
        if record is None:
            raise MAPEDProtocolError(
                f"MAPED run was not found: {canonical}", status_code=404
            )
        return record

    def _emit(self, record: _RunRecord, event: dict[str, Any]) -> None:
        with record.condition:
            if record.terminal:
                return
            complete = {
                "contractVersion": CONTRACT_VERSION,
                "runID": record.run_id,
                "sequence": len(record.events) + 1,
                "timestamp": _utc_now(),
                **event,
            }
            record.events.append(complete)
            if event.get("type") in {"completed", "cancelled", "failed"}:
                record.terminal = True
            record.condition.notify_all()

    def _execute_run(self, record: _RunRecord) -> None:
        request = record.request
        gpu = int(request["backend"]["gpu_index"])
        owner = f"run:{record.run_id}"
        if not self._device_arbiter.acquire_run(
            gpu,
            owner,
            record.cancel_requested,
            lambda: self._emit(
                record,
                {
                    "type": "accepted",
                    "mode": "automaticAlignment",
                    "gpuIndex": gpu,
                    "queueState": "waiting_for_device",
                },
            ),
        ):
            self._emit(
                record,
                {
                    "type": "cancelled",
                    "stage": None,
                    "detail": (
                        "MAPED cancelled while waiting for the CUDA device; "
                        "no cache files were changed."
                    ),
                },
            )
            return
        try:
            self._execute_run_on_device(record)
        finally:
            self._device_arbiter.release(gpu, owner)

    def _execute_run_on_device(self, record: _RunRecord) -> None:
        request = record.request
        try:
            cache_started = time.perf_counter()
            cache = self.validate_cache(
                {
                    "cacheIdentity": request["cacheIdentity"],
                    "cacheIdentitySHA256": request["cacheIdentitySHA256"],
                    "validation": request.get("validation"),
                }
            )
            if cache["state"] == "hit":
                receipt = json.loads(json.dumps(cache["receipt"]))
                receipt["cachedOpenSeconds"] = time.perf_counter() - cache_started
                self._emit(
                    record,
                    {
                        "type": "progress",
                        "stage": "ready",
                        "completedUnits": 1,
                        "totalUnits": 1,
                        "detail": "Opened validated MAPED cache",
                        "elapsedSeconds": receipt["cachedOpenSeconds"],
                    },
                )
                self._emit(record, {"type": "completed", "receipt": receipt})
                return
            if record.cancel_requested.is_set():
                raise _RunCancelled
            record.working_directory.parent.mkdir(parents=True, exist_ok=True)
            if record.working_directory.exists():
                shutil.rmtree(record.working_directory)
            record.working_directory.mkdir()

            def progress(value: dict[str, Any]) -> None:
                stage = value.get("stage")
                if stage not in _RUN_STAGES:
                    raise MAPEDProtocolError(f"Unknown MAPED progress stage: {stage}")
                self._emit(record, {"type": "progress", **value})

            outcome = self._runner(
                request,
                progress,
                record.cancel_requested,
                record.working_directory,
            )
            if record.cancel_requested.is_set():
                raise _RunCancelled
            receipt = self._complete_cache(request, outcome, record.working_directory)
            self._emit(
                record,
                {
                    "type": "progress",
                    "stage": "ready",
                    "completedUnits": 1,
                    "totalUnits": 1,
                    "detail": "Finalized validated MAPED cache",
                    "elapsedSeconds": receipt["coldRunSeconds"],
                },
            )
            self._emit(record, {"type": "completed", "receipt": receipt})
        except _RunCancelled:
            self._remove_incomplete(record.working_directory)
            self._emit(
                record,
                {
                    "type": "cancelled",
                    "stage": self._last_stage(record),
                    "detail": (
                        "MAPED cancelled after the CUDA worker stopped; "
                        "incomplete output was removed."
                    ),
                },
            )
        except Exception as exc:  # backend failures must become terminal events
            self._remove_incomplete(record.working_directory)
            if isinstance(exc, MAPEDProtocolError):
                code = exc.code
                recovery = exc.recovery
                stage = exc.stage or self._last_stage(record)
            else:
                code = "backendUnavailable"
                recovery = "Check the CUDA environment and run MAPED again."
                stage = self._last_stage(record)
            self._emit(
                record,
                {
                    "type": "failed",
                    "failure": {
                        "code": code,
                        "message": str(exc),
                        "recoverySuggestion": recovery,
                        "stage": stage,
                        "technicalDetail": f"{type(exc).__name__}: {exc}",
                    },
                },
            )

    @staticmethod
    def _last_stage(record: _RunRecord) -> str | None:
        for event in reversed(record.events):
            if event.get("type") == "progress":
                return str(event.get("stage"))
        return None

    @staticmethod
    def _remove_incomplete(path: Path) -> None:
        if path.name.startswith(".") and path.name.endswith(".incomplete") and path.exists():
            shutil.rmtree(path)

    def _complete_cache(
        self,
        request: dict[str, Any],
        outcome: dict[str, Any],
        working_directory: Path,
    ) -> dict[str, Any]:
        identity = request["cacheIdentity"]
        cache_hash = request["cacheIdentitySHA256"]
        coordinates = identity["orderedTilts"]
        alignment = outcome.get("automaticAlignment") or {}
        validation = outcome.get("validation") or {}
        _validate_automatic_shifts(alignment, coordinates)
        _validate_validation(validation)
        output_name = str(outcome.get("outputFile", "merged.npy"))
        output = (working_directory / output_name).resolve()
        try:
            output.relative_to(working_directory.resolve())
        except ValueError as exc:
            raise MAPEDProtocolError(
                "MAPED output escaped the incomplete cache directory."
            ) from exc
        if not output.is_file():
            raise MAPEDProtocolError(
                "MAPED worker did not create its declared output.", code="outputWriteFailed"
            )
        output_hash = outcome.get("outputSHA256") or _sha256(output)
        if not _is_sha256(output_hash):
            raise MAPEDProtocolError("MAPED worker returned an invalid output SHA-256.")
        output_stat = output.stat()
        output_shape = [int(value) for value in outcome.get("outputShape", [])]
        if len(output_shape) != 4 or any(value <= 0 for value in output_shape):
            raise MAPEDProtocolError("MAPED output shape must contain four positive dimensions.")
        if outcome.get("outputDtype") != "float32":
            raise MAPEDProtocolError("MAPED v0.1 output dtype must be float32.")

        final_directory = working_directory.parent / cache_hash
        if final_directory.exists():
            shutil.rmtree(final_directory)
        final_output = final_directory / output.name
        executed_devices = outcome.get("executedDevices")
        if not isinstance(executed_devices, list) or len(executed_devices) != 1:
            raise MAPEDProtocolError(
                "MAPED v0.1 requires exactly one recorded CUDA execution device.",
                code="invalidProvenance",
            )
        receipt = {
            "cache": {
                "identity": identity,
                "state": "complete",
                "createdAt": _utc_now(),
                "executedDevice": executed_devices[0],
                "backendEnvironment": outcome.get("backendEnvironment", {}),
                "outputPath": str(final_output),
                "outputSHA256": output_hash,
                "outputShape": output_shape,
                "outputDtype": "float32",
                "automaticAlignment": alignment,
                "stageTimings": outcome.get("stageTimings", []),
                "validation": validation,
            },
            "coldRunSeconds": outcome.get("coldRunSeconds"),
            "cachedOpenSeconds": None,
        }
        manifest = {
            "schemaVersion": CACHE_VERSION,
            "receipt": receipt,
            "integrity": {
                "outputByteCount": int(output_stat.st_size),
                "outputModificationNanoseconds": int(output_stat.st_mtime_ns),
                "products": outcome.get("products"),
                "algorithmTimings": outcome.get("algorithmTimings", {}),
            },
            "referenceIdentity": outcome.get("referenceIdentity"),
        }
        temporary_manifest = working_directory / "manifest.json.tmp"
        temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
        os.replace(temporary_manifest, working_directory / "manifest.json")
        os.replace(working_directory, final_directory)
        final_stat = final_output.stat()
        manifest["integrity"]["outputModificationNanoseconds"] = int(
            final_stat.st_mtime_ns
        )
        manifest_path = final_directory / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        response_receipt = json.loads(json.dumps(receipt))
        response_receipt["resultProducts"] = self._result_products(
            final_directory, identity, manifest["integrity"]
        )
        return response_receipt

    def _collection(self, identity: str) -> dict[str, Any]:
        with self._inventory_lock:
            collection = self._collections.get(identity)
        if collection is None:
            raise MAPEDProtocolError(
                "MAPED collection identity is unknown or stale; inventory the folder again."
            )
        return collection

    @staticmethod
    def _included_tilts(
        collection: dict[str, Any], requested: Any
    ) -> list[dict[str, Any]]:
        inventory = collection["tilts"]
        if requested is None:
            requested_ids = [item["datasetID"] for item in inventory]
        elif isinstance(requested, list) and all(
            isinstance(value, str) and value for value in requested
        ):
            requested_ids = requested
        else:
            raise MAPEDProtocolError(
                "includedDatasetIDs must be a non-empty ordered string list."
            )
        if not requested_ids or len(requested_ids) != len(set(requested_ids)):
            raise MAPEDProtocolError(
                "includedDatasetIDs must be non-empty and contain no duplicates."
            )
        requested_set = set(requested_ids)
        selected = [item for item in inventory if item["datasetID"] in requested_set]
        selected_ids = [item["datasetID"] for item in selected]
        if selected_ids != requested_ids:
            unknown = requested_set.difference(item["datasetID"] for item in inventory)
            if unknown:
                raise MAPEDProtocolError(
                    f"Unknown included MAPED datasets: {sorted(unknown)}"
                )
            raise MAPEDProtocolError(
                "includedDatasetIDs must preserve canonical inventory order."
            )
        if len(selected) not in SUPPORTED_TILT_COUNTS:
            raise MAPEDProtocolError(
                "MAPED execution requires an explicitly selected 2, 3, 5, or 7 tilt subset; "
                f"found {len(selected)}."
            )
        return selected

    def _cache_root(self, identity: dict[str, Any]) -> Path:
        parents = {
            Path(str(source["masterPath"])).expanduser().resolve().parent
            for source in identity["orderedSources"]
        }
        if len(parents) != 1:
            raise MAPEDProtocolError(
                "MAPED cache requires every tilt master to share one collection folder."
            )
        folder = parents.pop()
        try:
            folder.relative_to(self.data_folder)
        except ValueError as exc:
            raise MAPEDProtocolError(
                "MAPED cache folder is outside the configured data root.",
                status_code=403,
            ) from exc
        return folder / "maped-results"

    def _validate_cache_identity(self, identity: dict[str, Any]) -> None:
        expected_keys = {
            "schemaVersion",
            "algorithmVersion",
            "orderedSources",
            "orderedCalibrations",
            "orderedTilts",
            "parameters",
            "mode",
            "backend",
            "implementationRevision",
        }
        if set(identity) != expected_keys:
            raise MAPEDProtocolError("MAPED cache identity fields do not match v0.1.")
        if identity["schemaVersion"] != CACHE_VERSION:
            raise MAPEDProtocolError("Unsupported MAPED cache version.")
        if identity["algorithmVersion"] != ALGORITHM_VERSION:
            raise MAPEDProtocolError("Unsupported MAPED algorithm version.")
        if identity["implementationRevision"] != self.implementation_revision:
            raise MAPEDProtocolError(
                "MAPED cache implementation revision does not match this service."
            )
        if identity["mode"] != "automaticAlignment":
            raise MAPEDProtocolError("MAPED v0.1 supports automaticAlignment only.")
        self._validate_parameters(identity["parameters"])
        backend = identity["backend"]
        if not isinstance(backend, dict) or backend.get("kind") != "remote_cuda":
            raise MAPEDProtocolError("MAPED cache requires a remote_cuda backend.")
        if not str(backend.get("profile_id", "")).strip():
            raise MAPEDProtocolError("MAPED cache requires a remote profile_id.")
        try:
            gpu = int(backend["gpu_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise MAPEDProtocolError(
                "MAPED cache requires an explicit CUDA gpu_index."
            ) from exc
        if gpu < 0:
            raise MAPEDProtocolError("MAPED CUDA gpu_index must be nonnegative.")
        sources = identity["orderedSources"]
        tilts = identity["orderedTilts"]
        calibrations = identity["orderedCalibrations"]
        if (
            not isinstance(sources, list)
            or len(sources) not in SUPPORTED_TILT_COUNTS
            or not isinstance(tilts, list)
            or len(tilts) != len(sources)
            or not isinstance(calibrations, list)
            or len(calibrations) != len(sources)
        ):
            raise MAPEDProtocolError(
                "MAPED cache identity has an unsupported or inconsistent tilt count."
            )
        for source, calibration in zip(sources, calibrations, strict=True):
            if not isinstance(source, dict) or not _is_sha256(
                source.get("sourceIdentitySHA256")
            ):
                raise MAPEDProtocolError("MAPED cache source identity is invalid.")
            _validate_calibration_identity(
                calibration, source["sourceIdentitySHA256"]
            )
        self._cache_root(identity)

    def _validate_validation_request(self, validation: dict[str, Any]) -> None:
        if not isinstance(validation, dict):
            raise MAPEDProtocolError("MAPED validation request must be an object.")
        kind = validation.get("kind", "integrity_only")
        if kind not in _VALIDATION_KINDS:
            raise MAPEDProtocolError("MAPED validation kind is not recognized.")
        if kind == "integrity_only":
            if any(
                validation.get(name) is not None
                for name in ("referencePath", "referenceSHA256", "parity")
            ):
                raise MAPEDProtocolError(
                    "Integrity-only validation cannot carry reference or parity data."
                )
            return
        reference_path = Path(str(validation.get("referencePath", ""))).expanduser()
        if not reference_path.is_absolute():
            reference_path = self.data_folder / reference_path
        reference_path = reference_path.resolve()
        try:
            reference_path.relative_to(self.data_folder)
        except ValueError as exc:
            raise MAPEDProtocolError(
                "MAPED parity reference is outside the configured data root.",
                status_code=403,
            ) from exc
        if not _is_sha256(validation.get("referenceSHA256")):
            raise MAPEDProtocolError(
                "Reference parity requires the exact reference SHA-256."
            )

    @staticmethod
    def _validate_contract(request: dict[str, Any]) -> None:
        if request.get("contractVersion") != CONTRACT_VERSION:
            raise MAPEDProtocolError("Unsupported MAPED contract version.")

    def _validate_backend(self, backend: dict[str, Any]) -> int:
        if backend.get("kind") != "remote_cuda":
            raise MAPEDProtocolError("MAPED remote service requires explicit remote_cuda.")
        if not str(backend.get("profile_id", "")).strip():
            raise MAPEDProtocolError("MAPED requires an explicit remote profile_id.")
        gpu = backend.get("gpu_index")
        if gpu is None:
            raise MAPEDProtocolError("MAPED requires an explicit CUDA gpu_index.")
        gpu = int(gpu)
        if gpu not in self._available_gpus():
            raise MAPEDProtocolError(
                f"CUDA GPU {gpu} is unavailable.",
                code="backendUnavailable",
                recovery="Connect to an available CUDA GPU and try again.",
            )
        return gpu

    @staticmethod
    def _validate_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
        expected = {
            "batchSize": 16,
            "detectorBin": 1,
            "scanBin": 1,
            "originSigmaPixels": 1,
            "diffractionEdgeBlendPixels": 2,
            "realSpaceIterations": 20,
            "hanningFilter": True,
            "realSpacePaddingPixels": 2,
            "realSpaceEdgeBlendPixels": 5,
            "realSpacePadValue": "median",
            "shiftMethod": "bilinear",
            "accumulatorPolicy": "float32_accumulator",
            "outputDtype": "float32",
        }
        if parameters != expected:
            raise MAPEDProtocolError(
                "MAPED v0.1 requires the validated 4D-STEM parameter contract."
            )
        return dict(parameters)

    @staticmethod
    def _preview_cuda(master: Path, gpu: int) -> tuple[np.ndarray, np.ndarray]:
        import cupy as cp

        from quantem.gpu.io import load

        with cp.cuda.Device(gpu):
            result = load(
                master,
                backend="cuda",
                device=gpu,
                dtype="native",
                output="native",
                stack=True,
                verbose=False,
            )
            data = cp.asarray(result.data)
            if data.ndim != 4:
                raise MAPEDProtocolError(f"MAPED tilt must be 4-D; got {data.shape}.")
            rows, columns, detector_rows, detector_columns = data.shape
            flat = data.reshape(rows * columns, detector_rows * detector_columns)
            mean_diffraction = flat.mean(axis=0, dtype=cp.float32).reshape(
                detector_rows, detector_columns
            )
            bright_field = flat.mean(axis=1, dtype=cp.float32).reshape(rows, columns)
            output = cp.asnumpy(bright_field), cp.asnumpy(mean_diffraction)
            del flat, data, result, bright_field, mean_diffraction
            cp.get_default_memory_pool().free_all_blocks()
            return output

    @staticmethod
    def _read_diffraction_cuda(
        master: Path,
        row_start: int,
        row_stop: int,
        column_start: int,
        column_stop: int,
        gpu: int,
    ) -> np.ndarray:
        import cupy as cp

        from quantem.gpu.io import load

        positions = [
            (row, column)
            for row in range(row_start, row_stop)
            for column in range(column_start, column_stop)
        ]
        with cp.cuda.Device(gpu):
            result = load(
                master,
                backend="cuda",
                device=gpu,
                dtype="native",
                output="native",
                scan_indices=positions,
                index_mode="scan",
                stack=True,
                verbose=False,
            )
            frames = cp.asarray(result.data).reshape(len(positions), *result.data.shape[-2:])
            image = frames.mean(axis=0, dtype=cp.float32)
            output = cp.asnumpy(image)
            del frames, image, result
            cp.get_default_memory_pool().free_all_blocks()
            return output

    def _run_cuda_process(
        self,
        request: dict[str, Any],
        progress: Callable[[dict[str, Any]], None],
        cancel_requested: threading.Event,
        working_directory: Path,
    ) -> dict[str, Any]:
        context = multiprocessing.get_context("spawn")
        messages = context.Queue()
        process = context.Process(
            target=_cuda_maped_worker,
            args=(request, str(working_directory), messages),
            daemon=False,
        )
        process.start()
        outcome: dict[str, Any] | None = None
        error: str | None = None
        while process.is_alive():
            if cancel_requested.is_set():
                process.terminate()
                process.join(timeout=10)
                if process.is_alive():
                    process.kill()
                    process.join()
                raise _RunCancelled
            try:
                message = messages.get(timeout=0.1)
            except queue.Empty:
                continue
            if message.get("type") == "progress":
                progress(message["value"])
            elif message.get("type") == "result":
                outcome = message["value"]
            elif message.get("type") == "error":
                error = message["detail"]
        process.join()
        while True:
            try:
                message = messages.get_nowait()
            except queue.Empty:
                break
            if message.get("type") == "progress":
                progress(message["value"])
            elif message.get("type") == "result":
                outcome = message["value"]
            elif message.get("type") == "error":
                error = message["detail"]
        if cancel_requested.is_set():
            raise _RunCancelled
        if process.exitcode != 0 or error is not None:
            raise MAPEDProtocolError(
                error or f"MAPED CUDA worker exited with code {process.exitcode}.",
                code="mergeFailed",
                recovery="Inspect the CUDA worker error and run MAPED again.",
            )
        if outcome is None:
            raise MAPEDProtocolError("MAPED CUDA worker returned no result.")
        return outcome


def _cuda_maped_worker(
    request: dict[str, Any], working_directory: str, messages: Any
) -> None:
    """Execute one MAPED run in an isolated CUDA process."""
    try:
        import platform
        import sys

        import torch

        from quantem.diffraction import MAPEDTorch
        from quantem.gpu.io import load

        algorithm_module = sys.modules[MAPEDTorch.__module__]
        runtime_revision = _git_revision(Path(algorithm_module.__file__).resolve().parent)
        if runtime_revision is None:
            raise RuntimeError(
                "Installed quantem MAPED revision cannot be verified. Use the "
                "validated quantem checkout or set up an immutable package revision."
            )
        if runtime_revision != QUANTEM_ALGORITHM_REVISION:
            raise RuntimeError(
                "Installed quantem MAPED revision changed: "
                f"expected {QUANTEM_ALGORITHM_REVISION}, got {runtime_revision}."
            )
        backend = request["backend"]
        gpu = int(backend["gpu_index"])
        device = torch.device(f"cuda:{gpu}")
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        parameters = request["parameters"]
        tilts = request["orderedTiltInputs"]
        files = [item["sourceIdentity"]["masterPath"] for item in tilts]
        coordinates = [item["tilt"] for item in tilts]
        stage_timings: list[dict[str, Any]] = []
        algorithm_timings: dict[str, Any] = {}
        started = time.perf_counter()
        loaded_count = 0
        report_load_progress = True

        def emit(
            stage: str,
            completed: int,
            total: int,
            detail: str,
            elapsed: float | None = None,
        ) -> None:
            messages.put(
                {
                    "type": "progress",
                    "value": {
                        "stage": stage,
                        "completedUnits": completed,
                        "totalUnits": total,
                        "detail": detail,
                        "elapsedSeconds": elapsed,
                    },
                }
            )

        def read(path: str) -> torch.Tensor:
            nonlocal loaded_count
            result = load(
                path,
                backend="cuda",
                device=gpu,
                dtype="native",
                output="torch",
                stack=True,
                verbose=False,
            )
            if not torch.is_tensor(result.data):
                raise TypeError("CUDA load did not return a Torch tensor.")
            if report_load_progress:
                loaded_count += 1
                emit(
                    "load",
                    loaded_count,
                    len(files),
                    f"Loaded tilt {loaded_count} of {len(files)}",
                    time.perf_counter() - started,
                )
            return result.data

        emit("inventory", 1, 1, "Validated ordered tilt collection", 0.0)
        maped = MAPEDTorch.from_files(files, read=read, device=str(device))
        stage_started = time.perf_counter()
        maped.preprocess(plot_summary=False)
        torch.cuda.synchronize(device)
        report_load_progress = False
        preprocess_seconds = time.perf_counter() - stage_started
        stage_timings.append({"stage": "load", "seconds": preprocess_seconds})
        algorithm_timings["preprocessSeconds"] = preprocess_seconds

        registration_started = time.perf_counter()
        operation_started = time.perf_counter()
        maped.diffraction_origin(
            sigma=float(parameters["originSigmaPixels"]), plot_origins=False
        )
        torch.cuda.synchronize(device)
        algorithm_timings["diffractionOriginSeconds"] = (
            time.perf_counter() - operation_started
        )
        emit("registration", 1, 3, "Located diffraction origins")

        operation_started = time.perf_counter()
        maped.diffraction_align(
            edge_blend=float(parameters["diffractionEdgeBlendPixels"]),
            plot_aligned=False,
        )
        torch.cuda.synchronize(device)
        algorithm_timings["diffractionAlignmentSeconds"] = (
            time.perf_counter() - operation_started
        )
        emit("registration", 2, 3, "Aligned diffraction space")

        operation_started = time.perf_counter()
        maped.real_space_align(
            num_iter=int(parameters["realSpaceIterations"]),
            hanning_filter=bool(parameters["hanningFilter"]),
            padding=int(parameters["realSpacePaddingPixels"]),
            edge_blend=float(parameters["realSpaceEdgeBlendPixels"]),
            pad_val=str(parameters["realSpacePadValue"]),
            shift_method=str(parameters["shiftMethod"]),
            plot_aligned=False,
        )
        torch.cuda.synchronize(device)
        algorithm_timings["realSpaceRegistrationSeconds"] = (
            time.perf_counter() - operation_started
        )
        registration_seconds = time.perf_counter() - registration_started
        stage_timings.append(
            {"stage": "registration", "seconds": registration_seconds}
        )
        emit("registration", 3, 3, "Aligned real space", registration_seconds)

        merge_profile: dict[str, Any] = {}
        operation_started = time.perf_counter()
        emit("merge", 0, 1, "Merging aligned tilts")
        merged = maped.merge_datasets(
            shift_method="bilinear",
            dtype=torch.float32,
            plot_result=False,
            batch_size=int(parameters["batchSize"]),
            accumulator_device=device,
            compute_summaries=True,
            verbose=False,
            profile_timings=merge_profile,
        )
        torch.cuda.synchronize(device)
        merge_seconds = time.perf_counter() - operation_started
        stage_timings.append({"stage": "merge", "seconds": merge_seconds})
        algorithm_timings["mergeProfile"] = merge_profile
        emit("merge", 1, 1, "Merged aligned tilts", merge_seconds)

        work = Path(working_directory)
        output_path = work / "merged.npy"
        save_started = time.perf_counter()
        output = np.lib.format.open_memmap(
            output_path,
            mode="w+",
            dtype=np.float32,
            shape=tuple(merged.tensor.shape),
        )
        finite = True
        total_rows = int(merged.tensor.shape[0])
        for row_start in range(0, total_rows, 8):
            row_stop = min(total_rows, row_start + 8)
            band = merged.tensor[row_start:row_stop]
            finite = finite and bool(torch.isfinite(band).all().item())
            output[row_start:row_stop] = band.detach().cpu().numpy()
            emit(
                "save",
                row_stop,
                total_rows,
                f"Saved rows {row_start}–{row_stop - 1}",
            )
        output.flush()
        del output
        candidate_bf = merged.im_bf_merged.detach().cpu().numpy()
        candidate_dp = merged.dp_mean_merged.detach().cpu().numpy()
        real_shifts = maped.real_space_shifts.detach().cpu().tolist()
        diffraction_shifts = maped.diffraction_shifts.detach().cpu().tolist()
        products_path = work / "products.npz"
        np.savez_compressed(
            products_path,
            bright_field=candidate_bf,
            mean_diffraction=candidate_dp,
            real_space_shifts=np.asarray(real_shifts, dtype=np.float32),
            diffraction_shifts=np.asarray(diffraction_shifts, dtype=np.float32),
        )
        save_seconds = time.perf_counter() - save_started
        stage_timings.append({"stage": "save", "seconds": save_seconds})

        validation_request = request.get("validation") or {"kind": "integrity_only"}
        if validation_request.get("kind") == "reference_parity":
            reference_path = Path(str(validation_request.get("referencePath", "")))
            reference_hash = str(validation_request.get("referenceSHA256", ""))
            if not reference_path.is_file() or _sha256(reference_path) != reference_hash:
                raise ValueError("MAPED parity reference path or SHA-256 does not match.")
            with np.load(reference_path, allow_pickle=False) as reference:
                indices = {
                    name: np.asarray(reference[f"sample_{name}"], dtype=np.int64)
                    for name in ("r", "c", "h", "w")
                }
                index_tensors = {
                    name: torch.from_numpy(value).to(device=device)
                    for name, value in indices.items()
                }
                samples = (
                    merged.tensor[
                        index_tensors["r"],
                        index_tensors["c"],
                        index_tensors["h"],
                        index_tensors["w"],
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                )
                sample_metrics = _difference(samples, np.asarray(reference["samples"]))
                bf_metrics = _difference(candidate_bf, np.asarray(reference["im_bf"]))
                dp_metrics = _difference(candidate_dp, np.asarray(reference["dp_mean"]))
            passed = all(
                metrics["maximumAbsoluteError"] == 0
                for metrics in (sample_metrics, bf_metrics, dp_metrics)
            )
            parity = {
                "sampleCount": int(samples.size),
                "sampleMeanAbsoluteError": sample_metrics["meanAbsoluteError"],
                "sampleMaximumAbsoluteError": sample_metrics[
                    "maximumAbsoluteError"
                ],
                "brightFieldMeanAbsoluteError": bf_metrics["meanAbsoluteError"],
                "brightFieldMaximumAbsoluteError": bf_metrics[
                    "maximumAbsoluteError"
                ],
                "meanDiffractionMeanAbsoluteError": dp_metrics[
                    "meanAbsoluteError"
                ],
                "meanDiffractionMaximumAbsoluteError": dp_metrics[
                    "maximumAbsoluteError"
                ],
                "passed": passed,
            }
            validation = {
                "kind": "reference_parity",
                "passed": passed,
                "parity": parity,
            }
            reference_identity = {
                "path": str(reference_path.resolve()),
                "sha256": reference_hash,
            }
        else:
            checks = {
                "allOutputValuesFinite": finite,
                "brightFieldFinite": bool(np.isfinite(candidate_bf).all()),
                "meanDiffractionFinite": bool(np.isfinite(candidate_dp).all()),
                "outputShapeMatches": list(merged.tensor.shape)
                == [
                    tilts[0]["scanShape"]["rows"],
                    tilts[0]["scanShape"]["columns"],
                    tilts[0]["detectorShape"]["rows"],
                    tilts[0]["detectorShape"]["columns"],
                ],
            }
            validation = {
                "kind": "integrity_only",
                "passed": all(checks.values()),
            }
            reference_identity = None
        _validate_validation(validation)
        output_hash = _sha256(output_path)
        products_hash = _sha256(products_path)
        properties = torch.cuda.get_device_properties(device)
        executed_device = {
            "backend": "cuda",
            "deviceName": properties.name,
            "gpuIndex": gpu,
            "driverVersion": None,
            "runtimeVersion": str(torch.version.cuda),
            "implementationRevision": request["implementationRevision"],
        }
        alignment = {
            "realSpaceShifts": [
                {
                    "tilt": coordinate,
                    "rowPixels": float(shift[0]),
                    "columnPixels": float(shift[1]),
                }
                for coordinate, shift in zip(coordinates, real_shifts, strict=True)
            ],
            "diffractionShifts": [
                {
                    "tilt": coordinate,
                    "rowPixels": float(shift[0]),
                    "columnPixels": float(shift[1]),
                }
                for coordinate, shift in zip(
                    coordinates, diffraction_shifts, strict=True
                )
            ],
        }
        messages.put(
            {
                "type": "result",
                "value": {
                    "outputFile": output_path.name,
                    "outputSHA256": output_hash,
                    "outputShape": list(merged.tensor.shape),
                    "outputDtype": "float32",
                    "products": {
                        "path": str(Path("products.npz")),
                        "sha256": products_hash,
                        "byteCount": products_path.stat().st_size,
                    },
                    "automaticAlignment": alignment,
                    "validation": validation,
                    "referenceIdentity": reference_identity,
                    "executedDevices": [executed_device],
                    "backendEnvironment": {
                        "hostname": platform.node(),
                        "python": platform.python_version(),
                        "torch": torch.__version__,
                        "cudaRuntime": str(torch.version.cuda),
                        "quantemRevision": runtime_revision,
                        "quantemAlgorithmRevision": QUANTEM_ALGORITHM_REVISION,
                        "quantemGPURevision": request["implementationRevision"],
                    },
                    "stageTimings": stage_timings,
                    "algorithmTimings": algorithm_timings,
                    "coldRunSeconds": time.perf_counter() - started,
                },
            }
        )
    except BaseException as exc:
        messages.put({"type": "error", "detail": f"{type(exc).__name__}: {exc}"})
        raise


__all__ = [
    "ALGORITHM_VERSION",
    "CACHE_VERSION",
    "CONTRACT_VERSION",
    "MAPEDProtocolError",
    "MAPEDProtocolService",
    "QUANTEM_ALGORITHM_REVISION",
]
