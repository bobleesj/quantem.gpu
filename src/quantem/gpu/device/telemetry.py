"""Read driver telemetry without opening a compute context."""

import csv
import math
import os
import shutil
import subprocess
import time


def number(value: str) -> float | None:
    """Keep unsupported driver readings missing rather than reporting zero."""
    try:
        result = float(value.strip())
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def status(*, processes: bool = False) -> dict:
    """Read memory and utilization of all NVIDIA GPUs visible to the driver.

    This is a read-only driver query. It does not select a compute device,
    create a CUDA context, synchronize work or release allocator caches.
    GPU indices are physical driver indices, not the logical indices after
    ``CUDA_VISIBLE_DEVICES`` remapping. Missing readings are ``None``.

    Parameters
    ----------
    processes : bool, default False
        Include compute process IDs and their reported VRAM, keyed by GPU
        UUID. This uses a second bounded driver query. Graphics processes
        and driver allocations are not included in the process breakdown.

    Returns
    -------
    dict
        JSON-compatible snapshot with ``gpus``, ``error``, ``timestamp``,
        ``query_ms`` and ``cuda_visible_devices``. Memory values are bytes;
        driver MiB readings are converted to bytes. Utilization is percent,
        temperature Celsius and power watts. The
        driver may reserve memory beyond its reported used and free values.
        Without NVIDIA telemetry, ``gpus`` is empty and ``error`` explains why.
        If requested, ``processes`` contains ``gpu_uuid``, ``pid`` and
        ``used_bytes``; ``process_error`` reports process-query failures
        separately, preserving valid device readings.

    Examples
    --------
    >>> from quantem.gpu import device
    >>> snapshot = device.status()
    >>> for gpu in snapshot["gpus"]:
    ...     print(gpu["index"], gpu["free_bytes"])
    """
    started = time.perf_counter()
    snapshot = {
        "gpus": [],
        "error": "",
        "timestamp": time.time(),
        "query_ms": 0.0,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    executable = shutil.which("nvidia-smi")
    if executable is None:
        snapshot["error"] = (
            "NVIDIA telemetry unavailable: nvidia-smi is not installed. "
            "Apple unified memory is not reported as dedicated VRAM."
        )
        return snapshot
    fields = (
        "index,uuid,name,pci.bus_id,memory.total,memory.used,memory.free,"
        "utilization.gpu,temperature.gpu,power.draw"
    )
    try:
        result = subprocess.run(
            [executable, f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
        for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
            if not row:
                continue
            if len(row) != 10:
                raise ValueError("Unexpected NVIDIA telemetry columns")
            total, used, free = [number(value) for value in row[4:7]]
            snapshot["gpus"].append(
                {
                    "index": int(row[0]),
                    "uuid": row[1].strip(),
                    "name": row[2].strip(),
                    "pci_bus_id": row[3].strip(),
                    "total_bytes": None if total is None else int(total * 2**20),
                    "used_bytes": None if used is None else int(used * 2**20),
                    "free_bytes": None if free is None else int(free * 2**20),
                    "utilization": number(row[7]),
                    "temperature": number(row[8]),
                    "power": number(row[9]),
                }
            )
        if not snapshot["gpus"]:
            snapshot["error"] = "No NVIDIA GPUs are exposed to this server."
    except subprocess.TimeoutExpired:
        snapshot["gpus"] = []
        snapshot["error"] = "NVIDIA telemetry timed out; the next sample will retry."
    except subprocess.CalledProcessError as exc:
        snapshot["gpus"] = []
        detail = (exc.stderr or exc.stdout or "Driver query failed").strip()
        snapshot["error"] = f"NVIDIA telemetry unavailable: {detail[:240]}"
    except (OSError, ValueError) as exc:
        snapshot["gpus"] = []
        snapshot["error"] = f"NVIDIA telemetry unavailable: {exc}"
    if processes and snapshot["gpus"]:
        snapshot["processes"] = []
        snapshot["process_error"] = ""
        try:
            result = subprocess.run(
                [
                    executable,
                    "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=2,
                check=True,
            )
            for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
                if not row:
                    continue
                if len(row) != 3:
                    raise ValueError("Unexpected NVIDIA process columns")
                used = number(row[2])
                snapshot["processes"].append(
                    {
                        "gpu_uuid": row[0].strip(),
                        "pid": int(row[1]),
                        "used_bytes": None if used is None else int(used * 2**20),
                    }
                )
        except subprocess.TimeoutExpired:
            snapshot["processes"] = []
            snapshot["process_error"] = (
                "Compute-user query timed out; retrying next sample."
            )
        except subprocess.CalledProcessError as exc:
            snapshot["processes"] = []
            detail = (exc.stderr or exc.stdout or "Driver query failed").strip()
            snapshot["process_error"] = (
                f"Compute-user telemetry unavailable: {detail[:240]}"
            )
        except (OSError, ValueError) as exc:
            snapshot["processes"] = []
            snapshot["process_error"] = f"Compute-user telemetry unavailable: {exc}"
    snapshot["timestamp"] = time.time()
    snapshot["query_ms"] = (time.perf_counter() - started) * 1000
    return snapshot
