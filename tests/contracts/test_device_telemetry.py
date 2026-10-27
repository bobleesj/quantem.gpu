"""Driver telemetry remains independent of compute-device selection."""

import subprocess
from types import SimpleNamespace

from quantem.gpu.device import telemetry

READINGS = (
    "0, GPU-first, Example GPU, 00000000:01:00.0, 16384, 4096, 12000, 75, 65, 180.5\n"
    "1, GPU-second, Example GPU, 00000000:02:00.0, 16384, 0, 16000, [N/A], [N/A], [N/A]\n"
)


def test_driver_ids_and_memory_survive_a_compute_visibility_mask(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(telemetry.shutil, "which", lambda name: "nvidia-smi")
    calls = []

    def query(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(stdout=READINGS)

    monkeypatch.setattr(telemetry.subprocess, "run", query)
    snapshot = telemetry.status()
    assert snapshot["cuda_visible_devices"] == "1"
    assert [gpu["index"] for gpu in snapshot["gpus"]] == [0, 1]
    assert snapshot["gpus"][0]["used_bytes"] == 4096 * 2**20
    # Read the driver's free value, not total minus used (it reserves memory).
    assert snapshot["gpus"][0]["free_bytes"] == 12000 * 2**20
    assert snapshot["gpus"][1]["utilization"] is None
    assert snapshot["gpus"][1]["used_bytes"] == 0
    assert len(calls) == 1
    assert "processes" not in snapshot
    assert calls[0][1]["timeout"] == 2
    assert "--format=csv,noheader,nounits" in calls[0][0]


def test_failed_query_has_no_stale_gpu_readings_and_recovers(monkeypatch):
    monkeypatch.setattr(telemetry.shutil, "which", lambda name: "nvidia-smi")

    def unavailable(*args, **kwargs):
        raise subprocess.TimeoutExpired("nvidia-smi", 2)

    monkeypatch.setattr(telemetry.subprocess, "run", unavailable)
    snapshot = telemetry.status()
    assert snapshot["gpus"] == []
    assert "timed out" in snapshot["error"]
    monkeypatch.setattr(
        telemetry.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=READINGS),
    )
    recovered = telemetry.status()
    assert len(recovered["gpus"]) == 2
    assert not recovered["error"]


def test_absent_driver_is_reported_as_unavailable(monkeypatch):
    monkeypatch.setattr(telemetry.shutil, "which", lambda name: None)
    snapshot = telemetry.status()
    assert snapshot["gpus"] == []
    assert "not installed" in snapshot["error"]


def test_optional_process_query_preserves_gpu_identity_and_unknown_memory(monkeypatch):
    monkeypatch.setattr(telemetry.shutil, "which", lambda name: "nvidia-smi")

    def query(args, **kwargs):
        assert kwargs["timeout"] == 2
        if args[1].startswith("--query-gpu="):
            return SimpleNamespace(stdout=READINGS)
        assert args[1] == "--query-compute-apps=gpu_uuid,pid,used_gpu_memory"
        return SimpleNamespace(stdout="GPU-second, 42, 512\nGPU-first, 50, [N/A]\n")

    monkeypatch.setattr(telemetry.subprocess, "run", query)
    snapshot = telemetry.status(processes=True)
    assert snapshot["process_error"] == ""
    assert snapshot["processes"] == [
        {"gpu_uuid": "GPU-second", "pid": 42, "used_bytes": 512 * 2**20},
        {"gpu_uuid": "GPU-first", "pid": 50, "used_bytes": None},
    ]


def test_process_query_failure_preserves_device_readings(monkeypatch):
    monkeypatch.setattr(telemetry.shutil, "which", lambda name: "nvidia-smi")

    def query(args, **kwargs):
        if args[1].startswith("--query-gpu="):
            return SimpleNamespace(stdout=READINGS)
        raise subprocess.CalledProcessError(1, args, stderr="Insufficient permissions")

    monkeypatch.setattr(telemetry.subprocess, "run", query)
    snapshot = telemetry.status(processes=True)
    assert len(snapshot["gpus"]) == 2
    assert not snapshot["error"]
    assert snapshot["processes"] == []
    assert "permissions" in snapshot["process_error"]
