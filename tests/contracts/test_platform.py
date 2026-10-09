"""Install extras ([cuda], [mps], [cpu]), the shared device= contract, the missing-runtime notice and the CPU load path."""

import platform
import tomllib
from pathlib import Path
from types import SimpleNamespace

import h5py
import hdf5plugin
import numpy as np
import pytest
import torch

from quantem.gpu import device, io
from quantem.gpu.device import select

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def fresh_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test sees the once-per-process lines as if nothing had printed yet."""
    monkeypatch.setattr(select, "_printed", set())


def machine(monkeypatch, *, system, arch, nvidia=False, installed=()):
    """Pretend, inside ``select`` only, to be a computer with this platform, GPU and runtime modules."""
    monkeypatch.setattr(select, "sys", SimpleNamespace(platform=system))
    monkeypatch.setattr(select, "platform", SimpleNamespace(machine=lambda: arch, platform=platform.platform))
    monkeypatch.setattr(select, "shutil", SimpleNamespace(which=lambda name: "/usr/bin/nvidia-smi" if nvidia else None))
    monkeypatch.setattr(select, "os", SimpleNamespace(path=SimpleNamespace(exists=lambda path: False)))
    monkeypatch.setattr(select, "importlib", SimpleNamespace(util=SimpleNamespace(
        find_spec=lambda name: object() if name in installed else None,
    )))


def accelerators(monkeypatch, *, cuda=0, mps=False):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda > 0)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    monkeypatch.setattr(select, "runtime_notice", lambda: None)


# ---


def test_extras_are_explicit_cuda_mps_and_cpu() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    extras = project["optional-dependencies"]
    assert extras["cpu"] == []
    assert extras["cuda"] == [
        "cupy-cuda13x[ctk]>=14.0; sys_platform != 'darwin' and platform_machine != 'ARM64'",
        "torch>=2.11; sys_platform != 'darwin'",
        "optuna<5",
        "scipy",
    ]
    assert extras["mps"] == [
        "mlx; sys_platform == 'darwin' and platform_machine == 'arm64'",
        "pyobjc-framework-Metal; sys_platform == 'darwin' and platform_machine == 'arm64'",
        "optuna<5",
        "scipy",
    ]
    assert "gpu" not in extras
    assert not any(name.startswith(("cupy", "mlx", "pyobjc", "optuna")) for name in project["dependencies"])


def test_auto_prefers_cuda_then_mps_then_cpu_and_says_which(monkeypatch, fresh_output, capsys) -> None:
    accelerators(monkeypatch, cuda=2, mps=True)
    assert device.resolve_device() == "cuda:0"
    accelerators(monkeypatch, mps=True)
    assert device.resolve_device("auto") == "mps"
    accelerators(monkeypatch)
    assert device.resolve_device(None) == "cpu"
    assert device.resolve_device("auto") == "cpu"
    assert capsys.readouterr().out.splitlines() == [
        'quantem.gpu: device="auto" selected cuda:0.',
        'quantem.gpu: device="auto" selected mps.',
        'quantem.gpu: device="auto" selected cpu.',
    ]


def test_explicit_devices_resolve_quietly_or_raise(monkeypatch, fresh_output, capsys) -> None:
    accelerators(monkeypatch, cuda=2)
    assert device.resolve_device("cuda") == "cuda:0"
    assert device.resolve_device("CUDA:1") == "cuda:1"
    assert device.resolve_device("cpu") == "cpu"
    assert capsys.readouterr().out == ""
    with pytest.raises(ValueError, match="2 CUDA device"):
        device.resolve_device("cuda:2")
    with pytest.raises(RuntimeError, match="MPS device is unavailable"):
        device.resolve_device("mps")
    with pytest.raises(ValueError, match="Unknown device"):
        device.resolve_device("gpu")


def test_nvidia_gpu_without_cupy_names_the_install_command(monkeypatch, fresh_output, capsys) -> None:
    machine(monkeypatch, system="linux", arch="x86_64", nvidia=True)
    notice = (
        "quantem.gpu: an NVIDIA GPU is present but CuPy is not installed, so the GPU is not used. "
        'Install it with: pip install "quantem.gpu[cuda]" (widget: pip install "quantem.widget[cuda]")'
    )
    assert device.runtime_notice() == notice
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    device.resolve_device("cpu")
    device.resolve_device("cpu")
    assert capsys.readouterr().out == notice + "\n"


def test_apple_silicon_without_metal_runtime_names_the_install_command(monkeypatch) -> None:
    machine(monkeypatch, system="darwin", arch="arm64", installed={"Metal"})
    assert device.runtime_notice() == (
        "quantem.gpu: this Mac's Apple GPU is present but mlx is not installed, so the GPU is not used. "
        'Install it with: pip install "quantem.gpu[mps]" (widget: pip install "quantem.widget[mps]")'
    )


@pytest.mark.parametrize(
    ("system", "arch", "nvidia", "installed"),
    [
        ("linux", "x86_64", False, ()),
        ("linux", "x86_64", True, ("cupy",)),
        ("darwin", "arm64", False, ("Metal", "mlx")),
        ("darwin", "x86_64", False, ()),
    ],
)
def test_no_notice_without_a_gpu_or_with_its_runtime(monkeypatch, system, arch, nvidia, installed) -> None:
    machine(monkeypatch, system=system, arch=arch, nvidia=nvidia, installed=installed)
    assert device.runtime_notice() is None


def test_cpu_device_loads_an_arina_master_as_the_numpy_counts(tmp_path) -> None:
    counts = np.random.default_rng(3).integers(0, 60, (4, 5, 8, 8), dtype=np.uint16)
    master = tmp_path / "scan_master.h5"
    with h5py.File(master, "w") as handle:
        handle.create_dataset(
            "entry/data/data", data=counts.reshape(-1, 8, 8), chunks=(1, 8, 8),
            **hdf5plugin.Bitshuffle(cname="lz4"),
        )
    with io.load(master, device="cpu", scan_shape=(4, 5), verbose=False) as loaded:
        assert loaded.residency == "host"
        np.testing.assert_array_equal(np.asarray(loaded), counts)
        np.testing.assert_array_equal(loaded[1, 2:4, 3].numpy(), counts[1, 2:4, 3])
        assert loaded[0, 0].device == torch.device("cpu")
    with pytest.raises(ValueError, match="contradicts"):
        io.load(master, device="cpu", backend="cuda")


def test_cpu_device_reopens_a_qem_file() -> None:
    fixture = ROOT / "tests/data/qem-v1"
    with io.load(fixture / "uint16.qem", device="cpu", verbose=False) as loaded:
        assert loaded.residency == "host"
        np.testing.assert_array_equal(loaded[...].numpy(), np.load(fixture / "uint16.npy"))
