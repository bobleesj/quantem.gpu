"""Install extras ([cuda], [mps], [cpu]), the missing-runtime notice, and the refused CPU load."""

import platform
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

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
    monkeypatch.setattr(select, "platform", SimpleNamespace(machine=lambda: arch, platform=platform.platform, system=lambda: system))
    monkeypatch.setattr(select, "shutil", SimpleNamespace(which=lambda name: "/usr/bin/nvidia-smi" if nvidia else None))
    monkeypatch.setattr(select, "os", SimpleNamespace(path=SimpleNamespace(exists=lambda path: False)))
    monkeypatch.setattr(select, "importlib", SimpleNamespace(util=SimpleNamespace(
        find_spec=lambda name: object() if name in installed else None,
    )))


# ---


def test_extras_are_explicit_cuda_mps_and_cpu() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    extras = project["optional-dependencies"]
    assert extras["cpu"] == []
    assert extras["cuda"] == [
        "cupy-cuda13x[ctk]>=14.0; sys_platform != 'darwin' and platform_machine != 'ARM64'",
        "torch>=2.11; sys_platform != 'darwin'",
        "optuna==4.9.0",
        "scipy",
    ]
    assert extras["mps"] == [
        "mlx; sys_platform == 'darwin' and platform_machine == 'arm64'",
        "pyobjc-framework-Metal; sys_platform == 'darwin' and platform_machine == 'arm64'",
        "optuna==4.9.0",
        "scipy",
    ]
    assert "gpu" not in extras
    assert not any(name.startswith(("cupy", "mlx", "pyobjc", "optuna")) for name in project["dependencies"])


def test_nvidia_gpu_without_cupy_names_the_install_command(monkeypatch, fresh_output, capsys) -> None:
    machine(monkeypatch, system="linux", arch="x86_64", nvidia=True)
    notice = (
        "quantem.gpu: an NVIDIA GPU is present but CuPy is not installed, so the GPU is not used. "
        'Install it with: pip install "quantem.gpu[cuda]" (widget: pip install "quantem.widget[cuda]")'
    )
    assert device.runtime_notice() == notice
    for _ in range(2):
        with pytest.raises(RuntimeError, match="No QuantEM GPU backend"):
            device.resolve("auto")
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


def test_cpu_device_is_refused_by_load(tmp_path) -> None:
    """There is no CPU load: ``device="cpu"`` names no GPU, so it raises before any file is read."""
    with pytest.raises(ValueError, match="device='cpu' is not supported"):
        io.load(tmp_path / "scan_master.h5", device="cpu")
