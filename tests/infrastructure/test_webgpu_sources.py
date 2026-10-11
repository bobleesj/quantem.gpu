import json
from importlib.resources import files


def source_text(name: str) -> str:
    """Read one domain-owned WebGPU source resource."""

    return files("quantem.gpu").joinpath(name).read_text(encoding="utf-8")


def test_webgpu_sources_are_shipped_and_readable() -> None:
    names = json.loads(source_text("webgpu/sources.json"))
    assert names
    for name in names:
        assert source_text(name).strip()


def test_webgpu_ssb_builds_its_pipelines_asynchronously() -> None:
    """Native 512/1024 SSB shaders are large: a synchronous pipeline build can freeze
    Chromium's renderer for minutes on a cold Metal shader cache."""

    source = source_text("ssb/webgpu/backend.ts")
    assert ".createComputePipeline(" not in source
    assert ".createComputePipelineAsync(" in source


def test_webgpu_namespace_is_build_resources_not_a_python_execution_backend() -> None:
    import quantem.gpu as gpu
    from quantem.gpu import webgpu

    assert "webgpu" not in gpu.__all__
    assert not hasattr(webgpu, "load")
