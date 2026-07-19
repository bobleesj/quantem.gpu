from __future__ import annotations

import pytest


def test_webgpu_sources_are_shipped_and_readable() -> None:
    from quantem.gpu import webgpu

    names = webgpu.source_names()

    assert "compute.ts" in names
    assert "bslz4.ts" in names
    assert "device.ts" in names
    assert "showptycho-ssb.ts" in names
    for name in names:
        text = webgpu.source_text(name)
        assert text.strip()


def test_webgpu_compute_source_tracks_vi_and_dpc_kernels() -> None:
    from quantem.gpu.webgpu import source_text

    source = source_text("compute.ts")

    assert "const maskedSumSrc" in source
    assert "export function buildDetectorMask" in source
    assert "export function buildFullDetectorMask" in source
    assert "export function buildScanMask" in source
    assert "maskedSumBuffer(mask: Uint32Array)" in source
    assert "const maskedComSrc" in source
    assert "maskedCoM(mask: Uint32Array" in source
    assert "const DPC_MEAN_WGSL" in source
    assert "const DPC_COMPONENT_WGSL" in source
    assert "maskedDpcBuffer(mask: Uint32Array" in source
    assert "maskedDpc(mask: Uint32Array" in source
    assert "subgroupAdd" in source


def test_webgpu_showptycho_source_tracks_ssb_engine() -> None:
    from quantem.gpu.webgpu import source_text

    source = source_text("showptycho-ssb.ts")

    assert 'from "./device"' in source
    assert 'from "./h5reader"' in source
    assert 'from "./bslz4"' in source
    assert "export class ShowPtychoWebGPUSSB" in source
    assert "const SUPPORTED_SSB_SIZES = [128, 256, 512, 1024]" in source
    assert "makeSsbShader" in source
    assert "WebGPU SSB buffers are not ready after setup" in source


def test_webgpu_source_rejects_unknown_names() -> None:
    from quantem.gpu.webgpu import source_text

    with pytest.raises(ValueError, match="Unknown WebGPU source"):
        source_text("missing.ts")
