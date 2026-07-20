from __future__ import annotations

import pytest


def test_webgpu_sources_are_shipped_and_readable() -> None:
    from quantem.gpu import webgpu

    names = webgpu.source_names()

    assert "compute.ts" in names
    assert "bslz4.ts" in names
    assert "device.ts" in names
    assert "local-h5.ts" in names
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
    assert "const SUBTRACT_FROM_TOTAL_WGSL" in source
    assert "detectorComplementIndices(mask: Uint32Array)" in source
    assert "totalSumBuffer()" in source
    assert "const maskedComSrc" in source
    assert "maskedCoM(mask: Uint32Array" in source
    assert "const DPC_MEAN_WGSL" in source
    assert "const DPC_COMPONENT_WGSL" in source
    assert "maskedDpcBuffer(mask: Uint32Array" in source
    assert "maskedDpc(mask: Uint32Array" in source
    assert "getDevice(): GPUDevice" in source
    assert "readFloatBuffer(buf: GPUBuffer" in source
    assert "checksumFrames(scanIndices: number[])" in source
    assert "enc.copyBufferToBuffer(ch.buffer, byteOffset, rb, 0, byteLength)" in source
    assert "const bad = this.badPx.length ? new Set(this.badPx) : null;" in source
    assert "bad?.has(i) ? 0" in source
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


def test_webgpu_h5reader_keeps_single_pass_block_metadata_parse() -> None:
    from quantem.gpu.webgpu import source_text

    source = source_text("h5reader.ts")

    assert "const meta = new Uint32Array(framesThisChunk * nBlocksPerFrame * 2);" in source
    assert "meta[m++] = addr + pos + 4;" in source
    assert "meta[i] -= rangeStart;" in source
    assert "const frameBlockMeta: number[][] = new Array(nFrames);" not in source
    assert "readFrameBlockMeta(f, 0, []);" not in source
    assert "export interface Bslz4SelectedBlockVolume" in source
    assert "export interface Bslz4SelectedBlockMetadata" in source
    assert "export interface H5BlockIndexMetadata" in source
    assert "export function readH5VolumeFromBlockIndex" in source
    assert "export function readH5BlockIndexMetadata" in source
    assert "export function readBslz4SelectedBlockVolume" in source
    assert "export function readBslz4SelectedBlockMetadata" in source
    assert "QBSLZ4S1" in source
    assert "QH5IDX01" in source


def test_webgpu_bslz4_uses_fused_integer_to_uint8_decoder() -> None:
    from quantem.gpu.webgpu import source_text

    source = source_text("bslz4.ts")

    assert 'type IntegralSrcDtype = "uint8" | "uint16" | "uint32";' in source
    assert "export interface Bslz4BatchProfile" in source
    assert "variant: string;" in source
    assert 'const fused = dtype === "uint8" && srcDtype !== "float32";' in source
    assert "BSLZ4_LOW8_ONLY" in source
    assert "BSLZ4_COOP_LOW8" in source
    assert "BSLZ4_FRAME_LOW8" in source
    assert "BSLZ4_LOW8_U32_SHARED" in source
    assert "BSLZ4_SINGLE_PARSE_LOW8" in source
    assert "BSLZ4_UPLOAD_WRITEBUFFER" in source
    assert "BSLZ4_UPLOAD_MAPPED" in source
    assert "BSLZ4_UPLOAD_COMBINED" in source
    assert "FUSED_LOW8_WGSL" in source
    assert "FUSED_COOP_LOW8_WGSL" in source
    assert "FUSED_FRAME_COOP_LOW8_WGSL" in source
    assert "FUSED_FRAME_U32_LOW8_WGSL" in source
    assert "FUSED_FRAME_SINGLEPARSE_LOW8_WGSL" in source
    assert "fused-low8-experimental" in source
    assert "fused-coop-low8-experimental" in source
    assert "fused-frame-coop-low8-experimental" in source
    assert "fused-frame-u32-low8-experimental" in source
    assert "fused-frame-singleparse-low8-experimental" in source
    assert "uploadViaMapped" in source
    assert "stageUploadCopies" in source
    assert "profile.uploadMs" in source
    assert "profile.gpuWaitMs" in source
    assert 'const nbits = srcDtype === "uint32" ? 32 : srcDtype === "uint16" ? 16 : 8;' in source
    assert 'fused ? fusedBuild(device, s, srcDtype as IntegralSrcDtype, raws![i])' in source
    assert 'PASS1+PASS2_U8SRC' in source
    assert "export interface Bslz4MaskedSumSpec" in source
    assert "sourceStartScan?: number;" in source
    assert "frameStart?: number;" in source
    assert "frameCount?: number;" in source
    assert "selectedBlockIds?: number[];" in source
    assert "export function maskedSumBlockIds" in source
    assert "export function selectedBlockIdsCover" in source
    assert "export function sliceMaskedSumSpecsByScanRegion" in source
    assert "selectedGroups: number;" in source
    assert "groupBlockTable: Uint32Array;" in source
    assert "MASKED_SUM_LOW8_PIXEL_WGSL" in source
    assert "MASKED_SUM_LOW8_GROUPMASK_WGSL" in source
    assert "__QT_BSLZ4_MASKED_SUM_WG" in source
    assert "__QT_BSLZ4_MASKED_SUM_GROUPMASK" in source
    assert "__QT_BSLZ4_MASKED_SUM_COMPACT_SHARED" in source
    assert "return scanCount > 256 * 256;" in source
    assert "return scanCount > 512 * 512;" in source
    assert "pixel-wg${wgSize}" in source
    assert "groupmask-wg${wgSize}" in source
    assert "compactSh" in source
    assert "function sameBlockIds" in source
    assert "function sliceCompactedBslz4Frames" in source
    assert "function compactSelectedBslz4Blocks" in source
    assert "function compactBslz4Blocks" in source
    assert "decodeBslz4MaskedSumLow8Batch" in source


def test_webgpu_local_h5_source_tracks_show4dstem_loader_contract() -> None:
    from quantem.gpu.webgpu import source_text

    source = source_text("local-h5.ts")

    assert "export function setShow4DSTEMLocalFiles" in source
    assert "export function show4DSTEMHasLocalFiles" in source
    assert "export async function collectShow4DSTEMLocalH5Files" in source
    assert "export async function loadShow4DSTEMLocalH5Master" in source
    assert "export async function loadShow4DSTEMLocalH5MaskedSum" in source
    assert "readBslz4SelectedBlockMetadata" in source
    assert "readBslz4SelectedBlockVolume" in source
    assert "chooseSelectedBlockSidecars" in source
    assert "filterSelectedBlockSidecarsForScanRegion" in source
    assert "scanSpanIntersectsRegion" in source
    assert "chooseMaskedSumProductBatch" in source
    assert "localSidecarCandidatesForMaster" in source
    assert "localBlockIndexFor" in source
    assert 'sourceMode: "native-h5" | "selected-block-sidecar"' in source
    assert "const READ_WORKER_SOURCE" in source
    assert "new Blob([READ_WORKER_SOURCE]" in source
    assert "new Worker(workerUrl!)" in source
    assert 'new Worker(workerUrl!, { type: "module" })' not in source
    assert "function parseVolumeFromFrameIndex(buffer, index)" in source
    assert "function parseVolumeFromBlockIndex(buffer, indexBuffer, name)" in source
    assert "blockIndexFile" in source
    assert "self.postMessage({ id, name, volume: parsed.volume" in source
    assert "const frameIndexedDataItems = dataItems.length > 0" in source
    assert "const blockIndexedDataItems = dataItems.length > 0" in source
    assert "? (!region && (blockIndexedDataItems || frameIndexedDataItems) ? 8 : 2)" in source
    assert "readFiles(dataItems.map((item) => item.file), workerCount, frameIndexFor, localBlockIndexFor)" in source
    assert 'parseMode: blockIndexFiles === dataItems.length' in source
    assert '"block-index"' in source
    assert "decodeBslz4Batch" in source
    assert "decodeBslz4MaskedSumLow8Batch" in source
    assert "sliceMaskedSumSpecsByScanRegion" in source
    assert 'acquisitionMode: "local-file"' in source
    assert 'acquisitionMode: "local-file-product-first"' in source
    assert "localDataFilesForMaster" in source
    assert "readH5MasterInfo(await master.arrayBuffer()" in source
    assert "workerCount" in source
    assert "groupSize" in source
    assert "decodeVariant" in source


def test_webgpu_source_rejects_unknown_names() -> None:
    from quantem.gpu.webgpu import source_text

    with pytest.raises(ValueError, match="Unknown WebGPU source"):
        source_text("missing.ts")
