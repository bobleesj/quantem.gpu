from __future__ import annotations

from importlib.resources import files

import pytest


def source_text(name: str) -> str:
    """Read one domain-owned WebGPU source resource."""

    return files("quantem.gpu").joinpath(name).read_text(encoding="utf-8")


def test_webgpu_sources_are_shipped_and_readable() -> None:
    names = [
        "device/webgpu.ts",
        "detector/geometry.ts",
        "detector/compute/webgpu/backend.ts",
        "dpc/compute/webgpu/fft.ts",
        "dpc/compute/webgpu/kernels.ts",
        "ssb/compute/webgpu/backend.ts",
        "ssb/compute/webgpu/optimizer.ts",
        *(f"ssb/compute/webgpu/kernels/fft{size}.ts" for size in (128, 256, 512, 1024)),
    ]
    for name in names:
        text = source_text(name)
        assert text.strip()


def test_webgpu_compute_source_tracks_vi_and_dpc_kernels() -> None:
    detector_source = source_text("detector/compute/webgpu/backend.ts")
    dpc_source = source_text("dpc/compute/webgpu/kernels.ts")
    source = detector_source + "\n" + dpc_source

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
    assert "const DPC_COMPONENT_PAIR_WGSL" in source
    assert "const DPC_OUTPUT_ULP_CORRECT_WGSL" in source
    assert 'import { FFT_2D_SHADER } from "../../../dpc/compute/webgpu/fft";' in detector_source
    assert "const IDPC_PACK_WGSL" in source
    assert "const IDPC_POISSON_WGSL" in source
    assert "const IDPC_EXTRACT_WGSL" in source
    assert "maskedDpcBuffer(mask: Uint32Array" in source
    assert "maskedDpc(mask: Uint32Array" in source
    assert "maskedDpcPairBuffers(mask: Uint32Array" in source
    assert "maskedIDpcBuffer(" in source
    assert "maskedIDpc(" in source
    assert "gradFft[i] = vec2<f32>(gradRow, gradCol)" in source
    assert "let zMirrorConj = vec2<f32>(gradFft[mirror].x, -gradFft[mirror].y)" in source
    assert "runFFT2DInPlace" in source
    assert "getDevice(): GPUDevice" in source
    assert "readFloatBuffer(buf: GPUBuffer" in source
    assert "checksumFrames(scanIndices: number[])" in source
    assert "enc.copyBufferToBuffer(ch.buffer, byteOffset, rb, 0, byteLength)" in source
    assert "const bad = this.badPx.length ? new Set(this.badPx) : null;" in source
    assert "bad?.has(i) ? 0" in source
    assert "if (mode == 3u) { return data[gp]; }" in source
    assert "const values = new Uint32Array(mapped, 0, this.detSize);" in source
    assert "subgroupAdd" in source


def test_detector_geometry_source_is_shared() -> None:
    geometry = source_text("detector/geometry.ts")
    assert "export function diskMask" in geometry
    assert "export function annulusMask" in geometry


def test_webgpu_backend_source_tracks_ssb_engine() -> None:
    source = source_text("ssb/compute/webgpu/backend.ts")

    assert 'from "../../../device/webgpu"' in source
    assert 'from "../../../io/backends/webgpu/h5reader"' in source
    assert 'from "../../../io/backends/webgpu/bslz4"' in source
    assert "export class WebGPUSSBBackend" in source
    assert 'real_dtype: "float32"' in source
    assert 'complex_dtype: "complex64"' in source
    assert "requires the quantem.gpu float32/complex64 SSB precision contract" in source
    registry = source_text("ssb/compute/webgpu/kernels/index.ts")
    assert "SUPPORTED_SSB_SIZES = [128, 256, 512, 1024]" in registry
    assert "makeSsbShader" in source
    assert "WebGPU SSB buffers are not ready after setup" in source
    assert "scientific fitting never subsamples automatically" in source
    assert "BF clamped" not in source
    assert "const phi12Deg = phi12 * 180 / Math.PI" in source
    assert 'encoding="u4" remains reserved for packed 4-bit' in source
    assert 'raw === "<u4" || raw === ">u4" || raw === "|u4"' in source
    active_selector = source[
        source.index("function collectActiveBfIndices("):
        source.index("function packGeometry", source.index("function collectActiveBfIndices("))
    ]
    assert "for (let i = 0; i < cal.num_bf; i++)" in active_selector
    assert active_selector.index("for (let i = 0; i < cal.num_bf; i++)") < active_selector.index("const count =")
    assert "Math.floor((i + 0.5) * stride)" in active_selector
    assert "if (computeLoss) {" in source
    assert "if (computeLoss && bfCount === this.cal.num_bf)" not in source
    # Native 512/1024 SSB shaders are large. Synchronous pipeline creation can
    # freeze Chromium's renderer for minutes on a cold Metal shader cache.
    assert ".createComputePipeline(" not in source
    assert source.count(".createComputePipelineAsync(") >= 14


def test_webgpu_ssb_calibration_fails_explicitly() -> None:
    optimizer = source_text("ssb/compute/webgpu/optimizer.ts")

    assert "aberration optimization is not implemented" in optimizer
    assert "exact 200-trial plus Nelder-Mead workflow on CUDA or MPS" in optimizer
    assert "never substitutes a reduced objective" in optimizer


def test_webgpu_h5reader_keeps_single_pass_block_metadata_parse() -> None:
    source = source_text("io/backends/webgpu/h5reader.ts")

    assert "unsupported HDF5 chunk codec or uncompressed detector chunks" in source
    assert "validateBslz4ChunkHeader" in source
    assert "const meta = new Uint32Array(framesThisChunk * nBlocksPerFrame * 2);" in source
    assert "meta[m++] = addr + pos + 4;" in source
    assert "meta[i] -= rangeStart;" in source
    assert "const frameBlockMeta: number[][] = new Array(nFrames);" not in source
    assert "readFrameBlockMeta(f, 0, []);" not in source
    assert "export interface Bslz4SelectedBlockVolume" in source
    assert "export interface Bslz4SelectedBlockMetadata" in source
    assert "export interface H5BlockIndexMetadata" in source
    assert "dataFileIndexes?: number[];" in source
    assert "dataFileCount?: number;" in source
    assert "function readDataFileIndexes(buffer: ArrayBuffer): number[]" in source
    assert "bytes[i + 12] !== 46" in source
    assert "const dataFileIndexes = readDataFileIndexes(buffer);" in source
    assert "const dataFileCount = dataFileIndexes.length" in source
    assert "export function readH5VolumeFromBlockIndex" in source
    assert "export function readH5BlockIndexMetadata" in source
    assert "export function readBslz4SelectedBlockVolume" in source
    assert "export function readBslz4SelectedBlockMetadata" in source
    assert "QBSLZ4S1" in source
    assert "QH5IDX01" in source
    assert '"<u4" is an HDF5/NumPy source-dtype spelling' in source
    assert "/(^|[<>|])u4|uint32|int32/" in source


def test_webgpu_bslz4_uses_fused_integer_to_uint8_decoder() -> None:

    source = source_text("io/backends/webgpu/bslz4.ts")

    assert 'type IntegralSrcDtype = "uint8" | "uint16" | "uint32";' in source
    assert 'type SourceDtype = "uint8" | "uint16" | "uint32" | "float32";' in source
    assert 'type DecodeDtype = "uint8" | "uint16" | "uint32" | "float32";' in source
    assert "export interface Bslz4BatchProfile" in source
    assert "variant: string;" in source
    assert "function validateDecodeDtypes" in source
    assert "WebGPU dtype='uint32' requires a uint32 source." in source
    assert 'const fused = (dtype === "uint8" && srcDtype !== "float32") || fusedU16;' in source
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
    assert "mode: f32 ? 2 : nativeU32 ? 3 : u8 ? 1 : 0" in source
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
    assert "let STAGING_READY: Promise<void> = Promise.resolve();" in source
    assert "await STAGING_READY.catch(() => undefined);" in source
    assert "STAGING_READY = copyDone;" in source


def test_webgpu_local_h5_source_tracks_show4dstem_loader_contract() -> None:

    source = source_text("io/backends/webgpu/local-h5.ts")

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
    assert "detBin?: number;" in source
    assert "const ZERO_BAD_PIXELS_WGSL" in source
    assert "const DETECTOR_BIN_WGSL" in source
    assert "async function binDetectorChunks" in source
    assert "decoded.buffers.forEach((buffer) => buffer.destroy())" in source
    assert "badPixelClearSpecs(badPixels" in source
    assert "Detector shape ${vol.detRows}x${vol.detCols} is not divisible by detBin=${detBin}" in source
    assert 'type DecodeDtypeRequest = DecodeDtype | "u1" | "u2" | "u4" | "u32" | "uint4" | "native" | "auto";' in source
    assert "decodeDtype?: DecodeDtypeRequest;" in source
    assert "normalizeDecodeDtypeRequest" in source
    assert 'if (token === "u32") return "uint32";' in source
    assert "WebGPU decodeDtype='u4' means packed 4-bit counts" in source
    assert "WebGPU decodeDtype='uint32' requires a uint32 HDF5 source." in source
    assert "if (mode == 3u)" in source
    assert "Product-first WebGPU masked sums currently use the low-8 decode" in source
    assert "load the native uint32 stack with decodeDtype='native'" in source
    assert "detBinMs" in source
    assert "badPixels: detBin > 1 ? new Uint32Array(0) : badPixels" in source


def test_webgpu_has_no_top_level_compatibility_namespace() -> None:
    import quantem.gpu as gpu

    assert "webgpu" not in gpu.__all__
    with pytest.raises(AttributeError):
        getattr(gpu, "webgpu")
