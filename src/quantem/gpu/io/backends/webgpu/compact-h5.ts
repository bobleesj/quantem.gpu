/// <reference types="@webgpu/types" />
// Exact QuantEM compact-HDF5 residency for browser WebGPU.
//
// The logical 4D tensor is never allocated. Each raw-LZ4 shard decodes into its
// final bit-packed GPUBuffer, while a u32 descriptor buffer provides exact
// selected-diffraction and virtual-detector random access.

import {
  getGPUInfo,
  isSoftwareGPUAdapter,
  requireHardwareGPUDevice,
} from "../../../device/webgpu";
import {
  DPC_COMPONENT_PAIR_WGSL,
  DPC_MEAN_WGSL,
  DPC_OUTPUT_MEAN_WGSL,
  DPC_OUTPUT_ULP_CORRECT_WGSL,
  IDPC_EXTRACT_WGSL,
  IDPC_PACK_WGSL,
  IDPC_POISSON_WGSL,
} from "../../../dpc/backends/webgpu/kernels";
import { FFT_2D_SHADER } from "../../../dpc/backends/webgpu/fft";
import {
  EXACT_INTEGER_COM_WGSL,
  planExactIntegerCoM,
} from "../../../detector/backends/webgpu/exact-com";
import { StreamingSha256 } from "./logical-pixel-hash";
import { metadataSha256, requireMatchingResidentReceipt, type CompactH5ResidentReceipt } from "./resident-contract";
export type { CompactH5ResidentReceipt } from "./resident-contract";

const CONTAINER_MAGIC = [0x51, 0x47, 0x50, 0x55, 0x48, 0x35, 0x00, 0x01];
const INDEX_MAGIC_V1 = [0x51, 0x47, 0x49, 0x58, 0x00, 0x00, 0x00, 0x01];
const INDEX_MAGIC_V3 = [0x51, 0x47, 0x49, 0x58, 0x00, 0x00, 0x00, 0x03];
const PRELUDE_BYTES = 24;
const INDEX_HEADER_V1_BYTES = 36;
const INDEX_HEADER_V3_BYTES = 44;
const SHARD_RECORD_BYTES = 96;
const SCAN_TILE_V1 = 128;
const SCAN_TILE_V3 = 32;
const V3_CHECKPOINT_TILES = 32;
const V3_WIDTHS_PER_WORD = 8;
const MAX_INLINE_DPC_SAMPLE_VISITS = 1 << 28;

export interface CompactH5ShardIndex {
  payloadOffset: number;
  payloadBytes: number;
  lengthsOffset: number;
  lengthsBytes: number;
  widthsOffset: number;
  widthsBytes: number;
  decodedBytes: number;
  descriptorCount: number;
  chunkCount: number;
  decodedSha256: string;
  encodedEnvelopeSha256: string | null;
}

export interface CompactH5DetectorCalibration {
  readonly schema: "quantem.gpu.detector-calibration/v1";
  readonly sourceIdentitySha256: string;
  readonly detectorCenterPx: readonly [number, number];
  readonly brightFieldRadiusPx: number;
  readonly dpcRotationDegrees: number | null;
  readonly dpcComponentOrderExchanged: boolean | null;
  readonly method: string;
}

export interface CompactH5PreparedDpcMoments {
  readonly schema: "quantem.gpu.prepared-dpc-moments/v1";
  readonly fileOffset: number;
  readonly fileBytes: number;
  readonly sha256: string;
  readonly workingUint8Sha256: string;
  readonly detectorMaskSha256: string;
  readonly scanCount: number;
  readonly selectedDetectorPixels: number;
  readonly detectorColumns: number;
  readonly totalBound: string;
  readonly rowMomentBound: string;
  readonly columnMomentBound: string;
  readonly narrowInteger: boolean;
  readonly narrowProducts: boolean;
}

export interface CompactH5PreparedDetectorProduct {
  readonly name: "bf" | "abf" | "adf";
  readonly centerPx: readonly [number, number];
  readonly innerRadiusPx: number;
  readonly outerRadiusPx: number;
  readonly selectedDetectorPixels: number;
  readonly maskFileOffset: number;
  readonly maskFileBytes: number;
  readonly maskSha256: string;
  readonly valuesFileOffset: number;
  readonly valuesFileBytes: number;
  readonly valuesSha256: string;
}

export interface CompactH5PreparedDetectorProducts {
  readonly schema: "quantem.gpu.prepared-detector-products/v1";
  readonly workingUint8Sha256: string;
  readonly detectorMaskSha256: string;
  readonly products: readonly CompactH5PreparedDetectorProduct[];
}

export interface CompactH5Index {
  sourceBytes: number;
  sourceName: string;
  schemaVersion: 1 | 3;
  shape: readonly [number, number, number, number];
  scansPerShard: number;
  scanTile: 128 | 32;
  headerEncoding: 0 | 1;
  payloadChunkBytes: 128 | 0;
  payloadCodec: "raw-lz4" | "direct-bitpacked-u32";
  sourceIdentitySha256: string;
  sourceRawLogicalSha256: string | null;
  workingDtype: "uint8" | "uint16";
  detectorCalibration: CompactH5DetectorCalibration | null;
  preparedDpcMoments: CompactH5PreparedDpcMoments | null;
  preparedDetectorProducts: CompactH5PreparedDetectorProducts | null;
  excludedDetectorPixels: Uint32Array;
  maskedDetectorPixelsSha256: string | null;
  maskedDetectorRawValues: Uint16Array | null;
  rawReconstructionAvailable: boolean;
  shards: readonly CompactH5ShardIndex[];
  residentBytes: number;
  manifest: Readonly<Record<string, unknown>>;
  /** Populated only after complete authenticated residency and exact raw access. */
  residentReceipt?: CompactH5ResidentReceipt | null;
}

export interface CompactH5TrustedQualificationShardV1 {
  readonly ordinal: number;
  readonly payloadOffset: number;
  readonly payloadBytes: number;
  readonly payloadSha256: string;
  readonly headerOffset: number;
  readonly headerBytes: number;
  readonly headerSha256: string;
}

export interface CompactH5TrustedQualificationV1 {
  readonly schema: "quantem.gpu.compact-h5-trusted-qualification/v1";
  readonly qualificationOrigin: "writer-close-reopen-whole-file-sha256";
  readonly headerValidation: "canonical-cpu-all-shards";
  readonly sealedWholeFileSha256: string;
  readonly sourceBytes: number;
  readonly sourceIdentitySha256: string;
  readonly prefixBytes: number;
  readonly prefixSha256: string;
  readonly shards: readonly CompactH5TrustedQualificationShardV1[];
}

export interface WebGPUCompactH5LoadProfile {
  schemaVersion: 1 | 3;
  adapterInfo: string;
  softwareAdapter: boolean;
  sourceBytes: number;
  residentBytes: number;
  logicalDenseAllocationBytes: 0;
  metadataMs: number;
  pipelineCompileMs: number;
  sourceReadMs: number;
  gpuUploadSubmissionMs: number;
  gpuUploadFenceMs: number;
  gpuUploadMode: "mapped-at-creation" | "queue-write-buffer";
  descriptorPreparationMs: number;
  gpuDecodeWallMs: number;
  compactHeaderValidationMs: number;
  compactHeaderValidationMode: "gpu-structural" | "trusted-sidecar-exact-header-digest" | "not-applicable";
  decodedIntegrityMs: number;
  encodedIntegrityMs: number;
  wholeFileIntegrityMs: number;
  trustedPrefixIntegrityMs: number;
  residentReadyMs: number;
  maximumTransientBytes: number;
  preparedDpcBytes: number;
  preparedDpcReadMs: number;
  preparedDpcPrimeMs: number;
  preparedDetectorProductBytes: number;
  preparedDetectorProductReadMs: number;
  decodedShardSha256Checks: number;
  encodedShardSha256Checks: number;
  directPayloadSha256Checks: number;
  directHeaderSha256Checks: number;
  wholeFileSha256Checks: 0 | 1;
  trustedPrefixSha256Checks: 0 | 1;
  sessionQualificationReused: 0 | 1;
  sealedWholeFileSha256: string | null;
  integrityMode: "decoded-sha256" | "authenticated-encoded-envelope" | "whole-file-plus-direct-payload-sha256" | "trusted-sidecar-plus-direct-range-sha256" | "trusted-sidecar-plus-session-qualified-direct-range";
  fftDispatchCount: 0;
}

export interface WebGPUCompactH5DetectorMetrics {
  mode: "rebase" | "delta" | "prepared";
  changedDetectorPixels: number;
  addressingMode: "inline-header" | "resolved-v3-tiles";
  dispatchChunks: number;
  wallMs: number;
  gpuMs: number | null;
  fftDispatchCount: 0;
}

export interface WebGPUCompactH5ExactMomentSnapshot {
  readonly wordOrder: "little-endian-u32-pairs";
  readonly wordsPerScan: 8;
  readonly layout: readonly [
    "total_lo", "total_hi", "row_lo", "row_hi",
    "column_lo", "column_hi", "padding_0", "padding_1",
  ];
  readonly scanCount: number;
  readonly selectedDetectorPixels: number;
  readonly totalBound: string;
  readonly rowMomentBound: string;
  readonly columnMomentBound: string;
  readonly narrowInteger: boolean;
  readonly narrowProducts: boolean;
  readonly words: Uint32Array;
}

export interface CompactH5ByteSource {
  readonly size: number;
  readonly name?: string;
  readRange(offset: number, byteCount: number): Promise<Uint8Array>;
}

export type CompactH5Source = (Blob & { name?: string }) | CompactH5ByteSource;

interface ResidentShard {
  payload: GPUBuffer;
  descriptors: GPUBuffer;
}

interface CompactShardBytes {
  compressedBytes: Uint8Array;
  lengthBytes: Uint8Array;
  widthBytes: Uint8Array;
  encodedDigest: Promise<string> | null;
  elapsedMs: number;
}

interface DirectCompactShardBytes {
  payloadBytes: Uint8Array;
  headerBytes: Uint8Array;
  payloadDigest: Promise<string> | null;
  headerDigest: Promise<string> | null;
  elapsedMs: number;
}

/**
 * Records successful qualification of immutable browser File/Blob objects.
 *
 * A Blob is immutable for the lifetime of its object. After every direct
 * payload and header range has matched a trusted whole-file qualification,
 * later loads of that exact object can reuse the proof while still rereading
 * and uploading every byte. URL-backed sources are deliberately excluded
 * because their contents can change behind the same object.
 */
export class CompactH5SessionQualificationCache {
  private readonly verified = new WeakMap<Blob, Set<string>>();

  has(source: CompactH5Source, sealedWholeFileSha256: string): boolean {
    return typeof Blob !== "undefined" && source instanceof Blob
      && this.verified.get(source)?.has(sealedWholeFileSha256) === true;
  }

  record(source: CompactH5Source, sealedWholeFileSha256: string): void {
    if (typeof Blob === "undefined" || !(source instanceof Blob)) return;
    const seals = this.verified.get(source) ?? new Set<string>();
    seals.add(sealedWholeFileSha256);
    this.verified.set(source, seals);
  }
}

interface CompactIdpcPipelines {
  pack: GPUComputePipeline;
  poisson: GPUComputePipeline;
  extract: GPUComputePipeline;
  bitReverseRows: GPUComputePipeline;
  bitReverseColumns: GPUComputePipeline;
  butterflyRows: GPUComputePipeline;
  butterflyColumns: GPUComputePipeline;
  normalize: GPUComputePipeline;
}

type CompactDpcPlan = ReturnType<typeof planExactIntegerCoM>;

const COMPACT_LZ4_WGSL = /* wgsl */ `
struct DecodeConfig {
  decodedBytes: u32,
  chunkBytes: u32,
  chunkCount: u32,
  compressedBytes: u32,
};
@group(0) @binding(0) var<storage, read> compressed: array<u32>;
@group(0) @binding(1) var<storage, read> inputOffsets: array<u32>;
@group(0) @binding(2) var<storage, read_write> decoded: array<u32>;
@group(0) @binding(3) var<storage, read_write> status: array<u32>;
@group(0) @binding(4) var<uniform> config: DecodeConfig;

fn readByte(address: u32) -> u32 {
  return extractBits(compressed[address >> 2u], (address & 3u) * 8u, 8u);
}
fn readU16LE(address: u32) -> u32 {
  let word = address >> 2u;
  let lane = address & 3u;
  if (lane == 3u) {
    return extractBits(compressed[word], 24u, 8u)
      | (extractBits(compressed[word + 1u], 0u, 8u) << 8u);
  }
  return extractBits(compressed[word], lane * 8u, 16u);
}
fn readLocalByte(values: ptr<function, array<u32, 32>>, address: u32) -> u32 {
  return extractBits((*values)[address >> 2u], (address & 3u) * 8u, 8u);
}
fn writeLocalByte(values: ptr<function, array<u32, 32>>, address: u32, value: u32) {
  let word = address >> 2u;
  let shift = (address & 3u) * 8u;
  (*values)[word] = ((*values)[word] & ~(0xffu << shift)) | ((value & 0xffu) << shift);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let chunk = gid.x;
  if (chunk >= config.chunkCount) { return; }
  var input = inputOffsets[chunk];
  let inputEnd = inputOffsets[chunk + 1u];
  let outputStart = chunk * config.chunkBytes;
  let expected = min(config.chunkBytes, config.decodedBytes - outputStart);
  var localDecoded: array<u32, 32>;
  for (var word = 0u; word < 32u; word += 1u) {
    localDecoded[word] = 0u;
  }
  var output = 0u;
  var error = select(0u, 1u, input >= inputEnd || expected == 0u || inputEnd > config.compressedBytes);
  var tokenCount = 0u;
  loop {
    if (error != 0u || input >= inputEnd || output >= expected) { break; }
    tokenCount += 1u;
    if (tokenCount > expected) { error = 2u; break; }
    let token = readByte(input);
    input += 1u;
    var literalCount = token >> 4u;
    if (literalCount == 15u) {
      var extension = 255u;
      loop {
        if (extension != 255u) { break; }
        if (input >= inputEnd) { error = 3u; break; }
        extension = readByte(input);
        input += 1u;
        if (literalCount > 0xffffffffu - extension) { error = 3u; break; }
        literalCount += extension;
      }
    }
    if (error != 0u) { break; }
    if (literalCount > inputEnd - input || literalCount > expected - output) {
      error = 4u;
      break;
    }
    for (var i = 0u; i < literalCount; i += 1u) {
      writeLocalByte(&localDecoded, output + i, readByte(input + i));
    }
    input += literalCount;
    output += literalCount;
    if (output == expected || input == inputEnd) { break; }
    if (inputEnd - input < 2u) { error = 5u; break; }
    let matchOffset = readU16LE(input);
    input += 2u;
    if (matchOffset == 0u || matchOffset > output) {
      error = 0x60000000u | (min(matchOffset, 0xffffu) << 8u) | min(output, 0xffu);
      break;
    }
    var matchCount = token & 15u;
    if (matchCount == 15u) {
      var extension = 255u;
      loop {
        if (extension != 255u) { break; }
        if (input >= inputEnd) { error = 7u; break; }
        extension = readByte(input);
        input += 1u;
        if (matchCount > 0xfffffffbu - extension) { error = 7u; break; }
        matchCount += extension;
      }
    }
    if (error != 0u) { break; }
    matchCount += 4u;
    if (matchCount > expected - output) { error = 8u; break; }
    for (var i = 0u; i < matchCount; i += 1u) {
      let value = readLocalByte(&localDecoded, output - matchOffset);
      writeLocalByte(&localDecoded, output, value);
      output += 1u;
    }
  }
  status[chunk] = select(9u, error, error != 0u);
  if (error == 0u && input == inputEnd && output == expected) {
    let outputWord = outputStart >> 2u;
    let wordCount = (expected + 3u) >> 2u;
    for (var word = 0u; word < wordCount; word += 1u) {
      decoded[outputWord + word] = localDecoded[word];
    }
    status[chunk] = 0u;
  }
}
`;

const COMPACT_DESCRIPTOR_WGSL = /* wgsl */ `
struct DescriptorConfig { count: u32, payloadWords: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read> descriptors: array<u32>;
@group(0) @binding(1) var<storage, read_write> status: atomic<u32>;
@group(0) @binding(2) var<uniform> config: DescriptorConfig;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= config.count) { return; }
  let descriptor = descriptors[index];
  let width = descriptor & 31u;
  let offset = descriptor >> 5u;
  var error = 0u;
  if (width > 16u) { error |= 1u; }
  if (offset >= (1u << 27u)) { error |= 2u; }
  let expectedNext = offset + width * 4u;
  if (index + 1u < config.count) {
    if ((descriptors[index + 1u] >> 5u) != expectedNext) { error |= 4u; }
  } else if (expectedNext != config.payloadWords) {
    error |= 8u;
  }
  if (error != 0u) { atomicOr(&status, error); }
}
`;

const COMPACT_V3_HEADER_VALIDATE_WGSL = /* wgsl */ `
struct HeaderConfig { detectorPixels: u32, tileCount: u32, payloadWords: u32, headerWordsPerPixel: u32 };
@group(0) @binding(0) var<storage, read> headers: array<u32>;
@group(0) @binding(1) var<storage, read> excluded: array<u32>;
@group(0) @binding(2) var<storage, read_write> status: atomic<u32>;
@group(0) @binding(3) var<uniform> config: HeaderConfig;

fn widthAt(headerBase: u32, checkpointWords: u32, tile: u32) -> u32 {
  let packed = headers[headerBase + checkpointWords + tile / 8u];
  return (packed >> ((tile & 7u) * 4u)) & 15u;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let checkpointWords = (config.tileCount + 31u) / 32u;
  let widthWords = (config.tileCount + 7u) / 8u;
  let entry = gid.x;
  if (entry >= config.detectorPixels * checkpointWords) { return; }
  let pixel = entry / checkpointWords;
  let checkpoint = entry - pixel * checkpointWords;
  if (checkpointWords + widthWords != config.headerWordsPerPixel) {
    atomicOr(&status, 1u);
    return;
  }
  let headerBase = pixel * config.headerWordsPerPixel;
  let payloadBase = headers[headerBase];
  if (checkpoint == 0u && pixel == 0u && payloadBase != 0u) { atomicOr(&status, 2u); }
  let firstTile = checkpoint * 32u;
  let lastTile = min(firstTile + 32u, config.tileCount);
  var segmentWords = 0u;
  for (var tile = firstTile; tile < lastTile; tile += 1u) {
    let width = widthAt(headerBase, checkpointWords, tile);
    if (width > 8u) { atomicOr(&status, 8u); }
    if (excluded[pixel] != 0u && width != 0u) { atomicOr(&status, 16u); }
    segmentWords += width;
  }
  let cumulativeStart = select(headers[headerBase + checkpoint], 0u, checkpoint == 0u);
  let cumulativeEnd = cumulativeStart + segmentWords;
  if (checkpoint + 1u < checkpointWords) {
    if (headers[headerBase + checkpoint + 1u] != cumulativeEnd) { atomicOr(&status, 4u); }
  }
  if (checkpoint + 1u == checkpointWords && config.tileCount % 8u != 0u) {
    let usedNibbles = config.tileCount % 8u;
    let tail = headers[headerBase + checkpointWords + widthWords - 1u];
    if ((tail >> (usedNibbles * 4u)) != 0u) { atomicOr(&status, 32u); }
  }
  if (checkpoint + 1u == checkpointWords) {
    let end = payloadBase + cumulativeEnd;
    if (pixel + 1u < config.detectorPixels) {
      let nextBase = headers[headerBase + config.headerWordsPerPixel];
      if (nextBase != end) { atomicOr(&status, 64u); }
    } else if (end != config.payloadWords) {
      atomicOr(&status, 128u);
    }
  }
}
`;

// QGIX v1 stores one expanded offset/width descriptor per 128-scan tile.
// QGIX v3 stores one base/checkpoint/packed-width header row per detector pixel.
// Both paths feed the same public scientific kernels without reinterpreting
// either binary schema or expanding v3 headers into a second resident index.
const COMPACT_SAMPLE_WGSL = /* wgsl */ `
fn v3Width(headerBase: u32, checkpointWords: u32, tile: u32) -> u32 {
  let packed = descriptors[headerBase + checkpointWords + tile / 8u];
  return (packed >> ((tile & 7u) * 4u)) & 15u;
}

fn sampleValue(pixel: u32, scan: u32, tileCount: u32, schemaVersion: u32) -> u32 {
  var width: u32;
  var index: u32;
  var shift: u32;
  if (schemaVersion == 3u) {
    let tile = scan / 32u;
    let checkpointWords = (tileCount + 31u) / 32u;
    let widthWords = (tileCount + 7u) / 8u;
    let headerBase = pixel * (checkpointWords + widthWords);
    width = v3Width(headerBase, checkpointWords, tile);
    if (width == 0u) { return 0u; }
    let checkpoint = tile / 32u;
    let pixelBase = descriptors[headerBase];
    index = pixelBase;
    if (checkpoint > 0u) {
      index += descriptors[headerBase + checkpoint];
    }
    var previousTile = checkpoint * 32u;
    loop {
      if (previousTile >= tile) { break; }
      index += v3Width(headerBase, checkpointWords, previousTile);
      previousTile += 1u;
    }
    let bit = (scan & 31u) * width;
    index += bit / 32u;
    shift = bit & 31u;
  } else {
    let descriptor = descriptors[pixel * tileCount + scan / 128u];
    width = descriptor & 31u;
    if (width == 0u) { return 0u; }
    let bit = (scan & 127u) * width;
    index = (descriptor >> 5u) + bit / 32u;
    shift = bit & 31u;
  }
  var value = payload[index] >> shift;
  if (shift + width > 32u) { value |= payload[index + 1u] << (32u - shift); }
  return value & ((1u << width) - 1u);
}
`;

const COMPACT_SELECTED_WGSL = /* wgsl */ `
struct SelectedConfig { scan: u32, tileCount: u32, pixelCount: u32, schemaVersion: u32, raw: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read> payload: array<u32>;
@group(0) @binding(1) var<storage, read> descriptors: array<u32>;
@group(0) @binding(2) var<storage, read> excluded: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<u32>;
@group(0) @binding(4) var<uniform> config: SelectedConfig;
${COMPACT_SAMPLE_WGSL}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pixel = gid.x;
  if (pixel >= config.pixelCount) { return; }
  output[pixel] = select(sampleValue(pixel, config.scan, config.tileCount, config.schemaVersion), 0u, config.raw == 0u && excluded[pixel] != 0u);
}
`;

const COMPACT_DETECTOR_WGSL = /* wgsl */ `
struct DetectorConfig {
  scanCount: u32,
  tileCount: u32,
  entryCount: u32,
  outputOffset: u32,
  rebase: u32,
  schemaVersion: u32,
  _b: u32,
  _c: u32,
};
@group(0) @binding(0) var<storage, read> payload: array<u32>;
@group(0) @binding(1) var<storage, read> descriptors: array<u32>;
@group(0) @binding(2) var<storage, read> entries: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read> previous: array<u32>;
@group(0) @binding(4) var<storage, read_write> next: array<u32>;
@group(0) @binding(5) var<uniform> config: DetectorConfig;
var<workgroup> partials: array<u32, 256>;
// Each workgroup covers exactly 32 consecutive scans. All 32 scan lanes for
// one entry therefore use the same packed width and tile word base. Resolve
// that descriptor once (the scanLane==0 invocation of each four-way entry
// subgroup) and share it, rather than walking the v3 checkpoint header 32
// times. Values and accumulation remain exact u32; only redundant addressing
// work is removed.
var<workgroup> entryWordBases: array<u32, 8>;
var<workgroup> entryWidths: array<u32, 8>;
var<workgroup> entrySigns: array<u32, 8>;

fn resolveV3TileWordBase(pixel: u32, tile: u32) -> vec2<u32> {
  let checkpointWords = (config.tileCount + 31u) / 32u;
  let widthWords = (config.tileCount + 7u) / 8u;
  let headerBase = pixel * (checkpointWords + widthWords);
  let packedWidth = descriptors[headerBase + checkpointWords + tile / 8u];
  let width = (packedWidth >> ((tile & 7u) * 4u)) & 15u;
  if (width == 0u) { return vec2<u32>(0u, 0u); }
  let checkpoint = tile / 32u;
  var wordBase = descriptors[headerBase];
  if (checkpoint > 0u) {
    wordBase += descriptors[headerBase + checkpoint];
  }
  var previousTile = checkpoint * 32u;
  loop {
    if (previousTile >= tile) { break; }
    let packed = descriptors[headerBase + checkpointWords + previousTile / 8u];
    wordBase += (packed >> ((previousTile & 7u) * 4u)) & 15u;
    previousTile += 1u;
  }
  return vec2<u32>(wordBase, width);
}

fn resolveV1TileWordBase(pixel: u32, scan: u32) -> vec2<u32> {
  let descriptor = descriptors[pixel * config.tileCount + scan / 128u];
  return vec2<u32>(descriptor >> 5u, descriptor & 31u);
}

@compute @workgroup_size(256)
fn main(
  @builtin(local_invocation_index) lane: u32,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  let scanLane = lane & 31u;
  let entryLane = lane >> 5u;
  let scan = group.x * 32u + scanLane;
  var partial = 0u;
  if (config.schemaVersion == 1u) {
    // The v1 descriptor already contains the direct tile address. Register
    // loads are cheaper than two full-workgroup barriers for every eight
    // detector pixels. Consecutive scan lanes still read consecutive bits.
    for (var entryIndex = entryLane; entryIndex < config.entryCount; entryIndex += 8u) {
      let entry = entries[entryIndex];
      let resolved = resolveV1TileWordBase(entry.x, scan);
      let width = resolved.y;
      if (scan < config.scanCount && width != 0u) {
        let bit = (scan & 127u) * width;
        let shift = bit & 31u;
        let wordIndex = resolved.x + bit / 32u;
        var value = payload[wordIndex] >> shift;
        if (shift + width > 32u) {
          value |= payload[wordIndex + 1u] << (32u - shift);
        }
        value &= (1u << width) - 1u;
        partial += select(0u - value, value, bitcast<i32>(entry.y) == 1i);
      }
    }
  } else {
    for (var block = 0u; block < config.entryCount; block += 8u) {
      let entryIndex = block + entryLane;
      if (scanLane == 0u) {
        if (entryIndex < config.entryCount) {
          let entry = entries[entryIndex];
          var resolved: vec2<u32>;
          if (config.schemaVersion == 3u) {
            resolved = resolveV3TileWordBase(entry.x, group.x);
          } else {
            resolved = resolveV1TileWordBase(entry.x, scan);
          }
          entryWordBases[entryLane] = resolved.x;
          entryWidths[entryLane] = resolved.y;
          entrySigns[entryLane] = entry.y;
        } else {
          entryWordBases[entryLane] = 0u;
          entryWidths[entryLane] = 0u;
          entrySigns[entryLane] = 0u;
        }
      }
      workgroupBarrier();
      let width = entryWidths[entryLane];
      if (scan < config.scanCount && entryIndex < config.entryCount && width != 0u) {
        let bit = select((scan & 127u) * width, scanLane * width, config.schemaVersion == 3u);
        let wordIndex = entryWordBases[entryLane] + bit / 32u;
        let shift = bit & 31u;
        var value = payload[wordIndex] >> shift;
        if (shift + width > 32u) {
          value |= payload[wordIndex + 1u] << (32u - shift);
        }
        value &= (1u << width) - 1u;
        partial += select(0u - value, value, bitcast<i32>(entrySigns[entryLane]) == 1i);
      }
      workgroupBarrier();
    }
  }
  partials[lane] = partial;
  workgroupBarrier();
  if (entryLane == 0u && scan < config.scanCount) {
    let outputIndex = config.outputOffset + scan;
    var value = select(previous[outputIndex], 0u, config.rebase != 0u);
    for (var subgroup = 0u; subgroup < 8u; subgroup += 1u) {
      value += partials[scanLane + subgroup * 32u];
    }
    next[outputIndex] = value;
  }
}
`;

const COMPACT_U32_TO_F32_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> source: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> count: vec4<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index < count.x) { output[index] = f32(source[index]); }
}
`;

const COMPACT_DETECTOR_RESOLVE_V3_WGSL = /* wgsl */ `
struct DetectorConfig {
  scanCount: u32,
  tileCount: u32,
  entryCount: u32,
  outputOffset: u32,
  rebase: u32,
  schemaVersion: u32,
  resolvedOffset: u32,
  entryOffset: u32,
};
@group(0) @binding(0) var<storage, read> descriptors: array<u32>;
@group(0) @binding(1) var<storage, read> entries: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> resolved: array<vec2<u32>>;
@group(0) @binding(3) var<uniform> config: DetectorConfig;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let tile = gid.x;
  let entryIndex = gid.y;
  if (tile >= config.tileCount || entryIndex >= config.entryCount) { return; }
  let pixel = entries[config.entryOffset + entryIndex].x;
  let checkpointWords = (config.tileCount + 31u) / 32u;
  let widthWords = (config.tileCount + 7u) / 8u;
  let headerBase = pixel * (checkpointWords + widthWords);
  let packedWidth = descriptors[headerBase + checkpointWords + tile / 8u];
  let width = (packedWidth >> ((tile & 7u) * 4u)) & 15u;
  var wordBase = 0u;
  if (width != 0u) {
    let checkpoint = tile / 32u;
    wordBase = descriptors[headerBase];
    if (checkpoint > 0u) {
      wordBase += descriptors[headerBase + checkpoint];
    }
    var previousTile = checkpoint * 32u;
    loop {
      if (previousTile >= tile) { break; }
      let packed = descriptors[headerBase + checkpointWords + previousTile / 8u];
      wordBase += (packed >> ((previousTile & 7u) * 4u)) & 15u;
      previousTile += 1u;
    }
  }
  resolved[config.resolvedOffset + entryIndex * config.tileCount + tile] = vec2<u32>(wordBase, width);
}
`;

const COMPACT_DETECTOR_RESOLVED_V3_WGSL = /* wgsl */ `
struct DetectorConfig {
  scanCount: u32,
  tileCount: u32,
  entryCount: u32,
  outputOffset: u32,
  rebase: u32,
  schemaVersion: u32,
  resolvedOffset: u32,
  entryOffset: u32,
};
@group(0) @binding(0) var<storage, read> payload: array<u32>;
@group(0) @binding(1) var<storage, read> resolved: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read> entries: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read> previous: array<u32>;
@group(0) @binding(4) var<storage, read_write> next: array<u32>;
@group(0) @binding(5) var<uniform> config: DetectorConfig;
var<workgroup> partials: array<u32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(local_invocation_index) lane: u32,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  let scanLane = lane & 31u;
  let entryLane = lane >> 5u;
  let scan = group.x * 32u + scanLane;
  var partial = 0u;
  if (scan < config.scanCount) {
    for (var index = entryLane; index < config.entryCount; index += 8u) {
      let address = resolved[config.resolvedOffset + index * config.tileCount + group.x];
      let width = address.y;
      if (width != 0u) {
        let bit = scanLane * width;
        let wordIndex = address.x + bit / 32u;
        let shift = bit & 31u;
        var value = payload[wordIndex] >> shift;
        if (shift + width > 32u) {
          value |= payload[wordIndex + 1u] << (32u - shift);
        }
        value &= (1u << width) - 1u;
        partial += select(0u - value, value, bitcast<i32>(entries[config.entryOffset + index].y) == 1i);
      }
    }
  }
  partials[lane] = partial;
  workgroupBarrier();
  if (entryLane == 0u && scan < config.scanCount) {
    let outputIndex = config.outputOffset + scan;
    var value = select(previous[outputIndex], 0u, config.rebase != 0u);
    for (var subgroup = 0u; subgroup < 8u; subgroup += 1u) {
      value += partials[scanLane + subgroup * 32u];
    }
    next[outputIndex] = value;
  }
}
`;

const COMPACT_DPC_MOMENTS_WGSL = /* wgsl */ `
${EXACT_INTEGER_COM_WGSL}
struct DpcConfig {
  scanCount: u32,
  tileCount: u32,
  entryCount: u32,
  outputOffset: u32,
  detectorColumns: u32,
  integerFlags: u32,
  schemaVersion: u32,
  _c: u32,
};
@group(0) @binding(0) var<storage, read> payload: array<u32>;
@group(0) @binding(1) var<storage, read> descriptors: array<u32>;
@group(0) @binding(2) var<storage, read> detectorPixels: array<u32>;
@group(0) @binding(3) var<storage, read_write> moments: array<vec4<u32>>;
@group(0) @binding(4) var<uniform> config: DpcConfig;
var<workgroup> partialSumsLo: array<u32, 128>;
var<workgroup> partialSumsHi: array<u32, 128>;
var<workgroup> partialRowsLo: array<u32, 128>;
var<workgroup> partialRowsHi: array<u32, 128>;
var<workgroup> partialColumnsLo: array<u32, 128>;
var<workgroup> partialColumnsHi: array<u32, 128>;
${COMPACT_SAMPLE_WGSL}

@compute @workgroup_size(128)
fn main(
  @builtin(local_invocation_index) lane: u32,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  let scanLane = lane & 31u;
  let entryLane = lane >> 5u;
  let scan = group.x * 32u + scanLane;
  var sumN = 0u;
  var rowMomentN = 0u;
  var columnMomentN = 0u;
  var sum = U64Words(0u, 0u);
  var rowMoment = U64Words(0u, 0u);
  var columnMoment = U64Words(0u, 0u);
  if (scan < config.scanCount) {
    for (var index = entryLane; index < config.entryCount; index += 4u) {
      let pixel = detectorPixels[index];
      let value = sampleValue(pixel, scan, config.tileCount, config.schemaVersion);
      let row = pixel / config.detectorColumns;
      let column = pixel - row * config.detectorColumns;
      if ((config.integerFlags & 1u) != 0u) {
        sumN += value;
        rowMomentN += value * row;
        columnMomentN += value * column;
      } else {
        sum = u64Add(sum, U64Words(value, 0u));
        if ((config.integerFlags & 2u) != 0u) {
          rowMoment = u64Add(rowMoment, U64Words(value * row, 0u));
          columnMoment = u64Add(columnMoment, U64Words(value * column, 0u));
        } else {
          rowMoment = u64Add(rowMoment, u64MultiplyU32(value, row));
          columnMoment = u64Add(columnMoment, u64MultiplyU32(value, column));
        }
      }
    }
  }
  partialSumsLo[lane] = select(sum.lo, sumN, (config.integerFlags & 1u) != 0u);
  partialSumsHi[lane] = sum.hi;
  partialRowsLo[lane] = select(rowMoment.lo, rowMomentN, (config.integerFlags & 1u) != 0u);
  partialRowsHi[lane] = rowMoment.hi;
  partialColumnsLo[lane] = select(columnMoment.lo, columnMomentN, (config.integerFlags & 1u) != 0u);
  partialColumnsHi[lane] = columnMoment.hi;
  workgroupBarrier();
  if (entryLane == 0u && scan < config.scanCount) {
    var total = U64Words(0u, 0u);
    var totalRow = U64Words(0u, 0u);
    var totalColumn = U64Words(0u, 0u);
    for (var subgroup = 0u; subgroup < 4u; subgroup += 1u) {
      let partialIndex = scanLane + subgroup * 32u;
      total = u64Add(total, U64Words(partialSumsLo[partialIndex], partialSumsHi[partialIndex]));
      totalRow = u64Add(totalRow, U64Words(partialRowsLo[partialIndex], partialRowsHi[partialIndex]));
      totalColumn = u64Add(totalColumn, U64Words(partialColumnsLo[partialIndex], partialColumnsHi[partialIndex]));
    }
    let outputIndex = (config.outputOffset + scan) * 2u;
    moments[outputIndex] = vec4<u32>(total.lo, total.hi, totalRow.lo, totalRow.hi);
    moments[outputIndex + 1u] = vec4<u32>(totalColumn.lo, totalColumn.hi, 0u, 0u);
  }
}
`;

const COMPACT_MOMENTS_TO_COM_WGSL = /* wgsl */ `
${EXACT_INTEGER_COM_WGSL}
@group(0) @binding(0) var<storage, read> moments: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read_write> com: array<f32>;
@group(0) @binding(2) var<uniform> config: vec4<u32>; // scan count, integer flags, 0, 0

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= config.x) { return; }
  let primary = moments[index * 2u];
  let secondary = moments[index * 2u + 1u];
  if ((config.y & 1u) != 0u) {
    com[index] = ratioU32Exact(primary.z, primary.x);
    com[config.x + index] = ratioU32Exact(secondary.x, primary.x);
  } else {
    let total = U64Words(primary.x, primary.y);
    com[index] = ratioU64(U64Words(primary.z, primary.w), total);
    com[config.x + index] = ratioU64(U64Words(secondary.x, secondary.y), total);
  }
}
`;

const COMPACT_REDUCE_FRAMES_WGSL = /* wgsl */ `
struct ReduceConfig {
  tileCount: u32,
  selectedScans: u32,
  detectorPixels: u32,
  schemaVersion: u32,
};
@group(0) @binding(0) var<storage, read> payload: array<u32>;
@group(0) @binding(1) var<storage, read> descriptors: array<u32>;
@group(0) @binding(2) var<storage, read> scans: array<u32>;
@group(0) @binding(3) var<storage, read> excluded: array<u32>;
@group(0) @binding(4) var<storage, read_write> output: array<atomic<u32>>;
@group(0) @binding(5) var<uniform> config: ReduceConfig;
${COMPACT_SAMPLE_WGSL}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pixel = gid.x;
  if (pixel >= config.detectorPixels || excluded[pixel] != 0u) { return; }
  var sum = 0u;
  for (var index = 0u; index < config.selectedScans; index += 1u) {
    sum += sampleValue(pixel, scans[index], config.tileCount, config.schemaVersion);
  }
  if (sum != 0u) { atomicAdd(&output[pixel], sum); }
}
`;

type CompactDetectorSubmission = { encoder: GPUCommandEncoder; completion: Promise<void> };
type CompactDetectorSelection = { mask: Uint8Array; selectedPixelCount: number; requestedAt: number };
// Widths are integral and bounded by the admitted packed schema. Avoid
// repeating exponentiation for every selected pixel of every drag update.
const MAXIMUM_PACKED_VALUES = Float64Array.from({ length: 33 }, (_, width) => 2 ** width - 1);

export class WebGPUCompactH5ResidentSource {
  readonly representation = "packed" as const;
  readonly residency = "device" as const;
  readonly metadata: CompactH5Index;
  readonly loadProfile: WebGPUCompactH5LoadProfile;
  readonly residentReceipt: CompactH5ResidentReceipt | null;
  readonly device: GPUDevice;
  readonly scanCount: number;
  readonly detSize: number;
  readonly mode = 3;
  badPx: Uint32Array;
  isReleased = false;

  get logicalBytes(): number {
    return this.metadata.shape.reduce((count, length) => count * length, 1)
      * (this.metadata.workingDtype === "uint8" ? 1 : 2);
  }

  get residentBytes(): number { return this.loadProfile.residentBytes; }

  private shards: ResidentShard[];
  private excluded: GPUBuffer;
  private maximumWidths: Uint8Array;
  private detectorOutputs: GPUBuffer[];
  private diffractionOutput: GPUBuffer;
  private detectorDisplayOutput: GPUBuffer;
  private detectorDisplayCount: GPUBuffer;
  private detectorDisplayBindGroups: GPUBindGroup[];
  private selectedPipeline: GPUComputePipeline;
  private detectorPipeline: GPUComputePipeline;
  private detectorResolveV3Pipeline: GPUComputePipeline;
  private detectorResolveV3Layout: GPUBindGroupLayout;
  private detectorResolvedV3Pipeline: GPUComputePipeline;
  private u32ToF32Pipeline: GPUComputePipeline;
  private dpcMomentPipeline: GPUComputePipeline;
  private dpcMomentLayout: GPUBindGroupLayout;
  private momentsToComPipeline: GPUComputePipeline;
  private dpcMeanPipeline: GPUComputePipeline;
  private dpcPairPipeline: GPUComputePipeline;
  private dpcOutputMeanPipeline: GPUComputePipeline;
  private dpcOutputUlpCorrectPipeline: GPUComputePipeline;
  private idpcPipelines: Promise<CompactIdpcPipelines> | null = null;
  private reduceFramesPipeline: Promise<GPUComputePipeline> | null = null;
  private fftDispatches = 0;
  private detectorLayout: GPUBindGroupLayout;
  private detectorMask: Uint8Array;
  private detectorDispatchStorage: {
    entries: GPUBuffer;
    parameters: GPUBuffer;
    parameterValues: Uint32Array<ArrayBuffer>;
    bindGroups: GPUBindGroup[][];
  } | null = null;
  private hasDetector = false;
  private activeDetectorOutput = 0;
  private detectorTimestampQuery: GPUQuerySet | null;
  private detectorTimestampResolve: GPUBuffer | null;
  private detectorTimestampReadback: GPUBuffer | null;
  private detectorTimestampMapPending = false;
  private pendingDetectorUpdates = new Set<Promise<WebGPUCompactH5DetectorMetrics>>();
  private pendingConsumerOperations = new Set<Promise<unknown>>();
  private lastDetectorSubmission: {
    mode: "rebase" | "delta" | "prepared";
    changedDetectorPixels: number;
    addedPixels: number;
    removedPixels: number;
  } = { mode: "rebase", changedDetectorPixels: 0, addedPixels: 0, removedPixels: 0 };
  private dpcCache: {
    detectorPixels: Uint32Array;
    moments: GPUBuffer;
    com: GPUBuffer;
    row: GPUBuffer;
    column: GPUBuffer;
    selectedPixels: number;
    plan: CompactDpcPlan;
  } | null = null;
  private dpcCacheQueueReady = false;
  private preparedDpcMoments: {
    detectorPixels: Uint32Array;
    buffer: GPUBuffer;
  } | null;
  private preparedDetectorProducts: Map<"bf" | "abf" | "adf", {
    mask: Uint8Array;
    buffer: GPUBuffer;
    selectedPixels: number;
  }>;

  private publishLifecycle(event: string, extra: Record<string, unknown> = {}): void {
    try {
      const target = globalThis as typeof globalThis & {
        __quantemCompactResidentLifecycle?: Array<Record<string, unknown>>;
      };
      const history = target.__quantemCompactResidentLifecycle ?? [];
      history.push({
        atMs: performance.now(),
        event,
        sourceIdentitySha256: this.metadata.sourceIdentitySha256,
        released: this.isReleased,
        pendingDetectorUpdates: this.pendingDetectorUpdates.size,
        pendingConsumerOperations: this.pendingConsumerOperations.size,
        ...extra,
      });
      if (history.length > 80) history.splice(0, history.length - 80);
      target.__quantemCompactResidentLifecycle = history;
    } catch {
      // Diagnostics must never affect scientific compute.
    }
  }

  constructor(args: {
    metadata: CompactH5Index;
    loadProfile: WebGPUCompactH5LoadProfile;
    device: GPUDevice;
    shards: ResidentShard[];
    excluded: GPUBuffer;
    maximumWidths: Uint8Array;
    detectorOutputs: GPUBuffer[];
    diffractionOutput: GPUBuffer;
    selectedPipeline: GPUComputePipeline;
    detectorPipeline: GPUComputePipeline;
    detectorResolveV3Pipeline: GPUComputePipeline;
    detectorResolveV3Layout: GPUBindGroupLayout;
    detectorResolvedV3Pipeline: GPUComputePipeline;
    u32ToF32Pipeline: GPUComputePipeline;
    dpcMomentPipeline: GPUComputePipeline;
    dpcMomentLayout: GPUBindGroupLayout;
    momentsToComPipeline: GPUComputePipeline;
    dpcMeanPipeline: GPUComputePipeline;
    dpcPairPipeline: GPUComputePipeline;
    dpcOutputMeanPipeline: GPUComputePipeline;
    dpcOutputUlpCorrectPipeline: GPUComputePipeline;
    detectorLayout: GPUBindGroupLayout;
    implementationRevision?: string | null;
    collectDetectorTimings?: boolean;
    preparedDpcMoments?: {
      detectorPixels: Uint32Array;
      buffer: GPUBuffer;
    } | null;
    preparedDetectorProducts?: Map<"bf" | "abf" | "adf", {
      mask: Uint8Array;
      buffer: GPUBuffer;
      selectedPixels: number;
    }>;
  }) {
    this.metadata = args.metadata;
    this.loadProfile = args.loadProfile;
    this.device = args.device;
    this.scanCount = args.metadata.shape[0] * args.metadata.shape[1];
    this.detSize = args.metadata.shape[2] * args.metadata.shape[3];
    this.badPx = new Uint32Array(args.metadata.excludedDetectorPixels);
    this.shards = args.shards;
    this.excluded = args.excluded;
    this.maximumWidths = args.maximumWidths;
    this.detectorOutputs = args.detectorOutputs;
    this.diffractionOutput = args.diffractionOutput;
    this.selectedPipeline = args.selectedPipeline;
    this.detectorPipeline = args.detectorPipeline;
    this.detectorResolveV3Pipeline = args.detectorResolveV3Pipeline;
    this.detectorResolveV3Layout = args.detectorResolveV3Layout;
    this.detectorResolvedV3Pipeline = args.detectorResolvedV3Pipeline;
    this.u32ToF32Pipeline = args.u32ToF32Pipeline;
    this.dpcMomentPipeline = args.dpcMomentPipeline;
    this.dpcMomentLayout = args.dpcMomentLayout;
    this.momentsToComPipeline = args.momentsToComPipeline;
    this.dpcMeanPipeline = args.dpcMeanPipeline;
    this.dpcPairPipeline = args.dpcPairPipeline;
    this.dpcOutputMeanPipeline = args.dpcOutputMeanPipeline;
    this.dpcOutputUlpCorrectPipeline = args.dpcOutputUlpCorrectPipeline;
    this.detectorLayout = args.detectorLayout;
    this.preparedDpcMoments = args.preparedDpcMoments ?? null;
    this.preparedDetectorProducts = args.preparedDetectorProducts ?? new Map();
    this.detectorMask = new Uint8Array(args.metadata.shape[2] * args.metadata.shape[3]);
    // Timing queries require CPU mappings after every reduction. Keep those
    // diagnostic transfers out of the normal resident interaction path.
    const timestampQuery = args.collectDetectorTimings === true
      && this.device.features.has("timestamp-query");
    this.detectorTimestampQuery = timestampQuery
      ? this.device.createQuerySet({ type: "timestamp", count: 2 })
      : null;
    this.detectorTimestampResolve = timestampQuery
      ? this.device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC })
      : null;
    this.detectorTimestampReadback = timestampQuery
      ? this.device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
      : null;
    this.detectorDisplayOutput = this.device.createBuffer({
      label: "compact persistent virtual detector display",
      size: this.scanCount * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.detectorDisplayCount = uploadBytes(
      this.device,
      new Uint32Array([this.scanCount, 0, 0, 0]),
      GPUBufferUsage.UNIFORM,
    );
    this.detectorDisplayBindGroups = this.detectorOutputs.map((source) => this.device.createBindGroup({
      layout: this.u32ToF32Pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: source } },
        { binding: 1, resource: { buffer: this.detectorDisplayOutput } },
        { binding: 2, resource: { buffer: this.detectorDisplayCount } },
      ],
    }));
    const retainedBytes = this.shards.reduce((bytes, shard) => bytes + shard.payload.size + shard.descriptors.size, 0)
      + (this.preparedDpcMoments?.buffer.size ?? 0)
      + [...this.preparedDetectorProducts.values()].reduce((bytes, product) => bytes + product.buffer.size, 0);
    if (retainedBytes !== this.metadata.residentBytes) {
      this.releaseResidentStorage();
      throw new Error("Allocated compact data and prepared products disagree with the complete resident byte plan.");
    }
    this.residentReceipt = compactReceipt(this.metadata, args.implementationRevision ?? null);
    this.metadata.residentReceipt = this.residentReceipt;
    this.publishLifecycle("created");
  }

  /** Current retained GPUBuffer bytes, including auxiliary display and DPC storage. */
  get ownedBufferBytes(): number {
    if (this.isReleased) return 0;
    const buffers = new Set([
      ...this.shards.flatMap(shard => [shard.payload, shard.descriptors]),
      this.excluded, this.diffractionOutput, ...this.detectorOutputs,
      this.detectorDisplayOutput, this.detectorDisplayCount,
      this.detectorTimestampResolve, this.detectorTimestampReadback,
      this.detectorDispatchStorage?.entries, this.detectorDispatchStorage?.parameters,
      this.preparedDpcMoments?.buffer,
      ...[...this.preparedDetectorProducts.values()].map(product => product.buffer),
      ...(this.dpcCache ? [this.dpcCache.moments, this.dpcCache.com,
        this.dpcCache.row, this.dpcCache.column] : []),
    ]);
    return [...buffers].reduce((bytes, buffer) => bytes + (buffer?.size ?? 0), 0);
  }

  /** Read retained raw v1 counts or restore authenticated v3 excluded values. */
  extractRawDiffraction(scanRow: number, scanColumn: number): Promise<Uint32Array> {
    this.requireResident();
    if (!this.metadata.rawReconstructionAvailable) {
      throw new Error("Raw values are unavailable for this compact source. Rebuild from original uint16 data with recoverable excluded pixels.");
    }
    return this.trackConsumerOperation((async () => {
      if (this.metadata.schemaVersion === 1) {
        const buffer = await this.extractSelectedDiffractionBuffer(scanRow, scanColumn, true);
        return readU32Buffer(this.device, buffer, this.detSize);
      }
      const values = await this.extractDiffraction(scanRow, scanColumn);
      for (let index = 0; index < this.metadata.excludedDetectorPixels.length; index++) {
        values[this.metadata.excludedDetectorPixels[index]] = this.metadata.maskedDetectorRawValues![index];
      }
      return values;
    })());
  }

  /** Convert source-bound prepared moments into persistent DPC display maps. */
  async primePreparedDpc(): Promise<boolean> {
    this.requireResident();
    if (!this.preparedDpcMoments) return false;
    const plan = this.planDpcMoments(
      this.preparedDpcMoments.detectorPixels,
      this.metadata.shape[3],
    );
    this.ensureDpcCache(this.preparedDpcMoments.detectorPixels, plan);
    await this.device.queue.onSubmittedWorkDone();
    this.dpcCacheQueueReady = true;
    return true;
  }

  async extractDiffractionBuffer(scanRow: number, scanColumn: number): Promise<GPUBuffer> {
    return this.extractSelectedDiffractionBuffer(scanRow, scanColumn, false);
  }

  private async extractSelectedDiffractionBuffer(scanRow: number, scanColumn: number, raw: boolean): Promise<GPUBuffer> {
    this.requireResident();
    const [scanRows, scanColumns, detectorRows, detectorColumns] = this.metadata.shape;
    requireIndex(scanRow, scanRows, "scan row");
    requireIndex(scanColumn, scanColumns, "scan column");
    const globalScan = scanRow * scanColumns + scanColumn;
    const shardIndex = Math.floor(globalScan / this.metadata.scansPerShard);
    const localScan = globalScan % this.metadata.scansPerShard;
    const pixelCount = detectorRows * detectorColumns;
    const tileCount = Math.ceil(this.metadata.scansPerShard / this.metadata.scanTile);
    const config = uploadBytes(
      this.device,
      new Uint32Array([localScan, tileCount, pixelCount, this.metadata.schemaVersion, raw ? 1 : 0, 0, 0, 0]),
      GPUBufferUsage.UNIFORM,
    );
    const bindGroup = this.device.createBindGroup({
      layout: this.selectedPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.shards[shardIndex].payload } },
        { binding: 1, resource: { buffer: this.shards[shardIndex].descriptors } },
        { binding: 2, resource: { buffer: this.excluded } },
        { binding: 3, resource: { buffer: this.diffractionOutput } },
        { binding: 4, resource: { buffer: config } },
      ],
    });
    const encoder = this.device.createCommandEncoder({ label: "compact selected diffraction" });
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.selectedPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(pixelCount / 256));
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();
    config.destroy();
    return this.diffractionOutput;
  }

  async extractDiffraction(scanRow: number, scanColumn: number): Promise<Uint32Array> {
    const buffer = await this.extractDiffractionBuffer(scanRow, scanColumn);
    return readU32Buffer(this.device, buffer, this.metadata.shape[2] * this.metadata.shape[3]);
  }

  getDevice(): GPUDevice {
    this.requireResident();
    return this.device;
  }

  frameAt(scanIndex: number): Promise<Float32Array> {
    const operation = (async () => {
      this.requireResident();
      requireIndex(scanIndex, this.scanCount, "scan index");
      const scanColumnCount = this.metadata.shape[1];
      const values = await this.extractDiffraction(
        Math.floor(scanIndex / scanColumnCount),
        scanIndex % scanColumnCount,
      );
      return Float32Array.from(values);
    })();
    return this.trackConsumerOperation(operation);
  }

  maskedSum(mask: Uint32Array): Promise<Float32Array> {
    const operation = (async () => {
      await this.updateVirtualDetector(mask);
      return Float32Array.from(await this.virtualDetectorValues());
    })();
    return this.trackConsumerOperation(operation);
  }

  maskedSumBuffer(mask: Uint32Array): { buffer: GPUBuffer; n: number } {
    this.requireResident();
    if (mask.length !== this.detSize) {
      throw new Error(`A compact virtual-detector mask requires ${this.detSize} row-major values; got ${mask.length}.`);
    }
    const selectedPixels = countSelectedPixels(mask, this.metadata.excludedDetectorPixels);
    void this.updateVirtualDetector(mask).catch((error) => {
      if (!this.isReleased) {
        console.error("Compact WebGPU detector update failed after submission", error);
      }
    });
    const output = this.device.createBuffer({
      label: "compact virtual detector display copy",
      size: this.scanCount * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    if (selectedPixels > 0) {
      const count = uploadBytes(
        this.device,
        new Uint32Array([this.scanCount, 0, 0, 0]),
        GPUBufferUsage.UNIFORM,
      );
      const bindGroup = this.device.createBindGroup({
        layout: this.u32ToF32Pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.virtualDetectorBuffer() } },
          { binding: 1, resource: { buffer: output } },
          { binding: 2, resource: { buffer: count } },
        ],
      });
      const encoder = this.device.createCommandEncoder({ label: "compact virtual detector display conversion" });
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.u32ToF32Pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(this.scanCount / 256));
      pass.end();
      this.device.queue.submit([encoder.finish()]);
      const completion = this.trackConsumerOperation(this.device.queue.onSubmittedWorkDone());
      void completion.then(
        () => count.destroy(),
        () => count.destroy(),
      );
    }
    return { buffer: output, n: selectedPixels };
  }

  /**
   * Update the exact compact detector and return a borrowed persistent f32
   * display buffer. Unlike ``maskedSumBuffer``, callers must not destroy this
   * buffer; it is owned by this resident source and reused for every drag state.
   */
  maskedSumDisplayBuffer(mask: Uint32Array): {
    buffer: GPUBuffer;
    n: number;
    path: "compact-rebase" | "compact-delta" | "compact-prepared";
    addedPixels: number;
    removedPixels: number;
    borrowed: true;
  } {
    return this.enqueueMaskedSumDisplay(mask);
  }

  /**
   * Submit one exact detector update across resident comparison sources.
   *
   * The display buffers remain borrowed from their sources. Submission and
   * completion are shared; the scientific mask and precision are unchanged.
   *
   * @example
   * const displays = WebGPUCompactH5ResidentSource.maskedSumDisplayBuffersBatch(sources, mask);
   */
  static maskedSumDisplayBuffersBatch(sources: WebGPUCompactH5ResidentSource[], mask: Uint32Array) {
    if (!sources.length) return [];
    if (new Set(sources).size !== sources.length) {
      throw new Error("A detector batch cannot update the same resident source twice. Deduplicate comparison sources before submitting.");
    }
    const device = sources[0].device;
    for (const source of sources) {
      source.requireResident();
      if (source.metadata.schemaVersion !== 1) {
        throw new Error("Batched detector updates currently require lossless pack format v1 sources. Use each source's maskedSumDisplayBuffer for lossless pack format v3.");
      }
      if (source.device !== device || source.detSize !== mask.length
        || source.metadata.shape[2] !== sources[0].metadata.shape[2]
        || source.metadata.shape[3] !== sources[0].metadata.shape[3]) {
        throw new Error("Comparison sources must share a WebGPU device and detector shape. Reload the comparison together.");
      }
    }
    let complete!: () => void;
    let fail!: (error: unknown) => void;
    const completion = new Promise<void>((resolve, reject) => { complete = resolve; fail = reject; });
    // An encoding failure can happen before the first source registers a
    // consumer. Keep that failure handled even when there are no consumers yet.
    void completion.catch(() => {});
    const encoder = device.createCommandEncoder({ label: "compact comparison detector batch" });
    try {
      const displays = sources.map(source => source.enqueueMaskedSumDisplay(mask, { encoder, completion }));
      device.queue.submit([encoder.finish()]);
      void device.queue.onSubmittedWorkDone().then(complete, fail);
      return displays;
    } catch (error) {
      // No partially encoded detector generation may become the next delta's
      // reference when the containing command buffer was not submitted.
      for (const source of sources) source.hasDetector = false;
      fail(error);
      throw error;
    }
  }

  private enqueueMaskedSumDisplay(mask: Uint32Array, submission?: CompactDetectorSubmission): ReturnType<WebGPUCompactH5ResidentSource["maskedSumDisplayBuffer"]> {
    this.requireResident();
    if (mask.length !== this.detSize) {
      throw new Error(`A compact virtual-detector mask requires ${this.detSize} row-major values; got ${mask.length}.`);
    }
    const selection = this.prepareDetectorSelection(mask);
    void this.submitVirtualDetector(selection, submission).catch((error) => {
      if (!this.isReleased) {
        console.error("Compact WebGPU detector update failed after submission", error);
      }
    });
    const detectorUpdate = this.lastDetectorSubmission;
    const encoder = submission?.encoder ?? this.device.createCommandEncoder({ label: "compact persistent detector display conversion" });
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.u32ToF32Pipeline);
    pass.setBindGroup(0, this.detectorDisplayBindGroups[this.activeDetectorOutput]);
    pass.dispatchWorkgroups(Math.ceil(this.scanCount / 256));
    pass.end();
    if (!submission) this.device.queue.submit([encoder.finish()]);
    void this.trackConsumerOperation(submission?.completion ?? this.device.queue.onSubmittedWorkDone()).catch(
      (error) => {
        if (!this.isReleased) {
          console.error("Compact WebGPU display conversion failed after submission", error);
        }
      },
    );
    return {
      buffer: this.detectorDisplayOutput,
      n: selection.selectedPixelCount,
      path: detectorUpdate.mode === "prepared"
        ? "compact-prepared"
        : detectorUpdate.mode === "rebase" ? "compact-rebase" : "compact-delta",
      addedPixels: detectorUpdate.addedPixels,
      removedPixels: detectorUpdate.removedPixels,
      borrowed: true as const,
    };
  }

  maskedDpcBuffer(
    mask: Uint32Array,
    detectorColumns: number,
    component: "row" | "col" | 0 | 1,
  ): { buffer: GPUBuffer; n: number } {
    this.requireResident();
    if (detectorColumns !== this.metadata.shape[3]) {
      throw new Error(
        `Compact DPC detector columns ${detectorColumns} do not match ${this.metadata.shape[3]}.`,
      );
    }
    if (mask.length !== this.detSize) {
      throw new Error(`A compact DPC mask requires ${this.detSize} row-major values; got ${mask.length}.`);
    }
    const selection = compactDetectorSelection(mask, this.metadata.excludedDetectorPixels);
    const selectedPixels = selection.indices.length;
    const plan = this.planDpcMoments(selection.indices, detectorColumns);
    if (selectedPixels === 0) {
      return {
        buffer: this.device.createBuffer({
          size: this.scanCount * 4,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        }),
        n: 0,
      };
    }
    const cache = this.ensureDpcCache(selection.indices, plan);
    const source = component === "row" || component === 0 ? cache.row : cache.column;
    return { buffer: this.copyScanBuffer(source), n: selectedPixels };
  }

  /**
   * Return the resident DPC component without allocating or copying a scan map.
   *
   * The returned buffer is borrowed from this compact source. Display clients
   * must not destroy it; it remains valid until ``releaseResidentStorage``.
   */
  maskedDpcDisplayBuffer(
    mask: Uint32Array,
    detectorColumns: number,
    component: "row" | "col" | 0 | 1,
  ): { buffer: GPUBuffer; n: number; borrowed: true; queueReady: boolean } {
    this.requireResident();
    if (detectorColumns !== this.metadata.shape[3]) {
      throw new Error(
        `Compact DPC detector columns ${detectorColumns} do not match ${this.metadata.shape[3]}.`,
      );
    }
    if (mask.length !== this.detSize) {
      throw new Error(`A compact DPC mask requires ${this.detSize} row-major values; got ${mask.length}.`);
    }
    const selection = compactDetectorSelection(mask, this.metadata.excludedDetectorPixels);
    const selectedPixels = selection.indices.length;
    if (selectedPixels === 0) {
      return {
        buffer: this.detectorDisplayOutput,
        n: 0,
        borrowed: true,
        queueReady: true,
      };
    }
    const plan = this.planDpcMoments(selection.indices, detectorColumns);
    const cache = this.ensureDpcCache(selection.indices, plan);
    const source = component === "row" || component === 0 ? cache.row : cache.column;
    return {
      buffer: source,
      n: selectedPixels,
      borrowed: true,
      queueReady: this.dpcCacheQueueReady,
    };
  }

  /**
   * Return a prepared full-detector DPC component in O(1) CPU time.
   *
   * ``primePreparedDpc`` authenticates the source-bound detector-pixel list
   * and builds these immutable row/column buffers before the resident source
   * is published. A display switch therefore does not need to rescan the same
   * detector mask or repeat the exact-arithmetic plan.
   */
  preparedDpcDisplayBuffer(
    component: "row" | "col" | 0 | 1,
  ): { buffer: GPUBuffer; n: number; borrowed: true; queueReady: true } | null {
    this.requireResident();
    const prepared = this.preparedDpcMoments;
    const cache = this.dpcCache;
    if (!prepared || !cache || cache.moments !== prepared.buffer || !this.dpcCacheQueueReady) {
      return null;
    }
    const source = component === "row" || component === 0 ? cache.row : cache.column;
    return {
      buffer: source,
      n: cache.selectedPixels,
      borrowed: true,
      queueReady: true,
    };
  }

  async maskedDpc(
    mask: Uint32Array,
    detectorColumns: number,
    component: "row" | "col" | 0 | 1,
  ): Promise<Float32Array> {
    const { buffer, n } = this.maskedDpcBuffer(mask, detectorColumns, component);
    const values = await readF32Buffer(this.device, buffer, this.scanCount);
    buffer.destroy();
    if (n === 0) values.fill(0);
    return values;
  }

  async exactDetectorMoments(
    mask: Uint32Array,
    detectorColumns: number,
  ): Promise<WebGPUCompactH5ExactMomentSnapshot> {
    this.requireResident();
    if (detectorColumns !== this.metadata.shape[3]) {
      throw new Error(
        `Compact exact-moment detector columns ${detectorColumns} do not match ${this.metadata.shape[3]}.`,
      );
    }
    if (mask.length !== this.detSize) {
      throw new Error(`A compact exact-moment mask requires ${this.detSize} row-major values; got ${mask.length}.`);
    }
    const selection = compactDetectorSelection(mask, this.metadata.excludedDetectorPixels);
    const plan = this.planDpcMoments(selection.indices, detectorColumns);
    const words = selection.indices.length === 0
      ? new Uint32Array(this.scanCount * 8)
      : await readU32Buffer(
        this.device,
        this.ensureDpcCache(selection.indices, plan).moments,
        this.scanCount * 8,
      );
    return {
      wordOrder: "little-endian-u32-pairs",
      wordsPerScan: 8,
      layout: [
        "total_lo", "total_hi", "row_lo", "row_hi",
        "column_lo", "column_hi", "padding_0", "padding_1",
      ],
      scanCount: this.scanCount,
      selectedDetectorPixels: selection.indices.length,
      totalBound: plan.totalBound.toString(),
      rowMomentBound: plan.rowMomentBound.toString(),
      columnMomentBound: plan.columnMomentBound.toString(),
      narrowInteger: plan.narrowInteger === 1,
      narrowProducts: plan.narrowProducts === 1,
      words,
    };
  }

  get fftDispatchCount(): number {
    return this.fftDispatches;
  }

  async maskedIDpcBuffer(
    mask: Uint32Array,
    detectorColumns: number,
    scanRows: number,
    scanColumns: number,
    rotationDegrees = 0,
    useTranspose = false,
  ): Promise<{ buffer: GPUBuffer; n: number }> {
    this.requireResident();
    if (scanRows * scanColumns !== this.scanCount) {
      throw new Error(
        `Compact WebGPU iDPC scan shape ${scanRows}x${scanColumns} does not match ${this.scanCount} scan positions.`,
      );
    }
    if (!isPowerOfTwo(scanRows) || !isPowerOfTwo(scanColumns)) {
      throw new Error(`Compact WebGPU iDPC requires power-of-two scan dimensions; got ${scanRows}x${scanColumns}.`);
    }
    if (detectorColumns !== this.metadata.shape[3] || mask.length !== this.detSize) {
      throw new Error("Compact iDPC detector shape does not match this source.");
    }
    const selection = compactDetectorSelection(mask, this.metadata.excludedDetectorPixels);
    const plan = this.planDpcMoments(selection.indices, detectorColumns);
    const selectedPixels = selection.indices.length;
    const output = this.device.createBuffer({
      label: "compact iDPC phase",
      size: this.scanCount * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    if (selectedPixels === 0) return { buffer: output, n: 0 };

    const dpc = this.ensureDpcCache(selection.indices, plan);
    const pipelines = await this.ensureIdpcPipelines();
    const complexBytes = this.scanCount * 8;
    const rowFft = this.device.createBuffer({
      label: "compact iDPC row FFT",
      size: complexBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const columnFft = this.device.createBuffer({
      label: "compact iDPC column FFT",
      size: complexBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const phaseFft = this.device.createBuffer({
      label: "compact iDPC phase FFT",
      size: complexBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const packBytes = new ArrayBuffer(32);
    const packU32 = new Uint32Array(packBytes);
    const packF32 = new Float32Array(packBytes);
    const theta = rotationDegrees * Math.PI / 180;
    packU32[0] = this.scanCount;
    packU32[1] = useTranspose ? 1 : 0;
    packF32[4] = Math.cos(theta);
    packF32[5] = Math.sin(theta);
    const packParameters = uploadBytes(this.device, new Uint8Array(packBytes), GPUBufferUsage.UNIFORM);
    const poissonParameters = uploadBytes(
      this.device,
      new Uint32Array([scanColumns, scanRows, this.scanCount, 0]),
      GPUBufferUsage.UNIFORM,
    );
    const extractParameters = uploadBytes(
      this.device,
      new Uint32Array([this.scanCount, 0, 0, 0]),
      GPUBufferUsage.UNIFORM,
    );
    const encoder = this.device.createCommandEncoder({ label: "compact iDPC" });
    encodeComputePass(
      encoder,
      pipelines.pack,
      this.device.createBindGroup({
        layout: pipelines.pack.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: dpc.row } },
          { binding: 1, resource: { buffer: dpc.column } },
          { binding: 2, resource: { buffer: rowFft } },
          { binding: 3, resource: { buffer: columnFft } },
          { binding: 4, resource: { buffer: packParameters } },
        ],
      }),
      Math.ceil(this.scanCount / 256),
    );
    const temporaryBuffers = [rowFft, columnFft, phaseFft, packParameters, poissonParameters, extractParameters];
    let fftDispatches = 0;
    fftDispatches += this.encodeFft2d(encoder, pipelines, rowFft, scanColumns, scanRows, false, temporaryBuffers);
    fftDispatches += this.encodeFft2d(encoder, pipelines, columnFft, scanColumns, scanRows, false, temporaryBuffers);
    encodeComputePass(
      encoder,
      pipelines.poisson,
      this.device.createBindGroup({
        layout: pipelines.poisson.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: rowFft } },
          { binding: 1, resource: { buffer: columnFft } },
          { binding: 2, resource: { buffer: phaseFft } },
          { binding: 3, resource: { buffer: poissonParameters } },
        ],
      }),
      Math.ceil(this.scanCount / 256),
    );
    fftDispatches += this.encodeFft2d(encoder, pipelines, phaseFft, scanColumns, scanRows, true, temporaryBuffers);
    encodeComputePass(
      encoder,
      pipelines.extract,
      this.device.createBindGroup({
        layout: pipelines.extract.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: phaseFft } },
          { binding: 1, resource: { buffer: output } },
          { binding: 2, resource: { buffer: extractParameters } },
        ],
      }),
      Math.ceil(this.scanCount / 256),
    );
    this.device.queue.submit([encoder.finish()]);
    this.fftDispatches += fftDispatches;
    retireBuffers(this.device, temporaryBuffers);
    return { buffer: output, n: selectedPixels };
  }

  async maskedIDpc(
    mask: Uint32Array,
    detectorColumns: number,
    scanRows: number,
    scanColumns: number,
    rotationDegrees = 0,
    useTranspose = false,
  ): Promise<Float32Array> {
    const { buffer, n } = await this.maskedIDpcBuffer(
      mask,
      detectorColumns,
      scanRows,
      scanColumns,
      rotationDegrees,
      useTranspose,
    );
    const values = await readF32Buffer(this.device, buffer, this.scanCount);
    buffer.destroy();
    if (n === 0) values.fill(0);
    return values;
  }

  async maskedCoM(
    mask: Uint32Array,
    detectorColumns: number,
  ): Promise<{ comY: Float32Array; comX: Float32Array }> {
    this.requireResident();
    if (detectorColumns !== this.metadata.shape[3] || mask.length !== this.detSize) {
      throw new Error("Compact CoM shape does not match this detector.");
    }
    const selection = compactDetectorSelection(mask, this.metadata.excludedDetectorPixels);
    const plan = this.planDpcMoments(selection.indices, detectorColumns);
    if (selection.indices.length === 0) {
      return { comY: new Float32Array(this.scanCount), comX: new Float32Array(this.scanCount) };
    }
    const cache = this.ensureDpcCache(selection.indices, plan);
    const flat = await readF32Buffer(this.device, cache.com, this.scanCount * 2);
    return {
      comY: flat.slice(0, this.scanCount),
      comX: flat.slice(this.scanCount),
    };
  }

  async reduceFrames(scanMask: Uint32Array, mean = true): Promise<Float32Array> {
    this.requireResident();
    if (scanMask.length !== this.scanCount) {
      throw new Error(`A compact scan mask requires ${this.scanCount} row-major values; got ${scanMask.length}.`);
    }
    const scansByShard: number[][] = Array.from({ length: this.shards.length }, () => []);
    let selectedScans = 0;
    for (let scan = 0; scan < scanMask.length; scan++) {
      const value = scanMask[scan];
      if (value !== 0 && value !== 1) throw new Error("A compact scan mask must contain only zero or one.");
      if (value === 0) continue;
      const shard = Math.floor(scan / this.metadata.scansPerShard);
      scansByShard[shard].push(scan % this.metadata.scansPerShard);
      selectedScans++;
    }
    const result = new Float32Array(this.detSize);
    if (selectedScans === 0) return result;
    for (let pixel = 0; pixel < this.maximumWidths.length; pixel++) {
      const maximumValue = (1n << BigInt(this.maximumWidths[pixel])) - 1n;
      if (maximumValue * BigInt(selectedScans) > 0xffffffffn) {
        throw new Error(
          `This compact scan reduction can sum detector pixel ${pixel} beyond exact u32 output. `
          + "Use fewer scan positions or a future u64 reduction path.",
        );
      }
    }
    const pipeline = await this.ensureReduceFramesPipeline();
    const output = this.device.createBuffer({
      label: "compact scan-region diffraction sum",
      size: this.detSize * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const temporaryBuffers: GPUBuffer[] = [];
    const encoder = this.device.createCommandEncoder({ label: "compact scan-region diffraction" });
    encoder.clearBuffer(output);
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    const tileCount = Math.ceil(this.metadata.scansPerShard / this.metadata.scanTile);
    for (let shardIndex = 0; shardIndex < scansByShard.length; shardIndex++) {
      const localScans = scansByShard[shardIndex];
      if (localScans.length === 0) continue;
      const scanBuffer = uploadBytes(this.device, Uint32Array.from(localScans), GPUBufferUsage.STORAGE);
      const parameters = uploadBytes(
        this.device,
        new Uint32Array([tileCount, localScans.length, this.detSize, this.metadata.schemaVersion]),
        GPUBufferUsage.UNIFORM,
      );
      temporaryBuffers.push(scanBuffer, parameters);
      pass.setBindGroup(0, this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.shards[shardIndex].payload } },
          { binding: 1, resource: { buffer: this.shards[shardIndex].descriptors } },
          { binding: 2, resource: { buffer: scanBuffer } },
          { binding: 3, resource: { buffer: this.excluded } },
          { binding: 4, resource: { buffer: output } },
          { binding: 5, resource: { buffer: parameters } },
        ],
      }));
      pass.dispatchWorkgroups(Math.ceil(this.detSize / 256));
    }
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    const sums = await readU32Buffer(this.device, output, this.detSize);
    const denominator = mean ? selectedScans : 1;
    for (let pixel = 0; pixel < this.detSize; pixel++) result[pixel] = sums[pixel] / denominator;
    output.destroy();
    retireBuffers(this.device, temporaryBuffers);
    return result;
  }

  private ensureDpcCache(detectorPixels: Uint32Array, plan: CompactDpcPlan) {
    if (this.dpcCache && equalU32(this.dpcCache.detectorPixels, detectorPixels)) return this.dpcCache;
    this.clearDpcCache();
    const selectedPixels = detectorPixels.length;
    const integerFlags = plan.narrowInteger | (plan.narrowProducts << 1);
    const usePreparedMoments = Boolean(
      this.preparedDpcMoments
      && equalU32(this.preparedDpcMoments.detectorPixels, detectorPixels),
    );
    const sampleVisits = this.scanCount * selectedPixels;
    if (!usePreparedMoments && sampleVisits > MAX_INLINE_DPC_SAMPLE_VISITS) {
      throw new Error(
        `Exact DPC needs ${sampleVisits.toLocaleString()} compact sample visits. `
        + "This source requires authenticated prepared DPC moments; the unsafe long-running fallback was not dispatched.",
      );
    }
    const selected = usePreparedMoments
      ? null
      : uploadBytes(this.device, detectorPixels, GPUBufferUsage.STORAGE);
    const moments = usePreparedMoments
      ? this.preparedDpcMoments!.buffer
      : this.device.createBuffer({
        label: "compact exact detector moments as paired u32 words",
        size: this.scanCount * 32,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
    const com = this.device.createBuffer({
      size: this.scanCount * 2 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const mean = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE });
    const rowResidualMean = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE });
    const columnResidualMean = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE });
    const row = this.device.createBuffer({
      size: this.scanCount * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const column = this.device.createBuffer({
      size: this.scanCount * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const count = uploadBytes(
      this.device,
      new Uint32Array([this.scanCount, integerFlags, 0, 0]),
      GPUBufferUsage.UNIFORM,
    );
    const rowCount = uploadBytes(
      this.device,
      new Uint32Array([this.scanCount, 0, 0, 0]),
      GPUBufferUsage.UNIFORM,
    );
    const columnCount = uploadBytes(
      this.device,
      new Uint32Array([this.scanCount, 1, 0, 0]),
      GPUBufferUsage.UNIFORM,
    );
    const alignment = Math.max(256, this.device.limits.minUniformBufferOffsetAlignment);
    let parameterBuffer: GPUBuffer | null = null;
    if (!usePreparedMoments) {
      const parameters = new ArrayBuffer(alignment * this.shards.length);
      for (let shard = 0; shard < this.shards.length; shard++) {
        new Uint32Array(parameters, shard * alignment, 8).set([
          this.metadata.scansPerShard,
          Math.ceil(this.metadata.scansPerShard / this.metadata.scanTile),
          selectedPixels,
          shard * this.metadata.scansPerShard,
          this.metadata.shape[3],
          integerFlags,
          this.metadata.schemaVersion, 0,
        ]);
      }
      parameterBuffer = uploadBytes(
        this.device,
        new Uint8Array(parameters),
        GPUBufferUsage.UNIFORM,
      );
    }
    const encoder = this.device.createCommandEncoder({ label: "compact DPC components" });
    if (!usePreparedMoments) {
      const momentPass = encoder.beginComputePass();
      momentPass.setPipeline(this.dpcMomentPipeline);
      for (let shardIndex = 0; shardIndex < this.shards.length; shardIndex++) {
        const shard = this.shards[shardIndex];
        momentPass.setBindGroup(0, this.device.createBindGroup({
          layout: this.dpcMomentLayout,
          entries: [
            { binding: 0, resource: { buffer: shard.payload } },
            { binding: 1, resource: { buffer: shard.descriptors } },
            { binding: 2, resource: { buffer: selected! } },
            { binding: 3, resource: { buffer: moments } },
            { binding: 4, resource: { buffer: parameterBuffer!, size: 32 } },
          ],
        }), [shardIndex * alignment]);
        momentPass.dispatchWorkgroups(Math.ceil(this.metadata.scansPerShard / 32));
      }
      momentPass.end();
    }
    const comPass = encoder.beginComputePass();
    comPass.setPipeline(this.momentsToComPipeline);
    comPass.setBindGroup(0, this.device.createBindGroup({
      layout: this.momentsToComPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: moments } },
        { binding: 1, resource: { buffer: com } },
        { binding: 2, resource: { buffer: count } },
      ],
    }));
    comPass.dispatchWorkgroups(Math.ceil(this.scanCount / 256));
    comPass.end();
    const meanPass = encoder.beginComputePass();
    meanPass.setPipeline(this.dpcMeanPipeline);
    meanPass.setBindGroup(0, this.device.createBindGroup({
      layout: this.dpcMeanPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: com } },
        { binding: 1, resource: { buffer: mean } },
        { binding: 2, resource: { buffer: count } },
      ],
    }));
    meanPass.dispatchWorkgroups(1);
    meanPass.end();
    const pairPass = encoder.beginComputePass();
    pairPass.setPipeline(this.dpcPairPipeline);
    pairPass.setBindGroup(0, this.device.createBindGroup({
      layout: this.dpcPairPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: com } },
        { binding: 1, resource: { buffer: mean } },
        { binding: 2, resource: { buffer: row } },
        { binding: 3, resource: { buffer: column } },
        { binding: 4, resource: { buffer: count } },
      ],
    }));
    pairPass.dispatchWorkgroups(Math.ceil(this.scanCount / 256));
    pairPass.end();
    const correctionPass = encoder.beginComputePass();
    correctionPass.setPipeline(this.dpcOutputMeanPipeline);
    correctionPass.setBindGroup(0, this.device.createBindGroup({
      layout: this.dpcOutputMeanPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: row } },
        { binding: 1, resource: { buffer: rowResidualMean } },
        { binding: 2, resource: { buffer: rowCount } },
      ],
    }));
    correctionPass.dispatchWorkgroups(1);
    correctionPass.setBindGroup(0, this.device.createBindGroup({
      layout: this.dpcOutputMeanPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: column } },
        { binding: 1, resource: { buffer: columnResidualMean } },
        { binding: 2, resource: { buffer: columnCount } },
      ],
    }));
    correctionPass.dispatchWorkgroups(1);
    correctionPass.setPipeline(this.dpcOutputUlpCorrectPipeline);
    correctionPass.setBindGroup(0, this.device.createBindGroup({
      layout: this.dpcOutputUlpCorrectPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: row } },
        { binding: 1, resource: { buffer: rowResidualMean } },
        { binding: 2, resource: { buffer: mean } },
        { binding: 3, resource: { buffer: rowCount } },
      ],
    }));
    correctionPass.dispatchWorkgroups(Math.ceil(this.scanCount / 256));
    correctionPass.setBindGroup(0, this.device.createBindGroup({
      layout: this.dpcOutputUlpCorrectPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: column } },
        { binding: 1, resource: { buffer: columnResidualMean } },
        { binding: 2, resource: { buffer: mean } },
        { binding: 3, resource: { buffer: columnCount } },
      ],
    }));
    correctionPass.dispatchWorkgroups(Math.ceil(this.scanCount / 256));
    correctionPass.end();
    this.device.queue.submit([encoder.finish()]);
    void this.device.queue.onSubmittedWorkDone().then(() => {
      selected?.destroy();
      mean.destroy();
      rowResidualMean.destroy();
      columnResidualMean.destroy();
      count.destroy();
      rowCount.destroy();
      columnCount.destroy();
      parameterBuffer?.destroy();
    });
    this.dpcCache = { detectorPixels, moments, com, row, column, selectedPixels, plan };
    return this.dpcCache;
  }

  private ensureIdpcPipelines(): Promise<CompactIdpcPipelines> {
    if (!this.idpcPipelines) {
      const fftModule = this.device.createShaderModule({ code: FFT_2D_SHADER });
      const compile = (code: string, entryPoint = "main") => this.device.createComputePipelineAsync({
        layout: "auto",
        compute: { module: this.device.createShaderModule({ code }), entryPoint },
      });
      const fft = (entryPoint: string) => this.device.createComputePipelineAsync({
        layout: "auto",
        compute: { module: fftModule, entryPoint },
      });
      this.idpcPipelines = Promise.all([
        compile(IDPC_PACK_WGSL),
        compile(IDPC_POISSON_WGSL),
        compile(IDPC_EXTRACT_WGSL),
        fft("bitReverseRows"),
        fft("bitReverseCols"),
        fft("butterflyRows"),
        fft("butterflyCols"),
        fft("normalize2D"),
      ]).then(([
        pack,
        poisson,
        extract,
        bitReverseRows,
        bitReverseColumns,
        butterflyRows,
        butterflyColumns,
        normalize,
      ]) => ({
        pack,
        poisson,
        extract,
        bitReverseRows,
        bitReverseColumns,
        butterflyRows,
        butterflyColumns,
        normalize,
      }));
    }
    return this.idpcPipelines;
  }

  private ensureReduceFramesPipeline(): Promise<GPUComputePipeline> {
    if (!this.reduceFramesPipeline) {
      this.reduceFramesPipeline = this.device.createComputePipelineAsync({
        layout: "auto",
        compute: {
          module: this.device.createShaderModule({ code: COMPACT_REDUCE_FRAMES_WGSL }),
          entryPoint: "main",
        },
      });
    }
    return this.reduceFramesPipeline;
  }

  private encodeFft2d(
    encoder: GPUCommandEncoder,
    pipelines: CompactIdpcPipelines,
    data: GPUBuffer,
    width: number,
    height: number,
    inverse: boolean,
    temporaryBuffers: GPUBuffer[],
  ): number {
    const workgroupsX = Math.ceil(width / 16);
    const workgroupsY = Math.ceil(height / 16);
    let dispatches = 0;
    const dispatch = (pipeline: GPUComputePipeline, stageCount: number, stage: number, rowAxis: boolean) => {
      const bytes = new ArrayBuffer(24);
      const u32 = new Uint32Array(bytes);
      const f32 = new Float32Array(bytes);
      u32[0] = width;
      u32[1] = height;
      u32[2] = stageCount;
      u32[3] = stage;
      f32[4] = inverse ? 1 : -1;
      u32[5] = rowAxis ? 1 : 0;
      const parameters = uploadBytes(this.device, new Uint8Array(bytes), GPUBufferUsage.UNIFORM);
      temporaryBuffers.push(parameters);
      encodeComputePass(
        encoder,
        pipeline,
        this.device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: parameters } },
            { binding: 1, resource: { buffer: data } },
          ],
        }),
        workgroupsX,
        workgroupsY,
      );
      dispatches++;
    };
    const widthStages = Math.log2(width);
    const heightStages = Math.log2(height);
    dispatch(pipelines.bitReverseRows, widthStages, 0, true);
    for (let stage = 0; stage < widthStages; stage++) {
      dispatch(pipelines.butterflyRows, widthStages, stage, true);
    }
    dispatch(pipelines.bitReverseColumns, heightStages, 0, false);
    for (let stage = 0; stage < heightStages; stage++) {
      dispatch(pipelines.butterflyColumns, heightStages, stage, false);
    }
    if (inverse) dispatch(pipelines.normalize, heightStages, 0, false);
    return dispatches;
  }

  private planDpcMoments(detectorPixels: Uint32Array, detectorColumns: number): CompactDpcPlan {
    return planExactIntegerCoM(
      detectorPixels,
      detectorPixels.length,
      detectorColumns,
      this.metadata.workingDtype === "uint8" ? 1 : 0,
    );
  }

  private copyScanBuffer(source: GPUBuffer): GPUBuffer {
    const output = this.device.createBuffer({
      size: this.scanCount * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const encoder = this.device.createCommandEncoder({ label: "compact scan output copy" });
    encoder.copyBufferToBuffer(source, 0, output, 0, this.scanCount * 4);
    this.device.queue.submit([encoder.finish()]);
    return output;
  }

  private clearDpcCache(): void {
    if (
      this.dpcCache?.moments
      && this.dpcCache.moments !== this.preparedDpcMoments?.buffer
    ) {
      this.dpcCache.moments.destroy();
    }
    this.dpcCache?.com.destroy();
    this.dpcCache?.row.destroy();
    this.dpcCache?.column.destroy();
    this.dpcCache = null;
    this.dpcCacheQueueReady = false;
  }

  updateVirtualDetector(mask: Uint8Array | Uint32Array): Promise<WebGPUCompactH5DetectorMetrics> {
    return this.submitVirtualDetector(this.prepareDetectorSelection(mask));
  }

  private submitVirtualDetector(selection: CompactDetectorSelection, submission?: CompactDetectorSubmission): Promise<WebGPUCompactH5DetectorMetrics> {
    this.requireResident();
    const requestedAt = selection.requestedAt;
    const pending = this.runVirtualDetectorUpdate(selection, submission);
    const submissionCpuMs = performance.now() - requestedAt;
    this.pendingDetectorUpdates.add(pending);
    void pending.then(
      (metrics) => {
        this.pendingDetectorUpdates.delete(pending);
        try {
          const target = globalThis as typeof globalThis & {
            __quantemCompactDetectorMetrics?: Array<Record<string, unknown>>;
          };
          const history = target.__quantemCompactDetectorMetrics ?? [];
          history.push({
            atMs: performance.now(),
            requestToCompletionMs: performance.now() - requestedAt,
            submissionCpuMs,
            sourceIdentitySha256: this.metadata.sourceIdentitySha256,
            ...metrics,
          });
          if (history.length > 240) history.splice(0, history.length - 240);
          target.__quantemCompactDetectorMetrics = history;
        } catch {
          // Diagnostics must never affect scientific compute.
        }
      },
      () => {
        this.pendingDetectorUpdates.delete(pending);
        this.hasDetector = false;
      },
    );
    return pending;
  }

  private trackConsumerOperation<T>(operation: Promise<T>): Promise<T> {
    this.pendingConsumerOperations.add(operation);
    void operation.then(
      () => { this.pendingConsumerOperations.delete(operation); },
      () => { this.pendingConsumerOperations.delete(operation); },
    );
    return operation;
  }

  private prepareDetectorSelection(mask: Uint8Array | Uint32Array): CompactDetectorSelection {
    const requestedAt = performance.now();
    this.requireResident();
    const pixelCount = this.maximumWidths.length;
    if (mask.length !== pixelCount) {
      throw new Error(`A compact virtual-detector mask requires ${pixelCount} row-major values; got ${mask.length}.`);
    }
    const normalized = new Uint8Array(pixelCount);
    for (let pixel = 0; pixel < pixelCount; pixel++) {
      const value = Number(mask[pixel]);
      if (value !== 0 && value !== 1) {
        throw new Error("A compact virtual-detector mask must contain only zero or one.");
      }
      normalized[pixel] = value;
    }
    for (const pixel of this.metadata.excludedDetectorPixels) normalized[pixel] = 0;
    let maximumSum = 0;
    let selectedPixelCount = 0;
    for (let pixel = 0; pixel < pixelCount; pixel++) {
      if (!normalized[pixel]) continue;
      selectedPixelCount++;
      // Every admitted value is at most u32. Reject immediately above u32,
      // so this bound calculation is an exact JavaScript integer throughout.
      const maximumValue = MAXIMUM_PACKED_VALUES[this.maximumWidths[pixel]];
      maximumSum += maximumValue;
      if (maximumSum > 0xffffffff) {
        throw new Error(
          `This detector can sum to ${maximumSum}, beyond exact u32 output. `
          + "Use a narrower detector or a future u64 reduction path.",
        );
      }
    }
    return { mask: normalized, selectedPixelCount, requestedAt };
  }

  private runVirtualDetectorUpdate(selection: CompactDetectorSelection, submission?: CompactDetectorSubmission): Promise<WebGPUCompactH5DetectorMetrics> {
    const { mask: normalized, selectedPixelCount } = selection;
    const pixelCount = normalized.length;
    let preparedProduct: {
      mask: Uint8Array;
      buffer: GPUBuffer;
      selectedPixels: number;
    } | null = null;
    for (const candidate of this.preparedDetectorProducts.values()) {
      let matches = candidate.mask.length === normalized.length;
      for (let pixel = 0; matches && pixel < normalized.length; pixel++) {
        matches = candidate.mask[pixel] === normalized[pixel];
      }
      if (matches) {
        preparedProduct = candidate;
        break;
      }
    }
    if (preparedProduct) {
      let addedPixels = 0;
      let removedPixels = 0;
      for (let pixel = 0; pixel < pixelCount; pixel++) {
        if (normalized[pixel] === this.detectorMask[pixel]) continue;
        if (normalized[pixel]) addedPixels++;
        else removedPixels++;
      }
      const changedDetectorPixels = addedPixels + removedPixels;
      if (this.hasDetector && changedDetectorPixels === 0) {
        this.lastDetectorSubmission = {
          mode: "delta", changedDetectorPixels: 0, addedPixels: 0, removedPixels: 0,
        };
        return Promise.resolve({
          mode: "delta",
          changedDetectorPixels: 0,
          addressingMode: "inline-header",
          dispatchChunks: 0,
          wallMs: 0,
          gpuMs: null,
          fftDispatchCount: 0,
        });
      }
      const nextOutput = 1 - this.activeDetectorOutput;
      const encoder = submission?.encoder ?? this.device.createCommandEncoder({
        label: "compact prepared virtual detector activation",
      });
      encoder.copyBufferToBuffer(
        preparedProduct.buffer,
        0,
        this.detectorOutputs[nextOutput],
        0,
        this.scanCount * 4,
      );
      const started = performance.now();
      if (!submission) this.device.queue.submit([encoder.finish()]);
      this.activeDetectorOutput = nextOutput;
      this.detectorMask = normalized;
      this.hasDetector = true;
      this.lastDetectorSubmission = {
        mode: "prepared", changedDetectorPixels, addedPixels, removedPixels,
      };
      return (submission?.completion ?? this.device.queue.onSubmittedWorkDone()).then(() => ({
        mode: "prepared",
        changedDetectorPixels,
        addressingMode: "inline-header",
        dispatchChunks: 0,
        wallMs: performance.now() - started,
        gpuMs: null,
        fftDispatchCount: 0,
      }));
    }

    let rebase = !this.hasDetector;
    const changed: number[] = [];
    let addedPixels = 0;
    let removedPixels = 0;
    for (let pixel = 0; pixel < pixelCount; pixel++) {
      if ((rebase && normalized[pixel] !== 0) || (!rebase && normalized[pixel] !== this.detectorMask[pixel])) {
        changed.push(pixel);
        if (normalized[pixel]) addedPixels++;
        else removedPixels++;
      }
    }
    // A distant detector jump can remove nearly the entire old mask and add the
    // entire new one. Recomputing the new mask from zero is exact and processes
    // fewer packed detector planes whenever that selected set is smaller than
    // the signed delta. Small drags continue to use the much cheaper delta path.
    if (!rebase && changed.length > selectedPixelCount) {
      rebase = true;
      changed.length = 0;
      addedPixels = 0;
      removedPixels = 0;
      for (let pixel = 0; pixel < pixelCount; pixel++) {
        if (!normalized[pixel]) continue;
        changed.push(pixel);
        addedPixels++;
      }
    }
    this.lastDetectorSubmission = {
      mode: rebase ? "rebase" : "delta",
      changedDetectorPixels: changed.length,
      addedPixels,
      removedPixels,
    };
    if (changed.length === 0) {
      this.detectorMask = normalized;
      this.hasDetector = true;
      let completion = Promise.resolve();
      if (rebase) {
        // An empty detector has no entries to dispatch, but its scientific
        // result is zero, not the previous nonempty detector's retained sum.
        const nextOutput = 1 - this.activeDetectorOutput;
        const encoder = submission?.encoder ?? this.device.createCommandEncoder({ label: "compact empty detector" });
        encoder.clearBuffer(this.detectorOutputs[nextOutput]);
        if (!submission) this.device.queue.submit([encoder.finish()]);
        this.activeDetectorOutput = nextOutput;
        completion = submission?.completion ?? this.device.queue.onSubmittedWorkDone();
      }
      return completion.then(() => ({
        mode: rebase ? "rebase" : "delta",
        changedDetectorPixels: 0,
        addressingMode: "inline-header",
        dispatchChunks: 0,
        wallMs: 0,
        gpuMs: null,
        fftDispatchCount: 0,
      }));
    }
    const packed = new Uint32Array(changed.length * 2);
    for (let index = 0; index < changed.length; index++) {
      const pixel = changed[index];
      packed[index * 2] = pixel;
      packed[index * 2 + 1] = normalized[pixel] ? 1 : 0xffffffff;
    }
    const alignment = Math.max(256, this.device.limits.minUniformBufferOffsetAlignment);
    const parametersBytes = alignment * this.shards.length;
    // Fixed-capacity v1 dispatch storage keeps hundreds of immutable shard
    // bindings out of the pointer-rate path. Queue writes are ordered before
    // their dispatch, so later detector updates cannot overwrite earlier work.
    if (this.metadata.schemaVersion === 1 && !this.detectorDispatchStorage) {
      const entries = this.device.createBuffer({
        label: "compact retained detector entries",
        size: pixelCount * 8,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      let parameters: GPUBuffer | null = null;
      try {
        parameters = this.device.createBuffer({
          label: "compact retained detector parameters",
          size: parametersBytes,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        const parameterBuffer = parameters;
        const bindGroups = this.detectorOutputs.map((previous, current) =>
          this.shards.map(shard => this.device.createBindGroup({
            layout: this.detectorLayout,
            entries: [
              { binding: 0, resource: { buffer: shard.payload } },
              { binding: 1, resource: { buffer: shard.descriptors } },
              { binding: 2, resource: { buffer: entries } },
              { binding: 3, resource: { buffer: previous } },
              { binding: 4, resource: { buffer: this.detectorOutputs[1 - current] } },
              { binding: 5, resource: { buffer: parameterBuffer, size: 32 } },
            ],
          })),
        );
        this.detectorDispatchStorage = {
          entries, parameters, parameterValues: new Uint32Array(parametersBytes / 4), bindGroups,
        };
      } catch (error) {
        entries.destroy();
        parameters?.destroy();
        throw error;
      }
    }
    const retainedDispatch = this.detectorDispatchStorage;
    const entries = retainedDispatch?.entries ?? uploadBytes(this.device, packed, GPUBufferUsage.STORAGE);
    if (retainedDispatch) this.device.queue.writeBuffer(entries, 0, packed);
    const tileCount = Math.ceil(this.metadata.scansPerShard / this.metadata.scanTile);
    // A full detector teleport can change thousands of pixels. Resolve as many
    // addresses as the device can hold in one buffer, then bind only the current
    // shard's slice. Per-shard slices stay below maxStorageBufferBindingSize even
    // when the combined address slab is larger than that binding limit. The cap
    // bounds transient memory to 256 MiB for the 64-shard 512x512 fixture while
    // collapsing its common far taps from tens of passes to one.
    const resolvedBytesPerEntryPerShard = tileCount * 8;
    const resolvedChunkLimit = Math.max(1, Math.min(
      16384,
      Math.floor(
        this.device.limits.maxBufferSize
        / Math.max(1, this.shards.length * resolvedBytesPerEntryPerShard),
      ),
      Math.floor(
        this.device.limits.maxStorageBufferBindingSize
        / Math.max(1, resolvedBytesPerEntryPerShard),
      ),
    ));
    const resolvedChunkEntries = Math.min(changed.length, resolvedChunkLimit);
    const resolvedBytes = this.shards.length * resolvedChunkEntries * tileCount * 8;
    const useResolvedV3 = this.metadata.schemaVersion === 3
      && resolvedChunkEntries > 0
      && resolvedBytes <= this.device.limits.maxBufferSize
      && resolvedChunkEntries * resolvedBytesPerEntryPerShard
        <= this.device.limits.maxStorageBufferBindingSize;
    const chunks = useResolvedV3
      ? Array.from({ length: Math.ceil(changed.length / resolvedChunkLimit) }, (_, chunk) => ({
        start: chunk * resolvedChunkLimit,
        count: Math.min(resolvedChunkLimit, changed.length - chunk * resolvedChunkLimit),
      }))
      : [{ start: 0, count: changed.length }];
    const parameterBuffers: GPUBuffer[] = [];
    const resolved = useResolvedV3 ? this.device.createBuffer({
      label: "compact detector resolved v3 tile addresses",
      size: Math.max(8, resolvedBytes),
      usage: GPUBufferUsage.STORAGE,
    }) : null;
    const timestampQuery = this.detectorTimestampQuery;
    const timestampResolve = this.detectorTimestampResolve;
    const timestampReadback = this.detectorTimestampReadback;
    const collectGpuTimestamp = Boolean(
      timestampQuery
      && timestampResolve
      && timestampReadback
      && chunks.length === 1
      && !this.detectorTimestampMapPending
    );
    const encoder = submission?.encoder ?? this.device.createCommandEncoder({ label: "compact virtual detector" });
    let currentOutput = this.activeDetectorOutput;
    for (let chunkIndex = 0; chunkIndex < chunks.length; chunkIndex++) {
      const chunk = chunks[chunkIndex];
      const parameters = retainedDispatch?.parameterValues ?? new Uint32Array(parametersBytes / 4);
      for (let shardIndex = 0; shardIndex < this.shards.length; shardIndex++) {
        const offset = shardIndex * alignment / 4;
        parameters[offset] = this.metadata.scansPerShard;
        parameters[offset + 1] = tileCount;
        parameters[offset + 2] = chunk.count;
        parameters[offset + 3] = shardIndex * this.metadata.scansPerShard;
        parameters[offset + 4] = rebase && chunkIndex === 0 ? 1 : 0;
        parameters[offset + 5] = this.metadata.schemaVersion;
        parameters[offset + 6] = 0;
        parameters[offset + 7] = chunk.start;
      }
      const parameterBuffer = retainedDispatch?.parameters
        ?? uploadBytes(this.device, new Uint8Array(parameters.buffer), GPUBufferUsage.UNIFORM);
      if (retainedDispatch) this.device.queue.writeBuffer(parameterBuffer, 0, parameters);
      else parameterBuffers.push(parameterBuffer);
      if (resolved) {
        const resolvePass = encoder.beginComputePass();
        resolvePass.setPipeline(this.detectorResolveV3Pipeline);
        for (let shardIndex = 0; shardIndex < this.shards.length; shardIndex++) {
          resolvePass.setBindGroup(0, this.device.createBindGroup({
            layout: this.detectorResolveV3Layout,
            entries: [
              { binding: 0, resource: { buffer: this.shards[shardIndex].descriptors } },
              { binding: 1, resource: { buffer: entries } },
              { binding: 2, resource: {
                buffer: resolved,
                offset: shardIndex * chunk.count * resolvedBytesPerEntryPerShard,
                size: chunk.count * resolvedBytesPerEntryPerShard,
              } },
              { binding: 3, resource: { buffer: parameterBuffer, size: 32 } },
            ],
          }), [shardIndex * alignment]);
          resolvePass.dispatchWorkgroups(Math.ceil(tileCount / 16), Math.ceil(chunk.count / 16));
        }
        resolvePass.end();
      }
      const nextOutput = 1 - currentOutput;
      const pass = encoder.beginComputePass(collectGpuTimestamp ? {
        timestampWrites: {
          querySet: timestampQuery!,
          beginningOfPassWriteIndex: 0,
          endOfPassWriteIndex: 1,
        },
      } : undefined);
      pass.setPipeline(resolved ? this.detectorResolvedV3Pipeline : this.detectorPipeline);
      for (let shardIndex = 0; shardIndex < this.shards.length; shardIndex++) {
        const shard = this.shards[shardIndex];
        const bindGroup = retainedDispatch?.bindGroups[currentOutput][shardIndex] ?? this.device.createBindGroup({
          layout: this.detectorLayout,
          entries: [
            { binding: 0, resource: { buffer: shard.payload } },
            { binding: 1, resource: resolved ? {
              buffer: resolved,
              offset: shardIndex * chunk.count * resolvedBytesPerEntryPerShard,
              size: chunk.count * resolvedBytesPerEntryPerShard,
            } : { buffer: shard.descriptors } },
            { binding: 2, resource: { buffer: entries } },
            { binding: 3, resource: { buffer: this.detectorOutputs[currentOutput] } },
            { binding: 4, resource: { buffer: this.detectorOutputs[nextOutput] } },
            { binding: 5, resource: { buffer: parameterBuffer, size: 32 } },
          ],
        });
        pass.setBindGroup(0, bindGroup, [shardIndex * alignment]);
        pass.dispatchWorkgroups(Math.ceil(this.metadata.scansPerShard / 32));
      }
      pass.end();
      currentOutput = nextOutput;
    }
    if (collectGpuTimestamp) {
      encoder.resolveQuerySet(timestampQuery!, 0, 2, timestampResolve!, 0);
      encoder.copyBufferToBuffer(timestampResolve!, 0, timestampReadback!, 0, 16);
    }
    const started = performance.now();
    if (collectGpuTimestamp) this.detectorTimestampMapPending = true;
    try {
      if (!submission) this.device.queue.submit([encoder.finish()]);
    } catch (error) {
      if (collectGpuTimestamp) this.detectorTimestampMapPending = false;
      if (!retainedDispatch) entries.destroy();
      for (const parameterBuffer of parameterBuffers) parameterBuffer.destroy();
      resolved?.destroy();
      throw error;
    }
    this.activeDetectorOutput = currentOutput;
    this.detectorMask = normalized;
    this.hasDetector = true;
    // Encoding is synchronous so the batch owner can reject the entire batch
    // before submission. Only completion and optional timing readback are async.
    return (async (): Promise<WebGPUCompactH5DetectorMetrics> => {
      let wallMs = 0;
      let gpuMs: number | null = null;
      try {
        await (submission?.completion ?? this.device.queue.onSubmittedWorkDone());
        wallMs = performance.now() - started;
        if (collectGpuTimestamp && !this.isReleased) {
          await timestampReadback!.mapAsync(GPUMapMode.READ);
          const timestamps = new BigUint64Array(timestampReadback!.getMappedRange().slice(0));
          timestampReadback!.unmap();
          gpuMs = Number(timestamps[1] - timestamps[0]) / 1_000_000;
        }
      } finally {
        if (collectGpuTimestamp) {
          this.detectorTimestampMapPending = false;
          if (this.isReleased) {
            timestampQuery!.destroy();
            timestampResolve!.destroy();
            timestampReadback!.destroy();
          }
        }
        if (!retainedDispatch) entries.destroy();
        for (const parameterBuffer of parameterBuffers) parameterBuffer.destroy();
        resolved?.destroy();
      }
      return {
        mode: rebase ? "rebase" : "delta",
        changedDetectorPixels: changed.length,
        addressingMode: resolved ? "resolved-v3-tiles" : "inline-header",
        dispatchChunks: chunks.length,
        wallMs,
        gpuMs,
        fftDispatchCount: 0,
      };
    })();
  }

  async rebaseVirtualDetector(mask: Uint8Array | Uint32Array): Promise<WebGPUCompactH5DetectorMetrics> {
    this.hasDetector = false;
    return this.updateVirtualDetector(mask);
  }

  /** Wait for every submitted detector update and consumer readback to finish. */
  async quiesce(): Promise<void> {
    this.publishLifecycle("quiesce-start");
    while (this.pendingDetectorUpdates.size > 0 || this.pendingConsumerOperations.size > 0) {
      await Promise.allSettled([
        ...this.pendingDetectorUpdates,
        ...this.pendingConsumerOperations,
      ]);
    }
    await this.device.queue.onSubmittedWorkDone();
    this.publishLifecycle("quiesce-end");
  }

  virtualDetectorBuffer(): GPUBuffer {
    this.requireResident();
    if (!this.hasDetector) throw new Error("Run updateVirtualDetector before reading a result.");
    return this.detectorOutputs[this.activeDetectorOutput];
  }

  async virtualDetectorValues(): Promise<Uint32Array> {
    return readU32Buffer(this.device, this.virtualDetectorBuffer(), this.scanCount);
  }

  releaseResidentStorage(): void {
    if (this.isReleased) return;
    if (this.pendingDetectorUpdates.size > 0 || this.pendingConsumerOperations.size > 0) {
      throw new Error(
        "Compact WebGPU storage still has pending operations. Await quiesce() before release.",
      );
    }
    this.publishLifecycle("release-start");
    this.isReleased = true;
    for (const shard of this.shards) {
      shard.payload.destroy();
      shard.descriptors.destroy();
    }
    this.shards = [];
    this.excluded.destroy();
    this.diffractionOutput.destroy();
    this.detectorDisplayOutput.destroy();
    this.detectorDisplayCount.destroy();
    this.detectorDisplayBindGroups = [];
    this.detectorDispatchStorage?.entries.destroy();
    this.detectorDispatchStorage?.parameters.destroy();
    this.detectorDispatchStorage = null;
    this.clearDpcCache();
    this.preparedDpcMoments?.buffer.destroy();
    this.preparedDpcMoments = null;
    for (const product of this.preparedDetectorProducts.values()) {
      product.buffer.destroy();
    }
    this.preparedDetectorProducts.clear();
    for (const output of this.detectorOutputs) output.destroy();
    this.detectorOutputs = [];
    if (!this.detectorTimestampMapPending) {
      this.detectorTimestampQuery?.destroy();
      this.detectorTimestampResolve?.destroy();
      this.detectorTimestampReadback?.destroy();
    }
    this.detectorTimestampQuery = null;
    this.detectorTimestampResolve = null;
    this.detectorTimestampReadback = null;
    this.maximumWidths = new Uint8Array(0);
    this.detectorMask = new Uint8Array(0);
    this.publishLifecycle("release-end");
  }

  dispose(): void {
    this.releaseResidentStorage();
  }

  private requireResident(): void {
    if (this.isReleased) {
      throw new Error("The compact WebGPU source has been released. Load it again before requesting scientific output.");
    }
  }
}

export async function probeCompactH5HttpSource(url: string): Promise<CompactH5ByteSource | null> {
  const response = await fetch(url, { headers: { Range: `bytes=0-${PRELUDE_BYTES - 1}` } });
  if (response.status !== 206) {
    await response.body?.cancel();
    return null;
  }
  const sourceBytes = parseContentRange(response.headers.get("Content-Range"), 0, PRELUDE_BYTES);
  const prelude = new Uint8Array(await response.arrayBuffer());
  if (prelude.byteLength !== PRELUDE_BYTES) {
    throw new Error(`Compact HDF5 HTTP probe ended after ${prelude.byteLength} bytes; expected ${PRELUDE_BYTES}.`);
  }
  if (!matchesMagic(prelude, 0, CONTAINER_MAGIC)) return null;
  const sourceName = compactSourceName(url);
  let cachedPrelude: Uint8Array | null = prelude;
  return {
    size: sourceBytes,
    name: sourceName,
    readRange: async (offset, byteCount) => {
      if (cachedPrelude && offset === 0 && byteCount === PRELUDE_BYTES) {
        const result = cachedPrelude;
        cachedPrelude = null;
        return result;
      }
      return readHttpRange(url, sourceBytes, offset, byteCount);
    },
  };
}

export async function parseCompactH5Index(source: CompactH5Source): Promise<CompactH5Index> {
  const prelude = await readSourceRange(source, 0, PRELUDE_BYTES);
  if (prelude.byteLength !== PRELUDE_BYTES || !matchesMagic(prelude, 0, CONTAINER_MAGIC)) {
    throw new Error("The selected file has no QuantEM compact HDF5 user-block index.");
  }
  const preludeView = new DataView(prelude.buffer, prelude.byteOffset, prelude.byteLength);
  const headerBytes = preludeView.getUint32(8, true);
  const headerCrc32 = preludeView.getUint32(12, true);
  const binaryOffset = preludeView.getUint32(16, true);
  const binaryBytes = preludeView.getUint32(20, true);
  if (headerBytes === 0 || headerBytes > binaryOffset - PRELUDE_BYTES) {
    throw new Error(`Compact JSON header length ${headerBytes} is invalid.`);
  }
  if (binaryOffset < PRELUDE_BYTES + headerBytes || binaryBytes < INDEX_HEADER_V1_BYTES + 36
      || binaryOffset > source.size || binaryBytes > source.size - binaryOffset) {
    throw new Error("Compact binary index range is outside the selected file.");
  }
  const [header, binary] = await Promise.all([
    readSourceRange(source, PRELUDE_BYTES, headerBytes),
    readSourceRange(source, binaryOffset, binaryBytes),
  ]);
  if (crc32(header) !== headerCrc32) throw new Error("Compact JSON header failed its CRC-32 check.");
  let manifest: Record<string, unknown>;
  try {
    manifest = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(header));
  } catch (error) {
    throw new Error(`Compact JSON header is not valid UTF-8 JSON: ${String(error)}`);
  }
  const view = new DataView(binary.buffer, binary.byteOffset, binary.byteLength);
  let schemaVersion: 1 | 3;
  let shardCount: number;
  let payloadChunkBytes: 128 | 0;
  let shape: [number, number, number, number];
  let scansPerShard: number;
  let scanTile: 128 | 32;
  let headerEncoding: 0 | 1;
  let cursor: number;
  if (matchesMagic(binary, 0, INDEX_MAGIC_V1)) {
    if (binary.byteLength < INDEX_HEADER_V1_BYTES) throw new Error("Compact v1 binary index is truncated.");
    schemaVersion = 1;
    shardCount = view.getUint32(8, true);
    payloadChunkBytes = view.getUint32(12, true) as 128;
    shape = [
      view.getUint32(16, true), view.getUint32(20, true),
      view.getUint32(24, true), view.getUint32(28, true),
    ];
    scansPerShard = view.getUint32(32, true);
    scanTile = SCAN_TILE_V1;
    headerEncoding = 0;
    cursor = INDEX_HEADER_V1_BYTES;
    if (payloadChunkBytes !== 128) throw new Error(`Compact v1 requires 128-byte raw-LZ4 chunks; got ${payloadChunkBytes}.`);
  } else if (matchesMagic(binary, 0, INDEX_MAGIC_V3)) {
    if (binary.byteLength < INDEX_HEADER_V3_BYTES) throw new Error("Compact v3 binary index is truncated.");
    schemaVersion = 3;
    shardCount = view.getUint32(8, true);
    const reserved = view.getUint32(12, true);
    shape = [
      view.getUint32(16, true), view.getUint32(20, true),
      view.getUint32(24, true), view.getUint32(28, true),
    ];
    scansPerShard = view.getUint32(32, true);
    scanTile = view.getUint32(36, true) as 32;
    headerEncoding = view.getUint32(40, true) as 1;
    payloadChunkBytes = 0;
    cursor = INDEX_HEADER_V3_BYTES;
    if (reserved !== 0 || scanTile !== SCAN_TILE_V3 || headerEncoding !== 1) {
      throw new Error("Compact v3 requires reserved=0, scan_tile=32, and compact header encoding 1.");
    }
  } else {
    throw new Error("Unsupported compact binary index version.");
  }
  if (shardCount === 0 || scansPerShard === 0 || shape.some((value) => value === 0)) {
    throw new Error("Compact shape, shard count, and scans per shard must be positive.");
  }
  if (shape[0] * shape[1] !== shardCount * scansPerShard) {
    throw new Error("Compact shards do not cover the complete scan plane.");
  }
  if (schemaVersion === 3 && scansPerShard % scanTile !== 0) {
    throw new Error("Compact v3 requires complete 32-scan tiles.");
  }
  const detectorPixels = shape[2] * shape[3];
  if (cursor + 4 > binary.byteLength) throw new Error("Compact detector mask is truncated.");
  const maskCount = view.getUint32(cursor, true);
  cursor += 4;
  if (maskCount > detectorPixels || cursor + maskCount * 4 + 32 > binary.byteLength) {
    throw new Error("Compact detector mask is invalid.");
  }
  const excluded = new Uint32Array(maskCount);
  const excludedSet = new Set<number>();
  for (let index = 0; index < maskCount; index++) {
    const pixel = view.getUint32(cursor, true);
    cursor += 4;
    if (pixel >= detectorPixels || excludedSet.has(pixel)) throw new Error("Compact detector mask is invalid.");
    excluded[index] = pixel;
    excludedSet.add(pixel);
  }
  if (schemaVersion === 3) {
    for (let index = 1; index < excluded.length; index++) {
      if (excluded[index] < excluded[index - 1]) {
        throw new Error("Compact v3 detector mask must use ordered row-major pixel indices.");
      }
    }
  }
  const sourceIdentitySha256 = bytesToHex(binary.subarray(cursor, cursor + 32));
  cursor += 32;
  const tileCount = Math.ceil(scansPerShard / scanTile);
  const shards: CompactH5ShardIndex[] = [];
  const ranges: Array<{ start: number; end: number; label: string }> = [];
  let residentBytes = 0;
  for (let shardIndex = 0; shardIndex < shardCount; shardIndex++) {
    if (cursor + SHARD_RECORD_BYTES > binary.byteLength) throw new Error(`Compact shard record ${shardIndex} is truncated.`);
    const record: CompactH5ShardIndex = {
      payloadOffset: safeU64(view, cursor, `shard ${shardIndex} payload offset`),
      payloadBytes: safeU64(view, cursor + 8, `shard ${shardIndex} payload bytes`),
      lengthsOffset: safeU64(view, cursor + 16, `shard ${shardIndex} length offset`),
      lengthsBytes: safeU64(view, cursor + 24, `shard ${shardIndex} length bytes`),
      widthsOffset: safeU64(view, cursor + 32, `shard ${shardIndex} width offset`),
      widthsBytes: safeU64(view, cursor + 40, `shard ${shardIndex} width bytes`),
      decodedBytes: safeU64(view, cursor + 48, `shard ${shardIndex} decoded bytes`),
      descriptorCount: view.getUint32(cursor + 56, true),
      chunkCount: view.getUint32(cursor + 60, true),
      decodedSha256: bytesToHex(binary.subarray(cursor + 64, cursor + 96)),
      encodedEnvelopeSha256: null,
    };
    cursor += SHARD_RECORD_BYTES;
    if (schemaVersion === 1) {
      if (record.descriptorCount !== detectorPixels * tileCount || record.widthsBytes !== record.descriptorCount) {
        throw new Error(`Compact shard ${shardIndex} descriptor coverage is invalid.`);
      }
      if (record.payloadBytes === 0 || record.lengthsBytes !== record.chunkCount
          || record.decodedBytes === 0 || record.decodedBytes % 4 !== 0
          || record.chunkCount !== Math.ceil(record.decodedBytes / payloadChunkBytes)) {
        throw new Error(`Compact shard ${shardIndex} raw-LZ4 metadata is invalid.`);
      }
    } else {
      const checkpointWords = Math.ceil(tileCount / V3_CHECKPOINT_TILES);
      const widthWords = Math.ceil(tileCount / V3_WIDTHS_PER_WORD);
      const expectedHeaderWords = detectorPixels * (checkpointWords + widthWords);
      if (record.descriptorCount !== expectedHeaderWords || record.widthsBytes !== record.descriptorCount * 4) {
        throw new Error(`Compact v3 shard ${shardIndex} compact-header coverage is invalid.`);
      }
      if (record.lengthsOffset !== 0 || record.lengthsBytes !== 0 || record.chunkCount !== 0) {
        throw new Error(`Compact v3 shard ${shardIndex} unexpectedly contains raw-LZ4 metadata.`);
      }
      if (record.payloadBytes === 0 || record.payloadBytes !== record.decodedBytes || record.payloadBytes % 4 !== 0) {
        throw new Error(`Compact v3 shard ${shardIndex} direct payload is not a nonempty uint32 range.`);
      }
    }
    for (const [label, offset, bytes] of [
      ["payload", record.payloadOffset, record.payloadBytes],
      ["chunk lengths", record.lengthsOffset, record.lengthsBytes],
      ["descriptor widths", record.widthsOffset, record.widthsBytes],
    ] as const) {
      if (offset > source.size || bytes > source.size - offset) throw new Error(`Compact shard ${shardIndex} ${label} range is outside the file.`);
      ranges.push({ start: offset, end: offset + bytes, label: `shard ${shardIndex} ${label}` });
    }
    residentBytes += record.decodedBytes + (schemaVersion === 1 ? record.descriptorCount * 4 : record.widthsBytes);
    if (!Number.isSafeInteger(residentBytes)) throw new Error("Compact resident byte count exceeds exact JavaScript integer range.");
    shards.push(record);
  }
  if (cursor !== binary.byteLength) throw new Error("Compact binary index has trailing bytes.");
  ranges.sort((a, b) => a.start - b.start);
  for (let index = 1; index < ranges.length; index++) {
    if (ranges[index].start < ranges[index - 1].end) throw new Error(`Compact file ranges overlap between ${ranges[index - 1].label} and ${ranges[index].label}.`);
  }
  const protectedPrefixBytes = Math.min(
    ...ranges.filter((range) => range.start > 0).map((range) => range.start),
  );
  const validatedManifest = validateManifest(
    manifest,
    source.size,
    protectedPrefixBytes,
    shape,
    shardCount,
    scansPerShard,
    sourceIdentitySha256,
    excluded,
    schemaVersion,
    scanTile,
    headerEncoding,
    shards,
  );
  residentBytes += validatedManifest.preparedDpcMoments?.fileBytes ?? 0;
  residentBytes += validatedManifest.preparedDetectorProducts?.products.reduce(
    (total, product) => total + product.valuesFileBytes,
    0,
  ) ?? 0;
  if (!Number.isSafeInteger(residentBytes)) {
    throw new Error("Compact resident byte count exceeds exact JavaScript integer range.");
  }
  if (schemaVersion === 3) {
    if ("shards" in manifest) {
      throw new Error("Compact v3 embeds an unsupported JSON shard table.");
    }
  } else {
    const manifestShards = manifest.shards;
    if (!Array.isArray(manifestShards) || manifestShards.length !== shards.length) {
      throw new Error("Compact JSON shard list is incomplete.");
    }
    for (let shardIndex = 0; shardIndex < shards.length; shardIndex++) {
      const record = manifestShards[shardIndex];
      const shard = shards[shardIndex];
      if (!record || typeof record !== "object" || Array.isArray(record)) {
        throw new Error(`Compact JSON shard ${shardIndex} is not an object.`);
      }
      const json = record as Record<string, unknown>;
      const expected: Record<string, number | string> = {
        ordinal: shardIndex,
        payload_file_offset: shard.payloadOffset,
        payload_file_bytes: shard.payloadBytes,
        lengths_file_offset: shard.lengthsOffset,
        lengths_file_bytes: shard.lengthsBytes,
        descriptor_widths_file_offset: shard.widthsOffset,
        descriptor_widths_file_bytes: shard.widthsBytes,
        payload_decoded_bytes: shard.decodedBytes,
        descriptor_count: shard.descriptorCount,
        payload_chunk_count: shard.chunkCount,
        payload_decoded_sha256: shard.decodedSha256,
      };
      for (const [key, value] of Object.entries(expected)) {
        if (json[key] !== value) throw new Error(`Compact JSON shard ${shardIndex} ${key} disagrees with the binary index.`);
      }
      const encoded = json.encoded_envelope_sha256;
      if (encoded !== undefined && (typeof encoded !== "string" || !/^[0-9a-f]{64}$/.test(encoded))) {
        throw new Error(`Compact JSON shard ${shardIndex} encoded-envelope SHA-256 is invalid.`);
      }
      shard.encodedEnvelopeSha256 = typeof encoded === "string" ? encoded : null;
    }
  }
  return {
    sourceBytes: source.size,
    sourceName: source.name || "compact-source.h5",
    schemaVersion,
    shape,
    scansPerShard,
    scanTile,
    headerEncoding,
    payloadChunkBytes,
    payloadCodec: schemaVersion === 1 ? "raw-lz4" : "direct-bitpacked-u32",
    sourceIdentitySha256,
    sourceRawLogicalSha256: typeof manifest.source_raw_logical_sha256 === "string" ? manifest.source_raw_logical_sha256 : null,
    workingDtype: manifest.working_dtype as "uint8" | "uint16",
    detectorCalibration: validatedManifest.detectorCalibration,
    preparedDpcMoments: validatedManifest.preparedDpcMoments,
    preparedDetectorProducts: validatedManifest.preparedDetectorProducts,
    excludedDetectorPixels: excluded,
    maskedDetectorPixelsSha256: validatedManifest.maskedDetectorPixelsSha256,
    maskedDetectorRawValues: validatedManifest.maskedDetectorRawValues,
    rawReconstructionAvailable: schemaVersion === 1 ? (
      manifest.working_dtype === "uint16"
      && manifest.masked_detector_payload_policy === "retained_exactly_in_payload"
    ) : (
      excluded.length === 0
      || (validatedManifest.maskedDetectorPixelsSha256 !== null && validatedManifest.maskedDetectorRawValues !== null)
    ),
    shards,
    residentBytes,
    manifest,
  };
}

export async function loadCompactH5WebGPU(
  source: CompactH5Source,
  options: {
    device?: GPUDevice;
    shouldCancel?: () => boolean;
    integrity?: "auto" | "decoded-sha256";
    expectedWholeFileSha256?: string;
    trustedQualification?: CompactH5TrustedQualificationV1;
    sessionQualificationCache?: CompactH5SessionQualificationCache;
    /** Trusted generated-viewer receipt, never a sidecar selected alongside the source. */
    expectedReceipt?: CompactH5ResidentReceipt;
    /** Exact build identity, with patch digest when uncommitted. */
    implementationRevision?: string;
    /** Opt into per-update GPU timestamps and their diagnostic CPU readbacks. */
    collectDetectorTimings?: boolean;
  } = {},
): Promise<WebGPUCompactH5ResidentSource> {
  const totalStart = performance.now();
  const metadataStart = performance.now();
  const metadata = await parseCompactH5Index(source);
  const implementationRevision = options.implementationRevision ?? null;
  if (implementationRevision !== null && !implementationRevision.trim()) {
    throw new Error("Implementation revision must name the exact build or be omitted.");
  }
  if (options.expectedReceipt !== undefined) {
    requireMatchingResidentReceipt(compactReceipt(metadata, implementationRevision), options.expectedReceipt);
  }
  const metadataMs = performance.now() - metadataStart;
  let wholeFileIntegrityMs = 0;
  let trustedPrefixIntegrityMs = 0;
  let wholeFileSha256Checks: 0 | 1 = 0;
  let trustedPrefixSha256Checks: 0 | 1 = 0;
  let trustedQualification: CompactH5TrustedQualificationV1 | null = null;
  let sessionQualificationReused = false;
  if (metadata.schemaVersion === 3) {
    if (options.expectedWholeFileSha256 !== undefined && options.trustedQualification !== undefined) {
      throw new Error("QGIX v3 accepts either whole-file qualification or one trusted sidecar, not both.");
    }
    if (options.trustedQualification !== undefined) {
      trustedQualification = validateTrustedQualification(metadata, options.trustedQualification);
      const integrityStart = performance.now();
      const observed = await sha256SourceRange(source, 0, trustedQualification.prefixBytes);
      trustedPrefixIntegrityMs = performance.now() - integrityStart;
      trustedPrefixSha256Checks = 1;
      if (observed !== trustedQualification.prefixSha256) {
        throw new Error(
          `QGIX v3 trusted prefix SHA-256 is ${observed}, expected ${trustedQualification.prefixSha256}.`,
        );
      }
      sessionQualificationReused = options.sessionQualificationCache?.has(
        source,
        trustedQualification.sealedWholeFileSha256,
      ) === true;
    } else {
      if (!isSha256(options.expectedWholeFileSha256)) {
        throw new Error(
          "QGIX v3 qualification requires expectedWholeFileSha256 or an explicitly trusted qualification sidecar.",
        );
      }
      const integrityStart = performance.now();
      const observed = await sha256Source(source);
      wholeFileIntegrityMs = performance.now() - integrityStart;
      wholeFileSha256Checks = 1;
      if (observed !== options.expectedWholeFileSha256) {
        throw new Error(`QGIX v3 whole-file SHA-256 is ${observed}, expected ${options.expectedWholeFileSha256}.`);
      }
    }
  }
  const device = options.device || await requireHardwareGPUDevice("Compact HDF5 loading");
  const shouldCancel = options.shouldCancel || (() => false);
  const useEncodedIntegrity = options.integrity !== "decoded-sha256"
    && metadata.shards.every((shard) => shard.encodedEnvelopeSha256 !== null);
  if (shouldCancel()) throw new Error("Compact WebGPU load was cancelled before allocation.");
  for (const [shardIndex, shard] of metadata.shards.entries()) {
    if (shard.decodedBytes > device.limits.maxBufferSize || shard.decodedBytes > device.limits.maxStorageBufferBindingSize
        || shard.descriptorCount * 4 > device.limits.maxBufferSize
        || shard.descriptorCount * 4 > device.limits.maxStorageBufferBindingSize) {
      throw new Error(`Compact shard ${shardIndex} exceeds this adapter's storage-buffer limit; no partial source was published.`);
    }
  }

  const pipelineStart = performance.now();
  const decodePipelinePromise = device.createComputePipelineAsync({
    layout: "auto",
    compute: { module: device.createShaderModule({ code: COMPACT_LZ4_WGSL }), entryPoint: "main" },
  });
  const descriptorPipelinePromise = device.createComputePipelineAsync({
    layout: "auto",
    compute: { module: device.createShaderModule({ code: COMPACT_DESCRIPTOR_WGSL }), entryPoint: "main" },
  });
  const v3HeaderValidationPipelinePromise = device.createComputePipelineAsync({
    layout: "auto",
    compute: { module: device.createShaderModule({ code: COMPACT_V3_HEADER_VALIDATE_WGSL }), entryPoint: "main" },
  });
  const selectedPipelinePromise = device.createComputePipelineAsync({
    layout: "auto",
    compute: { module: device.createShaderModule({ code: COMPACT_SELECTED_WGSL }), entryPoint: "main" },
  });
  const detectorLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: 32 } },
    ],
  });
  const detectorPipelinePromise = device.createComputePipelineAsync({
    layout: device.createPipelineLayout({ bindGroupLayouts: [detectorLayout] }),
    compute: { module: device.createShaderModule({ code: COMPACT_DETECTOR_WGSL }), entryPoint: "main" },
  });
  const detectorResolveV3Layout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: 32 } },
    ],
  });
  const detectorResolveV3PipelinePromise = device.createComputePipelineAsync({
    layout: device.createPipelineLayout({ bindGroupLayouts: [detectorResolveV3Layout] }),
    compute: { module: device.createShaderModule({ code: COMPACT_DETECTOR_RESOLVE_V3_WGSL }), entryPoint: "main" },
  });
  const detectorResolvedV3PipelinePromise = device.createComputePipelineAsync({
    layout: device.createPipelineLayout({ bindGroupLayouts: [detectorLayout] }),
    compute: { module: device.createShaderModule({ code: COMPACT_DETECTOR_RESOLVED_V3_WGSL }), entryPoint: "main" },
  });
  const u32ToF32PipelinePromise = device.createComputePipelineAsync({
    layout: "auto",
    compute: { module: device.createShaderModule({ code: COMPACT_U32_TO_F32_WGSL }), entryPoint: "main" },
  });
  const dpcMomentLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: 32 } },
    ],
  });
  const dpcMomentPipelinePromise = device.createComputePipelineAsync({
    layout: device.createPipelineLayout({ bindGroupLayouts: [dpcMomentLayout] }),
    compute: { module: device.createShaderModule({ code: COMPACT_DPC_MOMENTS_WGSL }), entryPoint: "main" },
  });
  const momentsToComPipelinePromise = device.createComputePipelineAsync({
    layout: "auto",
    compute: { module: device.createShaderModule({ code: COMPACT_MOMENTS_TO_COM_WGSL }), entryPoint: "main" },
  });
  const dpcMeanPipelinePromise = device.createComputePipelineAsync({
    layout: "auto",
    compute: { module: device.createShaderModule({ code: DPC_MEAN_WGSL }), entryPoint: "main" },
  });
  const dpcPairPipelinePromise = device.createComputePipelineAsync({
    layout: "auto",
    compute: { module: device.createShaderModule({ code: DPC_COMPONENT_PAIR_WGSL }), entryPoint: "main" },
  });
  const dpcOutputMeanPipelinePromise = device.createComputePipelineAsync({
    layout: "auto",
    compute: { module: device.createShaderModule({ code: DPC_OUTPUT_MEAN_WGSL }), entryPoint: "main" },
  });
  const dpcOutputUlpCorrectPipelinePromise = device.createComputePipelineAsync({
    layout: "auto",
    compute: { module: device.createShaderModule({ code: DPC_OUTPUT_ULP_CORRECT_WGSL }), entryPoint: "main" },
  });
  const [
    decodePipeline,
    descriptorPipeline,
    v3HeaderValidationPipeline,
    selectedPipeline,
    detectorPipeline,
    detectorResolveV3Pipeline,
    detectorResolvedV3Pipeline,
    u32ToF32Pipeline,
    dpcMomentPipeline,
    momentsToComPipeline,
    dpcMeanPipeline,
    dpcPairPipeline,
    dpcOutputMeanPipeline,
    dpcOutputUlpCorrectPipeline,
  ] = await Promise.all([
    decodePipelinePromise, descriptorPipelinePromise, v3HeaderValidationPipelinePromise, selectedPipelinePromise, detectorPipelinePromise,
    detectorResolveV3PipelinePromise, detectorResolvedV3PipelinePromise,
    u32ToF32PipelinePromise, dpcMomentPipelinePromise, momentsToComPipelinePromise,
    dpcMeanPipelinePromise, dpcPairPipelinePromise,
    dpcOutputMeanPipelinePromise, dpcOutputUlpCorrectPipelinePromise,
  ]);
  const pipelineCompileMs = performance.now() - pipelineStart;
  const residentShards: ResidentShard[] = [];
  const detectorPixels = metadata.shape[2] * metadata.shape[3];
  const tileCount = Math.ceil(metadata.scansPerShard / metadata.scanTile);
  const excludedSet = new Set(metadata.excludedDetectorPixels);
  const maximumWidths = new Uint8Array(detectorPixels);
  let sourceReadMs = 0;
  let gpuUploadSubmissionMs = 0;
  let gpuUploadFenceMs = 0;
  let descriptorPreparationMs = 0;
  let gpuDecodeWallMs = 0;
  let decodedIntegrityMs = 0;
  let encodedIntegrityMs = 0;
  let compactHeaderValidationMs = 0;
  let maximumTransientBytes = 0;
  let preparedDpcBytes = 0;
  let preparedDpcReadMs = 0;
  let preparedDpcPrimeMs = 0;
  let preparedDetectorProductBytes = 0;
  let preparedDetectorProductReadMs = 0;
  if (metadata.schemaVersion === 3) {
    const excludedValues = new Uint32Array(detectorPixels);
    for (const pixel of metadata.excludedDetectorPixels) excludedValues[pixel] = 1;
    const excluded = uploadBytes(device, excludedValues, GPUBufferUsage.STORAGE);
    maximumWidths.fill(8);
    for (const pixel of metadata.excludedDetectorPixels) maximumWidths[pixel] = 0;
    const validationBatchSize = Math.min(8, metadata.shards.length);
    const pendingReads = new Map<number, Promise<DirectCompactShardBytes>>();
    const preparedDpcRead = metadata.preparedDpcMoments
      ? (async () => {
        const started = performance.now();
        const bytes = await readSourceRange(
          source,
          metadata.preparedDpcMoments!.fileOffset,
          metadata.preparedDpcMoments!.fileBytes,
        );
        const digest = await sha256Hex(bytes);
        return { bytes, digest, elapsedMs: performance.now() - started };
      })()
      : null;
    const preparedDetectorProductsRead = metadata.preparedDetectorProducts
      ? (async () => {
        const started = performance.now();
        const products = await Promise.all(metadata.preparedDetectorProducts!.products.map(
          async (product) => {
            const [mask, values] = await Promise.all([
              readSourceRange(source, product.maskFileOffset, product.maskFileBytes),
              readSourceRange(source, product.valuesFileOffset, product.valuesFileBytes),
            ]);
            const [maskDigest, valuesDigest] = await Promise.all([
              sha256Hex(mask),
              sha256Hex(values),
            ]);
            if (maskDigest !== product.maskSha256) {
              throw new Error(
                `Compact prepared ${product.name.toUpperCase()} mask SHA-256 is ${maskDigest}, expected ${product.maskSha256}.`,
              );
            }
            if (valuesDigest !== product.valuesSha256) {
              throw new Error(
                `Compact prepared ${product.name.toUpperCase()} values SHA-256 is ${valuesDigest}, expected ${product.valuesSha256}.`,
              );
            }
            let selected = 0;
            for (let pixel = 0; pixel < mask.length; pixel++) {
              if (mask[pixel] !== 0 && mask[pixel] !== 1) {
                throw new Error(`Compact prepared ${product.name.toUpperCase()} mask is not binary.`);
              }
              if (excludedSet.has(pixel) && mask[pixel] !== 0) {
                throw new Error(`Compact prepared ${product.name.toUpperCase()} mask selects an excluded detector pixel.`);
              }
              selected += mask[pixel];
            }
            if (selected !== product.selectedDetectorPixels) {
              throw new Error(
                `Compact prepared ${product.name.toUpperCase()} mask selects ${selected} pixels, expected ${product.selectedDetectorPixels}.`,
              );
            }
            return { product, mask, values };
          },
        ));
        return { products, elapsedMs: performance.now() - started };
      })()
      : null;
    let nextRead = 0;
    let preparedDpcResident: {
      detectorPixels: Uint32Array;
      buffer: GPUBuffer;
    } | null = null;
    const preparedDetectorProductsResident = new Map<"bf" | "abf" | "adf", {
      mask: Uint8Array;
      buffer: GPUBuffer;
      selectedPixels: number;
    }>();
    const readDirectShard = async (shardIndex: number): Promise<DirectCompactShardBytes> => {
      const shard = metadata.shards[shardIndex];
      const started = performance.now();
      const rangeStart = Math.min(shard.payloadOffset, shard.widthsOffset);
      const rangeEnd = Math.max(
        shard.payloadOffset + shard.payloadBytes,
        shard.widthsOffset + shard.widthsBytes,
      );
      const envelope = await readSourceRange(source, rangeStart, rangeEnd - rangeStart);
      const payloadBytes = envelope.subarray(
        shard.payloadOffset - rangeStart,
        shard.payloadOffset - rangeStart + shard.payloadBytes,
      );
      const headerBytes = envelope.subarray(
        shard.widthsOffset - rangeStart,
        shard.widthsOffset - rangeStart + shard.widthsBytes,
      );
      return {
        payloadBytes,
        headerBytes,
        payloadDigest: sessionQualificationReused ? null : sha256Hex(payloadBytes),
        headerDigest: trustedQualification && !sessionQualificationReused
          ? sha256Hex(headerBytes)
          : null,
        elapsedMs: performance.now() - started,
      };
    };
    const fillReadAhead = (): void => {
      while (nextRead < metadata.shards.length && pendingReads.size < validationBatchSize) {
        pendingReads.set(nextRead, readDirectShard(nextRead));
        nextRead++;
      }
    };
    fillReadAhead();
    try {
      for (let batchStart = 0; batchStart < metadata.shards.length; batchStart += validationBatchSize) {
        if (shouldCancel()) throw new Error(`Compact WebGPU load was cancelled before direct shard ${batchStart}.`);
        const batchEnd = Math.min(metadata.shards.length, batchStart + validationBatchSize);
        const readWaitStart = performance.now();
        const batchReads = await Promise.all(Array.from(
          { length: batchEnd - batchStart },
          (_, offset) => {
            const shardIndex = batchStart + offset;
            const pending = pendingReads.get(shardIndex);
            if (!pending) throw new Error(`Compact WebGPU direct shard ${shardIndex} has no pending source read.`);
            return pending;
          },
        ));
        sourceReadMs += performance.now() - readWaitStart;
        for (let shardIndex = batchStart; shardIndex < batchEnd; shardIndex++) pendingReads.delete(shardIndex);
        fillReadAhead();
        const prepared = batchReads.map((bytes, offset) => {
          const shardIndex = batchStart + offset;
          const shard = metadata.shards[shardIndex];
          const uploadStart = performance.now();
          // Direct-v3 payloads are already GPU-native u32 words. Large
          // mapped-at-creation buffers make Chromium copy every shard through
          // a JS-visible mapping before unmap; queue.writeBuffer uses Dawn's
          // optimized upload path and preserves queue ordering for validation
          // and all later detector kernels.
          const payload = uploadBytesViaQueue(
            device,
            bytes.payloadBytes,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
          );
          gpuUploadSubmissionMs += performance.now() - uploadStart;
          const headers = uploadBytes(device, bytes.headerBytes, GPUBufferUsage.STORAGE);
          const status = device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
          });
          device.queue.writeBuffer(status, 0, new Uint32Array([0]));
          const checkpointWords = Math.ceil(tileCount / V3_CHECKPOINT_TILES);
          const widthWords = Math.ceil(tileCount / V3_WIDTHS_PER_WORD);
          const parameters = uploadBytes(device, new Uint32Array([
            detectorPixels,
            tileCount,
            shard.payloadBytes / 4,
            checkpointWords + widthWords,
          ]), GPUBufferUsage.UNIFORM);
          const readback = device.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
          return { shardIndex, shard, bytes, payload, headers, status, parameters, readback };
        });
        const batchTransientBytes = prepared.reduce(
          (total, item) => total + item.bytes.payloadBytes.byteLength + item.bytes.headerBytes.byteLength + 8,
          0,
        );
        const nextReadBytes = metadata.shards.slice(batchEnd, batchEnd + validationBatchSize).reduce(
          (total, shard) => total + shard.payloadBytes + shard.widthsBytes,
          0,
        );
        maximumTransientBytes = Math.max(maximumTransientBytes, batchTransientBytes + nextReadBytes);
        let admitted = false;
        try {
          if (!trustedQualification) {
            const encoder = device.createCommandEncoder({ label: `compact v3 shards ${batchStart}-${batchEnd - 1} validation` });
            for (const item of prepared) {
              const pass = encoder.beginComputePass();
              pass.setPipeline(v3HeaderValidationPipeline);
              pass.setBindGroup(0, device.createBindGroup({
                layout: v3HeaderValidationPipeline.getBindGroupLayout(0),
                entries: [
                  { binding: 0, resource: { buffer: item.headers } },
                  { binding: 1, resource: { buffer: excluded } },
                  { binding: 2, resource: { buffer: item.status } },
                  { binding: 3, resource: { buffer: item.parameters } },
                ],
              }));
              const checkpointWords = Math.ceil(tileCount / V3_CHECKPOINT_TILES);
              pass.dispatchWorkgroups(Math.ceil(detectorPixels * checkpointWords / 256));
              pass.end();
              encoder.copyBufferToBuffer(item.status, 0, item.readback, 0, 4);
            }
            const validationStart = performance.now();
            device.queue.submit([encoder.finish()]);
            await Promise.all(prepared.map((item) => item.readback.mapAsync(GPUMapMode.READ)));
            compactHeaderValidationMs += performance.now() - validationStart;
            for (const item of prepared) {
              const status = new Uint32Array(item.readback.getMappedRange())[0];
              item.readback.unmap();
              if (status !== 0) {
                throw new Error(`Compact v3 shard ${item.shardIndex} failed GPU compact-header validation with status 0x${status.toString(16)}.`);
              }
            }
          }
          gpuDecodeWallMs += 0;
          descriptorPreparationMs += 0;
          if (!sessionQualificationReused) {
            const integrityStart = performance.now();
            const payloadDigests = await Promise.all(prepared.map((item) => item.bytes.payloadDigest!));
            const headerDigests = trustedQualification
              ? await Promise.all(prepared.map((item) => item.bytes.headerDigest!))
              : [];
            decodedIntegrityMs += performance.now() - integrityStart;
            for (let index = 0; index < prepared.length; index++) {
              const item = prepared[index];
              if (payloadDigests[index] !== item.shard.decodedSha256) {
                throw new Error(`Compact v3 shard ${item.shardIndex} direct payload SHA-256 is ${payloadDigests[index]}, expected ${item.shard.decodedSha256}.`);
              }
              if (trustedQualification) {
                const expectedHeader = trustedQualification.shards[item.shardIndex].headerSha256;
                if (headerDigests[index] !== expectedHeader) {
                  throw new Error(
                    `Compact v3 shard ${item.shardIndex} direct header SHA-256 is ${headerDigests[index]}, expected ${expectedHeader}.`,
                  );
                }
              }
            }
          }
          for (const item of prepared) residentShards.push({ payload: item.payload, descriptors: item.headers });
          admitted = true;
        } finally {
          for (const item of prepared) {
            item.status.destroy();
            item.parameters.destroy();
            item.readback.destroy();
            if (!admitted) {
              item.payload.destroy();
              item.headers.destroy();
            }
          }
        }
      }
      if (!sessionQualificationReused && trustedQualification) {
        options.sessionQualificationCache?.record(
          source,
          trustedQualification.sealedWholeFileSha256,
        );
      }
      const uploadFenceStart = performance.now();
      await device.queue.onSubmittedWorkDone();
      gpuUploadFenceMs += performance.now() - uploadFenceStart;
      if (metadata.preparedDpcMoments && preparedDpcRead) {
        const prepared = await preparedDpcRead;
        preparedDpcReadMs = prepared.elapsedMs;
        if (prepared.digest !== metadata.preparedDpcMoments.sha256) {
          throw new Error(
            `Compact prepared DPC SHA-256 is ${prepared.digest}, expected ${metadata.preparedDpcMoments.sha256}.`,
          );
        }
        preparedDpcBytes = prepared.bytes.byteLength;
        maximumTransientBytes = Math.max(maximumTransientBytes, preparedDpcBytes);
        const detectorSelection = new Uint32Array(
          detectorPixels - metadata.excludedDetectorPixels.length,
        );
        let selectionCursor = 0;
        for (let pixel = 0; pixel < detectorPixels; pixel++) {
          if (!excludedSet.has(pixel)) detectorSelection[selectionCursor++] = pixel;
        }
        const uploadStart = performance.now();
        preparedDpcResident = {
          detectorPixels: detectorSelection,
          buffer: uploadBytesViaQueue(
            device,
            prepared.bytes,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
          ),
        };
        gpuUploadSubmissionMs += performance.now() - uploadStart;
      }
      if (metadata.preparedDetectorProducts && preparedDetectorProductsRead) {
        const prepared = await preparedDetectorProductsRead;
        preparedDetectorProductReadMs = prepared.elapsedMs;
        preparedDetectorProductBytes = prepared.products.reduce(
          (total, product) => total + product.mask.byteLength + product.values.byteLength,
          0,
        );
        maximumTransientBytes = Math.max(maximumTransientBytes, preparedDetectorProductBytes);
        const uploadStart = performance.now();
        for (const item of prepared.products) {
          preparedDetectorProductsResident.set(item.product.name, {
            mask: item.mask,
            buffer: uploadBytesViaQueue(
              device,
              item.values,
              GPUBufferUsage.COPY_SRC,
            ),
            selectedPixels: item.product.selectedDetectorPixels,
          });
        }
        gpuUploadSubmissionMs += performance.now() - uploadStart;
      }
      const detectorOutputs = [0, 1].map((index) => device.createBuffer({
        label: `compact detector output ${index}`,
        size: metadata.shape[0] * metadata.shape[1] * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      }));
      const diffractionOutput = device.createBuffer({
        label: "compact selected diffraction output",
        size: detectorPixels * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
      const profile: WebGPUCompactH5LoadProfile = {
        schemaVersion: 3,
        adapterInfo: getGPUInfo(),
        softwareAdapter: isSoftwareGPUAdapter(),
        sourceBytes: metadata.sourceBytes,
        residentBytes: metadata.residentBytes,
        logicalDenseAllocationBytes: 0,
        metadataMs,
        pipelineCompileMs,
        sourceReadMs,
        gpuUploadSubmissionMs,
        gpuUploadFenceMs,
        gpuUploadMode: "queue-write-buffer",
        descriptorPreparationMs: 0,
        gpuDecodeWallMs: 0,
        compactHeaderValidationMs,
        compactHeaderValidationMode: trustedQualification
          ? "trusted-sidecar-exact-header-digest"
          : "gpu-structural",
        decodedIntegrityMs,
        encodedIntegrityMs: 0,
        wholeFileIntegrityMs,
        trustedPrefixIntegrityMs,
        residentReadyMs: performance.now() - totalStart,
        maximumTransientBytes,
        preparedDpcBytes,
        preparedDpcReadMs,
        preparedDpcPrimeMs,
        preparedDetectorProductBytes,
        preparedDetectorProductReadMs,
        decodedShardSha256Checks: 0,
        encodedShardSha256Checks: 0,
        directPayloadSha256Checks: sessionQualificationReused ? 0 : metadata.shards.length,
        directHeaderSha256Checks: trustedQualification && !sessionQualificationReused
          ? metadata.shards.length
          : 0,
        wholeFileSha256Checks,
        trustedPrefixSha256Checks,
        sessionQualificationReused: sessionQualificationReused ? 1 : 0,
        sealedWholeFileSha256: trustedQualification
          ? trustedQualification.sealedWholeFileSha256
          : options.expectedWholeFileSha256!,
        integrityMode: sessionQualificationReused
          ? "trusted-sidecar-plus-session-qualified-direct-range"
          : trustedQualification
            ? "trusted-sidecar-plus-direct-range-sha256"
            : "whole-file-plus-direct-payload-sha256",
        fftDispatchCount: 0,
      };
      const resident = new WebGPUCompactH5ResidentSource({
        metadata, loadProfile: profile, device, shards: residentShards, excluded, maximumWidths, implementationRevision,
        collectDetectorTimings: options.collectDetectorTimings,
        detectorOutputs, diffractionOutput, selectedPipeline, detectorPipeline,
        detectorResolveV3Pipeline, detectorResolveV3Layout, detectorResolvedV3Pipeline,
        u32ToF32Pipeline,
        dpcMomentPipeline, dpcMomentLayout, momentsToComPipeline, dpcMeanPipeline, dpcPairPipeline,
        dpcOutputMeanPipeline, dpcOutputUlpCorrectPipeline, detectorLayout,
        preparedDpcMoments: preparedDpcResident,
        preparedDetectorProducts: preparedDetectorProductsResident,
      });
      if (preparedDpcResident) {
        const primeStart = performance.now();
        await resident.primePreparedDpc();
        preparedDpcPrimeMs = performance.now() - primeStart;
        profile.preparedDpcPrimeMs = preparedDpcPrimeMs;
      }
      profile.residentReadyMs = performance.now() - totalStart;
      return resident;
    } catch (error) {
      await Promise.allSettled(pendingReads.values());
      for (const shard of residentShards) {
        shard.payload.destroy();
        shard.descriptors.destroy();
      }
      preparedDpcResident?.buffer.destroy();
      for (const product of preparedDetectorProductsResident.values()) {
        product.buffer.destroy();
      }
      await Promise.allSettled(
        preparedDetectorProductsRead ? [preparedDetectorProductsRead] : [],
      );
      excluded.destroy();
      throw error;
    }
  }
  const readShard = async (shardIndex: number): Promise<CompactShardBytes> => {
    const shard = metadata.shards[shardIndex];
    const readStart = performance.now();
    const rangeStart = Math.min(shard.payloadOffset, shard.lengthsOffset, shard.widthsOffset);
    const rangeEnd = Math.max(
      shard.payloadOffset + shard.payloadBytes,
      shard.lengthsOffset + shard.lengthsBytes,
      shard.widthsOffset + shard.widthsBytes,
    );
    const envelope = await readSourceRange(source, rangeStart, rangeEnd - rangeStart);
    const encodedDigest = useEncodedIntegrity ? sha256Hex(envelope) : null;
    const compressedBytes = envelope.subarray(
      shard.payloadOffset - rangeStart,
      shard.payloadOffset - rangeStart + shard.payloadBytes,
    );
    const lengthBytes = envelope.subarray(
      shard.lengthsOffset - rangeStart,
      shard.lengthsOffset - rangeStart + shard.lengthsBytes,
    );
    const widthBytes = envelope.subarray(
      shard.widthsOffset - rangeStart,
      shard.widthsOffset - rangeStart + shard.widthsBytes,
    );
    return {
      compressedBytes,
      lengthBytes,
      widthBytes,
      encodedDigest,
      elapsedMs: performance.now() - readStart,
    };
  };
  // Decode several shards per queue submission. On mobile WebGPU, waiting for
  // one mapped status buffer per shard serialized 64 queue round trips and was
  // much slower than the actual raw-LZ4 work. Four concurrent source reads and
  // one validation barrier per batch keep the transient budget bounded while
  // allowing storage, command preparation, and the previous GPU batch to
  // overlap.
  const validationBatchSize = Math.min(4, metadata.shards.length);
  const readAhead = validationBatchSize;
  const pendingReads = new Map<number, Promise<CompactShardBytes>>();
  let nextRead = 0;
  const fillReadAhead = (): void => {
    while (nextRead < metadata.shards.length && pendingReads.size < readAhead) {
      pendingReads.set(nextRead, readShard(nextRead));
      nextRead++;
    }
  };
  fillReadAhead();
  try {
    for (let batchStart = 0; batchStart < metadata.shards.length; batchStart += validationBatchSize) {
      if (shouldCancel()) throw new Error(`Compact WebGPU load was cancelled before shard ${batchStart}.`);
      const batchEnd = Math.min(metadata.shards.length, batchStart + validationBatchSize);
      const readWaitStart = performance.now();
      const batchReads = await Promise.all(Array.from(
        { length: batchEnd - batchStart },
        (_, offset) => {
          const shardIndex = batchStart + offset;
          const pendingRead = pendingReads.get(shardIndex);
          if (!pendingRead) throw new Error(`Compact WebGPU shard ${shardIndex} has no pending source read.`);
          return pendingRead;
        },
      ));
      sourceReadMs += performance.now() - readWaitStart;
      for (let shardIndex = batchStart; shardIndex < batchEnd; shardIndex++) pendingReads.delete(shardIndex);
      fillReadAhead();

      const prepared = batchReads.map((currentBytes, batchOffset) => {
        const shardIndex = batchStart + batchOffset;
        const shard = metadata.shards[shardIndex];
        const { compressedBytes, lengthBytes, widthBytes } = currentBytes;
        const descriptorStart = performance.now();
        const descriptors = new Uint32Array(shard.descriptorCount);
        let payloadWord = 0;
        for (let descriptorIndex = 0; descriptorIndex < widthBytes.length; descriptorIndex++) {
          const width = widthBytes[descriptorIndex];
          const pixel = Math.floor(descriptorIndex / tileCount);
          if (width > 16) throw new Error(`Compact shard ${shardIndex} descriptor ${descriptorIndex} exceeds exact uint16 width.`);
          if (metadata.workingDtype === "uint8" && width > 8 && !excludedSet.has(pixel)) {
            throw new Error(`Compact shard ${shardIndex} uses ${width} bits for nonexcluded detector pixel ${pixel}, contradicting its legacy uint8 manifest.`);
          }
          if (payloadWord >= 1 << 27) throw new Error(`Compact shard ${shardIndex} exceeds the 27-bit descriptor offset.`);
          descriptors[descriptorIndex] = (payloadWord << 5) | width;
          payloadWord += width * 4;
          if (width > maximumWidths[pixel]) maximumWidths[pixel] = width;
        }
        if (payloadWord * 4 !== shard.decodedBytes) throw new Error(`Compact shard ${shardIndex} widths do not cover its decoded payload.`);
        const compressedOffsets = new Uint32Array(lengthBytes.length + 1);
        for (let index = 0; index < lengthBytes.length; index++) compressedOffsets[index + 1] = compressedOffsets[index] + lengthBytes[index] + 1;
        if (compressedOffsets[compressedOffsets.length - 1] !== shard.payloadBytes) throw new Error(`Compact shard ${shardIndex} chunk lengths do not cover its compressed payload.`);
        const compressed = uploadBytes(device, compressedBytes, GPUBufferUsage.STORAGE);
        const offsets = uploadBytes(device, compressedOffsets, GPUBufferUsage.STORAGE);
        const descriptorBuffer = uploadBytes(device, descriptors, GPUBufferUsage.STORAGE);
        const decoded = device.createBuffer({
          label: `compact shard ${shardIndex} payload`,
          size: shard.decodedBytes,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        const decodeStatus = device.createBuffer({
          size: shard.chunkCount * 4,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        const descriptorStatus = device.createBuffer({
          size: 4,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(descriptorStatus, 0, new Uint32Array([0]));
        const decodeConfig = uploadBytes(
          device,
          new Uint32Array([shard.decodedBytes, metadata.payloadChunkBytes, shard.chunkCount, shard.payloadBytes]),
          GPUBufferUsage.UNIFORM,
        );
        const descriptorConfig = uploadBytes(
          device,
          new Uint32Array([shard.descriptorCount, payloadWord, 0, 0]),
          GPUBufferUsage.UNIFORM,
        );
        const statusReadback = device.createBuffer({
          size: shard.chunkCount * 4 + 4,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        const decodedReadback = useEncodedIntegrity ? null : device.createBuffer({
          size: shard.decodedBytes,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        descriptorPreparationMs += performance.now() - descriptorStart;
        return {
          shardIndex, shard, compressedBytes, lengthBytes, widthBytes, descriptors,
          compressedOffsets, compressed, offsets, descriptorBuffer, decoded,
          decodeStatus, descriptorStatus, decodeConfig, descriptorConfig,
          statusReadback, decodedReadback, encodedDigest: currentBytes.encodedDigest,
        };
      });
      const batchTransientBytes = prepared.reduce(
        (bytes, item) => bytes + item.compressedBytes.byteLength + item.lengthBytes.byteLength
          + item.widthBytes.byteLength + item.shard.decodedBytes * (useEncodedIntegrity ? 1 : 2)
          + item.descriptors.byteLength + item.compressedOffsets.byteLength
          + item.shard.chunkCount * 4,
        0,
      );
      const nextReadBytes = metadata.shards.slice(batchEnd, batchEnd + readAhead).reduce(
        (bytes, next) => bytes + next.payloadBytes + next.lengthsBytes + next.widthsBytes,
        0,
      );
      maximumTransientBytes = Math.max(maximumTransientBytes, batchTransientBytes + nextReadBytes);
      let admitted = false;
      try {
        const encoder = device.createCommandEncoder({ label: `compact shards ${batchStart}-${batchEnd - 1} decode` });
        for (const item of prepared) {
          const decodePass = encoder.beginComputePass();
          decodePass.setPipeline(decodePipeline);
          decodePass.setBindGroup(0, device.createBindGroup({
            layout: decodePipeline.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: item.compressed } },
              { binding: 1, resource: { buffer: item.offsets } },
              { binding: 2, resource: { buffer: item.decoded } },
              { binding: 3, resource: { buffer: item.decodeStatus } },
              { binding: 4, resource: { buffer: item.decodeConfig } },
            ],
          }));
          decodePass.dispatchWorkgroups(Math.ceil(item.shard.chunkCount / 64));
          decodePass.end();
          const descriptorPass = encoder.beginComputePass();
          descriptorPass.setPipeline(descriptorPipeline);
          descriptorPass.setBindGroup(0, device.createBindGroup({
            layout: descriptorPipeline.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: item.descriptorBuffer } },
              { binding: 1, resource: { buffer: item.descriptorStatus } },
              { binding: 2, resource: { buffer: item.descriptorConfig } },
            ],
          }));
          descriptorPass.dispatchWorkgroups(Math.ceil(item.shard.descriptorCount / 256));
          descriptorPass.end();
          encoder.copyBufferToBuffer(item.decodeStatus, 0, item.statusReadback, 0, item.shard.chunkCount * 4);
          encoder.copyBufferToBuffer(item.descriptorStatus, 0, item.statusReadback, item.shard.chunkCount * 4, 4);
        }
        const decodeStart = performance.now();
        device.queue.submit([encoder.finish()]);
        await Promise.all(prepared.map((item) => item.statusReadback.mapAsync(GPUMapMode.READ)));
        gpuDecodeWallMs += performance.now() - decodeStart;
        for (const item of prepared) {
          const statusValues = new Uint32Array(item.statusReadback.getMappedRange());
          for (let chunk = 0; chunk < item.shard.chunkCount; chunk++) {
            const status = statusValues[chunk];
            if (status === 0) continue;
            if ((status >>> 28) === 6) {
              const matchOffset = (status >>> 8) & 0xffff;
              const outputBytes = status & 0xff;
              throw new Error(
                `Compact shard ${item.shardIndex} raw-LZ4 chunk ${chunk} failed with decoder status 6 `
                + `(match offset ${matchOffset}, output ${outputBytes}).`,
              );
            }
            throw new Error(`Compact shard ${item.shardIndex} raw-LZ4 chunk ${chunk} failed with decoder status ${status}.`);
          }
          const descriptorError = statusValues[item.shard.chunkCount];
          if (descriptorError !== 0) throw new Error(`Compact shard ${item.shardIndex} failed GPU descriptor validation with status ${descriptorError}.`);
          item.statusReadback.unmap();
        }

        if (useEncodedIntegrity) {
          const integrityStart = performance.now();
          const digests = await Promise.all(prepared.map((item) => item.encodedDigest!));
          encodedIntegrityMs += performance.now() - integrityStart;
          for (let index = 0; index < prepared.length; index++) {
            const item = prepared[index];
            if (digests[index] !== item.shard.encodedEnvelopeSha256) {
              throw new Error(`Compact shard ${item.shardIndex} encoded-envelope SHA-256 is ${digests[index]}, expected ${item.shard.encodedEnvelopeSha256}.`);
            }
          }
        } else {
          const integrityStart = performance.now();
          const integrityEncoder = device.createCommandEncoder({ label: `compact shards ${batchStart}-${batchEnd - 1} integrity` });
          for (const item of prepared) {
            integrityEncoder.copyBufferToBuffer(item.decoded, 0, item.decodedReadback!, 0, item.shard.decodedBytes);
          }
          device.queue.submit([integrityEncoder.finish()]);
          await Promise.all(prepared.map((item) => item.decodedReadback!.mapAsync(GPUMapMode.READ)));
          const digests = await Promise.all(prepared.map((item) => sha256Hex(item.decodedReadback!.getMappedRange())));
          decodedIntegrityMs += performance.now() - integrityStart;
          for (let index = 0; index < prepared.length; index++) {
            const item = prepared[index];
            item.decodedReadback!.unmap();
            if (digests[index] !== item.shard.decodedSha256) {
              throw new Error(`Compact shard ${item.shardIndex} decoded SHA-256 is ${digests[index]}, expected ${item.shard.decodedSha256}.`);
            }
          }
        }
        for (const item of prepared) residentShards.push({ payload: item.decoded, descriptors: item.descriptorBuffer });
        admitted = true;
      } finally {
        for (const item of prepared) {
          item.compressed.destroy();
          item.offsets.destroy();
          item.decodeStatus.destroy();
          item.descriptorStatus.destroy();
          item.decodeConfig.destroy();
          item.descriptorConfig.destroy();
          item.statusReadback.destroy();
          item.decodedReadback?.destroy();
          if (!admitted) {
            item.decoded.destroy();
            item.descriptorBuffer.destroy();
          }
        }
      }
    }

    const excludedValues = new Uint32Array(detectorPixels);
    for (const pixel of metadata.excludedDetectorPixels) excludedValues[pixel] = 1;
    const excluded = uploadBytes(device, excludedValues, GPUBufferUsage.STORAGE);
    const scanCount = metadata.shape[0] * metadata.shape[1];
    const detectorOutputs = [0, 1].map((index) => device.createBuffer({
      label: `compact detector output ${index}`,
      size: scanCount * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    }));
    const diffractionOutput = device.createBuffer({
      label: "compact selected diffraction output",
      size: detectorPixels * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const profile: WebGPUCompactH5LoadProfile = {
      schemaVersion: 1,
      adapterInfo: getGPUInfo(),
      softwareAdapter: isSoftwareGPUAdapter(),
      sourceBytes: metadata.sourceBytes,
      residentBytes: metadata.residentBytes,
      logicalDenseAllocationBytes: 0,
      metadataMs,
      pipelineCompileMs,
      sourceReadMs,
      gpuUploadSubmissionMs,
      gpuUploadFenceMs,
      gpuUploadMode: "mapped-at-creation",
      descriptorPreparationMs,
      gpuDecodeWallMs,
      compactHeaderValidationMs: 0,
      compactHeaderValidationMode: "not-applicable",
      decodedIntegrityMs,
      encodedIntegrityMs,
      wholeFileIntegrityMs,
      trustedPrefixIntegrityMs: 0,
      residentReadyMs: performance.now() - totalStart,
      maximumTransientBytes,
      preparedDpcBytes,
      preparedDpcReadMs,
      preparedDpcPrimeMs,
      preparedDetectorProductBytes,
      preparedDetectorProductReadMs,
      decodedShardSha256Checks: useEncodedIntegrity ? 0 : metadata.shards.length,
      encodedShardSha256Checks: useEncodedIntegrity ? metadata.shards.length : 0,
      directPayloadSha256Checks: 0,
      directHeaderSha256Checks: 0,
      wholeFileSha256Checks,
      trustedPrefixSha256Checks: 0,
      sessionQualificationReused: 0,
      sealedWholeFileSha256: null,
      integrityMode: useEncodedIntegrity ? "authenticated-encoded-envelope" : "decoded-sha256",
      fftDispatchCount: 0,
    };
    return new WebGPUCompactH5ResidentSource({
      metadata,
      loadProfile: profile,
      implementationRevision,
      collectDetectorTimings: options.collectDetectorTimings,
      device,
      shards: residentShards,
      excluded,
      maximumWidths,
      detectorOutputs,
      diffractionOutput,
      selectedPipeline,
      detectorPipeline,
      detectorResolveV3Pipeline,
      detectorResolveV3Layout,
      detectorResolvedV3Pipeline,
      u32ToF32Pipeline,
      dpcMomentPipeline,
      dpcMomentLayout,
      momentsToComPipeline,
      dpcMeanPipeline,
      dpcPairPipeline,
      dpcOutputMeanPipeline,
      dpcOutputUlpCorrectPipeline,
      detectorLayout,
    });
  } catch (error) {
    await Promise.allSettled(pendingReads.values());
    for (const shard of residentShards) {
      shard.payload.destroy();
      shard.descriptors.destroy();
    }
    throw error;
  }
}

function compactReceipt(metadata: CompactH5Index, implementationRevision: string | null): CompactH5ResidentReceipt | null {
  const manifest = metadata.manifest;
  // Historical v1 uint8 files declare low-bit truncation. Preserve their viewing
  // path, but do not turn them into an exact raw contract by attaching metadata.
  if (!metadata.rawReconstructionAvailable || !isSha256(metadata.sourceRawLogicalSha256)) return null;
  const maskDigest = manifest.detector_mask_sha256;
  if (metadata.excludedDetectorPixels.length && !isSha256(maskDigest)) return null;
  const calibration = metadata.detectorCalibration;
  const logicalCount = metadata.shape.reduce((count, length) => count * length, 1);
  if (!Number.isSafeInteger(logicalCount * 2)) throw new Error("Source logical bytes exceed the exact integer range.");
  const shape = Object.freeze([...metadata.shape]) as unknown as CompactH5Index["shape"];
  return Object.freeze({
    schema: "quantem.gpu.4dstem-resident-receipt/v3",
    representation: "packed",
    sourceIdentitySHA256: metadata.sourceIdentitySha256,
    sourceShape: shape, workingShape: shape, sourceDtype: "uint16",
    workingDtype: metadata.workingDtype,
    sourceLogicalTensorBytes: logicalCount * 2,
    workingLogicalTensorBytes: logicalCount * (metadata.workingDtype === "uint8" ? 1 : 2),
    physicalResidentBytes: metadata.residentBytes,
    containerBytes: metadata.sourceBytes,
    storageSchema: String(manifest.schema),
    losslessExact: true, scanBin: 1, detectorBin: 1, crop: null,
    detectorMaskCount: metadata.excludedDetectorPixels.length,
    detectorMaskSHA256: isSha256(maskDigest) ? maskDigest : null,
    detectorMaskSchema: isSha256(maskDigest) ? "quantem.gpu.detector-mask-identity/opaque-v1" : null,
    calibrationSchema: calibration?.schema ?? null,
    calibrationSHA256: calibration ? metadataSha256(manifest.detector_calibration) : null,
    provenanceSchema: "quantem.gpu.packed-detector-h5-manifest/v1",
    provenanceSHA256: metadataSha256(manifest),
    sourceRawLogicalSHA256: metadata.sourceRawLogicalSha256,
    workingLogicalSHA256: metadata.schemaVersion === 1
      ? (isSha256(manifest.working_logical_sha256) ? manifest.working_logical_sha256 : null)
      : (isSha256(manifest.prepared_uint8_sha256) ? manifest.prepared_uint8_sha256 : null),
    implementationRevision,
  });
}

function validateManifest(
  manifest: Record<string, unknown>,
  sourceBytes: number,
  protectedPrefixBytes: number,
  shape: readonly number[],
  shardCount: number,
  scansPerShard: number,
  sourceIdentity: string,
  excluded: Uint32Array,
  schemaVersion: 1 | 3,
  scanTile: 128 | 32,
  headerEncoding: 0 | 1,
  shards: readonly CompactH5ShardIndex[],
): {
  detectorCalibration: CompactH5DetectorCalibration | null;
  preparedDpcMoments: CompactH5PreparedDpcMoments | null;
  preparedDetectorProducts: CompactH5PreparedDetectorProducts | null;
  maskedDetectorPixelsSha256: string | null;
  maskedDetectorRawValues: Uint16Array | null;
} {
  const expected: Record<string, unknown> = schemaVersion === 1 ? {
      schema: "quantem.gpu.packed-detector-h5/v1",
      status: "complete",
      source_dtype: "uint16",
      scan_bin: 1,
      detector_bin: 1,
      crop: null,
      shard_count: shardCount,
      scans_per_shard: scansPerShard,
      payload_chunk_bytes: 128,
      payload_chunk_codec: "independent raw LZ4 blocks",
      payload_chunk_length_codec: "uint8 encoded_bytes_minus_one",
      descriptor_codec: "uint8 five-bit widths",
      source_identity_sha256: sourceIdentity,
    } : {
      schema: "quantem.gpu.packed-detector-h5/v3",
      status: "complete",
      payload_codec: "direct-bitpacked-u32",
      source_dtype: "uint16",
      working_dtype: "uint8",
      working_value_definition: "all admitted source counts exactly; authenticated dead pixels set to zero",
      scan_bin: 1,
      detector_bin: 1,
      crop: null,
      shard_count: shardCount,
      scan_tile: scanTile,
      source_identity_sha256: sourceIdentity,
    };
  for (const [key, value] of Object.entries(expected)) {
    if (manifest[key] !== value) throw new Error(`Compact JSON contract field ${key} disagrees with the binary index.`);
  }
  if (!Array.isArray(manifest.source_shape) || manifest.source_shape.length !== 4
      || manifest.source_shape.some((value, index) => value !== shape[index])) {
    throw new Error("Compact JSON source_shape disagrees with the binary index.");
  }
  if (schemaVersion === 1 && manifest.working_dtype !== "uint8" && manifest.working_dtype !== "uint16") {
    throw new Error(`Compact working dtype must be uint8 or uint16; got ${String(manifest.working_dtype)}.`);
  }
  if (!Array.isArray(manifest.masked_detector_pixels)) throw new Error("Compact JSON detector mask is not a list.");
  const manifestPixels: number[] = [];
  if (schemaVersion === 1) {
    for (const coordinate of manifest.masked_detector_pixels) {
      if (!Array.isArray(coordinate) || coordinate.length !== 2
          || !Number.isInteger(coordinate[0]) || !Number.isInteger(coordinate[1])
          || coordinate[0] < 0 || coordinate[0] >= shape[2]
          || coordinate[1] < 0 || coordinate[1] >= shape[3]) {
        throw new Error(`Compact JSON detector coordinate ${JSON.stringify(coordinate)} is invalid.`);
      }
      manifestPixels.push(coordinate[0] * shape[3] + coordinate[1]);
    }
  } else {
    for (const pixel of manifest.masked_detector_pixels) {
      if (!Number.isInteger(pixel) || (pixel as number) < 0 || (pixel as number) >= shape[2] * shape[3]) {
        throw new Error("Compact v3 detector mask is invalid.");
      }
      manifestPixels.push(pixel as number);
    }
    for (let index = 1; index < manifestPixels.length; index++) {
      if (manifestPixels[index] < manifestPixels[index - 1]) {
        throw new Error("Compact v3 detector mask must use ordered row-major pixel indices.");
      }
    }
  }
  if (manifestPixels.length !== excluded.length
      || manifestPixels.some((pixel, index) => pixel !== excluded[index])) {
    throw new Error("Compact JSON and binary detector masks disagree.");
  }
  let maskedDetectorPixelsSha256: string | null = null;
  let maskedDetectorRawValues: Uint16Array | null = null;
  if (schemaVersion === 3) {
    if (headerEncoding !== 1) throw new Error("Compact v3 header encoding is unsupported.");
    for (const field of ["source_raw_logical_sha256", "prepared_uint8_sha256", "detector_mask_sha256"] as const) {
      if (!isSha256(manifest[field])) throw new Error(`Compact v3 ${field} is invalid.`);
    }
    const pixelDigest = manifest.masked_detector_pixels_sha256;
    if (pixelDigest !== undefined && pixelDigest !== null) {
      if (!isSha256(pixelDigest)) throw new Error("Compact v3 masked_detector_pixels_sha256 is invalid.");
      const orderedBytes = new Uint8Array(excluded.length * 4);
      const orderedView = new DataView(orderedBytes.buffer);
      for (let index = 0; index < excluded.length; index++) orderedView.setUint32(index * 4, excluded[index], true);
      const digest = new StreamingSha256();
      digest.update(orderedBytes);
      if (pixelDigest !== digest.digestHex()) {
        throw new Error("Compact v3 masked_detector_pixels_sha256 does not match the ordered binary-index detector pixels.");
      }
      maskedDetectorPixelsSha256 = pixelDigest;
    }
    const rawValues = manifest.masked_detector_raw_values;
    if (rawValues !== undefined && rawValues !== null) {
      if (!Array.isArray(rawValues) || rawValues.length !== excluded.length
          || rawValues.some((value) => !Number.isInteger(value) || (value as number) < 0 || (value as number) > 65535)) {
        throw new Error("Compact v3 masked_detector_raw_values must be exact uint16 values aligned one-to-one with the ordered detector mask.");
      }
      maskedDetectorRawValues = Uint16Array.from(rawValues as number[]);
    } else if (excluded.length === 0) {
      maskedDetectorRawValues = new Uint16Array(0);
    }
  }
  const detectorCalibration = parseDetectorCalibration(
    manifest.detector_calibration,
    shape,
    sourceIdentity,
  );
  const preparedDpcMoments = parsePreparedDpcMoments(
      manifest.prepared_dpc_moments,
      sourceBytes,
      shape,
      sourceIdentity,
      excluded,
      schemaVersion,
      shards,
      manifest.prepared_uint8_sha256,
      manifest.detector_mask_sha256,
    );
  return {
    detectorCalibration,
    preparedDpcMoments,
    preparedDetectorProducts: parsePreparedDetectorProducts(
      manifest.prepared_detector_products,
      sourceBytes,
      protectedPrefixBytes,
      shape,
      sourceIdentity,
      schemaVersion,
      shards,
      preparedDpcMoments,
      detectorCalibration,
      manifest.detector_calibration,
      manifest.prepared_uint8_sha256,
      manifest.detector_mask_sha256,
    ),
    maskedDetectorPixelsSha256,
    maskedDetectorRawValues,
  };
}

function parsePreparedDpcMoments(
  value: unknown,
  sourceBytes: number,
  shape: readonly number[],
  sourceIdentity: string,
  excluded: Uint32Array,
  schemaVersion: 1 | 3,
  shards: readonly CompactH5ShardIndex[],
  workingUint8Sha256: unknown,
  detectorMaskSha256: unknown,
): CompactH5PreparedDpcMoments | null {
  if (value === undefined || value === null) return null;
  if (schemaVersion !== 3) {
    throw new Error("Prepared DPC moments require compact QGIX v3.");
  }
  if (typeof value !== "object" || Array.isArray(value)) {
    throw new Error("Compact prepared DPC moments are not an object.");
  }
  const prepared = value as Record<string, unknown>;
  const scanCount = shape[0] * shape[1];
  const detectorPixels = shape[2] * shape[3];
  const excludedSet = new Set(excluded);
  const selectedDetectorPixels = detectorPixels - excluded.length;
  let rowMomentBound = 0;
  let columnMomentBound = 0;
  for (let pixel = 0; pixel < detectorPixels; pixel++) {
    if (excludedSet.has(pixel)) continue;
    rowMomentBound += Math.floor(pixel / shape[3]) * 255;
    columnMomentBound += (pixel % shape[3]) * 255;
  }
  const totalBound = selectedDetectorPixels * 255;
  const expectedLayout = [
    "total_lo", "total_hi", "row_lo", "row_hi",
    "column_lo", "column_hi", "padding_0", "padding_1",
  ];
  const expected: Record<string, unknown> = {
    schema: "quantem.gpu.prepared-dpc-moments/v1",
    source_identity_sha256: sourceIdentity,
    working_uint8_sha256: workingUint8Sha256,
    detector_mask_sha256: detectorMaskSha256,
    detector_selection: "all-nonexcluded-v1",
    scan_count: scanCount,
    selected_detector_pixels: selectedDetectorPixels,
    detector_columns: shape[3],
    dtype: "little-endian-u32",
    word_order: "little-endian-u32-pairs",
    words_per_scan: 8,
    total_bound: String(totalBound),
    row_moment_bound: String(rowMomentBound),
    column_moment_bound: String(columnMomentBound),
    narrow_integer: totalBound <= 0xffffffff,
    narrow_products: Math.max(rowMomentBound, columnMomentBound) <= 0xffffffff,
  };
  for (const [key, expectedValue] of Object.entries(expected)) {
    if (prepared[key] !== expectedValue) {
      throw new Error(`Compact prepared DPC field ${key} disagrees with the source.`);
    }
  }
  if (!Array.isArray(prepared.layout)
      || prepared.layout.length !== expectedLayout.length
      || prepared.layout.some((entry, index) => entry !== expectedLayout[index])) {
    throw new Error("Compact prepared DPC word layout is unsupported.");
  }
  const fileOffset = prepared.file_offset;
  const fileBytes = prepared.file_bytes;
  const expectedBytes = scanCount * 8 * 4;
  if (!Number.isSafeInteger(fileOffset) || !Number.isSafeInteger(fileBytes)
      || (fileOffset as number) < 0 || (fileOffset as number) % 4 !== 0
      || fileBytes !== expectedBytes
      || (fileOffset as number) > sourceBytes - (fileBytes as number)) {
    throw new Error("Compact prepared DPC byte range is invalid.");
  }
  const rangeEnd = (fileOffset as number) + (fileBytes as number);
  for (let shardIndex = 0; shardIndex < shards.length; shardIndex++) {
    const shard = shards[shardIndex];
    for (const [label, offset, bytes] of [
      ["payload", shard.payloadOffset, shard.payloadBytes],
      ["chunk lengths", shard.lengthsOffset, shard.lengthsBytes],
      ["descriptor widths", shard.widthsOffset, shard.widthsBytes],
    ] as const) {
      if (bytes > 0 && (fileOffset as number) < offset + bytes && offset < rangeEnd) {
        throw new Error(`Compact prepared DPC range overlaps shard ${shardIndex} ${label}.`);
      }
    }
  }
  if (!isSha256(prepared.sha256)) {
    throw new Error("Compact prepared DPC SHA-256 is invalid.");
  }
  return {
    schema: "quantem.gpu.prepared-dpc-moments/v1",
    fileOffset: fileOffset as number,
    fileBytes: fileBytes as number,
    sha256: prepared.sha256,
    workingUint8Sha256: workingUint8Sha256 as string,
    detectorMaskSha256: detectorMaskSha256 as string,
    scanCount,
    selectedDetectorPixels,
    detectorColumns: shape[3],
    totalBound: String(totalBound),
    rowMomentBound: String(rowMomentBound),
    columnMomentBound: String(columnMomentBound),
    narrowInteger: totalBound <= 0xffffffff,
    narrowProducts: Math.max(rowMomentBound, columnMomentBound) <= 0xffffffff,
  };
}

function parsePreparedDetectorProducts(
  value: unknown,
  sourceBytes: number,
  protectedPrefixBytes: number,
  shape: readonly number[],
  sourceIdentity: string,
  schemaVersion: 1 | 3,
  shards: readonly CompactH5ShardIndex[],
  preparedDpc: CompactH5PreparedDpcMoments | null,
  calibration: CompactH5DetectorCalibration | null,
  calibrationManifest: unknown,
  workingUint8Sha256: unknown,
  detectorMaskSha256: unknown,
): CompactH5PreparedDetectorProducts | null {
  if (value === undefined || value === null) return null;
  if (schemaVersion !== 3) {
    throw new Error("Prepared detector products require compact QGIX v3.");
  }
  if (typeof value !== "object" || Array.isArray(value)) {
    throw new Error("Compact prepared detector products are not an object.");
  }
  if (!calibration || typeof calibrationManifest !== "object" || Array.isArray(calibrationManifest)) {
    throw new Error("Compact prepared detector products require source-bound calibration.");
  }
  const prepared = value as Record<string, unknown>;
  const calibrationDigest = new StreamingSha256();
  calibrationDigest.update(new TextEncoder().encode(canonicalJson(calibrationManifest, true)));
  const expectedRoot: Record<string, unknown> = {
    schema: "quantem.gpu.prepared-detector-products/v1",
    source_identity_sha256: sourceIdentity,
    working_uint8_sha256: workingUint8Sha256,
    detector_mask_sha256: detectorMaskSha256,
    detector_calibration_sha256: calibrationDigest.digestHex(),
    detector_calibration_digest_encoding: "canonical-json-numbers-as-f64be-hex/v1",
    product_dtype: "little-endian-u32",
    mask_dtype: "uint8-binary-row-major",
    mask_rule: "quantem.gpu.detector-mask-inclusive/v1",
  };
  for (const [key, expected] of Object.entries(expectedRoot)) {
    if (prepared[key] !== expected) {
      throw new Error(`Compact prepared detector-products field ${key} disagrees with the source.`);
    }
  }
  for (const [key, expected] of [
    ["scan_shape", shape.slice(0, 2)],
    ["detector_shape", shape.slice(2, 4)],
    ["product_order", ["bf", "abf", "adf"]],
  ] as const) {
    const observed = prepared[key];
    if (!Array.isArray(observed) || observed.length !== expected.length
        || observed.some((entry, index) => entry !== expected[index])) {
      throw new Error(`Compact prepared detector-products field ${key} is invalid.`);
    }
  }
  const rawProducts = prepared.products;
  if (!Array.isArray(rawProducts) || rawProducts.length !== 3) {
    throw new Error("Compact prepared detector product list is incomplete.");
  }
  const names = ["bf", "abf", "adf"] as const;
  const bfRadius = calibration.brightFieldRadiusPx;
  const geometries = [
    [0, bfRadius],
    [0.5 * bfRadius, bfRadius],
    [bfRadius, 2 * bfRadius],
  ] as const;
  const detectorPixels = shape[2] * shape[3];
  const scanCount = shape[0] * shape[1];
  const ranges: Array<{ start: number; end: number; label: string }> = [];
  ranges.push({ start: 0, end: protectedPrefixBytes, label: "compact container prefix" });
  for (let shardIndex = 0; shardIndex < shards.length; shardIndex++) {
    const shard = shards[shardIndex];
    for (const [label, offset, bytes] of [
      ["payload", shard.payloadOffset, shard.payloadBytes],
      ["chunk lengths", shard.lengthsOffset, shard.lengthsBytes],
      ["descriptor widths", shard.widthsOffset, shard.widthsBytes],
    ] as const) {
      if (bytes > 0) ranges.push({ start: offset, end: offset + bytes, label: `shard ${shardIndex} ${label}` });
    }
  }
  if (preparedDpc) {
    ranges.push({
      start: preparedDpc.fileOffset,
      end: preparedDpc.fileOffset + preparedDpc.fileBytes,
      label: "prepared DPC moments",
    });
  }
  const products: CompactH5PreparedDetectorProduct[] = [];
  for (let ordinal = 0; ordinal < names.length; ordinal++) {
    const raw = rawProducts[ordinal];
    const name = names[ordinal];
    if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
      throw new Error(`Compact prepared detector product ${ordinal} is invalid.`);
    }
    const product = raw as Record<string, unknown>;
    const center = product.center_px;
    if (product.name !== name
        || !Array.isArray(center) || center.length !== 2
        || center[0] !== calibration.detectorCenterPx[0]
        || center[1] !== calibration.detectorCenterPx[1]
        || product.inner_radius_px !== geometries[ordinal][0]
        || product.outer_radius_px !== geometries[ordinal][1]) {
      throw new Error(`Compact prepared ${name.toUpperCase()} geometry disagrees with the calibration.`);
    }
    const selected = product.selected_detector_pixels;
    const maskOffset = product.mask_file_offset;
    const maskBytes = product.mask_file_bytes;
    const valuesOffset = product.values_file_offset;
    const valuesBytes = product.values_file_bytes;
    if (!Number.isSafeInteger(selected) || (selected as number) < 0 || (selected as number) > detectorPixels
        || !Number.isSafeInteger(maskOffset) || (maskOffset as number) < 0
        || maskBytes !== detectorPixels || (maskOffset as number) > sourceBytes - detectorPixels
        || !Number.isSafeInteger(valuesOffset) || (valuesOffset as number) < 0
        || valuesBytes !== scanCount * 4 || (valuesOffset as number) > sourceBytes - scanCount * 4) {
      throw new Error(`Compact prepared ${name.toUpperCase()} byte ranges are invalid.`);
    }
    if (!isSha256(product.mask_sha256) || !isSha256(product.values_sha256)) {
      throw new Error(`Compact prepared ${name.toUpperCase()} SHA-256 is invalid.`);
    }
    ranges.push(
      { start: maskOffset as number, end: (maskOffset as number) + detectorPixels, label: `prepared ${name} mask` },
      { start: valuesOffset as number, end: (valuesOffset as number) + scanCount * 4, label: `prepared ${name} values` },
    );
    products.push({
      name,
      centerPx: [center[0] as number, center[1] as number],
      innerRadiusPx: geometries[ordinal][0],
      outerRadiusPx: geometries[ordinal][1],
      selectedDetectorPixels: selected as number,
      maskFileOffset: maskOffset as number,
      maskFileBytes: detectorPixels,
      maskSha256: product.mask_sha256,
      valuesFileOffset: valuesOffset as number,
      valuesFileBytes: scanCount * 4,
      valuesSha256: product.values_sha256,
    });
  }
  ranges.sort((a, b) => a.start - b.start);
  for (let index = 1; index < ranges.length; index++) {
    if (ranges[index].start < ranges[index - 1].end) {
      throw new Error(`Compact ranges overlap between ${ranges[index - 1].label} and ${ranges[index].label}.`);
    }
  }
  return {
    schema: "quantem.gpu.prepared-detector-products/v1",
    workingUint8Sha256: workingUint8Sha256 as string,
    detectorMaskSha256: detectorMaskSha256 as string,
    products,
  };
}

function canonicalJson(value: unknown, numbersAsF64beHex = false): string {
  if (numbersAsF64beHex && typeof value === "number") {
    if (!Number.isFinite(value)) throw new Error("Detector calibration numbers must be finite.");
    const bytes = new Uint8Array(8);
    new DataView(bytes.buffer).setFloat64(0, value, false);
    const hex = Array.from(bytes, (entry) => entry.toString(16).padStart(2, "0")).join("");
    return JSON.stringify(`f64be:${hex}`);
  }
  if (value === null || typeof value !== "object") {
    const encoded = JSON.stringify(value);
    return encoded === undefined ? "null" : encoded;
  }
  if (Array.isArray(value)) {
    return `[${value.map((entry) => canonicalJson(entry, numbersAsF64beHex)).join(",")}]`;
  }
  const object = value as Record<string, unknown>;
  return `{${Object.keys(object).sort().map(
    (key) => `${JSON.stringify(key)}:${canonicalJson(object[key], numbersAsF64beHex)}`,
  ).join(",")}}`;
}

function isSha256(value: unknown): value is string {
  return typeof value === "string" && /^[0-9a-f]{64}$/.test(value);
}

function validateTrustedQualification(
  metadata: CompactH5Index,
  value: CompactH5TrustedQualificationV1,
): CompactH5TrustedQualificationV1 {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("QGIX v3 trusted qualification sidecar is not an object.");
  }
  if (value.schema !== "quantem.gpu.compact-h5-trusted-qualification/v1"
      || value.qualificationOrigin !== "writer-close-reopen-whole-file-sha256"
      || value.headerValidation !== "canonical-cpu-all-shards") {
    throw new Error("QGIX v3 trusted qualification sidecar schema or origin is unsupported.");
  }
  if (!isSha256(value.sealedWholeFileSha256) || !isSha256(value.prefixSha256)) {
    throw new Error("QGIX v3 trusted qualification sidecar SHA-256 fields are invalid.");
  }
  if (value.sourceBytes !== metadata.sourceBytes
      || value.sourceIdentitySha256 !== metadata.sourceIdentitySha256) {
    throw new Error("QGIX v3 trusted qualification sidecar belongs to a different source.");
  }
  const expectedPrefixBytes = Math.min(...metadata.shards.flatMap(
    (shard) => [shard.payloadOffset, shard.widthsOffset].filter((offset) => offset > 0),
  ));
  if (!Number.isSafeInteger(value.prefixBytes) || value.prefixBytes !== expectedPrefixBytes) {
    throw new Error(
      `QGIX v3 trusted qualification prefix is ${value.prefixBytes}, expected ${expectedPrefixBytes}.`,
    );
  }
  if (!Array.isArray(value.shards) || value.shards.length !== metadata.shards.length) {
    throw new Error("QGIX v3 trusted qualification shard list is incomplete.");
  }
  for (let ordinal = 0; ordinal < metadata.shards.length; ordinal++) {
    const expected = metadata.shards[ordinal];
    const record = value.shards[ordinal];
    if (!record || typeof record !== "object" || Array.isArray(record)
        || record.ordinal !== ordinal
        || record.payloadOffset !== expected.payloadOffset
        || record.payloadBytes !== expected.payloadBytes
        || record.payloadSha256 !== expected.decodedSha256
        || record.headerOffset !== expected.widthsOffset
        || record.headerBytes !== expected.widthsBytes
        || !isSha256(record.headerSha256)) {
      throw new Error(`QGIX v3 trusted qualification shard ${ordinal} disagrees with its binary index.`);
    }
  }
  return value;
}

function parseDetectorCalibration(
  value: unknown,
  shape: readonly number[],
  sourceIdentity: string,
): CompactH5DetectorCalibration | null {
  if (value === undefined || value === null) return null;
  if (typeof value !== "object" || Array.isArray(value)) {
    throw new Error("Compact detector calibration is not an object.");
  }
  const calibration = value as Record<string, unknown>;
  if (calibration.schema !== "quantem.gpu.detector-calibration/v1") {
    throw new Error("Compact detector calibration schema is unsupported.");
  }
  if (calibration.source_identity_sha256 !== sourceIdentity) {
    throw new Error("Compact detector calibration belongs to a different source.");
  }
  const center = calibration.detector_center_px;
  if (!Array.isArray(center) || center.length !== 2
      || !center.every((coordinate) => typeof coordinate === "number" && Number.isFinite(coordinate))
      || center[0] < 0 || center[0] >= shape[2]
      || center[1] < 0 || center[1] >= shape[3]) {
    throw new Error("Compact detector calibration center must be finite [row, column].");
  }
  const radius = calibration.bright_field_radius_px;
  if (typeof radius !== "number" || !Number.isFinite(radius)
      || radius <= 0 || radius > Math.hypot(shape[2], shape[3])) {
    throw new Error("Compact detector calibration bright-field radius is invalid.");
  }
  const rotation = calibration.dpc_rotation_degrees;
  const exchanged = calibration.dpc_component_order_exchanged;
  if ((rotation === undefined || rotation === null) !== (exchanged === undefined || exchanged === null)) {
    throw new Error("Compact DPC calibration requires both rotation and component order.");
  }
  if (rotation !== undefined && rotation !== null
      && (typeof rotation !== "number" || !Number.isFinite(rotation) || typeof exchanged !== "boolean")) {
    throw new Error("Compact DPC calibration is invalid.");
  }
  const method = calibration.method;
  if (typeof method !== "string" || method.trim().length === 0) {
    throw new Error("Compact detector calibration method is missing.");
  }
  return {
    schema: "quantem.gpu.detector-calibration/v1",
    sourceIdentitySha256: sourceIdentity,
    detectorCenterPx: [center[0], center[1]],
    brightFieldRadiusPx: radius,
    dpcRotationDegrees: typeof rotation === "number" ? rotation : null,
    dpcComponentOrderExchanged: typeof exchanged === "boolean" ? exchanged : null,
    method,
  };
}

function matchesMagic(bytes: Uint8Array, offset: number, magic: readonly number[]): boolean {
  return magic.every((value, index) => bytes[offset + index] === value);
}

function safeU64(view: DataView, offset: number, label: string): number {
  const value = view.getBigUint64(offset, true);
  if (value > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error(`${label} exceeds exact JavaScript integer range.`);
  return Number(value);
}

function requireIndex(value: number, size: number, label: string): void {
  if (!Number.isInteger(value) || value < 0 || value >= size) throw new Error(`${label} ${value} is outside 0 through ${size - 1}.`);
}

function countSelectedPixels(mask: Uint32Array, excludedPixels: Uint32Array): number {
  const excluded = new Set(excludedPixels);
  let count = 0;
  for (let pixel = 0; pixel < mask.length; pixel++) {
    const value = mask[pixel];
    if (value !== 0 && value !== 1) {
      throw new Error("A compact virtual-detector mask must contain only zero or one.");
    }
    if (value !== 0 && !excluded.has(pixel)) count++;
  }
  return count;
}

function compactDetectorSelection(
  mask: Uint32Array,
  excludedPixels: Uint32Array,
): { indices: Uint32Array } {
  const excluded = new Set(excludedPixels);
  const selected: number[] = [];
  for (let pixel = 0; pixel < mask.length; pixel++) {
    const value = mask[pixel];
    if (value !== 0 && value !== 1) {
      throw new Error("A compact detector mask must contain only zero or one.");
    }
    if (value !== 0 && !excluded.has(pixel)) selected.push(pixel);
  }
  return { indices: Uint32Array.from(selected) };
}

function equalU32(left: Uint32Array, right: Uint32Array): boolean {
  if (left.length !== right.length) return false;
  for (let index = 0; index < left.length; index++) {
    if (left[index] !== right[index]) return false;
  }
  return true;
}

function isPowerOfTwo(value: number): boolean {
  return Number.isInteger(value) && value > 0 && (value & (value - 1)) === 0;
}

function encodeComputePass(
  encoder: GPUCommandEncoder,
  pipeline: GPUComputePipeline,
  bindGroup: GPUBindGroup,
  workgroupsX: number,
  workgroupsY = 1,
): void {
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(workgroupsX, workgroupsY);
  pass.end();
}

function retireBuffers(device: GPUDevice, buffers: GPUBuffer[]): void {
  void device.queue.onSubmittedWorkDone()
    .catch(() => {})
    .finally(() => {
      for (const buffer of buffers) buffer.destroy();
    });
}

async function readSourceRange(source: CompactH5Source, offset: number, byteCount: number): Promise<Uint8Array> {
  const result = "readRange" in source
    ? await source.readRange(offset, byteCount)
    : new Uint8Array(await source.slice(offset, offset + byteCount).arrayBuffer());
  if (result.byteLength !== byteCount) throw new Error(`Compact HDF5 range at ${offset} ended before ${byteCount} bytes.`);
  return result;
}

async function readHttpRange(url: string, sourceBytes: number, offset: number, byteCount: number): Promise<Uint8Array> {
  if (!Number.isSafeInteger(offset) || offset < 0 || !Number.isSafeInteger(byteCount) || byteCount <= 0
      || offset > sourceBytes || byteCount > sourceBytes - offset) {
    throw new Error(`Compact HDF5 HTTP range ${offset}+${byteCount} is outside its ${sourceBytes}-byte source.`);
  }
  const response = await fetch(url, { headers: { Range: `bytes=${offset}-${offset + byteCount - 1}` } });
  if (response.status !== 206) {
    await response.body?.cancel();
    throw new Error(`Compact HDF5 HTTP source returned status ${response.status}; a byte-range server is required.`);
  }
  const total = parseContentRange(response.headers.get("Content-Range"), offset, byteCount);
  if (total !== sourceBytes) throw new Error(`Compact HDF5 HTTP source size changed from ${sourceBytes} to ${total} bytes.`);
  const result = new Uint8Array(await response.arrayBuffer());
  if (result.byteLength !== byteCount) throw new Error(`Compact HDF5 HTTP range at ${offset} ended before ${byteCount} bytes.`);
  return result;
}

function parseContentRange(value: string | null, offset: number, byteCount: number): number {
  const match = /^bytes (\d+)-(\d+)\/(\d+)$/.exec(value || "");
  const expectedEnd = offset + byteCount - 1;
  if (!match || Number(match[1]) !== offset || Number(match[2]) !== expectedEnd) {
    throw new Error(`Compact HDF5 HTTP source returned invalid Content-Range ${JSON.stringify(value)}.`);
  }
  const total = Number(match[3]);
  if (!Number.isSafeInteger(total) || total <= expectedEnd) {
    throw new Error(`Compact HDF5 HTTP source returned invalid total size ${match[3]}.`);
  }
  return total;
}

function compactSourceName(url: string): string {
  try {
    const path = new URL(url, globalThis.location?.href).pathname;
    return decodeURIComponent(path.split("/").filter(Boolean).pop() || "compact-source.h5");
  } catch {
    return url.split(/[/?#]/).filter(Boolean).pop() || "compact-source.h5";
  }
}

function uploadBytes(device: GPUDevice, values: ArrayBufferView, usage: GPUBufferUsageFlags): GPUBuffer {
  const size = Math.max(4, Math.ceil(values.byteLength / 4) * 4);
  const buffer = device.createBuffer({ size, usage, mappedAtCreation: true });
  new Uint8Array(buffer.getMappedRange()).set(
    new Uint8Array(values.buffer, values.byteOffset, values.byteLength),
  );
  buffer.unmap();
  return buffer;
}

function uploadBytesViaQueue(
  device: GPUDevice,
  values: ArrayBufferView,
  usage: GPUBufferUsageFlags,
): GPUBuffer {
  const size = Math.max(4, Math.ceil(values.byteLength / 4) * 4);
  const buffer = device.createBuffer({
    size,
    usage: usage | GPUBufferUsage.COPY_DST,
  });
  const upload = new Uint8Array(values.byteLength);
  upload.set(new Uint8Array(values.buffer, values.byteOffset, values.byteLength));
  device.queue.writeBuffer(
    buffer,
    0,
    upload,
  );
  return buffer;
}

async function readU32Buffer(device: GPUDevice, source: GPUBuffer, count: number): Promise<Uint32Array> {
  const readback = device.createBuffer({ size: count * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, 0, readback, 0, count * 4);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const result = new Uint32Array(readback.getMappedRange().slice(0));
  readback.unmap();
  readback.destroy();
  return result;
}

async function readF32Buffer(device: GPUDevice, source: GPUBuffer, count: number): Promise<Float32Array> {
  const readback = device.createBuffer({ size: count * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, 0, readback, 0, count * 4);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const result = new Float32Array(readback.getMappedRange().slice(0));
  readback.unmap();
  readback.destroy();
  return result;
}

async function sha256Source(source: CompactH5Source): Promise<string> {
  return sha256SourceRange(source, 0, source.size);
}

async function sha256SourceRange(
  source: CompactH5Source,
  offset: number,
  byteCount: number,
): Promise<string> {
  const digest = new StreamingSha256();
  const chunkBytes = 16 * 1024 * 1024;
  const end = offset + byteCount;
  for (let cursor = offset; cursor < end; cursor += chunkBytes) {
    digest.update(await readSourceRange(source, cursor, Math.min(chunkBytes, end - cursor)));
  }
  return digest.digestHex();
}

async function sha256Hex(bytes: ArrayBuffer | Uint8Array): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", bytes as BufferSource);
  return bytesToHex(new Uint8Array(digest));
}

function bytesToHex(bytes: Uint8Array): string {
  return Array.from(bytes, (value) => value.toString(16).padStart(2, "0")).join("");
}

let crcTable: Uint32Array | null = null;
function crc32(bytes: Uint8Array): number {
  if (!crcTable) {
    crcTable = new Uint32Array(256);
    for (let value = 0; value < 256; value++) {
      let entry = value;
      for (let bit = 0; bit < 8; bit++) entry = (entry & 1) ? (0xedb88320 ^ (entry >>> 1)) : (entry >>> 1);
      crcTable[value] = entry >>> 0;
    }
  }
  let crc = 0xffffffff;
  for (const value of bytes) crc = crcTable[(crc ^ value) & 0xff] ^ (crc >>> 8);
  return (crc ^ 0xffffffff) >>> 0;
}
