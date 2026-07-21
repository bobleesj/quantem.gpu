/// <reference types="@webgpu/types" />
// Local browser-file acquisition for native HDF5 Show4DSTEM sources.
//
// The URL path (`fetch(...).arrayBuffer()`) is convenient for served exports but
// becomes the dominant full-512 load cost in the browser. When a user grants the
// local folder, this module reads File objects through a small worker pool, parses
// the native HDF5 chunk index on the main thread, then feeds the same bslz4 WGSL
// decoder as the URL path. The microscope evidence and decompressor are unchanged.

import {
  decodeBslz4Batch,
  decodeBslz4MaskedSumLow8Batch,
  maskedSumBlockIds,
  selectedBlockIdsCover,
  sliceMaskedSumSpecsByScanRegion,
  type Bslz4Spec,
  type Bslz4BatchProfile,
  type Bslz4MaskedSumProfile,
  type Bslz4MaskedSumSpec,
} from "./bslz4";
import { getGPUInfo, isSoftwareGPUAdapter } from "./device";
import {
  readBslz4SelectedBlockMetadata,
  readBslz4SelectedBlockVolume,
  readH5VolumeFromBlockIndex,
  readH5MasterInfo,
  readH5Volume,
  readH5VolumeFromFrameIndex,
  type H5VolumeFrameIndex,
} from "./h5reader";

type SourceDtype = "uint8" | "uint16" | "uint32" | "float32";

export interface LocalH5GpuChunk {
  buffer: GPUBuffer;
  startScan: number;
  nScan: number;
}

export interface LocalH5LoadProfile {
  acquisitionMode: "local-file";
  totalMs: number;
  masterReadMs: number;
  readWaitMs: number;
  readWorkerMs: number;
  parseMs: number;
  packMs: number;
  decompressMs: number;
  uploadMs: number;
  uploadCopyWaitMs?: number;
  detBinMs?: number;
  decodeBuildMs: number;
  gpuWaitMs: number;
  decodeComputeWaitMs?: number;
  decodeProfileMs: number;
  fileGB: number;
  blockIndexGB?: number;
  compressedGB: number;
  decodeCompressedMB: number;
  sourceDtype: SourceDtype | "unknown";
  decodeDtype: "uint8" | "uint16" | "float32";
  detBin: number;
  sourceDetRows: number;
  sourceDetCols: number;
  outputDetRows: number;
  outputDetCols: number;
  chunks: number;
  frames: number;
  sourceFrames: number;
  scanRows: number;
  scanCols: number;
  outputRows: number;
  outputCols: number;
  badPixels: number;
  dataFilesExpected: number | null;
  workerCount: number;
  groupSize: number;
  decodeBatch: number;
  decodeVariant: string;
  decodeProfileSplit?: boolean;
  frameIndexFiles: number;
  blockIndexFiles: number;
  parseMode: "h5-btree" | "frame-index" | "block-index" | "mixed" | "selected-block-sidecar";
  adapterInfo: string;
  softwareAdapter: boolean;
  timestampQuery: boolean;
  subgroups: boolean;
  maxBufferGB: number | null;
  maxStorageBufferGB: number | null;
}

export interface LocalH5LoadResult {
  device: GPUDevice;
  chunks: LocalH5GpuChunk[];
  scanCount: number;
  detSize: number;
  detRows: number;
  detCols: number;
  sourceDetRows: number;
  sourceDetCols: number;
  detBin: number;
  mode: number;
  badPixels: Uint32Array;
  profile: LocalH5LoadProfile;
}

export interface LocalH5LoadOptions {
  scanRows: number;
  scanCols: number;
  scanRegion?: readonly [number, number, number, number] | null;
  embeddedBadPixelsJson?: string;
  decodeBatch?: number;
  groupSize?: number;
  workerCount?: number;
  detBin?: number;
}

export interface LocalH5MaskedSumOptions extends LocalH5LoadOptions {
  mask: Uint32Array;
  scanRegion?: readonly [number, number, number, number] | null;
  productBatch?: number;
}

export interface LocalH5MaskedSumProfile {
  acquisitionMode: "local-file-product-first" | "local-file-selected-block-sidecar";
  sourceMode: "native-h5" | "selected-block-sidecar";
  totalMs: number;
  masterReadMs: number;
  readWaitMs: number;
  readWorkerMs: number;
  readWallMs: number;
  parseMs: number;
  productMs: number;
  fileGB: number;
  compressedGB: number;
  sourceDtype: SourceDtype | "unknown";
  framesRead: number;
  framesComputed: number;
  scanRows: number;
  scanCols: number;
  outputRows: number;
  outputCols: number;
  badPixels: number;
  dataFilesExpected: number | null;
  workerCount: number;
  productBatch: number;
  sidecarFiles: number;
  sidecarCoverage: "used" | "unavailable" | "missing_blocks";
  frameIndexFiles: number;
  blockIndexFiles: number;
  parseMode: "h5-btree" | "frame-index" | "block-index" | "mixed" | "selected-block-sidecar";
  productProfile: Bslz4MaskedSumProfile;
  adapterInfo: string;
  softwareAdapter: boolean;
  timestampQuery: boolean;
  subgroups: boolean;
  maxBufferGB: number | null;
  maxStorageBufferGB: number | null;
}

export interface LocalH5MaskedSumResult {
  device: GPUDevice;
  buffer: GPUBuffer;
  scanRows: number;
  scanCols: number;
  profile: LocalH5MaskedSumProfile;
}

interface ReadJob {
  id: number;
  name: string;
  file: File;
  frameIndex?: H5VolumeFrameIndex;
  blockIndexFile?: File;
}

interface ReadResult {
  id: number;
  name: string;
  buffer?: ArrayBuffer;
  volume?: ParsedWorkerVolume;
  fileBytes: number;
  blockIndexBytes?: number;
  readMs: number;
  parseMs?: number;
  error?: string;
}

interface ParsedSpec extends Bslz4Spec {
  startScan: number;
  nScan: number;
}

interface ParsedWorkerVolume {
  detRows: number;
  detCols: number;
  detSize: number;
  blockElems: number;
  nBlocksPerFrame: number;
  srcDtype: SourceDtype;
  nFrames: number;
  chunks: Bslz4Spec[];
  chunkScanCounts: number[];
}

interface LocalDataFileItem {
  file: File;
  startScan?: number;
  nFrames?: number;
}

type Show4DSTEMFileSystemFileHandle = {
  kind?: "file";
  name: string;
  getFile: () => Promise<File>;
};

type Show4DSTEMFileSystemDirectoryHandle = {
  kind?: "directory";
  name: string;
  values?: () => AsyncIterable<Show4DSTEMFileSystemHandle>;
  entries?: () => AsyncIterable<[string, Show4DSTEMFileSystemHandle]>;
};

type Show4DSTEMFileSystemHandle = Show4DSTEMFileSystemFileHandle | Show4DSTEMFileSystemDirectoryHandle;

const localFilesByPath = new Map<string, File>();
const localFilesByName = new Map<string, File>();

function setLocalH5Debug(stage: string, extra: Record<string, unknown> = {}): void {
  if (typeof globalThis === "undefined") return;
  (globalThis as { __QT_LOCAL_H5_DEBUG?: unknown }).__QT_LOCAL_H5_DEBUG = {
    stage,
    ...extra,
    nowMs: Math.round(performance.now()),
  };
}

function normalisePath(path: string): string {
  const raw = path.split(/[?#]/, 1)[0] || path;
  try {
    return decodeURIComponent(raw).replace(/^[.][/]/, "").replace(/^[/]+/, "");
  } catch {
    return raw.replace(/^[.][/]/, "").replace(/^[/]+/, "");
  }
}

function basename(path: string): string {
  const clean = normalisePath(path);
  return clean.split("/").filter(Boolean).pop() || clean;
}

function stripPickedRoot(path: string): string {
  const parts = normalisePath(path).split("/").filter(Boolean);
  return parts.length > 1 ? parts.slice(1).join("/") : parts.join("/");
}

export function setShow4DSTEMLocalFiles(files: ArrayLike<File>): void {
  localFilesByPath.clear();
  localFilesByName.clear();
  for (let i = 0; i < files.length; i++) {
    const file = files[i] as File & { webkitRelativePath?: string };
    const rel = String(file.webkitRelativePath || "");
    if (rel) {
      localFilesByPath.set(normalisePath(rel), file);
      localFilesByPath.set(stripPickedRoot(rel), file);
    }
    localFilesByName.set(file.name, file);
  }
}

export function show4DSTEMHasLocalFiles(): boolean {
  return localFilesByPath.size > 0 || localFilesByName.size > 0;
}

export function clearShow4DSTEMLocalFiles(): void {
  localFilesByPath.clear();
  localFilesByName.clear();
}

export async function collectShow4DSTEMLocalH5Files(
  directory: Show4DSTEMFileSystemDirectoryHandle,
  maxFiles = 10000,
): Promise<File[]> {
  const files: File[] = [];
  const walk = async (dir: Show4DSTEMFileSystemDirectoryHandle): Promise<void> => {
    if (files.length >= maxFiles) return;
    if (typeof dir.values === "function") {
      for await (const handle of dir.values()) {
        if (files.length >= maxFiles) return;
        if (handle.kind === "directory" || "values" in handle || "entries" in handle) {
          await walk(handle as Show4DSTEMFileSystemDirectoryHandle);
        } else if (handle.kind === "file" || "getFile" in handle) {
          const file = await (handle as Show4DSTEMFileSystemFileHandle).getFile();
          if (/\.(h5|qh5idx|qbslz4)$/i.test(file.name)) files.push(file);
        }
      }
      return;
    }
    if (typeof dir.entries === "function") {
      for await (const [, handle] of dir.entries()) {
        if (files.length >= maxFiles) return;
        if (handle.kind === "directory" || "values" in handle || "entries" in handle) {
          await walk(handle as Show4DSTEMFileSystemDirectoryHandle);
        } else if (handle.kind === "file" || "getFile" in handle) {
          const file = await (handle as Show4DSTEMFileSystemFileHandle).getFile();
          if (/\.(h5|qh5idx|qbslz4)$/i.test(file.name)) files.push(file);
        }
      }
    }
  };
  await walk(directory);
  return files;
}

function localFileFor(path: string): File | null {
  const clean = normalisePath(path);
  return localFilesByPath.get(clean) || localFilesByName.get(basename(clean)) || null;
}

function frameIndexFor(path: string): H5VolumeFrameIndex | undefined {
  const manifest = typeof globalThis !== "undefined"
    ? (globalThis as { __QT_H5_LOCAL_FRAME_INDEX?: Record<string, H5VolumeFrameIndex> }).__QT_H5_LOCAL_FRAME_INDEX
    : undefined;
  if (!manifest) return undefined;
  const clean = normalisePath(path);
  return manifest[clean] || manifest[basename(clean)];
}

function localBlockIndexFor(path: string): File | undefined {
  const cleanBase = basename(path);
  const stem = cleanBase.replace(/\.h5$/i, "");
  return localFilesByName.get(`${stem}.qh5idx`)
    || localFilesByName.get(`${cleanBase}.qh5idx`)
    || localFilesByPath.get(`${normalisePath(path)}.qh5idx`)
    || undefined;
}

function masterDataPath(masterUrl: string, index: number): string {
  const clean = normalisePath(masterUrl);
  const base = clean.replace(/_master\.h5$/i, "");
  return `${base}_data_${String(index).padStart(6, "0")}.h5`;
}

function localDataFilesForMaster(masterUrl: string): File[] {
  const out: File[] = [];
  for (let i = 1; i < 10000; i++) {
    const file = localFileFor(masterDataPath(masterUrl, i));
    if (!file) break;
    out.push(file);
  }
  return out;
}

function localSidecarCandidatesForDataPath(dataPath: string): File[] {
  const stem = basename(dataPath).replace(/\.h5$/i, "");
  const seen = new Set<File>();
  const out: File[] = [];
  for (const file of localFilesByName.values()) {
    if (!/\.qbslz4$/i.test(file.name)) continue;
    if (!file.name.startsWith(stem)) continue;
    if (seen.has(file)) continue;
    seen.add(file);
    out.push(file);
  }
  return out.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
}

function localSidecarCandidatesForMaster(masterUrl: string): File[][] {
  const out: File[][] = [];
  for (let i = 1; i < 10000; i++) {
    const candidates = localSidecarCandidatesForDataPath(masterDataPath(masterUrl, i));
    if (candidates.length === 0) break;
    out.push(candidates);
  }
  return out;
}

interface SelectedBlockSidecarSpan {
  startScan: number;
  nFrames: number;
}

interface SelectedBlockSidecarSelection {
  files: File[];
  spans: SelectedBlockSidecarSpan[];
  coverage: "used" | "unavailable" | "missing_blocks";
}

async function chooseSelectedBlockSidecars(
  masterUrl: string,
  mask: Uint32Array,
  badPixels: Uint32Array,
): Promise<SelectedBlockSidecarSelection> {
  const candidateRows = localSidecarCandidatesForMaster(masterUrl);
  if (candidateRows.length === 0) return { files: [], spans: [], coverage: "unavailable" };
  const selected: File[] = [];
  const spans: SelectedBlockSidecarSpan[] = [];
  let sawMissing = false;
  let inferredStartScan = 0;
  for (const candidates of candidateRows) {
    let chosen: File | null = null;
    let chosenSpan: SelectedBlockSidecarSpan | null = null;
    for (const file of candidates) {
      try {
        const header = await file.slice(0, 64 * 1024).arrayBuffer();
        const meta = readBslz4SelectedBlockMetadata(header, file.name);
        const requested = maskedSumBlockIds(mask, badPixels, meta.blockElems);
        if (selectedBlockIdsCover(meta.selectedBlockIds, requested)) {
          chosen = file;
          chosenSpan = {
            startScan: Math.max(0, Math.round(Number(meta.startScan ?? inferredStartScan))),
            nFrames: Math.max(0, Math.round(Number(meta.nFrames || 0))),
          };
          break;
        }
        sawMissing = true;
      } catch {
        sawMissing = true;
      }
    }
    if (!chosen) {
      return { files: [], spans: [], coverage: sawMissing ? "missing_blocks" : "unavailable" };
    }
    selected.push(chosen);
    spans.push(chosenSpan || { startScan: inferredStartScan, nFrames: 0 });
    inferredStartScan = (chosenSpan?.startScan ?? inferredStartScan) + (chosenSpan?.nFrames ?? 0);
  }
  return { files: selected, spans, coverage: "used" };
}

function scanSpanIntersectsRegion(
  startScan: number,
  nFrames: number,
  scanCols: number,
  region: readonly [number, number, number, number],
): boolean {
  const stopScan = startScan + nFrames;
  const r0 = Math.max(0, Math.round(region[0]));
  const r1 = Math.max(r0, Math.round(region[1]));
  const c0 = Math.max(0, Math.round(region[2]));
  const c1 = Math.max(c0, Math.round(region[3]));
  for (let row = r0; row < r1; row++) {
    const rowStart = row * scanCols + c0;
    const rowStop = row * scanCols + c1;
    if (Math.max(startScan, rowStart) < Math.min(stopScan, rowStop)) return true;
  }
  return false;
}

function filterSelectedBlockSidecarsForScanRegion(
  selection: SelectedBlockSidecarSelection,
  scanCols: number,
  region: readonly [number, number, number, number] | null,
): SelectedBlockSidecarSelection {
  if (!region || selection.files.length === 0) return selection;
  const files: File[] = [];
  const spans: SelectedBlockSidecarSpan[] = [];
  for (let i = 0; i < selection.files.length; i++) {
    const span = selection.spans[i];
    if (scanSpanIntersectsRegion(span.startScan, span.nFrames, scanCols, region)) {
      files.push(selection.files[i]);
      spans.push(span);
    }
  }
  return { files, spans, coverage: selection.coverage };
}

const READ_WORKER_SOURCE = `
function readBE32(b, off) {
  return ((b[off] << 24) | (b[off + 1] << 16) | (b[off + 2] << 8) | b[off + 3]) >>> 0;
}

function parseVolumeFromFrameIndex(buffer, index) {
  const detRows = Math.round(Number(index.detRows));
  const detCols = Math.round(Number(index.detCols));
  const nFrames = Math.round(Number(index.nFrames));
  const srcDtype = index.srcDtype || "uint16";
  const detSize = detRows * detCols;
  const offsets = index.frameOffsets || [];
  if (offsets.length < nFrames) {
    throw new Error("HDF5 frame index has " + offsets.length + " offsets, expected " + nFrames + ".");
  }
  const srcBytes = srcDtype === "uint8" ? 1 : (srcDtype === "uint32" || srcDtype === "float32") ? 4 : 2;
  const fileBytes = new Uint8Array(buffer);
  const blockBytes = readBE32(fileBytes, offsets[0] + 8);
  const blockElems = blockBytes / srcBytes;
  const nBlocksPerFrame = Math.ceil(detSize / blockElems);
  const frameStep = Math.max(1, Math.floor((1024 * 1024 * 1024) / detSize));
  const chunks = [];
  const chunkScanCounts = [];
  const transfers = [buffer];
  for (let start = 0; start < nFrames; start += frameStep) {
    const stop = Math.min(nFrames, start + frameStep);
    const framesThisChunk = stop - start;
    let rangeStart = Number.POSITIVE_INFINITY;
    let rangeEnd = 0;
    const meta = new Uint32Array(framesThisChunk * nBlocksPerFrame * 2);
    let m = 0;
    for (let f = start; f < stop; f++) {
      const addr = offsets[f];
      rangeStart = Math.min(rangeStart, addr);
      let pos = 12;
      for (let b = 0; b < nBlocksPerFrame; b++) {
        const clen = readBE32(fileBytes, addr + pos);
        meta[m++] = addr + pos + 4;
        meta[m++] = clen;
        pos += 4 + clen;
      }
      rangeEnd = Math.max(rangeEnd, addr + pos);
    }
    for (let i = 0; i < meta.length; i += 2) {
      meta[i] -= rangeStart;
    }
    transfers.push(meta.buffer);
    chunks.push({
      compressed: fileBytes.subarray(rangeStart, rangeEnd),
      blockMeta: meta,
      nFrames: framesThisChunk,
      nBlocksPerFrame,
      blockElems,
      detSize,
    });
    chunkScanCounts.push(framesThisChunk);
  }
  return {
    volume: { detRows, detCols, detSize, blockElems, nBlocksPerFrame, srcDtype, nFrames, chunks, chunkScanCounts },
    transfers,
  };
}

function parseVolumeFromBlockIndex(buffer, indexBuffer, name) {
  const bytes = new Uint8Array(indexBuffer);
  const magic = new TextDecoder().decode(bytes.subarray(0, 8));
  if (magic !== "QH5IDX01") {
    throw new Error("HDF5 block-index sidecar for " + name + " is not a QH5IDX01 file.");
  }
  const dv = new DataView(indexBuffer);
  const jsonLen = dv.getUint32(8, true);
  const blockMetaWords = dv.getUint32(12, true);
  const jsonStart = 16;
  const jsonStop = jsonStart + jsonLen;
  const metaStart = Math.ceil(jsonStop / 4) * 4;
  if (jsonStop > indexBuffer.byteLength || metaStart + blockMetaWords * 4 > indexBuffer.byteLength) {
    throw new Error("HDF5 block-index sidecar for " + name + " is truncated.");
  }
  const meta = JSON.parse(new TextDecoder().decode(bytes.subarray(jsonStart, jsonStop)));
  const detRows = Math.round(Number(meta.detRows));
  const detCols = Math.round(Number(meta.detCols));
  const detSize = detRows * detCols;
  const nFrames = Math.round(Number(meta.nFrames));
  const blockElems = Math.round(Number(meta.blockElems));
  const nBlocksPerFrame = Math.round(Number(meta.nBlocksPerFrame));
  const srcDtype = meta.srcDtype || "uint16";
  const fileBytes = new Uint8Array(buffer);
  const allBlockMeta = new Uint32Array(indexBuffer, metaStart, blockMetaWords);
  const chunks = [];
  const chunkScanCounts = [];
  const rawChunks = meta.chunks || [];
  for (let i = 0; i < rawChunks.length; i++) {
    const chunk = rawChunks[i];
    const rangeStart = Math.max(0, Math.round(Number(chunk.rangeStart || 0)));
    const rangeEnd = Math.max(rangeStart, Math.round(Number(chunk.rangeEnd || 0)));
    const metaOffsetWords = Math.max(0, Math.round(Number(chunk.metaOffsetWords || 0)));
    const metaWords = Math.max(0, Math.round(Number(chunk.metaWords || 0)));
    const framesThisChunk = Math.max(0, Math.round(Number(chunk.nFrames || 0)));
    chunks.push({
      compressed: fileBytes.subarray(rangeStart, rangeEnd),
      blockMeta: allBlockMeta.subarray(metaOffsetWords, metaOffsetWords + metaWords),
      nFrames: framesThisChunk,
      nBlocksPerFrame,
      blockElems,
      detSize,
    });
    chunkScanCounts.push(framesThisChunk);
  }
  return {
    volume: { detRows, detCols, detSize, blockElems, nBlocksPerFrame, srcDtype, nFrames, chunks, chunkScanCounts },
    transfers: [buffer, indexBuffer],
  };
}

self.onmessage = async (event) => {
  const { id, name, file, frameIndex, blockIndexFile } = event.data;
  const t0 = performance.now();
  try {
    if (blockIndexFile) {
      const [buffer, indexBuffer] = await Promise.all([file.arrayBuffer(), blockIndexFile.arrayBuffer()]);
      const readMs = performance.now() - t0;
      const pt = performance.now();
      const parsed = parseVolumeFromBlockIndex(buffer, indexBuffer, name);
      self.postMessage({ id, name, volume: parsed.volume, fileBytes: file.size, blockIndexBytes: blockIndexFile.size, readMs, parseMs: performance.now() - pt }, parsed.transfers);
      return;
    }
    const buffer = await file.arrayBuffer();
    const readMs = performance.now() - t0;
    if (frameIndex) {
      const pt = performance.now();
      const parsed = parseVolumeFromFrameIndex(buffer, frameIndex);
      self.postMessage({ id, name, volume: parsed.volume, fileBytes: file.size, readMs, parseMs: performance.now() - pt }, parsed.transfers);
      return;
    }
    self.postMessage({ id, name, buffer, fileBytes: file.size, readMs }, [buffer]);
  } catch (err) {
    self.postMessage({ id, name, fileBytes: file ? file.size || 0 : 0, readMs: performance.now() - t0, error: err instanceof Error ? err.message : String(err) });
  }
};`;

let workerUrl: string | null = null;
let workers: Worker[] | null = null;

function workerPool(count: number): Worker[] {
  if (!workerUrl) workerUrl = URL.createObjectURL(new Blob([READ_WORKER_SOURCE], { type: "text/javascript" }));
  if (!workers || workers.length !== count) {
    workers?.forEach((worker) => worker.terminate());
    workers = Array.from({ length: count }, () => new Worker(workerUrl!));
  }
  return workers;
}

function readFiles(
  files: File[],
  workerCount: number,
  frameIndexLookup?: (name: string) => H5VolumeFrameIndex | undefined,
  blockIndexLookup?: (name: string) => File | undefined,
): Promise<ReadResult>[] {
  if (workerCount <= 0) {
    return files.map(async (file, id) => {
      const t0 = performance.now();
      const blockIndexFile = blockIndexLookup?.(file.name);
      if (blockIndexFile) {
        const [buffer, indexBuffer] = await Promise.all([file.arrayBuffer(), blockIndexFile.arrayBuffer()]);
        const readMs = performance.now() - t0;
        const pt = performance.now();
        return {
          id,
          name: file.name,
          volume: readH5VolumeFromBlockIndex(buffer, indexBuffer, file.name),
          fileBytes: file.size,
          blockIndexBytes: blockIndexFile.size,
          readMs,
          parseMs: performance.now() - pt,
        };
      }
      const buffer = await file.arrayBuffer();
      return { id, name: file.name, buffer, fileBytes: file.size, readMs: performance.now() - t0 };
    });
  }
  const results: Array<Promise<ReadResult>> = [];
  const resolves: Array<(value: ReadResult) => void> = [];
  const rejects: Array<(error: Error) => void> = [];
  for (let i = 0; i < files.length; i++) {
    results.push(new Promise((resolve, reject) => {
      resolves[i] = resolve;
      rejects[i] = reject;
    }));
  }
  let next = 0;
  const pump = (worker: Worker): void => {
    if (next >= files.length) return;
    const id = next++;
    const blockIndexFile = blockIndexLookup?.(files[id].name);
    const job: ReadJob = {
      id,
      name: files[id].name,
      file: files[id],
      blockIndexFile,
      frameIndex: blockIndexFile ? undefined : frameIndexLookup?.(files[id].name),
    };
    worker.onmessage = (event: MessageEvent<ReadResult>) => {
      const data = event.data;
      if (data.error) rejects[data.id](new Error(data.error));
      else resolves[data.id](data);
      pump(worker);
    };
    worker.onerror = (event: ErrorEvent) => {
      event.preventDefault();
      rejects[id](new Error(event.message || "local HDF5 worker read failed"));
      pump(worker);
    };
    worker.postMessage(job);
  };
  workerPool(workerCount).forEach(pump);
  return results;
}

function dataFileItemsForScanRegion(
  files: File[],
  scanRows: number,
  scanCols: number,
  scanRegion: readonly [number, number, number, number] | null,
): LocalDataFileItem[] {
  if (!scanRegion) return files.map((file) => ({ file }));
  const { region } = normaliseScanRegion(scanRows, scanCols, scanRegion);
  if (!region) return files.map((file) => ({ file }));
  const [r0, r1, c0, c1] = region;
  if (r0 === 0 && r1 === scanRows && c0 === 0 && c1 === scanCols) {
    return files.map((file) => ({ file }));
  }
  const items: LocalDataFileItem[] = [];
  let cursor = 0;
  for (const file of files) {
    const index = frameIndexFor(file.name);
    if (!index) return files.map((fallback) => ({ file: fallback }));
    const nFrames = Math.max(0, Math.round(index.nFrames));
    items.push({ file, startScan: cursor, nFrames });
    cursor += nFrames;
  }
  const firstSource = r0 * scanCols + c0;
  const lastSourceExclusive = (r1 - 1) * scanCols + c1;
  return items.filter((item) => {
    const start = item.startScan ?? 0;
    const stop = start + (item.nFrames ?? 0);
    return start < lastSourceExclusive && stop > firstSource;
  });
}

function parseEmbeddedBadPixels(raw: string | undefined): Uint32Array | null {
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw) as number[];
    return new Uint32Array(parsed);
  } catch {
    return null;
  }
}

function safeInt(value: number | undefined, fallback: number, min: number, max: number): number {
  const raw = Number(value);
  return Math.max(min, Math.min(max, Number.isFinite(raw) ? Math.round(raw) : fallback));
}

function chooseMaskedSumProductBatch(
  requested: number | undefined,
  outputScanCount: number,
  productSpecCount: number,
): number {
  if (requested !== undefined) return safeInt(requested, 1, 1, 16);
  // Cropped scan regions can become many small row-window specs. Batching a few
  // of those specs cuts WebGPU submit overhead without the high upload pressure
  // that hurts full 512/1024 sidecar loads.
  if (outputScanCount <= 256 * 256 && productSpecCount > 64) return 4;
  return 1;
}

function normaliseScanRegion(
  scanRows: number,
  scanCols: number,
  scanRegion: readonly [number, number, number, number] | null | undefined,
): { rows: number; cols: number; region: readonly [number, number, number, number] | null } {
  const rows = Math.max(1, Math.round(scanRows));
  const cols = Math.max(1, Math.round(scanCols));
  if (!scanRegion) return { rows, cols, region: null };
  const r0 = Math.max(0, Math.min(rows, Math.round(scanRegion[0])));
  const r1 = Math.max(0, Math.min(rows, Math.round(scanRegion[1])));
  const c0 = Math.max(0, Math.min(cols, Math.round(scanRegion[2])));
  const c1 = Math.max(0, Math.min(cols, Math.round(scanRegion[3])));
  if (r1 <= r0 || c1 <= c0) {
    throw new Error(`Invalid scan_region (${scanRegion.join(", ")}); expected (row_start, row_stop, col_start, col_stop) with non-empty bounds.`);
  }
  return { rows: r1 - r0, cols: c1 - c0, region: [r0, r1, c0, c1] };
}

function compactBslz4FrameWindow(spec: Bslz4Spec, frameStart: number, frameCount: number): Bslz4Spec {
  const firstFrame = Math.max(0, Math.min(spec.nFrames, Math.round(frameStart)));
  const nFrames = Math.max(0, Math.min(spec.nFrames - firstFrame, Math.round(frameCount)));
  if (firstFrame === 0 && nFrames === spec.nFrames) return spec;
  if (nFrames === 0) {
    return { ...spec, compressed: new Uint8Array(0), blockMeta: new Uint32Array(2), nFrames: 0 };
  }
  const firstBlock = firstFrame * spec.nBlocksPerFrame;
  const lastBlock = firstBlock + nFrames * spec.nBlocksPerFrame - 1;
  const rangeStart = spec.blockMeta[firstBlock * 2];
  const rangeEnd = spec.blockMeta[lastBlock * 2] + spec.blockMeta[lastBlock * 2 + 1];
  const blockMeta = new Uint32Array(nFrames * spec.nBlocksPerFrame * 2);
  for (let i = 0; i < nFrames * spec.nBlocksPerFrame; i++) {
    const source = (firstBlock + i) * 2;
    blockMeta[i * 2] = spec.blockMeta[source] - rangeStart;
    blockMeta[i * 2 + 1] = spec.blockMeta[source + 1];
  }
  return {
    ...spec,
    compressed: spec.compressed.subarray(rangeStart, rangeEnd),
    blockMeta,
    nFrames,
  };
}

function sliceFullStackSpecsByScanRegion(
  specs: ParsedSpec[],
  scanRows: number,
  scanCols: number,
  scanRegion: readonly [number, number, number, number] | null,
): ParsedSpec[] {
  if (!scanRegion) return specs;
  const { region } = normaliseScanRegion(scanRows, scanCols, scanRegion);
  if (!region) return specs;
  const [r0, r1, c0, c1] = region;
  if (r0 === 0 && r1 === scanRows && c0 === 0 && c1 === scanCols) return specs;
  const cropCols = c1 - c0;
  const out: ParsedSpec[] = [];
  for (let row = r0; row < r1; row++) {
    const rowSourceStart = row * scanCols + c0;
    const rowSourceStop = row * scanCols + c1;
    const rowOutputStart = (row - r0) * cropCols;
    for (const spec of specs) {
      const sourceStart = spec.startScan;
      const sourceStop = sourceStart + spec.nScan;
      const start = Math.max(rowSourceStart, sourceStart);
      const stop = Math.min(rowSourceStop, sourceStop);
      if (stop <= start) continue;
      const nScan = stop - start;
      out.push({
        ...compactBslz4FrameWindow(spec, start - sourceStart, nScan),
        startScan: rowOutputStart + (start - rowSourceStart),
        nScan,
      });
    }
  }
  return out;
}

function accumulateProfile(dst: Bslz4BatchProfile, src: Bslz4BatchProfile): void {
  if (src.variant) dst.variant = src.variant;
  dst.groups += src.groups;
  dst.specs += src.specs;
  dst.compressedMB += src.compressedMB;
  dst.uploadMs += src.uploadMs;
  dst.uploadCopyWaitMs = (dst.uploadCopyWaitMs ?? 0) + (src.uploadCopyWaitMs ?? 0);
  dst.buildMs += src.buildMs;
  dst.gpuWaitMs += src.gpuWaitMs;
  dst.decodeComputeWaitMs = (dst.decodeComputeWaitMs ?? 0) + (src.decodeComputeWaitMs ?? 0);
  dst.totalMs += src.totalMs;
  dst.profileSplit = Boolean(dst.profileSplit || src.profileSplit);
}

const ZERO_BAD_PIXELS_WGSL = `
@group(0) @binding(0) var<storage,read_write> src: array<u32>;
@group(0) @binding(1) var<storage,read> specs: array<u32>;  // wordOffset, clearMask pairs
@group(0) @binding(2) var<uniform> cfg: vec4<u32>;  // nScan, wordsPerFrame, nSpecs, unused
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let specIdx = gid.x;
  let scan = gid.y;
  if (specIdx >= cfg.z || scan >= cfg.x) { return; }
  let wordOffset = specs[specIdx * 2u];
  let clearMask = specs[specIdx * 2u + 1u];
  let index = scan * cfg.y + wordOffset;
  src[index] = src[index] & clearMask;
}`;

const DETECTOR_BIN_WGSL = `
@group(0) @binding(0) var<storage,read> src: array<u32>;
@group(0) @binding(1) var<storage,read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> cfg: vec4<u32>;   // nScan, srcRows, srcCols, detBin
@group(0) @binding(3) var<uniform> cfg2: vec4<u32>;  // outDetSize, srcDetSize, mode, outCols
fn sample(idx: u32, mode: u32) -> f32 {
  if (mode == 1u) {
    let word = src[idx >> 2u];
    return f32((word >> ((idx & 3u) * 8u)) & 255u);
  }
  if (mode == 0u) {
    let word = src[idx >> 1u];
    return f32((word >> ((idx & 1u) * 16u)) & 65535u);
  }
  return bitcast<f32>(src[idx]);
}
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let outPix = gid.x;
  let scan = gid.y;
  if (outPix >= cfg2.x || scan >= cfg.x) { return; }
  let outCols = cfg2.w;
  let outRow = outPix / outCols;
  let outCol = outPix - outRow * outCols;
  var sum = 0.0;
  for (var br = 0u; br < cfg.w; br = br + 1u) {
    for (var bc = 0u; bc < cfg.w; bc = bc + 1u) {
      let srcRow = outRow * cfg.w + br;
      let srcCol = outCol * cfg.w + bc;
      let detPix = srcRow * cfg.z + srcCol;
      sum = sum + sample(scan * cfg2.y + detPix, cfg2.z);
    }
  }
  dst[scan * cfg2.x + outPix] = bitcast<u32>(sum);
}`;

let detectorBinPipe: GPUComputePipeline | null = null;
let zeroBadPixelsPipe: GPUComputePipeline | null = null;

function getZeroBadPixelsPipe(device: GPUDevice): GPUComputePipeline {
  if (!zeroBadPixelsPipe) {
    zeroBadPixelsPipe = device.createComputePipeline({
      layout: "auto",
      compute: {
        module: device.createShaderModule({ code: ZERO_BAD_PIXELS_WGSL }),
        entryPoint: "main",
      },
    });
  }
  return zeroBadPixelsPipe;
}

function getDetectorBinPipe(device: GPUDevice): GPUComputePipeline {
  if (!detectorBinPipe) {
    detectorBinPipe = device.createComputePipeline({
      layout: "auto",
      compute: {
        module: device.createShaderModule({ code: DETECTOR_BIN_WGSL }),
        entryPoint: "main",
      },
    });
  }
  return detectorBinPipe;
}

function localUniform(device: GPUDevice, vals: number[]): GPUBuffer {
  const arr = new Uint32Array(vals);
  const buffer = device.createBuffer({
    size: Math.max(16, arr.byteLength),
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buffer, 0, arr.buffer, arr.byteOffset, arr.byteLength);
  return buffer;
}

function badPixelClearSpecs(badPixels: Uint32Array, detSize: number, mode: number): Uint32Array {
  const clear = new Map<number, number>();
  for (let i = 0; i < badPixels.length; i++) {
    const idx = badPixels[i];
    if (idx >= detSize) continue;
    let word = idx;
    let mask = 0;
    if (mode === 1) {
      word = idx >> 2;
      const shift = (idx & 3) * 8;
      mask = (~(0xff << shift)) >>> 0;
    } else if (mode === 0) {
      word = idx >> 1;
      const shift = (idx & 1) * 16;
      mask = (~(0xffff << shift)) >>> 0;
    }
    clear.set(word, (clear.get(word) ?? 0xffffffff) & mask);
  }
  const specs = new Uint32Array(Math.max(2, clear.size * 2));
  let pos = 0;
  for (const [word, mask] of clear) {
    specs[pos++] = word >>> 0;
    specs[pos++] = mask >>> 0;
  }
  return specs.subarray(0, pos);
}

async function binDetectorChunks(
  device: GPUDevice,
  sources: GPUBuffer[],
  sourceMode: number,
  nScans: number[],
  sourceDetRows: number,
  sourceDetCols: number,
  detBin: number,
  badPixels: Uint32Array,
): Promise<{ buffers: GPUBuffer[]; detRows: number; detCols: number; detSize: number; mode: number; ms: number }> {
  const outputDetRows = sourceDetRows / detBin;
  const outputDetCols = sourceDetCols / detBin;
  const outputDetSize = outputDetRows * outputDetCols;
  const sourceDetSize = sourceDetRows * sourceDetCols;
  const outs = sources.map((_, i) => device.createBuffer({
    size: Math.max(4, nScans[i] * outputDetSize * 4),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  }));
  const clearSpecs = badPixelClearSpecs(badPixels, sourceDetSize, sourceMode);
  const nClearSpecs = clearSpecs.length / 2;
  const wordsPerFrame = sourceMode === 1
    ? Math.ceil(sourceDetSize / 4)
    : sourceMode === 0
      ? Math.ceil(sourceDetSize / 2)
      : sourceDetSize;
  const temps: GPUBuffer[] = [];
  let clearSpecBuf: GPUBuffer | null = null;
  let zeroPipe: GPUComputePipeline | null = null;
  const zeroBindGroups: GPUBindGroup[] = [];
  if (nClearSpecs) {
    clearSpecBuf = device.createBuffer({
      size: Math.max(8, clearSpecs.byteLength),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    temps.push(clearSpecBuf);
    device.queue.writeBuffer(clearSpecBuf, 0, clearSpecs.buffer, clearSpecs.byteOffset, clearSpecs.byteLength);
    zeroPipe = getZeroBadPixelsPipe(device);
    for (let i = 0; i < sources.length; i++) {
      const cfg = localUniform(device, [nScans[i], wordsPerFrame, nClearSpecs, 0]);
      temps.push(cfg);
      zeroBindGroups.push(device.createBindGroup({
        layout: zeroPipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: sources[i] } },
          { binding: 1, resource: { buffer: clearSpecBuf } },
          { binding: 2, resource: { buffer: cfg } },
        ],
      }));
    }
  }
  const binPipe = getDetectorBinPipe(device);
  const binBindGroups: GPUBindGroup[] = [];
  for (let i = 0; i < sources.length; i++) {
    const cfg = localUniform(device, [nScans[i], sourceDetRows, sourceDetCols, detBin]);
    const cfg2 = localUniform(device, [outputDetSize, sourceDetSize, sourceMode, outputDetCols]);
    temps.push(cfg, cfg2);
    binBindGroups.push(device.createBindGroup({
      layout: binPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: sources[i] } },
        { binding: 1, resource: { buffer: outs[i] } },
        { binding: 2, resource: { buffer: cfg } },
        { binding: 3, resource: { buffer: cfg2 } },
      ],
    }));
  }
  const enc = device.createCommandEncoder();
  if (zeroPipe && zeroBindGroups.length) {
    const pass = enc.beginComputePass();
    pass.setPipeline(zeroPipe);
    for (let i = 0; i < zeroBindGroups.length; i++) {
      pass.setBindGroup(0, zeroBindGroups[i]);
      pass.dispatchWorkgroups(Math.ceil(nClearSpecs / 64), nScans[i]);
    }
    pass.end();
  }
  const pass = enc.beginComputePass();
  pass.setPipeline(binPipe);
  for (let i = 0; i < binBindGroups.length; i++) {
    pass.setBindGroup(0, binBindGroups[i]);
    pass.dispatchWorkgroups(Math.ceil(outputDetSize / 16), Math.ceil(nScans[i] / 16));
  }
  pass.end();
  const t0 = performance.now();
  device.queue.submit([enc.finish()]);
  await device.queue.onSubmittedWorkDone();
  const ms = performance.now() - t0;
  temps.forEach((buf) => buf.destroy());
  return { buffers: outs, detRows: outputDetRows, detCols: outputDetCols, detSize: outputDetSize, mode: 2, ms };
}

export async function loadShow4DSTEMLocalH5Master(
  masterUrl: string,
  options: LocalH5LoadOptions,
): Promise<LocalH5LoadResult | null> {
  if (!/_master\.h5(?:[?#].*)?$/i.test(masterUrl)) return null;
  const master = localFileFor(masterUrl);
  const dataFiles = localDataFilesForMaster(masterUrl);
  if (!master || dataFiles.length === 0) return null;

  const scanRows = Math.max(1, Math.round(options.scanRows));
  const scanCols = Math.max(1, Math.round(options.scanCols));
  const { rows: outputRows, cols: outputCols, region } = normaliseScanRegion(scanRows, scanCols, options.scanRegion);
  const scanCount = outputRows * outputCols;
  const detBin = safeInt(options.detBin, 1, 1, 16);
  const low8Only = typeof globalThis !== "undefined" && (globalThis as { __BSLZ4_LOW8_ONLY?: boolean }).__BSLZ4_LOW8_ONLY === true;
  const dataItems = dataFileItemsForScanRegion(dataFiles, scanRows, scanCols, region);
  const blockIndexedDataItems = dataItems.length > 0 && dataItems.every((item) => Boolean(localBlockIndexFor(item.file.name)));
  const frameIndexedDataItems = dataItems.length > 0 && dataItems.every((item) => Boolean(frameIndexFor(item.file.name)));
  const defaultWorkerCount = low8Only
    ? (!region && (blockIndexedDataItems || frameIndexedDataItems) ? 8 : 2)
    : Math.min(4, navigator.hardwareConcurrency || 4);
  const workerCount = safeInt(options.workerCount, defaultWorkerCount, 0, 8);
  const defaultGroupSize = low8Only ? (region && scanCount <= 256 * 256 ? 4 : 8) : 4;
  const groupSize = safeInt(options.groupSize, defaultGroupSize, 1, 16);
  const decodeBatch = safeInt(options.decodeBatch, low8Only ? (region && scanCount <= 256 * 256 ? 8 : 1) : 4, 1, 16);
  const t0 = performance.now();

  let badPixels = parseEmbeddedBadPixels(options.embeddedBadPixelsJson) || new Uint32Array(0);
  let totalFrames = 0;
  const mt = performance.now();
  setLocalH5Debug("product:read-master", { scanRows, scanCols, outputRows, outputCols });
  try {
    const info = readH5MasterInfo(await master.arrayBuffer(), master.name);
    if (badPixels.length === 0 && info.badPixels.length) badPixels = new Uint32Array(info.badPixels);
    totalFrames = Math.max(0, Math.round(Number(info.totalFrames || 0)));
  } catch {
    // Continue with shape metadata from the widget traits.
  }
  const masterReadMs = performance.now() - mt;

  const reads = readFiles(dataItems.map((item) => item.file), workerCount, frameIndexFor, localBlockIndexFor);
  const gpuChunks: LocalH5GpuChunk[] = [];
  let pendingDecode: Promise<{ device: GPUDevice; buffers: GPUBuffer[]; mode: number; profile: Bslz4BatchProfile } | null> | null = null;
  let pendingSpecs: ParsedSpec[] = [];
  let device: GPUDevice | null = null;
  let mode = 1;
  let detSize = 0;
  let sourceDetRows = 0;
  let sourceDetCols = 0;
  let outputDetRows = 0;
  let outputDetCols = 0;
  let frames = 0;
  let sourceFrames = 0;
  let fileBytes = 0;
  let blockIndexBytes = 0;
  let compressedBytes = 0;
  let readWaitMs = 0;
  let readWorkerMs = 0;
  let parseMs = 0;
  let packMs = 0;
  let frameIndexFiles = 0;
  let blockIndexFiles = 0;
  let sourceDtype: SourceDtype | "unknown" = "unknown";
  let decodeDtype: "uint8" | "uint16" | "float32" = "uint8";
  let decompressMs = 0;
  let detBinMs = 0;
  const decodeProfile: Bslz4BatchProfile = {
    variant: "",
    groups: 0,
    specs: 0,
    compressedMB: 0,
    uploadMs: 0,
    uploadCopyWaitMs: 0,
    buildMs: 0,
    gpuWaitMs: 0,
    decodeComputeWaitMs: 0,
    totalMs: 0,
    profileSplit: false,
  };
  const drain = async (): Promise<void> => {
    if (!pendingDecode) return;
    const dt = performance.now();
    const decoded = await pendingDecode;
    decompressMs += performance.now() - dt;
    if (!decoded) throw new Error("WebGPU unavailable for local HDF5 decode.");
    device = decoded.device;
    if (decoded.buffers.length) mode = detBin > 1 ? 2 : decoded.mode;
    accumulateProfile(decodeProfile, decoded.profile);
    let outputBuffers = decoded.buffers;
    if (detBin > 1 && decoded.buffers.length) {
      const binned = await binDetectorChunks(
        decoded.device,
        decoded.buffers,
        decoded.mode,
        pendingSpecs.map((spec) => spec.nScan),
        sourceDetRows,
        sourceDetCols,
        detBin,
        badPixels,
      );
      decoded.buffers.forEach((buffer) => buffer.destroy());
      outputBuffers = binned.buffers;
      detSize = binned.detSize;
      outputDetRows = binned.detRows;
      outputDetCols = binned.detCols;
      detBinMs += binned.ms;
    }
    for (let i = 0; i < decoded.buffers.length; i++) {
      const buffer = outputBuffers[i];
      const spec = pendingSpecs[i];
      gpuChunks.push({ buffer, startScan: spec.startScan, nScan: spec.nScan });
    }
    pendingDecode = null;
    pendingSpecs = [];
  };

  for (let g = 0; g < dataItems.length; g += groupSize) {
    const specs: ParsedSpec[] = [];
    for (let i = g; i < Math.min(g + groupSize, dataItems.length); i++) {
      const item = dataItems[i];
      const wt = performance.now();
      const read = await reads[i];
      readWaitMs += performance.now() - wt;
      readWorkerMs += read.readMs;
      fileBytes += read.fileBytes;
      blockIndexBytes += read.blockIndexBytes || 0;

      const pt = performance.now();
      const frameIndex = frameIndexFor(read.name);
      const vol = read.volume
        ? read.volume
        : frameIndex
          ? readH5VolumeFromFrameIndex(read.buffer!, read.name, frameIndex)
          : readH5Volume(read.buffer!, read.name);
      if (read.blockIndexBytes) blockIndexFiles += 1;
      else if (read.volume || frameIndex) frameIndexFiles += 1;
      parseMs += read.volume ? (read.parseMs ?? 0) : performance.now() - pt;
      if (sourceDtype !== "unknown" && sourceDtype !== vol.srcDtype) {
        throw new Error(`Mixed HDF5 source dtypes are not supported in one local load: ${sourceDtype} and ${vol.srcDtype}.`);
      }
      sourceDtype = vol.srcDtype;
      if (sourceDetRows && (sourceDetRows !== vol.detRows || sourceDetCols !== vol.detCols)) {
        throw new Error(`Mixed detector shapes are not supported in one local HDF5 load: ${sourceDetRows}x${sourceDetCols} and ${vol.detRows}x${vol.detCols}.`);
      }
      if (detBin > 1 && (vol.detRows % detBin !== 0 || vol.detCols % detBin !== 0)) {
        throw new Error(`Detector shape ${vol.detRows}x${vol.detCols} is not divisible by detBin=${detBin}.`);
      }
      if (detBin > 1 && vol.srcDtype === "uint32") {
        throw new Error("WebGPU detector-bin load currently supports uint8, uint16, and float32 sources; uint32 needs real-acquisition parity before enablement.");
      }
      // Count-audited low8 browse sources can be detector-binned directly from
      // the same lossless low8 decode. This preserves the explicit detBin
      // evidence policy while avoiding the slower full uint16 intermediate.
      decodeDtype = detBin > 1
        ? (vol.srcDtype === "float32" ? "float32" : low8Only ? "uint8" : vol.srcDtype === "uint16" ? "uint16" : "uint8")
        : vol.srcDtype === "float32"
          ? "float32"
          : "uint8";
      sourceDetRows = vol.detRows;
      sourceDetCols = vol.detCols;
      outputDetRows = detBin > 1 ? vol.detRows / detBin : vol.detRows;
      outputDetCols = detBin > 1 ? vol.detCols / detBin : vol.detCols;
      detSize = outputDetRows * outputDetCols;
      const fileSourceStart = item.startScan ?? sourceFrames;
      let chunkStart = fileSourceStart;
      vol.chunks.forEach((chunk, chunkIndex) => {
        const nScan = vol.chunkScanCounts[chunkIndex] ?? chunk.nFrames;
        specs.push({ ...chunk, startScan: chunkStart, nScan });
        chunkStart += nScan;
      });
      sourceFrames = item.startScan === undefined ? sourceFrames + vol.nFrames : Math.max(sourceFrames, fileSourceStart + vol.nFrames);
    }
    const ptPack = performance.now();
    const decodeSpecs = sliceFullStackSpecsByScanRegion(specs, scanRows, scanCols, region);
    packMs += performance.now() - ptPack;
    compressedBytes += decodeSpecs.reduce((n, spec) => n + spec.compressed.byteLength, 0);
    frames += decodeSpecs.reduce((n, spec) => n + spec.nScan, 0);
    await drain();
    if (decodeSpecs.length) {
      pendingSpecs = decodeSpecs;
      pendingDecode = decodeBslz4Batch(decodeSpecs, decodeDtype, sourceDtype === "unknown" ? "uint16" : sourceDtype, decodeBatch);
    }
  }
  await drain();
  const profileDevice = device as GPUDevice | null;
  if (!profileDevice) throw new Error("WebGPU unavailable for local HDF5 decode.");
  const profile: LocalH5LoadProfile = {
    acquisitionMode: "local-file",
    totalMs: Math.round(performance.now() - t0),
    masterReadMs: Math.round(masterReadMs),
    readWaitMs: Math.round(readWaitMs),
    readWorkerMs: Math.round(readWorkerMs),
    parseMs: Math.round(parseMs),
    packMs: Math.round(packMs),
    decompressMs: Math.round(decompressMs),
    uploadMs: Math.round(decodeProfile.uploadMs),
    uploadCopyWaitMs: Math.round(decodeProfile.uploadCopyWaitMs ?? 0),
    detBinMs: Math.round(detBinMs),
    decodeBuildMs: Math.round(decodeProfile.buildMs),
    gpuWaitMs: Math.round(decodeProfile.gpuWaitMs),
    decodeComputeWaitMs: Math.round(decodeProfile.decodeComputeWaitMs ?? 0),
    decodeProfileMs: Math.round(decodeProfile.totalMs),
    fileGB: +(fileBytes / 1e9).toFixed(2),
    blockIndexGB: +(blockIndexBytes / 1e9).toFixed(3),
    compressedGB: +(compressedBytes / 1e9).toFixed(2),
    decodeCompressedMB: Math.round(decodeProfile.compressedMB),
    sourceDtype,
    decodeDtype,
    detBin,
    sourceDetRows,
    sourceDetCols,
    outputDetRows,
    outputDetCols,
    chunks: gpuChunks.length,
    frames,
    sourceFrames: totalFrames || sourceFrames,
    scanRows,
    scanCols,
    outputRows,
    outputCols,
    badPixels: badPixels.length,
    dataFilesExpected: totalFrames > 0 ? dataFiles.length : null,
    workerCount,
    groupSize,
    decodeBatch,
    decodeVariant: decodeProfile.variant,
    decodeProfileSplit: Boolean(decodeProfile.profileSplit),
    frameIndexFiles,
    blockIndexFiles,
    parseMode: blockIndexFiles === dataItems.length
      ? "block-index"
      : frameIndexFiles === 0 && blockIndexFiles === 0
        ? "h5-btree"
        : frameIndexFiles === dataItems.length
          ? "frame-index"
          : "mixed",
    adapterInfo: getGPUInfo(),
    softwareAdapter: isSoftwareGPUAdapter(),
    timestampQuery: Boolean(profileDevice.features.has("timestamp-query")),
    subgroups: Boolean(profileDevice.features.has("subgroups" as GPUFeatureName)),
    maxBufferGB: +(Number(profileDevice.limits.maxBufferSize || 0) / 1e9).toFixed(2),
    maxStorageBufferGB: +(Number(profileDevice.limits.maxStorageBufferBindingSize || 0) / 1e9).toFixed(2),
  };
  if (frames < scanCount) {
    console.warn(`Local HDF5 load decoded ${frames} scan positions, fewer than target scan shape ${scanCount}.`);
  }
  return {
    device: profileDevice,
    chunks: gpuChunks,
    scanCount,
    detSize,
    detRows: outputDetRows,
    detCols: outputDetCols,
    sourceDetRows,
    sourceDetCols,
    detBin,
    mode,
    badPixels: detBin > 1 ? new Uint32Array(0) : badPixels,
    profile,
  };
}

export async function loadShow4DSTEMLocalH5MaskedSum(
  masterUrl: string,
  options: LocalH5MaskedSumOptions,
): Promise<LocalH5MaskedSumResult | null> {
  if (!/_master\.h5(?:[?#].*)?$/i.test(masterUrl)) return null;
  const master = localFileFor(masterUrl);
  const dataFiles = localDataFilesForMaster(masterUrl);
  if (!master) return null;

  const scanRows = Math.max(1, Math.round(options.scanRows));
  const scanCols = Math.max(1, Math.round(options.scanCols));
  const { rows: outputRows, cols: outputCols, region } = normaliseScanRegion(scanRows, scanCols, options.scanRegion);
  const outputScanCount = outputRows * outputCols;
  const workerCount = safeInt(options.workerCount, 0, 0, 16);
  const requestedProductBatch = options.productBatch ?? options.decodeBatch;
  const t0 = performance.now();

  let badPixels = parseEmbeddedBadPixels(options.embeddedBadPixelsJson) || new Uint32Array(0);
  let totalFrames = 0;
  const mt = performance.now();
  try {
    const info = readH5MasterInfo(await master.arrayBuffer(), master.name);
    if (badPixels.length === 0 && info.badPixels.length) badPixels = new Uint32Array(info.badPixels);
    totalFrames = Math.max(0, Math.round(Number(info.totalFrames || 0)));
  } catch {
    // Continue with shape metadata from the widget traits.
  }
  const masterReadMs = performance.now() - mt;

  const sidecarSelection = filterSelectedBlockSidecarsForScanRegion(
    await chooseSelectedBlockSidecars(masterUrl, options.mask, badPixels),
    scanCols,
    region,
  );
  setLocalH5Debug("product:select-source", {
    sidecarFiles: sidecarSelection.files.length,
    sidecarCoverage: sidecarSelection.coverage,
    dataFiles: dataFiles.length,
  });
  const sourceMode: "native-h5" | "selected-block-sidecar" = sidecarSelection.files.length ? "selected-block-sidecar" : "native-h5";
  const sourceFiles = sourceMode === "selected-block-sidecar" ? sidecarSelection.files : dataFiles;
  if (sourceFiles.length === 0) return null;
  const reads = readFiles(sourceFiles, workerCount);
  setLocalH5Debug("product:read-files", {
    sourceMode,
    sourceFiles: sourceFiles.length,
    workerCount,
  });
  const specs: Bslz4MaskedSumSpec[] = [];
  let fileBytes = 0;
  let compressedBytes = 0;
  let readWaitMs = 0;
  let readWorkerMs = 0;
  let parseMs = 0;
  let frameIndexFiles = 0;
  let sourceDtype: SourceDtype | "unknown" = "unknown";
  let frames = 0;
  const readWallStart = performance.now();
  for (let i = 0; i < sourceFiles.length; i++) {
    const wt = performance.now();
    const read = await reads[i];
    readWaitMs += performance.now() - wt;
    readWorkerMs += read.readMs;
    fileBytes += read.fileBytes;

    const pt = performance.now();
    if (sourceMode === "selected-block-sidecar") {
      if (!read.buffer) throw new Error(`Local selected-block worker did not return bytes for ${read.name}.`);
      const vol = readBslz4SelectedBlockVolume(read.buffer, read.name);
      parseMs += performance.now() - pt;
      if (sourceDtype !== "unknown" && sourceDtype !== vol.srcDtype) {
        throw new Error(`Mixed selected-block source dtypes are not supported in one local product load: ${sourceDtype} and ${vol.srcDtype}.`);
      }
      if (vol.srcDtype === "float32") {
        throw new Error("Product-first WebGPU selected-block masked sums currently require integer bslz4 sources.");
      }
      sourceDtype = vol.srcDtype;
      compressedBytes += vol.chunk.compressed.byteLength;
      const sourceStartScan = sidecarSelection.spans[i]?.startScan ?? frames;
      specs.push({ ...vol.chunk, startScan: sourceStartScan, sourceStartScan });
      frames += vol.nFrames;
    } else {
      const frameIndex = frameIndexFor(read.name);
      const vol = frameIndex
        ? readH5VolumeFromFrameIndex(read.buffer!, read.name, frameIndex)
        : readH5Volume(read.buffer!, read.name);
      if (frameIndex) frameIndexFiles += 1;
      parseMs += performance.now() - pt;
      if (sourceDtype !== "unknown" && sourceDtype !== vol.srcDtype) {
        throw new Error(`Mixed HDF5 source dtypes are not supported in one local product load: ${sourceDtype} and ${vol.srcDtype}.`);
      }
      if (vol.srcDtype === "float32") {
        throw new Error("Product-first WebGPU HDF5 masked sums currently require integer bslz4 sources; use the full-stack float32 path.");
      }
      sourceDtype = vol.srcDtype;
      const spec = vol.chunks[0];
      compressedBytes += spec.compressed.byteLength;
      specs.push({ ...spec, startScan: frames, sourceStartScan: frames });
      frames += vol.nFrames;
    }
    setLocalH5Debug("product:read-file", {
      sourceMode,
      index: i + 1,
      sourceFiles: sourceFiles.length,
      frames,
      compressedGB: +(compressedBytes / 1e9).toFixed(3),
    });
  }
  const readWallMs = performance.now() - readWallStart;
  const productSpecs = region
    ? sliceMaskedSumSpecsByScanRegion(specs, scanRows, scanCols, region)
    : specs;
  const framesComputed = productSpecs.reduce((n, spec) => n + (spec.frameCount ?? spec.nFrames), 0);
  const productBatch = chooseMaskedSumProductBatch(requestedProductBatch, outputScanCount, productSpecs.length);
  const pt = performance.now();
  setLocalH5Debug("product:dispatch", {
    sourceMode,
    specs: specs.length,
    productSpecs: productSpecs.length,
    framesComputed,
    productBatch,
  });
  const product = await decodeBslz4MaskedSumLow8Batch(productSpecs, options.mask, outputScanCount, badPixels, productBatch);
  const productMs = performance.now() - pt;
  if (!product) throw new Error("WebGPU unavailable for local HDF5 product-first masked sum.");
  setLocalH5Debug("product:done", { productMs: Math.round(productMs) });
  const profileDevice = product.device;
  const profile: LocalH5MaskedSumProfile = {
    acquisitionMode: sourceMode === "selected-block-sidecar" ? "local-file-selected-block-sidecar" : "local-file-product-first",
    sourceMode,
    totalMs: Math.round(performance.now() - t0),
    masterReadMs: Math.round(masterReadMs),
    readWaitMs: Math.round(readWaitMs),
    readWorkerMs: Math.round(readWorkerMs),
    readWallMs: Math.round(readWallMs),
    parseMs: Math.round(parseMs),
    productMs: Math.round(productMs),
    fileGB: +(fileBytes / 1e9).toFixed(2),
    compressedGB: +(compressedBytes / 1e9).toFixed(2),
    sourceDtype,
    framesRead: frames,
    framesComputed,
    scanRows,
    scanCols,
    outputRows,
    outputCols,
    badPixels: badPixels.length,
    dataFilesExpected: totalFrames > 0 ? Math.ceil(totalFrames / Math.max(1, frames / Math.max(1, sourceFiles.length))) : null,
    workerCount,
    productBatch,
    sidecarFiles: sourceMode === "selected-block-sidecar" ? sourceFiles.length : 0,
    sidecarCoverage: sourceMode === "selected-block-sidecar" ? "used" : sidecarSelection.coverage,
    frameIndexFiles,
    blockIndexFiles: 0,
    parseMode: sourceMode === "selected-block-sidecar" ? "selected-block-sidecar" : frameIndexFiles === 0 ? "h5-btree" : frameIndexFiles === sourceFiles.length ? "frame-index" : "mixed",
    productProfile: product.profile,
    adapterInfo: getGPUInfo(),
    softwareAdapter: isSoftwareGPUAdapter(),
    timestampQuery: Boolean(profileDevice.features.has("timestamp-query")),
    subgroups: Boolean(profileDevice.features.has("subgroups" as GPUFeatureName)),
    maxBufferGB: +(Number(profileDevice.limits.maxBufferSize || 0) / 1e9).toFixed(2),
    maxStorageBufferGB: +(Number(profileDevice.limits.maxStorageBufferBindingSize || 0) / 1e9).toFixed(2),
  };
  return { device: profileDevice, buffer: product.buffer, scanRows: outputRows, scanCols: outputCols, profile };
}
