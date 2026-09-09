import { source112Layout, source112Pipeline } from './source112-pipelines';
import { Source112WorkerReader } from "./source112-worker-reader";
import { Source112Ingress, type Source112IngressProfile } from './source112-ingress';
import { compactHuffmanShader, COMPACT_BUILD_WGSL, compactGroupLayout } from './source112-huffman-compact';
import { authenticatedRecords } from './source112-prefetch';
import { HUFFMAN_MIGRATION_WGSL, HUFFMAN_CHECKPOINT_WGSL } from './source112-huffman-migration';
import { SOURCE112_HUFFMAN64_WGSL, SOURCE112_HUFFMAN64_SUM_WGSL } from './source112-huffman64';
import { encodingBook, source112HuffmanBooks } from './source112-huffman-books';
import type { Uint32ImageView } from "../../../display/webgpu/borrowed-image";
/// <reference types="@webgpu/types" />
/**
 * Browser adapter for the preserved source112 pair-tANS and sparse-count archive.
 * This private format boundary admits complete native uint16 acquisitions at
 * 512 x 512 scan positions and 192 x 192 detector pixels. It does not generalize
 * reduce the scientific source. The optional Huffman64 representation is exact. Metadata is checked before GPU allocation;
 * each encoded record is authenticated once during bounded local-file loading.
 */
import { type RansByteSource, type RansPayloadProfile } from './rans-source';
import { SOURCE112_WGSL, SOURCE112_SUM_WGSL, SOURCE112_OFFSET_RESTART_WGSL, SOURCE112_OFFSET_RESTART_BUILD_WGSL } from './source112-kernels';
interface Component {
  name: string;
  offset: number;
  nbytes: number;
  dtype: string;
}
interface RecordInfo {
  chunk: number;
  acquisition: number;
  first_scan: number;
  scan_count: number;
  shard: number;
  file_offset: number;
  record_bytes: number;
  sha256: string;
  components: Component[];
}
interface GlobalInfo {
  file: string;
  dtype: string;
  shape: number[];
  nbytes: number;
  sha256: string;
}
interface Manifest {
  format: string;
  shape: number[];
  dtype: string;
  layout: {
    files: {
      name: string;
      nbytes: number;
    }[];
    chunks: RecordInfo[];
  };
  globals: Record<string, GlobalInfo>;
}
interface Group {
  payload: GPUBuffer;
  descriptors: GPUBuffer;
  records: RecordInfo[];
  bind: GPUBindGroup[];
  params: GPUBuffer[];
}
export interface Source112LoadOptions {
  representation?: 'tans' | 'huffman64';
  progressive?: boolean;
  maxAcquisitions?: number;
  onProgress?: (source: Source112ResidentSet) => void;
  yieldToInteraction?: () => Promise<void>;
}
const C = 17466, S = C * 32, ANS_WORDS = 82944;
const STORAGE = () => GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
const N = 512 * 512, Q = 192 * 192;
const hash = async (raw: ArrayBuffer) => [...new Uint8Array(await crypto.subtle.digest('SHA-256', raw))].map(v => v.toString(16).padStart(2, '0')).join('');
const validName = (name: string) => !!name && !name.split('/').some(p => !p || p === '.' || p === '..' || /[:\\?#]/.test(p));
/** File input grants remain the only source of payload bytes after selection. */
function fileSource(files: ArrayLike<File>): RansByteSource & {blob(name: string, start?: number, end?: number): Blob} {
  const byName = new Map<string, File>();
  for (const file of Array.from(files)) {
    if (byName.has(file.name))
      throw new Error(`Duplicate source file ${file.name}; select exactly one exported source112 folder.`);
    byName.set(file.name, file);
  }
  return {
    mode: 'local-folder', blob(name, start, end) {
      if (!validName(name))
        throw new Error(`Invalid source112 filename: ${name}`);
      const file = byName.get(name);
      if (!file)
        throw new Error(`Missing ${name}; select the complete exported source112 folder.`);
      const lo = start ?? 0, hi = end ?? file.size;
      if (!Number.isSafeInteger(lo) || !Number.isSafeInteger(hi) || lo < 0 || hi < lo || hi > file.size)
        throw new Error(`Invalid range for ${name}: ${lo}:${hi}.`);
      return file.slice(lo, hi);
    },
    async read(name, start, end) {
      const blob = this.blob(name, start, end);
      const bytes = await blob.arrayBuffer();
      if (bytes.byteLength !== blob.size) throw new Error(`Short source112 read: ${name}.`);
      return bytes;
    }
  };
}
/** Zero-copy payload views over an already validated folder, never a disk cache. */
export async function source112AcquisitionFiles(
  files: ArrayLike<File>, original: Manifest, globals: Record<string, ArrayBuffer>, acquisition: number,
): Promise<File[]> {
  if (!Number.isInteger(acquisition) || acquisition < 0 || acquisition >= original.shape[0])
    throw Error('Select an acquisition within the validated series.');
  const manifest = structuredClone(original);
  manifest.shape[0] = 1;
  const records = manifest.layout.chunks.slice(acquisition * 16, (acquisition + 1) * 16);
  const output: File[] = [], shards = [...new Set(records.map(record => record.shard))];
  const byShard = shards.map(shard => records.filter(record => record.shard === shard));
  manifest.layout.files = shards.map((shard, index) => {
    const rows = byShard[index];
    const first = rows[0].file_offset, end = rows[rows.length - 1].file_offset + rows[rows.length - 1].record_bytes;
    const name = original.layout.files[shard].name;
    const file = Array.from(files).find(file => file.name === name)!;
    output.push({name, size: end - first, slice: (lo = 0, hi = end - first) => file.slice(first + lo, first + hi)} as File);
    for (const row of rows) { row.shard = index; row.file_offset -= first; }
    return {name, nbytes: end - first};
  });
  records.forEach((record, i) => { record.chunk = i; record.acquisition = 0; });
  manifest.layout.chunks = records;
  for (const [name, raw] of Object.entries(globals)) {
    const spec = manifest.globals[name];
    const bytes = name === 'model_ids' ? raw.slice(acquisition * 4 * Q, (acquisition + 1) * 4 * Q) : raw;
    if (name === 'model_ids') spec.shape = [4, Q];
    spec.nbytes = bytes.byteLength; spec.sha256 = await hash(bytes);
    output.push(new File([bytes], spec.file));
  }
  output.push(new File([JSON.stringify(manifest)], 'manifest.json'));
  return output;
}
/**
 * Owns encoded source groups, exact uint32 products, and reusable float displays.
 * Acquisition adapters borrow this set; only its owner releases GPU resources.
 * Presentation normalization always starts from the authoritative count buffer.
 */
export class Source112ResidentSet {
  readonly shape = [512, 512, 192, 192] as const;
  readonly isSource112 = true;
  readonly scanCount = N;
  readonly detSize = Q;
  readonly computes: Source112DetectorCompute[];
  readonly acquisitionMode = 'local-folder';
  readonly checkpointMs = 0;
  private partitions?: Source112ResidentSet[];
  private loadingAbort?: AbortController;
  completion: Promise<void> = Promise.resolve();
  readonly acquisitionReadyMs: number[] = [];
  allAcquisitionsReadyMs = 0;
  get loadedAcquisitions() { return this.partitions?.length ?? this.acquisitionCount; }
  private loaded(index: number) {
    this.check();
    const child = this.partitions?.[index];
    if (!child) throw Error(`Acquisition ${index + 1} is still loading; select a ready image.`);
    return child;
  }
  badPx = new Uint32Array(0);
  get payloadBytes(): number { if (this.partitions) return this.partitions.reduce((n, child) => n + child.payloadBytes, 0); return this.groups.reduce((n, g) => n + g.payload.size, 0); }
  get readyMs() { return this.profile.readyMs; }
  get loadMs() { return this.profile.readyMs; }
  get loadProfile() { return this.profile; }
  readonly nativeDtype = 'uint16';
  private outputBuffer!: GPUBuffer;
  private errorBuffer!: GPUBuffer;
  get output() { if (this.partitions) throw Error("Use imageViewsU32 for partitioned resident images."); return this.outputBuffer; }
  get errors() { return this.errorBuffer; }
  readonly groups: Group[] = [];
  readonly profile: RansPayloadProfile & Source112IngressProfile & {
    payloadStagingBytes: number;
    peakPayloadStagingBytes: number;
    residentBytes: number;
    readyMs: number;
    records: number;
    restartPreparationMs: number;
    restartCacheBytes: number;
    representation: 'tans' | 'huffman64';
    restartCacheLayout: 'none' | 'tans128-offset' | 'huffman64-offset' | 'huffman64-compact';
    migrationHotNativeValues: number;
    migrationRetainedWords: number;
    peakResidentBytes: number;
    payloadHashMs: number;
    peakHostPrefetchBytes: number;
    payloadLoadMs: number;
    metadataAndSetupMs: number;
    payloadReadBackend: 'main-thread' | 'worker';
  } = {
      payloadHostCopyMs: 0, payloadMapPendingMs: 0, payloadMapBlockedMs: 0, payloadCopySubmitMs: 0, payloadFenceWaitMs: 0, payloadStagingBytes: 0, peakPayloadStagingBytes: 0, payloadReadMs: 0, payloadReadWaitMs: 0, payloadStageMs: 0, payloadChunks: 0, residentBytes: 0, readyMs: 0, records: 0, restartPreparationMs: 0, restartCacheBytes: 0, representation: 'tans', restartCacheLayout:'none', migrationHotNativeValues: 0, migrationRetainedWords: 0, peakResidentBytes: 0, payloadHashMs: 0, peakHostPrefetchBytes: 0, payloadLoadMs: 0, metadataAndSetupMs: 0, payloadReadBackend: 'main-thread'
    };
  private owned: GPUBuffer[] = [];
  private disposed = false;
  private preparing = false;
  private representation: 'tans' | 'huffman64' = 'tans';
  get storageRepresentation() { return this.representation; }
  private columns: Uint32Array[] = [];
  private selected: GPUBuffer[] = [];
  private pipelines: GPUComputePipeline[] = [];
  private denseSumPipeline!: GPUComputePipeline;
  private restartPipeline?: GPUComputePipeline;
  private restartEnabled = false;
  private restartPreparing = false;
  private restartOriginals?: { group: Group; descriptors: GPUBuffer; bind: GPUBindGroup[] }[];
  private restartProfile?: { offsetBytes: number; checkpointBytes: number; additionalBytes: number; groups: number; streams: number; segmentValues: number; preparationMs: number };
  private previousMask: Uint32Array | null = null;
  private displayBuffers = new Map<number, GPUBuffer>();
  private convertPipeline?: GPUComputePipeline;
  private convertUniforms?: GPUBuffer;
  private convertUniformData?: ArrayBuffer;
  private convertUniformStride?: number;
  private convertBindings?: Map<number, GPUBindGroup>;
  private layout!: GPUBindGroupLayout;
  private globals!: GPUBuffer[];
  private patternCounts?: GPUBuffer;
  private patternDisplays?: Map<number, GPUBuffer>;
  private patternParams?: GPUBuffer;
  private patternConvert?: GPUComputePipeline;
  private patternConvertBindings?: Map<number, GPUBindGroup>;
  private constructor(readonly device: GPUDevice, readonly acquisitionCount = 66) {
    this.computes = Array.from({ length: acquisitionCount }, (_, i) => new Source112DetectorCompute(this, i));
  }
  /** Lifetime logical high-water mark for owned buffers plus preparation scratch.
   * Renderer-owned resources and one-shot scientific readbacks are excluded. */
  private recordResidentPeak(preparationBytes = 0) {
    this.profile.peakResidentBytes = Math.max(this.profile.peakResidentBytes ?? 0, this.profile.residentBytes + preparationBytes + (this.profile.payloadStagingBytes ?? 0));
  }
  private buffer(size: number, usage: GPUBufferUsageFlags, raw?: ArrayBuffer): GPUBuffer {
    const bytes = Math.max(4, Math.ceil(size / 4) * 4);
    const buffer = this.device.createBuffer({ size: bytes, usage, mappedAtCreation: !!raw });
    this.owned.push(buffer);
    this.profile.residentBytes += bytes;
    this.recordResidentPeak();
    if (raw) {
      new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(raw));
      buffer.unmap();
    }
    return buffer;
  }
  private check() {
    if (this.disposed)
      throw new Error('Source112 source is closed; load the folder again.');
    if (this.preparing) throw new Error('Source112 is preparing its complete resident representation.');
  }
  /** Prepare each complete acquisition independently, retaining ready owners. */
  private static async loadAcquisitions(
    device: GPUDevice, files: ArrayLike<File>, manifest: Manifest,
    globals: Record<string, ArrayBuffer>, status: (text: string) => void,
    signal: AbortSignal | undefined, options: Source112LoadOptions,
    began: number, badPx: Uint32Array,
  ): Promise<Source112ResidentSet> {
    const count = options.maxAcquisitions ?? manifest.shape[0];
    if (!Number.isInteger(count) || count < 1 || count > manifest.shape[0])
      throw Error('Choose a positive acquisition count within the selected series.');
    const result = new Source112ResidentSet(device, count);
    result.partitions = [];
    result.badPx = new Uint32Array(badPx);
    result.loadingAbort = new AbortController();
    const abort = () => result.loadingAbort!.abort(signal?.reason);
    signal?.addEventListener('abort', abort, {once: true});
    if (signal?.aborted) abort();
    const load = async (acquisition: number) => {
      result.loadingAbort!.signal.throwIfAborted();
      await options.yieldToInteraction?.();
      const selected = await source112AcquisitionFiles(files, manifest, globals, acquisition);
      const child = await Source112ResidentSet.loadFiles(device, selected,
        () => status(`${result.loadedAcquisitions}/${count} acquisitions ready · loading acquisition ${acquisition + 1}`),
        result.loadingAbort!.signal, {representation: options.representation, yieldToInteraction: options.yieldToInteraction});
      if (result.disposed || result.loadingAbort!.signal.aborted) {
        child.destroy(); result.loadingAbort!.signal.throwIfAborted();
        throw Error('Resident source was closed during admission.');
      }
      result.partitions!.push(child);
      result.acquisitionReadyMs.push(performance.now() - began);
      // Admission owns only complete children; partial children clean themselves up.
      for (const key of ['residentBytes', 'records', 'payloadReadMs', 'payloadReadWaitMs', 'payloadHashMs',
        'payloadHostCopyMs', 'payloadMapPendingMs', 'payloadMapBlockedMs', 'payloadCopySubmitMs',
        'payloadFenceWaitMs', 'payloadStageMs', 'payloadChunks', 'payloadLoadMs', 'metadataAndSetupMs',
        'restartPreparationMs', 'restartCacheBytes', 'migrationHotNativeValues', 'migrationRetainedWords'] as const)
        result.profile[key] += child.profile[key];
      result.profile.peakPayloadStagingBytes = Math.max(result.profile.peakPayloadStagingBytes, child.profile.peakPayloadStagingBytes);
      result.profile.peakHostPrefetchBytes = Math.max(result.profile.peakHostPrefetchBytes, child.profile.peakHostPrefetchBytes);
      result.profile.peakResidentBytes = Math.max(result.profile.peakResidentBytes,
        result.profile.residentBytes - child.profile.residentBytes + child.profile.peakResidentBytes);
      result.profile.payloadReadBackend = child.profile.payloadReadBackend;
      result.profile.representation = child.profile.representation;
      result.profile.restartCacheLayout = child.profile.restartCacheLayout;
      result.representation = child.representation;
      status(`${result.loadedAcquisitions}/${count} acquisitions ready${result.loadedAcquisitions < count ? ' · you can interact while loading continues' : ''}`);
      options.onProgress?.(result);
    };
    try { await load(0); }
    catch (error) { result.destroy(); signal?.removeEventListener('abort', abort); throw error; }
    result.profile.readyMs = performance.now() - began;
    result.completion = (async () => {
      try {
        for (let i = 1; i < count; i++) await load(i);
        result.allAcquisitionsReadyMs = performance.now() - began;
      } finally { signal?.removeEventListener('abort', abort); }
    })();
    // Keep a handled completion immediately; the UI may attach after first-ready.
    void result.completion.catch(() => {});
    return result;
  }

  /** Admit the user-granted folder, releasing partial loads on failure. */
  static async loadFiles(device: GPUDevice, files: ArrayLike<File>, status: (text: string) => void = () => { }, signal?: AbortSignal, options: Source112LoadOptions = {}): Promise<Source112ResidentSet> {
    if (options.representation !== undefined && !['tans', 'huffman64'].includes(options.representation)) throw new Error('Select Source112 representation tans or huffman64.');
    const began = performance.now(), granted = fileSource(files);
    const source: RansByteSource = {
      mode: 'local-folder', async read(...args) { signal?.throwIfAborted(); const data = await granted.read(...args); signal?.throwIfAborted(); return data; }
    };
    const manifest: Manifest = JSON.parse(new TextDecoder().decode(await source.read('manifest.json')));
    const T = manifest.shape?.[0];
    if (!Number.isInteger(T) || T < 1 || T > 66 || manifest.format !== 'source112-tans1024-pair-v1' ||
        manifest.dtype !== '<u2' ||
        JSON.stringify(manifest.shape) !== JSON.stringify([T, 512, 512, 192, 192]) ||
        !Array.isArray(manifest.layout?.chunks) ||
        manifest.layout.chunks.length !== T * 16 ||
        !Array.isArray(manifest.layout.files) ||
        !manifest.globals)
      throw new Error('Select a complete native source112 acquisition series.');
    const shardNames = new Set<string>();
    for (const shard of manifest.layout.files) {
      if (!shard || typeof shard.name !== 'string' || !validName(shard.name) || shardNames.has(shard.name) || !Number.isSafeInteger(shard.nbytes) || shard.nbytes <= 0)
        throw new Error('Invalid source112 payload file metadata; re-export the archive.');
      shardNames.add(shard.name);
      const file = Array.from(files).find(file => file.name === shard.name);
      if (!file || file.size !== shard.nbytes)
        throw new Error(`Missing or incomplete ${shard.name}; select the complete exported folder with actual payload files (browser folder grants omit symbolic links).`);
    }
    const records = manifest.layout.chunks;
    const limit = Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize);
    const shardEnds = manifest.layout.files.map(() => 0);
    for (const [i, record] of records.entries()) {
      if (!record || !Array.isArray(record.components))
        throw new Error(`Invalid source112 record ${i}.`);
      const shard = manifest.layout.files[record.shard];
      if (!Number.isSafeInteger(record.shard) || !Number.isSafeInteger(record.file_offset) || record.file_offset % 4 || !shard || !Number.isSafeInteger(shard.nbytes))
        throw new Error(`Invalid byte address in source112 record ${i}.`);
      if (record.components.some(component => !component || typeof component.name !== 'string'))
        throw new Error(`Invalid source112 component in record ${i}.`);
      if (record.components.length !== 4 || new Set(record.components.map(c => c?.name)).size !== 4)
        throw new Error(`Source112 record ${i} requires exactly four distinct source components.`);
      if (record.chunk !== i ||
          record.acquisition !== Math.floor(i / 16) ||
          record.first_scan !== i % 16 * 16384 ||
          record.scan_count !== 16384 ||
          !shard ||
          !validName(shard.name) ||
          !Number.isSafeInteger(record.record_bytes) ||
          record.record_bytes <= 0 ||
          record.record_bytes > Math.min(limit, 256 * 1024 * 1024) ||
          record.record_bytes % 4 ||
          record.file_offset < 0 ||
          record.file_offset + record.record_bytes > shard.nbytes)
        throw new Error(`Invalid native source record ${i}; re-export the preserved archive.`);
      for (const name of ['dense', 'dense_offsets', 'sparse', 'sparse_offsets']) {
        const c = record.components.find(c => c.name === name);
        if (!c ||
            c.dtype !== '<u4' ||
            !Number.isSafeInteger(c.offset) ||
            !Number.isSafeInteger(c.nbytes) ||
            c.nbytes <= 0 ||
            c.offset < 0 ||
            c.offset % 4 ||
            c.nbytes % 4 ||
            c.offset + c.nbytes > record.record_bytes)
          throw new Error(`Invalid ${name} extent in record ${i}.`);
      }
      if (record.file_offset !== shardEnds[record.shard])
        throw new Error(`Overlapping or missing source112 record ${i}.`);
      shardEnds[record.shard] = record.file_offset + record.record_bytes;
      let componentEnd = 0;
      for (const name of ['dense', 'dense_offsets', 'sparse', 'sparse_offsets']) {
        const component = record.components.find(c => c.name === name)!;
        if (component.offset !== Math.ceil(componentEnd / 64) * 64)
          throw new Error(`Overlapping or misplaced source112 component ${name} in record ${i}.`);
        componentEnd = component.offset + component.nbytes;
      }
      if (record.record_bytes !== Math.ceil(componentEnd / 4096) * 4096)
        throw new Error(`Invalid padded record extent in source112 record ${i}.`);
      if (!/^[a-f0-9]{64}$/.test(record.sha256))
        throw new Error(`Invalid source112 record digest ${i}.`);
      for (const [name, bytes] of [['dense_offsets', 628776], ['sparse_offsets', 698332]] as const) {
        if (record.components.find(c => c.name === name)!.nbytes !== bytes)
          throw new Error(`Invalid compact seek extent in record ${i}: ${name}.`);
      }
    }
    if (shardEnds.some((end, index) => end !== manifest.layout.files[index].nbytes))
      throw new Error('Source112 records do not cover their complete payload files.');
    // Authenticate and validate the small metadata before any device allocation.
    const rawGlobals: Record<string, ArrayBuffer> = {};
    const globalTypes: Record<string, {
      dtype: string;
      shape: number[];
      bytes: number;
    }> = {
      dense_columns: { dtype: '<u4', shape: [17466], bytes: 17466 * 4 },
      sparse_columns: { dtype: '<u4', shape: [19398], bytes: 19398 * 4 },
      model_ids: { dtype: '|u1', shape: [T * 4, Q], bytes: T * 4 * Q },
      decoding: { dtype: '<u4', shape: [81, 1024], bytes: 81 * 1024 * 4 },
      hardware: { dtype: '|u1', shape: [Q], bytes: Q },
      valid: { dtype: '|u1', shape: [Q], bytes: Q },
    };
    const globalNames = [
      'dense_columns', 'sparse_columns', 'model_ids', 'decoding', 'hardware', 'valid'
    ];
    for (const name of globalNames) {
      const spec = manifest.globals[name], expected = globalTypes[name];
      if (!spec || typeof spec.file !== 'string' || !validName(spec.file) || shardNames.has(spec.file)
        || spec.dtype !== expected.dtype || JSON.stringify(spec.shape) !== JSON.stringify(expected.shape)
        || spec.nbytes !== expected.bytes || !/^[a-f0-9]{64}$/.test(spec.sha256))
        throw new Error(`Invalid source112 metadata ${name}; re-export the native source.`);
      const raw = await source.read(spec.file);
      if (raw.byteLength !== expected.bytes || await hash(raw) !== spec.sha256)
        throw new Error(`Source112 metadata checksum failed: ${name}.`);
      rawGlobals[name] = raw;
    }
    const columns = [new Uint32Array(rawGlobals.dense_columns), new Uint32Array(rawGlobals.sparse_columns)];
    const seen = new Uint8Array(Q);
    for (const list of columns)
      for (const q of list) {
        if (q >= Q || seen[q]++)
          throw new Error('Dense/sparse columns must partition the native detector exactly.');
      }
    const hardware = new Uint8Array(rawGlobals.hardware), modelIds = new Uint8Array(rawGlobals.model_ids);
    if (hardware.some(value => value > 1) || columns[1].some(q => hardware[q]))
      throw new Error('Hardware counts must use preserved dense literal streams.');
    for (let model = 0;model < T * 4;model++)
      for (const q of columns[0]) {
        const id = modelIds[model * Q + q];
        if ((id >= 81 && id !== 255) || (hardware[q] && id !== 255))
          throw new Error('Invalid source112 model ID or hardware literal stream.');
      }
    let badPx = new Uint32Array(0);
    if (rawGlobals.valid) {
      const valid = new Uint8Array(rawGlobals.valid);
      if (valid.some(value => value > 1))
        throw new Error('Invalid native detector-validity mask.');
      badPx = Uint32Array.from(Array.from(valid, (_, q) => q).filter(q => !valid[q]));
    }
    signal?.throwIfAborted();
    if (options.progressive) {
      return this.loadAcquisitions(device, files, manifest, rawGlobals, status, signal, options, began, badPx);
    }
    // The constructor owns no device resources; all allocations share cleanup.
    const result = new Source112ResidentSet(device, T);
    let scopes = 0;
    try {
      device.pushErrorScope("out-of-memory");
      scopes++;
      device.pushErrorScope("validation");
      scopes++;
      result.outputBuffer = result.buffer(T * N * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
      result.errorBuffer = result.buffer(4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
      result.columns = columns;
      result.badPx = badPx;
      const cols = result.columns.map(a => result.buffer(a.byteLength, GPUBufferUsage.STORAGE, a.buffer as ArrayBuffer));
      const ids = result.buffer(rawGlobals.model_ids.byteLength, GPUBufferUsage.STORAGE, rawGlobals.model_ids);
      const tables = result.buffer(rawGlobals.decoding.byteLength, GPUBufferUsage.STORAGE, rawGlobals.decoding);
      result.selected = result.columns.map(a => result.buffer(a.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST));
      const layout = source112Layout(device);
      result.layout = layout;
      result.globals = [cols[0], cols[1], ids, tables];
      result.pipelines = await Promise.all(['decode_dense', 'decode_sparse'].map(
        entryPoint => source112Pipeline(device, SOURCE112_WGSL, entryPoint)));
      result.denseSumPipeline = await source112Pipeline(device, SOURCE112_SUM_WGSL, 'decode_dense');
      const plans: RecordInfo[][] = [];
      let bytes = limit;
      for (const record of records) {
        if (bytes + record.record_bytes > limit) {
          plans.push([]);
          bytes = 0;
        }
        plans[plans.length - 1].push(record);
        bytes += record.record_bytes;
      }
      const payloadBegan = performance.now();
      result.profile.metadataAndSetupMs = payloadBegan - began;
      const ingress = new Source112Ingress(device, result.profile, bytes => {
        result.profile.payloadStagingBytes = bytes;
        result.profile.peakPayloadStagingBytes = Math.max(result.profile.peakPayloadStagingBytes, bytes);
        result.recordResidentPeak();
      }, signal);
      let ingressFailure = false;
      let reader: Source112WorkerReader | undefined;
      try {
        if (typeof Worker !== 'undefined') {
          reader = new Source112WorkerReader(signal);
          result.profile.payloadReadBackend = 'worker';
        }
        for (const [groupIndex, groupRecords] of plans.entries()) {
          const size = groupRecords.reduce((n, r) => n + r.record_bytes, 0);
          status(`encoded group ${groupIndex + 1}/${plans.length}`);
          const payload = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
          result.owned.push(payload);
          result.profile.residentBytes += size;
          result.recordResidentPeak();
          const desc = new Uint32Array(groupRecords.length * 12);
          let offset = 0;
          let i = 0;
          for await (const { record, raw } of authenticatedRecords(groupRecords, record => {
            const name = manifest.layout.files[record.shard].name;
            return reader
              ? reader.read(granted.blob(name, record.file_offset, record.file_offset + record.record_bytes))
              : source.read(name, record.file_offset, record.file_offset + record.record_bytes);
          }, result.profile, signal)) {
            signal?.throwIfAborted();
            await options.yieldToInteraction?.();
            signal?.throwIfAborted();
            const uploadStarted = performance.now();
            await ingress.write(payload, offset, raw);
            result.profile.payloadStageMs += performance.now() - uploadStarted;
            const component = (name: string) => record.components.find(c => c.name === name)!;
            desc.set(['dense', 'dense_offsets', 'sparse', 'sparse_offsets'].map(n => (offset + component(n).offset) / 4), i * 12);
            desc.set([
              Math.floor(record.chunk / 4) * Q, record.acquisition * N + record.first_scan, component('dense').nbytes / 4, component('sparse').nbytes / 4, 0, 81, 0, 0
            ], i * 12 + 4);
            offset += record.record_bytes;
            result.profile.records++;
            i++;
          }
          signal?.throwIfAborted();
          const descriptors = result.buffer(desc.byteLength, GPUBufferUsage.STORAGE, desc.buffer);
          const params = [0, 1].map(() => result.buffer(64, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST));
          const bind = options.representation === 'huffman64' ? [] : [0, 1].map(kind => device.createBindGroup({
            layout, entries: [
              payload, descriptors, cols[kind], ids, tables, result.selected[kind], result.output, result.errors, params[kind]
            ].map((buffer, binding) => ({ binding, resource: { buffer } }))
          }));
          result.groups.push({ payload, descriptors, records: groupRecords, bind, params });
        }
        await ingress.finish();
      } catch (error) { ingressFailure = true; throw error; }
      finally {
        reader?.dispose();
        try { await ingress.dispose(); }
        catch (error) { if (!ingressFailure) throw error; }
      }
      await device.queue.onSubmittedWorkDone();
      result.profile.payloadLoadMs = performance.now() - payloadBegan;
      signal?.throwIfAborted();
      let error: GPUError | null = null;
      while (scopes) {
        scopes--;
        error = (await device.popErrorScope()) || error;
      }
      if (error)
        throw new Error(`Complete source112 GPU admission failed: ${error.message}`);
      if (options.representation === 'huffman64') {
        await options.yieldToInteraction?.();
        signal?.throwIfAborted();
        await result.prepareHuffman64(new Uint32Array(rawGlobals.decoding), status, signal);
        await options.yieldToInteraction?.();
        signal?.throwIfAborted();
        await result.prepareCompactHuffman(status, signal);
      } else {
        status('Preparing exact GPU restart cache and stream offsets…');
        await result.prepareDenseRestarts((done, total) => {
          status(`Preparing exact GPU restart cache and stream offsets: group ${done}/${total}`);
        }, signal);
        result.setDenseRestartsEnabled(true);
      }
      signal?.throwIfAborted();
      result.profile.readyMs = performance.now() - began;
      return result;
    }
    catch (error) {
      result.destroy();
      while (scopes) {
        scopes--;
        await device.popErrorScope().catch(() => null);
      }
      throw error;
    }
  }
  /** Prepare a complete lossless hybrid generation before loadFiles returns.
   * Any failure closes the partially migrated owner; there is no format fallback. */
  private async prepareHuffman64(decoding: Uint32Array, status: (message: string) => void, signal?: AbortSignal) {
    const state = this, device = this.device;
    const books = source112HuffmanBooks(decoding);
    const options = { cursorBits: 11 as const, signal, status };
    state.check();
    if (state.representation === 'huffman64')
        throw Error('Huffman migration is already complete; reload the original folder before changing its codec.');
    if (state.restartPreparing)
        throw Error('Wait for the existing source preparation to finish.');
    const book = encodingBook(books), bits = options.cursorBits ?? 11;
    if (![11, 12, 13].includes(bits))
        throw Error('Choose a checked cursor width of 11, 12, or 13 bits.');
    const sourceBytes = () => [...new Set(state.owned)].reduce((n, b) => n + b.size, 0);
    // Small series need an explicit overlap allowance: the replacement encoded
    // group coexists with its original until verified. Bound this by one source
    // group plus metadata scratch, independently of the total series length.
    const budget = sourceBytes() + state.groups.reduce((n, g) => n + g.records.length * (48 + 17466 * 32 * 16), 0)
      + Math.max(0, ...state.groups.map(g => g.payload.size)) + 16 * 1024 * 1024;
    if (!Number.isSafeInteger(budget) || budget <= 0)
        throw Error('Supply a finite positive resident allocation budget in bytes.');
    state.preparing = true;
    state.restartPreparing = true;
    const temporary = new Set<GPUBuffer>();
    let scopes = 0, lost = false, completed = false, peak = sourceBytes();
    void device.lost.then(() => { lost = true; });
    const began = performance.now(), reports: Array<Record<string, unknown>> = [];
    const alive = () => { if (state.disposed || lost)
        throw Error('Source closed or device lost during migration; reload the folder.'); options.signal?.throwIfAborted(); };
    const resident = () => sourceBytes() + [...temporary].reduce((n, b) => n + b.size, 0);
    const allocate = (bytes: number, usage: number, raw?: Uint32Array) => {
        alive();
        bytes = Math.max(4, Math.ceil(bytes / 4) * 4);
        if (!Number.isSafeInteger(bytes) || bytes > device.limits.maxBufferSize || ((usage & GPUBufferUsage.STORAGE) && bytes > device.limits.maxStorageBufferBindingSize))
            throw Error(`Migration allocation ${bytes} exceeds this adapter's binding limit.`);
        if (resident() + bytes > budget)
            throw Error(`Migration needs ${resident() + bytes} logical bytes, above budget ${budget}; release other owned resources or reload with adequate headroom.`);
        const buffer = device.createBuffer({ size: bytes, usage });
        temporary.add(buffer);
        peak = Math.max(peak, resident());
        state.profile.peakResidentBytes = Math.max(state.profile.peakResidentBytes ?? 0, peak);
        if (raw)
            device.queue.writeBuffer(buffer, 0, raw.buffer as ArrayBuffer, raw.byteOffset, raw.byteLength);
        return buffer;
    };
    const retire = (buffer: GPUBuffer) => { temporary.delete(buffer); const i = state.owned.indexOf(buffer); if (i >= 0) {
        state.owned.splice(i, 1);
        state.profile.residentBytes -= buffer.size;
    } buffer.destroy(); };
    const adopt = (buffer: GPUBuffer) => { if (!temporary.delete(buffer))
        throw Error('Cannot adopt an unowned migration buffer.'); state.owned.push(buffer); state.profile.residentBytes += buffer.size; };
    const push = () => { device.pushErrorScope('out-of-memory'); device.pushErrorScope('validation'); scopes += 2; };
    const checkErrors = async () => { await device.queue.onSubmittedWorkDone(); alive(); let failure = ''; while (scopes) {
        scopes--;
        const error = await device.popErrorScope();
        if (error)
            failure += error.message + '\n';
    } if (failure)
        throw Error(`GPU migration rejected: ${failure}`); push(); };
    const read = async (buffer: GPUBuffer, words: number) => {
        const out = allocate(words * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
        try {
            const encoder = device.createCommandEncoder();
            encoder.copyBufferToBuffer(buffer, 0, out, 0, words * 4);
            device.queue.submit([encoder.finish()]);
            await out.mapAsync(GPUMapMode.READ);
            alive();
            const data = new Uint32Array(out.getMappedRange()).slice();
            out.unmap();
            return data;
        }
        finally {
            retire(out);
        }
    };
    const makeBind = (g: Group, tables: GPUBuffer) => [0, 1].map(kind => device.createBindGroup({ layout: state.layout, entries: [g.payload, g.descriptors, state.globals[kind], state.globals[2], tables, state.selected[kind], state.output, state.errors, g.params[kind]].map((buffer, binding) => ({ binding, resource: { buffer } })) }));
    try {
        push();
        alive();
        // Compile every final scientific route BEFORE releasing the original cache.
        const layout = source112Layout(device, 5);
        const entries = ['copy_words', 'rank_columns', 'plan', 'prefix_blocks', 'prefix_records', 'finish_starts', 'write_lengths', 'encode', 'copy_sparse', 'verify', 'verify_sparse'];
        const pipes = new Map<string, GPUComputePipeline>();
        for (const entryPoint of entries)
            pipes.set(entryPoint, await source112Pipeline(device, HUFFMAN_MIGRATION_WGSL, entryPoint, 5));
        for (const entryPoint of ['build_checkpoints', 'verify_checkpoints'])
            pipes.set(entryPoint, await source112Pipeline(device, HUFFMAN_CHECKPOINT_WGSL, entryPoint, 5));
        const gather = await source112Pipeline(device, SOURCE112_HUFFMAN64_WGSL, 'decode_dense');
        const sum = await source112Pipeline(device, SOURCE112_HUFFMAN64_SUM_WGSL, 'decode_dense_huffman64');
        await checkErrors();
        await device.queue.onSubmittedWorkDone(); alive();
        const uniform = allocate(32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST), dummy = allocate(4, STORAGE());
        const combined = allocate((ANS_WORDS + 10240) * 4, STORAGE());
        const migrationBind = (old: GPUBuffer, desc: GPUBuffer, meta: GPUBuffer, payload: GPUBuffer, stats: GPUBuffer) => device.createBindGroup({ layout, entries: [old, desc, state.globals[0], state.globals[2], combined, meta, payload, stats, uniform].map((buffer, binding) => ({ binding, resource: { buffer } })) });
        const dispatch = (entry: string, bind: GPUBindGroup, params: Uint32Array, count: number, records = 1) => {
            alive();
            device.queue.writeBuffer(uniform, 0, params.buffer as ArrayBuffer, params.byteOffset, params.byteLength);
            const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
            pass.setPipeline(pipes.get(entry)!);
            pass.setBindGroup(0, bind);
            if (entry === 'rank_columns' || entry === 'prefix_records')
                pass.dispatchWorkgroups(records);
            else {
                const groups = Math.ceil(count / 64);
                const x = Math.min(groups, 32768);
                pass.dispatchWorkgroups(x, Math.ceil(groups / x), records);
            }
            pass.end();
            device.queue.submit([encoder.finish()]);
        };
        const copyModule = device.createShaderModule({ code: `
@group(0) @binding(0) var<storage,read> inputWords:array<u32>;
@group(0) @binding(1) var<storage,read_write> outputWords:array<u32>;
@group(0) @binding(2) var<uniform> sizes:vec4u;
@compute @workgroup_size(64) fn copy(@builtin(global_invocation_id) id:vec3u){if(id.x<sizes.x){outputWords[id.x]=inputWords[id.x];}}` });
        const copyPipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: copyModule, entryPoint: 'copy' } });
        const copyUniform = allocate(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
        const copyRaw = (input: GPUBuffer, output: GPUBuffer, words: number) => {
            device.queue.writeBuffer(copyUniform, 0, new Uint32Array([words, 0, 0, 0]));
            const binding = device.createBindGroup({ layout: copyPipeline.getBindGroupLayout(0), entries: [input, output, copyUniform].map((buffer, binding) => ({ binding, resource: { buffer } })) });
            const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
            pass.setPipeline(copyPipeline);
            pass.setBindGroup(0, binding);
            pass.dispatchWorkgroups(Math.ceil(words / 64));
            pass.end();
            device.queue.submit([encoder.finish()]);
        };
        copyRaw(state.globals[3], combined, ANS_WORDS);
        device.queue.writeBuffer(combined, ANS_WORDS * 4, book.buffer as ArrayBuffer);
        await checkErrors();
        const phase1: Array<{
            group: Group;
            headers: Uint32Array;
            permanentWords: number;
            sparseOffsetWords: number;
            literalStreams: number[];
        }> = [];
        for (const [index, g] of state.groups.entries()) {
            alive();
            options.status?.(`Huffman migration: verify group ${index + 1}/${state.groups.length}`);
            const R = g.records.length;
            if (!R)
                throw Error('Empty source group.');
            const headerCopy = allocate(g.records.length * 48, STORAGE());
            copyRaw(g.descriptors, headerCopy, g.records.length * 12);
            const oldHeaders = await read(headerCopy, g.records.length * 12);
            retire(headerCopy);
            for (let r = 0; r < R; r++)
                if (oldHeaders[r * 12 + 8] !== 0 || oldHeaders[r * 12 + 9] !== 81)
                    throw Error('Private migration requires the canonical 81-model table at offset zero.');
            const permanentWords = R * (20 + C + S), prefixBase = permanentWords;
            const metaInit = new Uint32Array(R * 20);
            metaInit.set(oldHeaders);
            for (let r = 0; r < R; r++) {
                const m = R * 12 + r * 8;
                metaInit[r * 12 + 10] = m;
                metaInit[r * 12 + 11] = R * (20 + C) + r * S;
                metaInit.set([0, R * 20 + r * C, 0, 0, bits, 64, 7 * bits + 5, 1], m);
            }
            const descriptors = allocate((permanentWords + R * C) * 4, STORAGE(), metaInit);
            const stats = allocate((16 + R * 9) * 4, STORAGE());
            const sparseLengths = g.records.map(r => r.components.find(c => c.name === 'sparse_offsets')!.nbytes / 4);
            if (sparseLengths.some(n => n !== sparseLengths[0]))
                throw Error('Unexpected sparse offset layout; keep source unchanged and inspect metadata.');
            const sparseOffsetWords = sparseLengths[0];
            let params = new Uint32Array([R, C, S, 1, prefixBase, sparseOffsetWords, 0, 0]);
            let bind = migrationBind(g.payload, g.descriptors, descriptors, dummy, stats);
            dispatch('rank_columns', bind, params, 1, R);
            dispatch('plan', bind, params, S, R);
            dispatch('prefix_blocks', bind, params, C, R);
            dispatch('prefix_records', bind, params, 1, R);
            await checkErrors();
            let counters = await read(stats, 16 + R * 9);
            if (counters[0])
                throw Error(`Group ${index} plan rejected (${counters[0]}).`);
            const headers = await read(descriptors, R * 20);
            let words = 0;
            for (let r = 0; r < R; r++) {
                const b = r * 12;
                headers[b] = words;
                words += headers[b + 6];
                headers[b + 1] = words;
                words += C + S / 4;
                headers[b + 2] = words;
                words += headers[b + 7];
                headers[b + 3] = words;
                words += sparseOffsetWords;
            }
            const payload = allocate(words * 4, STORAGE());
            device.queue.writeBuffer(descriptors, 0, headers.buffer as ArrayBuffer);
            params = new Uint32Array([R, C, S, words, prefixBase, sparseOffsetWords, 0, 0]);
            bind = migrationBind(g.payload, g.descriptors, descriptors, payload, stats);
            dispatch('finish_starts', bind, params, S, R);
            dispatch('write_lengths', bind, params, S / 4, R);
            dispatch('encode', bind, params, S, R);
            const sparseWords = Math.max(sparseOffsetWords, ...Array.from({ length: R }, (_, r) => headers[r * 12 + 7]));
            dispatch('copy_sparse', bind, params, sparseWords, R);
            dispatch('verify', bind, params, S, R);
            dispatch('verify_sparse', bind, params, sparseWords, R);
            await checkErrors();
            counters = await read(stats, 16 + R * 9);
            if (counters[0])
                throw Error(`Group ${index} native/copy gate rejected (${counters[0]}).`);
            for (let r = 0; r < R; r++) {
                const k = 16 + r * 9;
                if (counters[k + 1] !== counters[k] * 512 || counters[k + 2] !== counters[k + 3] || counters[k + 4] !== headers[r * 12 + 7] || counters[k + 5] !== sparseOffsetWords)
                    throw Error(`Group ${index}, record ${r}: incomplete native or byte-copy coverage.`);
            }
            reports.push({ group: index, records: R, oldPayloadBytes: g.payload.size, newPayloadBytes: payload.size, hotNativeValues: Array.from({ length: R }, (_, r) => counters[17 + r * 9]).reduce((a, b) => a + b, 0), retainedWords: Array.from({ length: R }, (_, r) => counters[19 + r * 9]).reduce((a, b) => a + b, 0) });
            // Queue and GPU scopes have completed. No caller can submit through the source.
            const oldPayload = g.payload, oldDescriptors = g.descriptors;
            g.payload = payload;
            g.descriptors = descriptors;
            g.bind = [];
            adopt(payload);
            adopt(descriptors);
            retire(oldPayload);
            retire(oldDescriptors);
            retire(stats);
            phase1.push({ group: g, headers, permanentWords, sparseOffsetWords, literalStreams: Array.from({ length: R }, (_, r) => counters[24 + r * 9]) });
        }
        let cacheBytes = 0;
        for (const [index, item] of phase1.entries()) {
            alive();
            options.status?.(`Huffman migration: checked checkpoints ${index + 1}/${phase1.length}`);
            const { group: g, permanentWords, sparseOffsetWords } = item, R = g.records.length;
            const headers = item.headers.slice();
            let words = permanentWords;
            for (let r = 0; r < R; r++) {
                const m = headers[r * 12 + 10], hotColumns = headers[m];
                const hotBits = hotColumns * 32 * (7 * bits + 5);
                if (hotBits >= 2 ** 32)
                    throw Error('Packed checkpoint bit addressing exceeds u32.');
                headers[m + 2] = words;
                words += Math.ceil(hotBits / 32);
                headers[m + 3] = words;
                words += (C - hotColumns) * 32 * 3;
            }
            const descriptors = allocate(words * 4, STORAGE());
            const encoder = device.createCommandEncoder();
            encoder.copyBufferToBuffer(g.descriptors, 0, descriptors, 0, permanentWords * 4);
            device.queue.submit([encoder.finish()]);
            device.queue.writeBuffer(descriptors, 0, headers.buffer as ArrayBuffer);
            const stats = allocate((16 + R * 9) * 4, STORAGE());
            const bind = migrationBind(g.payload, g.descriptors, descriptors, dummy, stats);
            const params = new Uint32Array([R, C, S, g.payload.size / 4, 0, sparseOffsetWords, 1, 0]);
            dispatch('build_checkpoints', bind, params, S, R);
            dispatch('verify_checkpoints', bind, params, S, R);
            await checkErrors();
            const counters = await read(stats, 16 + R * 9);
            if (counters[0])
                throw Error(`Group ${index} checkpoint gate rejected (${counters[0]}); no mixed generation was exposed.`);
            const literalStreams = item.literalStreams;
            for (let r = 0; r < R; r++)
                if (counters[22 + r * 9] !== S - literalStreams[r] || counters[23 + r * 9] !== S - literalStreams[r])
                    throw Error(`Group ${index}, record ${r}: incomplete checkpoint coverage.`);
            cacheBytes += descriptors.size;
            const old = g.descriptors;
            g.descriptors = descriptors;
            adopt(descriptors);
            retire(old);
            retire(stats);
        }
        // Final table has the LUT immediately after the original ANS table. Migration
        // uses a separate book-first table, which is never installed as a decoder table.
        const finalTables = allocate((ANS_WORDS + 2048) * 4, STORAGE());
        copyRaw(state.globals[3], finalTables, ANS_WORDS);
        device.queue.writeBuffer(finalTables, ANS_WORDS * 4, book.buffer as ArrayBuffer, 8192 * 4, 2048 * 4);
        await checkErrors();
        const newBinds = state.groups.map(g => makeBind(g, finalTables));
        await checkErrors();
        while (scopes) {
            scopes--;
            const error = await device.popErrorScope();
            if (error)
                throw Error(`Final decoder admission failed: ${error.message}`);
        }
        alive();
        const oldTables = state.globals[3];
        state.globals[3] = finalTables;
        adopt(finalTables);
        retire(oldTables);
        for (let i = 0; i < state.groups.length; i++)
            state.groups[i].bind = newBinds[i];
        state.pipelines[0] = gather;
        state.restartPipeline = sum;
        state.restartEnabled = true;
        state.previousMask = null;
        state.restartProfile = { offsetBytes: state.groups.reduce((n, g) => n + g.records.length * 17466 * 32 * 4, 0), checkpointBytes: cacheBytes - state.groups.reduce((n, g) => n + g.records.length * (20 * 4 + C * 4 + S * 4), 0), additionalBytes: cacheBytes, groups: state.groups.length, streams: state.groups.reduce((n, g) => n + g.records.length * 17466 * 32, 0), segmentValues: 64, preparationMs: performance.now() - began };
        state.profile.restartCacheBytes = cacheBytes;
        state.profile.restartPreparationMs = performance.now() - began;
        state.representation = 'huffman64';
        state.profile.representation = 'huffman64';
        state.profile.restartCacheLayout = 'huffman64-offset';
        state.profile.migrationHotNativeValues = reports.reduce((n, r) => n + Number(r.hotNativeValues), 0);
        state.profile.migrationRetainedWords = reports.reduce((n, r) => n + Number(r.retainedWords), 0);
        state.profile.peakResidentBytes = Math.max(state.profile.peakResidentBytes ?? 0, peak, state.profile.residentBytes);
        state.restartPreparing = false;
        state.preparing = false;
        completed = true;
        return;
    }
    catch (error) {
        state.destroy();
        throw error;
    }
    finally {
        if (!completed) {
            try {
                await device.queue.onSubmittedWorkDone();
            }
            catch { /* Device loss still requires owned-buffer cleanup. */ }
        }
        while (scopes) {
            scopes--;
            try {
                await device.popErrorScope();
            }
            catch { /* Preserve the original error. */ }
        }
        for (const buffer of temporary)
            buffer.destroy();
        temporary.clear();
        if (!completed) {
            state.restartPreparing = false;
            state.preparing = false;
        }
    }
}

  /** Allocate and build an optional exact dense-stream restart cache on the GPU.
   * Original payloads and canonical sums remain unchanged. Failed preparation
   * leaves all original descriptor bindings active and releases new allocations.
   */
  async prepareDenseRestarts(status: (done: number, total: number) => void = () => {}, signal?: AbortSignal) {
    if (this.representation === 'huffman64') throw new Error('Reload the folder to change a Huffman64 source representation.');
    this.check();
    if (this.restartProfile) return this.restartProfile;
    if (this.restartPreparing) throw new Error('Dense restart preparation is already running.');
    const device = this.device, denseCount = this.columns[0].length;
    const limit = Math.min(device.limits.maxBufferSize, device.limits.maxStorageBufferBindingSize);
    const plans = this.groups.map(group => ({ group, bytes: group.records.length * (12 + 32 * denseCount * 4) * 4 }));
    if (plans.some(plan => plan.bytes > limit)) throw new Error('Dense restart descriptor cache exceeds this device binding limit.');
    this.restartPreparing = true;
    const started = performance.now();
    const replacements: { group: Group; descriptors: GPUBuffer; bind: GPUBindGroup[] }[] = [];
    const temporary: GPUBuffer[] = [];
    const allocated: GPUBuffer[] = [];
    let scopes = 0, committed = false;
    const openScopes = () => { device.pushErrorScope('out-of-memory'); device.pushErrorScope('validation'); scopes += 2; };
    const closeScopes = async () => {
      const errors: string[] = [];
      for (let i = 0; i < 2; i++) { const error = await device.popErrorScope(); scopes--; if (error) errors.push(error.message); }
      if (errors.length) throw new Error(errors.join('\n'));
    };
    try {
      await device.queue.onSubmittedWorkDone(); this.check(); signal?.throwIfAborted();
      openScopes();
      const buildLayout = device.createBindGroupLayout({ entries: Array.from({ length: 9 }, (_, binding) => ({
        binding, visibility: GPUShaderStage.COMPUTE,
        buffer: { type: binding === 8 ? 'uniform' : binding === 1 || binding >= 6 ? 'storage' : 'read-only-storage' },
      })) as GPUBindGroupLayoutEntry[] });
      const buildModule = device.createShaderModule({ code: SOURCE112_OFFSET_RESTART_BUILD_WGSL });
      const decodeModule = device.createShaderModule({ code: SOURCE112_OFFSET_RESTART_WGSL.split('p.mode').join('0u') });
      for (const module of [buildModule, decodeModule]) {
        const errors = (await module.getCompilationInfo()).messages.filter(message => message.type === 'error');
        if (errors.length) throw new Error(errors.map(error => error.message).join('\n'));
      }
      const buildPipeline = await device.createComputePipelineAsync({
        layout: device.createPipelineLayout({ bindGroupLayouts: [buildLayout] }),
        compute: { module: buildModule, entryPoint: 'build_dense_restarts' },
      });
      const decodePipeline = await device.createComputePipelineAsync({
        layout: device.createPipelineLayout({ bindGroupLayouts: [this.layout] }),
        compute: { module: decodeModule, entryPoint: 'decode_dense_restart' },
      });
      const params = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const errors = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
      const readErrors = device.createBuffer({ size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      temporary.push(params, errors, readErrors);
      this.recordResidentPeak(temporary.reduce((n, buffer) => n + buffer.size, 0));
      await closeScopes();
      const [dense, sparse, ids, tables] = this.globals;
      for (let index = 0; index < plans.length; index++) {
        this.check(); signal?.throwIfAborted();
        const { group, bytes } = plans[index];
        openScopes();
        const descriptors = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        allocated.push(descriptors);
        this.recordResidentPeak([...temporary, ...allocated].reduce((n, buffer) => n + buffer.size, 0));
        // Host-side metadata only; no stream payload is read or decoded here.
        const words = new Uint32Array(group.records.length * 12);
        let offset = 0;
        group.records.forEach((record, i) => {
          const component = (name: string) => record.components.find(c => c.name === name)!;
          words.set(['dense', 'dense_offsets', 'sparse', 'sparse_offsets'].map(name => (offset + component(name).offset) / 4), i * 12);
          words.set([Math.floor(record.chunk / 4) * Q, record.acquisition * N + record.first_scan,
            component('dense').nbytes / 4, component('sparse').nbytes / 4, 0, 81,
            words.length + i * 32 * denseCount * 4, words.length + i * 32 * denseCount * 4 + 32 * denseCount * 3], i * 12 + 4);
          offset += record.record_bytes;
        });
        device.queue.writeBuffer(descriptors, 0, words);
        device.queue.writeBuffer(params, 0, new Uint32Array([0, group.records.length, denseCount, denseCount, denseCount, 0, 0, Q, 0, Q, 0, 0, 0, 0, 0, 0]));
        const makeBind = (layout: GPUBindGroupLayout, kind: number, uniform: GPUBuffer, errorBuffer: GPUBuffer) => device.createBindGroup({
          layout, entries: [group.payload, descriptors, kind === 0 ? dense : sparse, ids, tables, this.selected[kind], this.output, errorBuffer, uniform]
            .map((buffer, binding) => ({ binding, resource: { buffer } })),
        });
        const buildBind = makeBind(buildLayout, 0, params, errors);
        const bind = [0, 1].map(kind => makeBind(this.layout, kind, group.params[kind], this.errors));
        const encoder = device.createCommandEncoder();
        encoder.clearBuffer(errors);
        const pass = encoder.beginComputePass(); pass.setPipeline(buildPipeline); pass.setBindGroup(0, buildBind);
        pass.dispatchWorkgroups(Math.ceil(denseCount / 64), 32, group.records.length); pass.end();
        encoder.copyBufferToBuffer(errors, 0, readErrors, 0, 4);
        device.queue.submit([encoder.finish()]);
        // Bound preprocessing to one group at a time, including error readback.
        await device.queue.onSubmittedWorkDone();
        await closeScopes();
        await readErrors.mapAsync(GPUMapMode.READ);
        const errorBits = new Uint32Array(readErrors.getMappedRange())[0]; readErrors.unmap();
        if (errorBits) throw new Error(`Dense restart preparation rejected group ${index} (decoder status ${errorBits}).`);
        this.check(); signal?.throwIfAborted();
        replacements.push({ group, descriptors, bind });
        status(index + 1, plans.length);
      }
      this.check(); signal?.throwIfAborted();
      this.restartOriginals = replacements.map(({ group }) => ({ group, descriptors: group.descriptors, bind: group.bind }));
      for (const replacement of replacements) {
        replacement.group.descriptors = replacement.descriptors;
        replacement.group.bind = replacement.bind;
        this.owned.push(replacement.descriptors);
        this.profile.residentBytes += replacement.descriptors.size;
        this.recordResidentPeak();
      }
      this.restartPipeline = decodePipeline;
      this.restartEnabled = false;
      this.restartProfile = {
        offsetBytes: plans.reduce((sum, plan) => sum + plan.group.records.length * 32 * denseCount * 4, 0),
        checkpointBytes: plans.reduce((sum, plan) => sum + plan.group.records.length * 32 * denseCount * 3 * 4, 0),
        additionalBytes: plans.reduce((sum, plan) => sum + plan.bytes, 0),
        groups: plans.length, streams: this.groups.reduce((sum, group) => sum + group.records.length * 32 * denseCount, 0),
        segmentValues: 128, preparationMs: performance.now() - started };
      this.profile.restartPreparationMs = this.restartProfile.preparationMs;
      this.profile.restartCacheBytes = this.restartProfile.additionalBytes;
      this.profile.restartCacheLayout = 'tans128-offset';
      committed = true;
      return this.restartProfile;
    } finally {
      // Drain before releasing failed or temporary GPU resources.
      await device.queue.onSubmittedWorkDone().catch(() => {});
      while (scopes) { scopes--; await device.popErrorScope().catch(() => null); }
      temporary.forEach(buffer => buffer.destroy());
      if (!committed) allocated.forEach(buffer => buffer.destroy());
      this.restartPreparing = false;
    }
  }
  /** Compact verified Huffman checkpoints before the complete source is exposed.
   * Encoded payload is unchanged. A failed partial metadata migration closes the owner.
   * 64-value segments leave display headroom while preserving every source count.
   * The 2 GiB logical bound covers conversion scratch; GPU scopes
   * enforce actual allocation success without assuming driver memory retirement. */
  private async prepareCompactHuffman(status: (message: string) => void, signal?: AbortSignal) {
      const device = this.device, priorPreparationMs = this.profile.restartPreparationMs;
      const options = { signal, maxAdditionalBytes: 2 * 1024 ** 3 };
      this.check();
      options.signal?.throwIfAborted();
      if (this.storageRepresentation !== 'huffman64' || this.restartPreparing || this.restartProfile?.segmentValues !== 64 || this.profile.restartCacheLayout === 'huffman64-compact')
          throw Error('Load an unmodified production Huffman64 source first.');
      const groups = this.groups.slice(), payloads = groups.map(g => g.payload), originals = groups.map(g => g.descriptors), initial = [...new Set(this.owned)].reduce((n, b) => n + b.size, 0), budget = initial + options.maxAdditionalBytes;
      const temporary = new Set<GPUBuffer>();
      let retiredAny = false, complete = false, lost = false, scopes = 0, peak = initial;
      const began = performance.now();
      void device.lost.then(() => { lost = true; });
      const alive = () => { if (lost || this.disposed || groups.length !== this.groups.length || groups.some((g, i) => this.groups[i] !== g || g.payload !== payloads[i]))
          throw Error('Source closed or payload generation changed; reload the folder.'); options.signal?.throwIfAborted(); };
      const resident = () => [...new Set(this.owned)].reduce((n, b) => n + b.size, 0) + [...temporary].reduce((n, b) => n + b.size, 0);
      const allocate = (bytes: number, usage: number, data?: Uint32Array) => { alive(); bytes = Math.max(4, Math.ceil(bytes / 4) * 4); if (!Number.isSafeInteger(bytes) || bytes > device.limits.maxBufferSize || ((usage & GPUBufferUsage.STORAGE) && bytes > device.limits.maxStorageBufferBindingSize) || resident() + bytes > budget)
          throw Error(`Compact cache admission exceeds logical admission budget: ${resident() + bytes} > ${budget}.`); const b = device.createBuffer({ size: bytes, usage }); temporary.add(b); peak = Math.max(peak, resident()); this.profile.peakResidentBytes = Math.max(this.profile.peakResidentBytes, peak); if (data)
          device.queue.writeBuffer(b, 0, data.buffer as ArrayBuffer, data.byteOffset, data.byteLength); return b; };
      const retire = (b: GPUBuffer) => { temporary.delete(b); const i = this.owned.indexOf(b); if (i >= 0) {
          this.owned.splice(i, 1);
          this.profile.residentBytes -= b.size;
      } b.destroy(); };
      const adopt = (b: GPUBuffer) => { if (!temporary.delete(b))
          throw Error('Candidate metadata is not owned.'); this.owned.push(b); this.profile.residentBytes += b.size; };
      const push = () => { device.pushErrorScope('out-of-memory'); device.pushErrorScope('validation'); scopes += 2; };
      const drain = async () => { await device.queue.onSubmittedWorkDone(); alive(); let failure = ''; while (scopes) {
          scopes--;
          const e = await device.popErrorScope();
          if (e)
              failure += e.message + '\n';
      } if (failure)
          throw Error(failure); };
      const read = async (b: GPUBuffer, bytes: number) => { const out = allocate(bytes, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ); try {
          const e = device.createCommandEncoder();
          e.copyBufferToBuffer(b, 0, out, 0, bytes);
          device.queue.submit([e.finish()]);
          await out.mapAsync(GPUMapMode.READ);
          alive();
          try {
              return new Uint32Array(out.getMappedRange()).slice();
          }
          finally {
              out.unmap();
          }
      }
      finally {
          retire(out);
      } };
      const compile = async (code: string, entryPoint: string, firstWritable: 5 | 6 = 6) => {
          const pipeline = await source112Pipeline(device, code, entryPoint, firstWritable);
          alive();
          return pipeline;
      };
      this.preparing = true;
      this.restartPreparing = true;
      push();
      try {
          const plans = [];
          let finalDelta = 0, requiredAdditional = 0, records = 0;
          for (let i = 0; i < groups.length; i++) {
              const g = groups[i], n = g.records.length, old = await read(originals[i], n * 80), p = compactGroupLayout(old, n, 64);
              requiredAdditional = Math.max(requiredAdditional, finalDelta + p.bytes + 256);
              finalDelta += p.bytes - originals[i].size;
              records += n;
              plans.push(p);
          }
          if (requiredAdditional > options.maxAdditionalBytes)
              throw Error(`Compact64 needs ${requiredAdditional} additional logical bytes before old-cache retirement; allowed ${options.maxAdditionalBytes}.`);
          const code = compactHuffmanShader(64), sum = await compile(code.replace(/\bp\.mode\b/g, '0u'), 'decode_dense_huffman64'), gather = await compile(code, 'decode_dense'), sparse = await compile(code, 'decode_sparse');
          const buildLayout = source112Layout(device, 5);
          const build = await compile(COMPACT_BUILD_WGSL, 'build_compact', 5), counter = allocate(16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST), errors = allocate(4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST), uniform = allocate(64, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
          for (let i = 0; i < groups.length; i++) {
              alive();
              const g = groups[i];
              if (g.descriptors !== originals[i])
                  throw Error('Cache generation changed during admission.');
              status(`Preparing compact Huffman checkpoints: group ${i + 1}/${groups.length}`);
              alive();
              const candidate = allocate(plans[i].bytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, plans[i].headers);
              const bind = device.createBindGroup({ layout: buildLayout, entries: [g.payload, originals[i], this.globals[0], this.globals[2], this.globals[3], candidate, counter, errors, uniform].map((buffer, binding) => ({ binding, resource: { buffer } })) });
              let counts = new Uint32Array(4);
              for (let verify = 0; verify < 2; verify++) {
                  device.queue.writeBuffer(uniform, 0, new Uint32Array([0, g.records.length, 17466, 17466, 0, 0, 0, Q, 0, Q, 0, 64, verify, 0, 0, 0]));
                  const e = device.createCommandEncoder();
                  e.clearBuffer(counter);
                  e.clearBuffer(errors);
                  const pass = e.beginComputePass();
                  pass.setPipeline(build);
                  pass.setBindGroup(0, bind);
                  pass.dispatchWorkgroups(Math.ceil(17466 * 32 / 64), 1, g.records.length);
                  pass.end();
                  device.queue.submit([e.finish()]);
                  // An invalid command buffer can leave both counters and error flags zero.
                  // Surface scoped GPU validation before interpreting those values as a decoder failure.
                  await drain();
                  push();
                  counts = await read(counter, 16);
                  const error = await read(errors, 4);
                  await drain();
                  push();
                  if (error[0] || counts[0] !== g.records.length * (17466 * 32) || counts[1] + counts[2] + counts[3] !== counts[0])
                      throw Error(`Group ${i} compact verification ${verify} failed: ${error[0]}, ${counts}.`);
              }
              await drain();
              alive();
              push();
              // Only verified metadata retires its original; science remains blocked until every group is ready.
              g.descriptors = candidate;
              adopt(candidate);
              retiredAny = true;
              retire(originals[i]);
              this.profile.restartCacheBytes += candidate.size - originals[i].size;
          }
          const binds = groups.map(g => [0, 1].map(k => device.createBindGroup({ layout: this.layout, entries: [g.payload, g.descriptors, this.globals[k], this.globals[2], this.globals[3], this.selected[k], this.output, this.errors, g.params[k]].map((buffer, binding) => ({ binding, resource: { buffer } })) })));
          retire(counter);
          retire(errors);
          retire(uniform);
          await drain();
          alive();
          const cacheBytes = plans.reduce((n, p) => n + p.bytes, 0), offsetBytes = records * ((17466 * 32) * 2 + 17466 * 4), checkpointBytes = cacheBytes - records * (80 + 17466 * 4) - offsetBytes;
          // Publication is synchronous: no caller can observe mixed readers and metadata between awaits.
          groups.forEach((g, i) => { g.bind = binds[i]; });
          this.pipelines[0] = gather;
          this.pipelines[1] = sparse;
          this.restartPipeline = sum;
          this.restartEnabled = true;
          this.previousMask = null;
          this.restartProfile = { offsetBytes, checkpointBytes, additionalBytes: cacheBytes, groups: groups.length, streams: records * (17466 * 32), segmentValues: 64, preparationMs: priorPreparationMs + performance.now() - began };
          this.profile.restartCacheBytes = cacheBytes;
          this.profile.restartPreparationMs = priorPreparationMs + performance.now() - began;
          this.preparing = false;
          this.restartPreparing = false;
          complete = true;
          this.profile.restartCacheLayout = 'huffman64-compact';
          return;
      }
      catch (error) {
          if (retiredAny)
              this.destroy();
          throw error;
      }
      finally {
          if (!complete) {
              try {
                  await device.queue.onSubmittedWorkDone();
              }
              catch { }
          }
          while (scopes) {
              scopes--;
              await device.popErrorScope().catch(() => null);
          }
          for (const b of temporary)
              b.destroy();
          temporary.clear();
          if (!complete) {
              this.preparing = false;
              this.restartPreparing = false;
          }
      }
  }

  /** Diagnostic opt-in: preparation does not change the active decode path. */
  setDenseRestartsEnabled(enabled: boolean) {
    if (this.representation === 'huffman64') throw new Error('Reload the folder to change a Huffman64 source representation.');
    this.check();
    if (enabled && !this.restartPipeline) throw new Error('Prepare dense restarts before enabling them.');
    this.restartEnabled = enabled;
  }
  /** Release only the optional cache, restoring the original descriptor bindings. */
  async releaseDenseRestarts() {
    if (this.representation === 'huffman64') throw new Error('Reload the folder to change a Huffman64 source representation.');
    this.check();
    if (this.restartPreparing) throw new Error('Wait for dense restart preparation before releasing it.');
    this.restartEnabled = false;
    await this.device.queue.onSubmittedWorkDone(); this.check();
    for (const original of this.restartOriginals ?? []) {
      const cache = original.group.descriptors;
      original.group.descriptors = original.descriptors; original.group.bind = original.bind;
      const index = this.owned.indexOf(cache); if (index >= 0) this.owned.splice(index, 1);
      this.profile.residentBytes -= cache.size; cache.destroy();
    }
    this.restartOriginals = undefined; this.restartPipeline = undefined; this.restartProfile = undefined;
    this.profile.restartCacheBytes = 0;
    this.profile.restartCacheLayout = 'none';
  }
  denseRestartStatus(): {enabled: boolean; profile: Source112ResidentSet["restartProfile"] | null} { if (this.partitions) return { enabled: this.partitions.every(child => child.denseRestartStatus().enabled), profile: null }; return { enabled: !!this.restartEnabled, profile: this.restartProfile ?? null }; }

  /** Submit an exact full or incremental all66 detector image update. */
  integrate(mask: Uint32Array): {
    added: number;
    removed: number;
    full: boolean;
  } {
    this.check();
    if (this.partitions) {
      let delta = {added: 0, removed: 0, full: false};
      this.partitions.forEach((child, i) => { const result = child.integrate(mask); if (i === 0) delta = result; });
      this.previousMask = mask.slice();
      return delta;
    }
    if (mask.length !== Q)
      throw new Error(`Detector mask must contain ${Q} native pixels.`);
    const full = !this.previousMask, lists: number[][] = [[], []];
    let added = 0, removed = 0;
    for (let kind = 0;kind < 2;kind++)
      for (let rank = 0;rank < this.columns[kind].length;rank++) {
        const q = this.columns[kind][rank], now = !!mask[q], before = !!this.previousMask?.[q];
        if (now && !before) {
          lists[kind].push(rank);
          added++;
        }
        else if (!now && before) {
          lists[kind].push(rank | 0x1000000);
          removed++;
        }
      }
    const encoder = this.device.createCommandEncoder();
    if (full)
      encoder.clearBuffer(this.output);
    for (let kind = 0;kind < 2;kind++)
      if (lists[kind].length) {
        this.device.queue.writeBuffer(this.selected[kind], 0, new Uint32Array(lists[kind]));
        for (const group of this.groups) {
          const params = new Uint32Array([
            0, group.records.length, this.columns[kind].length, kind === 0 ? 17466 : 19399, lists[kind].length, 0, 0, Q, 0, Q, 0, 0, 0, 0, 0, 0
          ]);
          this.device.queue.writeBuffer(group.params[kind], 0, params);
          const pass = encoder.beginComputePass();
          const restarts = kind === 0 && this.restartEnabled;
          pass.setPipeline(restarts ? this.restartPipeline! : kind === 0 ? this.denseSumPipeline : this.pipelines[kind]);
          pass.setBindGroup(0, group.bind[kind]);
          pass.dispatchWorkgroups(Math.ceil(lists[kind].length / (restarts ? 16 : 64)), 32, group.records.length);
          pass.end();
        }
      }
    this.device.queue.submit([encoder.finish()]);
    this.previousMask = mask.slice();
    return { added, removed, full };
  }
  applyDelta(add: Uint32Array, sub: Uint32Array) {
    if (!this.previousMask)
      throw new Error('Source112 delta requires an initial full detector mask.');
    const next = this.previousMask.slice();
    for (let q = 0;q < Q;q++) {
      if (add[q])
        next[q] = 1;
      if (sub[q])
        next[q] = 0;
    }
    return this.integrate(next);
  }
  private convertDisplayIndices(indices: number[], area: number): GPUBuffer[] {
    this.check();
    for (const index of indices)
      if (!Number.isInteger(index) || index < 0 || index >= this.acquisitionCount)
        throw new Error('Invalid acquisition index.');
    if (!indices.length)
      return [];
    if (!this.convertPipeline)
      this.convertPipeline = this.device.createComputePipeline({
        layout: 'auto', compute: {
          module: this.device.createShaderModule({
            code: `
struct P { base:u32, n:u32, area:f32, pad:u32 }
@group(0) @binding(0) var<storage,read> counts:array<u32>;
@group(0) @binding(1) var<storage,read_write> display:array<f32>;
@group(0) @binding(2) var<uniform> p:P;
@compute @workgroup_size(256) fn convert(@builtin(global_invocation_id) g:vec3u){if(g.x<p.n){display[g.x]=f32(counts[p.base+g.x])/p.area;}}`
          }), entryPoint: 'convert'
        }
      });
    // Lazy initialization also permits updating an already resident instance.
    if (!this.convertUniforms) {
      const alignment = this.device.limits.minUniformBufferOffsetAlignment;
      this.convertUniformStride = Math.ceil(16 / alignment) * alignment;
      this.convertUniformData = new ArrayBuffer(this.acquisitionCount * this.convertUniformStride);
      this.convertUniforms = this.buffer(this.convertUniformData.byteLength, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
      this.convertBindings = new Map();
    }
    const stride = this.convertUniformStride!;
    const data = this.convertUniformData!;
    const words = new Uint32Array(data), floats = new Float32Array(data);
    const buffers = indices.map(index => {
      let display = this.displayBuffers.get(index);
      if (!display) {
        display = this.buffer(N * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
        this.displayBuffers.set(index, display);
      }
      if (!this.convertBindings!.has(index))
        this.convertBindings!.set(index, this.device.createBindGroup({
          layout: this.convertPipeline!.getBindGroupLayout(0), entries: [
            { binding: 0, resource: { buffer: this.output } },
            { binding: 1, resource: { buffer: display } },
            { binding: 2, resource: { buffer: this.convertUniforms!, offset: index * stride, size: 16 } },
          ],
        }));
      const word = index * stride / 4;
      words[word] = index * N;
      words[word + 1] = N;
      floats[word + 2] = area;
      return display;
    });
    // Queue order separates consecutive conversions even when they share a slab.
    this.device.queue.writeBuffer(this.convertUniforms, 0, data);
    const encoder = this.device.createCommandEncoder(), pass = encoder.beginComputePass();
    pass.setPipeline(this.convertPipeline);
    for (const index of indices) {
      pass.setBindGroup(0, this.convertBindings!.get(index)!);
      pass.dispatchWorkgroups(N / 256);
    }
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    return buffers;
  }
  /** Borrow native count images for immediate GPU display without conversion.
   * The source retains ownership; consumers must neither write nor destroy them.
   */
  imageViewsU32(indices: number[], divisor: number): Uint32ImageView[] {
    this.check();
    if (this.partitions) return indices.map(index => this.loaded(index).imageViewsU32([0], divisor)[0]);
    if (!Number.isFinite(divisor) || !Number.isFinite(Math.fround(divisor)) || Math.fround(divisor) <= 0)
      throw new Error('Image mean divisor must be positive and finite in float32.');
    if (indices.some(index => !Number.isInteger(index) || index < 0 || index >= this.acquisitionCount))
      throw new Error('Select native acquisition indices within the loaded series.');
    return indices.map(index => ({ device: this.device, buffer: this.output,
      byteOffset: index * N * 4, count: N, divisor }));
  }
  imageBuffersF32(indices: number[]): GPUBuffer[] {
    if (this.partitions) return indices.map(index => {
      const buffer = this.loaded(index).imageBuffersF32([0])[0];
      this.displayBuffers.set(index, buffer); return buffer;
    });
    return this.convertDisplayIndices(indices, 1);
  }
  normalizeDisplayBuffers(buffers: GPUBuffer[], area: number) {
    // Always divide authoritative counts, never an already normalized display.
    this.check();
    if (this.partitions) {
      for (const buffer of buffers) {
        const index = [...this.displayBuffers].find(([, b]) => b === buffer)?.[0];
        if (index === undefined) throw Error('Only owned display copies can be normalized.');
        this.loaded(index).normalizeDisplayBuffers([buffer], area);
      }
      return;
    }
    if (!Number.isFinite(area) || area <= 0)
      throw new Error('Detector area must be positive.');
    const indices = buffers.map(buffer => {
      const entry = [...this.displayBuffers].find(([, value]) => value === buffer);
      if (!entry)
        throw new Error('Only owned display copies can be normalized.');
      return entry[0];
    });
    this.convertDisplayIndices(indices, area);
  }
  async readImage(i: number) { return Float32Array.from(await this.readImageU32(i)); }
  private encodePattern(encoder: GPUCommandEncoder, acquisition: number, scan: number, out: GPUBuffer) {
    this.check();
    if (!Number.isInteger(acquisition) || acquisition < 0 || acquisition >= this.acquisitionCount
      || !Number.isInteger(scan) || scan < 0 || scan >= N)
      throw new Error('Select a native acquisition and scan position.');
    const chunk = acquisition * 16 + Math.floor(scan / 16384), group = this.groups.find(g => g.records.some(r => r.chunk === chunk))!, recordFirst = group.records.findIndex(r => r.chunk === chunk);
    encoder.clearBuffer(out);
    for (let kind = 0;kind < 2;kind++) {
      const count = this.columns[kind].length;
      this.device.queue.writeBuffer(group.params[kind], 0, new Uint32Array([
        recordFirst, 1, count, kind === 0 ? 17466 : 19399, count, 1, scan % 16384, Q, 0, Q, 0, 0, 0, 0, 0, 0
      ]));
      const [dense, sparse, ids, tables] = this.globals;
      const bind = this.device.createBindGroup({
        layout: this.layout, entries: [
          group.payload, group.descriptors, kind === 0 ? dense : sparse, ids, tables, this.selected[kind], out, this.errors, group.params[kind]
        ].map((buffer, binding) => ({ binding, resource: { buffer } }))
      });
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.pipelines[kind]);
      pass.setBindGroup(0, bind);
      pass.dispatchWorkgroups(Math.ceil(count / 64), 1, 1);
      pass.end();
    }
  }
  /** Borrow the newest validity-filtered float32 pattern on this device.
   * Each acquisition has a separate reusable output; submit its display before
   * requesting the next scan position for that acquisition.
   * The source owns the reusable buffers and releases them on destroy(). */
  patternBuffer(acquisition: number, scan: number): GPUBuffer {
    this.check();
    if (this.partitions) return this.loaded(acquisition).patternBuffer(0, scan);
    if (!Number.isInteger(acquisition) || acquisition < 0 || acquisition >= this.acquisitionCount
      || !Number.isInteger(scan) || scan < 0 || scan >= N)
      throw new Error('Select a native acquisition and scan position.');
    if (!this.patternCounts) {
      const start = this.owned.length;
      try {
        this.patternCounts = this.buffer(Q * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
        this.patternParams = this.buffer(16, GPUBufferUsage.UNIFORM, new Uint32Array([Q, 0, 0, 0]).buffer);
        this.patternConvert = this.device.createComputePipeline({layout: 'auto', compute: {
          module: this.device.createShaderModule({code: `
@group(0) @binding(0) var<storage, read> counts: array<u32>;
@group(0) @binding(1) var<storage, read_write> display: array<f32>;
@group(0) @binding(2) var<storage, read> errors: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4u;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= params.x) { return; }
  if (errors[0] != 0u) { display[id.x] = -1.0; return; }
  display[id.x] = f32(counts[id.x]);
}`}), entryPoint: 'main'}});
      } catch (error) {
        for (const buffer of this.owned.splice(start)) {
          this.profile.residentBytes -= buffer.size;
          buffer.destroy();
        }
        this.patternCounts = undefined; this.patternParams = undefined;
        this.patternConvert = undefined;
        throw error;
      }
    }
    const displays = this.patternDisplays ??= new Map<number, GPUBuffer>();
    const bindings = this.patternConvertBindings ??= new Map<number, GPUBindGroup>();
    let display = displays.get(acquisition);
    if (!display) {
      const start = this.owned.length;
      try {
        display = this.buffer(Q * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
        const bind = this.device.createBindGroup({layout: this.patternConvert!.getBindGroupLayout(0),
          entries: [this.patternCounts, display, this.errors, this.patternParams!]
            .map((buffer, binding) => ({binding, resource: {buffer}}))});
        displays.set(acquisition, display);
        bindings.set(acquisition, bind);
      } catch (error) {
        for (const buffer of this.owned.splice(start)) {
          this.profile.residentBytes -= buffer.size;
          buffer.destroy();
        }
        throw error;
      }
    }
    const encoder = this.device.createCommandEncoder();
    this.encodePattern(encoder, acquisition, scan, this.patternCounts);
    // Apply the existing display validity policy without changing encoded counts.
    for (const q of this.badPx) if (q < Q) encoder.clearBuffer(this.patternCounts, q * 4, 4);
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.patternConvert!); pass.setBindGroup(0, bindings.get(acquisition)!);
    pass.dispatchWorkgroups(Math.ceil(Q / 64)); pass.end();
    this.device.queue.submit([encoder.finish()]);
    return display;
  }
  async pattern(acquisition: number, scan: number): Promise<Float32Array> {
    this.check();
    if (this.partitions) return this.loaded(acquisition).pattern(0, scan);
    if (!Number.isInteger(acquisition) || acquisition < 0 || acquisition >= this.acquisitionCount || !Number.isInteger(scan) || scan < 0 || scan >= N)
      throw new Error('Select a native acquisition and scan position.');
    const out = this.device.createBuffer({ size: Q * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST }), read = this.device.createBuffer({ size: Q * 4 + 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    try {
      const encoder = this.device.createCommandEncoder();
      this.encodePattern(encoder, acquisition, scan, out);
      encoder.copyBufferToBuffer(out, 0, read, 0, Q * 4);
      encoder.copyBufferToBuffer(this.errors, 0, read, Q * 4, 4);
      this.device.queue.submit([encoder.finish()]);
      await read.mapAsync(GPUMapMode.READ);
      const snapshot = new Uint32Array(read.getMappedRange());
      if (snapshot[Q])
        throw new Error(`Source112 decoder rejected source (status ${snapshot[Q]}); reload verified data.`);
      return Float32Array.from(snapshot.subarray(0, Q));
    }
    finally {
      read.destroy();
      out.destroy();
    }
  }
  dispose() { this.destroy(); }
  async readImageU32(acquisition: number): Promise<Uint32Array> {
    this.check();
    if (this.partitions) return this.loaded(acquisition).readImageU32(0);
    if (!Number.isInteger(acquisition) || acquisition < 0 || acquisition >= this.acquisitionCount)
      throw new Error('Select acquisition0 through65.');
    const buffer = this.device.createBuffer({ size: N * 4 + 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    try {
      const encoder = this.device.createCommandEncoder();
      encoder.copyBufferToBuffer(this.output, acquisition * N * 4, buffer, 0, N * 4);
      encoder.copyBufferToBuffer(this.errors, 0, buffer, N * 4, 4);
      this.device.queue.submit([encoder.finish()]);
      await buffer.mapAsync(GPUMapMode.READ);
      const snapshot = new Uint32Array(buffer.getMappedRange());
      if (snapshot[N])
        throw new Error(`Source112 decoder rejected source (status ${snapshot[N]}); reload verified data.`);
      return snapshot.slice(0, N);
    }
    finally {
      buffer.destroy();
    }
  }
  async errorStatus(): Promise<number> {
    this.check();
    if (this.partitions) return (await Promise.all(this.partitions.map(child => child.errorStatus()))).reduce((a, b) => a | b, 0);
    const buffer = this.device.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    try {
      const encoder = this.device.createCommandEncoder();
      encoder.copyBufferToBuffer(this.errors, 0, buffer, 0, 4);
      this.device.queue.submit([encoder.finish()]);
      await buffer.mapAsync(GPUMapMode.READ);
      return new Uint32Array(buffer.getMappedRange())[0];
    }
    finally {
      buffer.destroy();
    }
  }
  /** Release all owned buffers, including cached display uniforms, exactly once. */
  destroy() {
    if (this.disposed)
      return;
    this.disposed = true;
    this.loadingAbort?.abort();
    for (const child of this.partitions ?? []) child.destroy();
    if (this.partitions) this.partitions.length = 0;
    this.restartEnabled = false;
    this.restartPipeline = undefined;
    this.restartOriginals = undefined;
    this.restartProfile = undefined;
    this.profile.restartCacheBytes = 0;
    this.profile.restartCacheLayout = 'none';
    for (const buffer of this.owned)
      buffer.destroy();
    this.owned = [];
    this.groups.length = 0;
    this.patternCounts = undefined; this.patternParams = undefined;
    this.patternConvert = undefined; this.patternDisplays?.clear(); this.patternConvertBindings?.clear();
    this.previousMask = null;
  }
}
/** Acquisition adapter used by the stock detector batch contract. */
export class Source112DetectorCompute {
  readonly isRansResident = true;
  readonly isSource112Resident = true;
  readonly scanCount = N;
  readonly detSize = Q;
  readonly mode = 2;
  get badPx() { return this.set.badPx; }
  effective(mask: Uint32Array) {
    const m = mask.slice();
    for (const q of this.badPx)
      m[q] = 0;
    return m;
  }
  constructor(readonly set: Source112ResidentSet, readonly tilt: number) { }
  getDevice() { return this.set.device; }
  async maskedSum(mask: Uint32Array) { this.set.integrate(this.effective(mask)); return this.set.readImage(this.tilt); }
  maskedSumBuffer(mask: Uint32Array) { this.set.integrate(this.effective(mask)); return { buffer: this.set.imageBuffersF32([this.tilt])[0], n: N }; }
  frameAtBuffer(scan: number) {
    return {buffer: this.set.patternBuffer(this.tilt, scan), n: Q};
  }
  async frameAt(scan: number) {
    const frame = await this.set.pattern(this.tilt, scan);
    for (const q of this.badPx)
      frame[q] = 0;
    return frame;
  }
  dispose() { }
}
