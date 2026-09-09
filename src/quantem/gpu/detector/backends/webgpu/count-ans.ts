/** Admit a self-contained count-ANS v1 file to the existing resident rANS engine.
 * Only encoded tables are transformed on the host. Native counts are decoded
 * by the shared GPU recurrence, including literal streams and tail blocks.
 */
import type { RansByteSource } from "./rans-source";
import type { RansManifest } from "./rans";

// Incremental SHA-256 keeps payload verification bounded; WebCrypto.digest
// requires an entire multi-gigabyte section in one ArrayBuffer.
class SectionSHA256 {
  private state = new Uint32Array([0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]);
  private block = new Uint8Array(64);
  private words = new Uint32Array(64);
  private used = 0;
  private length = 0;
  private static readonly constants = new Uint32Array([
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
  ]);
  private transform(bytes: Uint8Array, offset: number): void {
    const rotate = (value: number, shift: number) => (value >>> shift) | (value << (32 - shift));
    const words = this.words;
    for (let i = 0; i < 16; i++) words[i] = (bytes[offset + i * 4] << 24) | (bytes[offset + i * 4 + 1] << 16) | (bytes[offset + i * 4 + 2] << 8) | bytes[offset + i * 4 + 3];
    for (let i = 16; i < 64; i++) {
      const a = words[i - 15], b = words[i - 2];
      words[i] = words[i - 16] + (rotate(a, 7) ^ rotate(a, 18) ^ (a >>> 3)) + words[i - 7] + (rotate(b, 17) ^ rotate(b, 19) ^ (b >>> 10));
    }
    let [a,b,c,d,e,f,g,h] = this.state;
    for (let i = 0; i < 64; i++) {
      const first = (h + (rotate(e, 6) ^ rotate(e, 11) ^ rotate(e, 25)) + ((e & f) ^ (~e & g)) + SectionSHA256.constants[i] + words[i]) >>> 0;
      const second = ((rotate(a, 2) ^ rotate(a, 13) ^ rotate(a, 22)) + ((a & b) ^ (a & c) ^ (b & c))) >>> 0;
      h=g; g=f; f=e; e=(d+first)>>>0; d=c; c=b; b=a; a=(first+second)>>>0;
    }
    const next = [a,b,c,d,e,f,g,h];
    for (let i = 0; i < 8; i++) this.state[i] += next[i];
  }
  update(bytes: Uint8Array): void {
    this.length += bytes.length;
    let offset = 0;
    if (this.used) {
      const count = Math.min(64 - this.used, bytes.length);
      this.block.set(bytes.subarray(0, count), this.used); this.used += count; offset += count;
      if (this.used === 64) { this.transform(this.block, 0); this.used = 0; }
    }
    for (; offset + 64 <= bytes.length; offset += 64) this.transform(bytes, offset);
    this.block.set(bytes.subarray(offset), this.used); this.used += bytes.length - offset;
  }
  finish(): string {
    const bitLength = BigInt(this.length) * 8n;
    this.update(new Uint8Array([128]));
    const padding = new Uint8Array((56 - this.used + 64) % 64 + 8);
    new DataView(padding.buffer).setBigUint64(padding.length - 8, bitLength);
    this.update(padding);
    return [...this.state].map(value => value.toString(16).padStart(8, "0")).join("");
  }
}

type Section = { offset: number; count: number; dtype: string; sha256: string };
type CountANS = {
  schema: string; codec: string; order: string; dtype: "uint8" | "uint16";
  shape: number[]; block_frames: number; scale: number; logical_sha256: string;
  metadata: Record<string, unknown>; sections: Record<string, Section>;
};
const types = { payload: ["u1", 1], offsets: ["<u8", 8], model_ids: ["<u4", 4], context_offsets: ["<u4", 4], symbols: ["<u2", 2], cumulative: ["<u2", 2], frequencies: ["<u2", 2], literal: ["u1", 1] } as const;
function requireANS(condition: unknown, detail: string): asserts condition {
  if (!condition) throw new Error(`Invalid count-ANS source: ${detail}. Select an intact quantem.gpu.count-ans.v1 file.`);
}
function parseManifest(text: string): CountANS {
  const value = JSON.parse(text);
  // JSON.parse alone would silently accept conflicting duplicate schema keys.
  const tokens = text.match(/"(?:\\.|[^"\\])*"|[{}\[\]:,]/g) ?? [];
  const stack: (Set<string> | null)[] = [];
  for (let i = 0; i < tokens.length; i++) {
    const token = tokens[i];
    if (token === "{") stack.push(new Set());
    else if (token === "[") stack.push(null);
    else if (token === "}" || token === "]") stack.pop();
    else if (token.startsWith('"') && tokens[i + 1] === ":") {
      const keys = stack[stack.length - 1]!; const key = JSON.parse(token);
      requireANS(!keys.has(key), `duplicate manifest field ${key}`); keys.add(key);
    }
  }
  return value;
}

/** Map a verified canonical file into encoded inputs for RansResidentSet. */
export async function countAnsFileSource(file: File, onStatus: (text: string) => void = () => {}): Promise<RansByteSource> {
  requireANS(file.size >= 65536, "truncated header");
  const header = new DataView(await file.slice(0, 24).arrayBuffer());
  const magic = [81,71,65,78,83,0,1,0];
  requireANS(magic.every((value, index) => header.getUint8(index) === value), "unsupported file magic");
  const manifestLength = Number(header.getBigUint64(8, true));
  requireANS(header.getBigUint64(16, true) === 65536n && manifestLength > 0 && manifestLength <= 65512, "invalid manifest bounds");
  const manifest = parseManifest(await file.slice(24, 24 + manifestLength).text());
  requireANS(manifest && manifest.schema === "quantem.gpu.count-ans.v1" && manifest.codec === "block-column-rans-byte-v1", "unsupported schema or entropy codec");
  requireANS(manifest.order === "scan_row,scan_column,detector_row,detector_column", "dimension order must be explicit row/column order");
  requireANS(manifest.dtype === "uint8" || manifest.dtype === "uint16", "unsupported native dtype");
  requireANS(Array.isArray(manifest.shape) && manifest.shape.length === 4 && manifest.shape.every(value => Number.isSafeInteger(value) && value > 0), "shape must have four positive dimensions");
  requireANS(Number.isInteger(manifest.block_frames) && manifest.block_frames > 0 && manifest.block_frames < 2 ** 30, "block length exceeds browser indexing");
  requireANS(Number.isInteger(manifest.scale) && manifest.scale >= 1 && manifest.scale <= 15, "unsupported probability precision");
  requireANS(/^[0-9a-f]{64}$/.test(manifest.logical_sha256), "missing logical count digest");
  requireANS(manifest.metadata && typeof manifest.metadata === "object" && !Array.isArray(manifest.metadata), "metadata must be an object");
  const [scanRows, scanCols, detRows, detCols] = manifest.shape;
  const scans = scanRows * scanCols, K = detRows * detCols, frames = manifest.block_frames, blocks = Math.ceil(scans / frames);
  requireANS(Number.isSafeInteger(scans) && scans < 2 ** 32 && Number.isSafeInteger(K) && K < 2 ** 24, "geometry exceeds browser indexing");
  requireANS(K * (manifest.dtype === "uint8" ? 255 : 65535) < 2 ** 32, "detector sums may exceed the browser uint32 image contract; use the CUDA uint64 product backend");
  const metadata = manifest.metadata;
  requireANS((metadata.scan_bin ?? 1) === 1 && (metadata.crop ?? null) === null && (metadata.detector_bin ?? 1) === 1, "pre-reduced provenance is not a full-resolution resident source");
  for (const name of ["source_shape", "working_shape"]) if (name in metadata) requireANS(JSON.stringify(metadata[name]) === JSON.stringify(manifest.shape), `${name} disagrees with stored geometry`);
  for (const name of ["source_dtype", "working_dtype"]) if (name in metadata) requireANS(metadata[name] === manifest.dtype, `${name} disagrees with native counts`);
  requireANS(manifest.sections && Object.keys(manifest.sections).sort().join() === Object.keys(types).sort().join(), "typed sections are incomplete");
  const arrays = new Map<string, ArrayBuffer>();
  let cursor = 65536;
  for (const [name, [dtype, itemsize]] of Object.entries(types)) {
    const section = manifest.sections[name];
    requireANS(section && Number.isSafeInteger(section.offset) && Number.isSafeInteger(section.count) && section.count >= 0 && section.dtype === dtype, `invalid ${name} section`);
    const size = section.count * itemsize;
    requireANS(Number.isSafeInteger(size) && section.offset === Math.ceil(cursor / 8) * 8 && section.offset + size <= file.size, `${name} bounds disagree with file`);
    requireANS(/^[0-9a-f]{64}$/.test(section.sha256), `${name} has no checksum`);
    onStatus(`Verifying count-ANS ${name}`);
    const digest = new SectionSHA256();
    if (name === "payload") {
      for (let offset = 0; offset < size; offset += 8 << 20) digest.update(new Uint8Array(await file.slice(section.offset + offset, section.offset + Math.min(size, offset + (8 << 20))).arrayBuffer()));
    } else {
      const bytes = await file.slice(section.offset, section.offset + size).arrayBuffer();
      arrays.set(name, bytes); digest.update(new Uint8Array(bytes));
    }
    requireANS(digest.finish() === section.sha256, `${name} checksum mismatch`);
    cursor = section.offset + size;
  }
  requireANS(cursor === file.size, "undeclared trailing bytes");
  const offsets = new BigUint64Array(arrays.get("offsets")!);
  const models = new Uint32Array(arrays.get("model_ids")!);
  const contexts = new Uint32Array(arrays.get("context_offsets")!);
  const symbols = new Uint16Array(arrays.get("symbols")!);
  const cumulative = new Uint16Array(arrays.get("cumulative")!);
  const frequencies = new Uint16Array(arrays.get("frequencies")!);
  const literal = new Uint8Array(arrays.get("literal")!);
  requireANS(offsets.length === blocks * K + 1 && models.length === blocks * K && offsets[0] === 0n && offsets[offsets.length - 1] === BigInt(manifest.sections.payload.count), "stream index does not partition the payload");
  requireANS(literal.length > 0 && contexts.length === literal.length + 1 && contexts[0] === 0 && contexts[contexts.length - 1] === symbols.length, "model contexts do not cover the symbol table");
  requireANS(symbols.length === cumulative.length && symbols.length === frequencies.length, "model table lengths differ");
  const maxCount = manifest.dtype === "uint8" ? 255 : 65535;
  const entries = new Uint32Array(Math.max(2, symbols.length * 2));
  for (let model = 0; model < literal.length; model++) {
    const first = contexts[model], stop = contexts[model + 1];
    requireANS(first <= stop && literal[model] <= 1, "invalid model interval or literal flag");
    if (literal[model]) { requireANS(first === stop, "literal models must not contain entropy entries"); continue; }
    let sum = 0;
    for (let index = first; index < stop; index++) {
      requireANS(frequencies[index] > 0 && cumulative[index] === sum && symbols[index] <= maxCount && (index === first || symbols[index] > symbols[index - 1]), "probability table does not partition sorted native symbols");
      entries[index * 2] = (cumulative[index] << 16) | symbols[index]; entries[index * 2 + 1] = frequencies[index];
      sum += frequencies[index];
    }
    requireANS(sum === 2 ** manifest.scale, "probabilities do not sum to the declared precision");
  }
  const blockMeta = [];
  for (let block = 0; block < blocks; block++) {
    const count = Math.min(frames, scans - block * frames);
    const first = block * K;
    const start = Number(offsets[first]), end = Number(offsets[first + K]);
    requireANS(end >= start && end - start < 2 ** 32, "encoded block exceeds uint32 local offsets");
    for (let column = 0; column < K; column++) {
      const stream = first + column, model = models[stream];
      requireANS(model < literal.length && offsets[stream + 1] >= offsets[stream], "invalid stream selector or offsets");
      const size = offsets[stream + 1] - offsets[stream];
      requireANS(literal[model] ? size === BigInt(count * 2) : size >= 4n, "stream cannot represent its declared block");
    }
    blockMeta.push({ index: block, bytes: end - start, model: block, byte_start: start, byte_end: end, frames: count });
  }
  const mapped: RansManifest = {
    scan_shape: [scanRows, scanCols], detector_shape: [detRows, detCols], source_metadata: metadata, native_dtype: manifest.dtype,
    tilts: [{ tilt: 0, K, frames, blocks, scale: manifest.scale, model_frames: frames, binary_lookup: true, payload_url: "payload", blocks_meta: blockMeta,
      models: blockMeta.map(block => ({ index: block.index, symbols: symbols.length, colmeta_url: `columns-${block.index}`, entries_url: "entries", lut_url: "lookup" })) }],
  };
  return { mode: "local-folder", async read(name, start, end) {
    if (name === "manifest.json") return new TextEncoder().encode(JSON.stringify(mapped)).buffer;
    if (name === "entries") return entries.buffer;
    if (name === "lookup") return new ArrayBuffer(4); // Binary lookup shares the recurrence without a byte-index LUT.
    if (name === "payload") {
      requireANS(start !== undefined && end !== undefined && start >= 0 && end >= start && end <= manifest.sections.payload.count, "invalid payload byte range");
      return file.slice(manifest.sections.payload.offset + start, manifest.sections.payload.offset + end).arrayBuffer();
    }
    const columns = /^columns-(\d+)$/.exec(name);
    if (columns) {
      const block = Number(columns[1]); const metadata = new Uint32Array(K * 3);
      for (let column = 0; column < K; column++) {
        const model = models[block * K + column];
        metadata.set([contexts[model], contexts[model + 1], literal[model]], column * 3);
      }
      return metadata.buffer;
    }
    const localOffsets = /^t0-offsets-(\d+)\.u32$/.exec(name);
    requireANS(localOffsets, `unknown encoded table ${name}`);
    const first = Number(localOffsets[1]) * K; const local = new Uint32Array(K + 1);
    for (let column = 0; column <= K; column++) local[column] = Number(offsets[first + column] - offsets[first]);
    return local.buffer;
  } };
}


/** Join compatible encoded files without decoding counts or changing their order. */
export async function countAnsFilesSource(files: ArrayLike<File>, onStatus: (text: string) => void = () => {}, badPixels: number[] = []): Promise<RansByteSource> {
  const ordered = Array.from(files);
  requireANS(ordered.length > 0, "select at least one count-ANS file");
  const sources: RansByteSource[] = [];
  const manifests: RansManifest[] = [];
  for (let index = 0; index < ordered.length; index++) {
    const source = await countAnsFileSource(ordered[index], text => onStatus(`${index + 1}/${ordered.length} ${ordered[index].name}: ${text}`));
    const manifest = JSON.parse(new TextDecoder().decode(await source.read("manifest.json"))) as RansManifest;
    if (index > 0) {
      const first = manifests[0];
      const shape = [...manifest.scan_shape!, ...manifest.detector_shape!];
      const expected = [...first.scan_shape!, ...first.detector_shape!];
      if (JSON.stringify(shape) !== JSON.stringify(expected) || manifest.native_dtype !== first.native_dtype) {
        throw new Error(`Count-ANS file ${ordered[index].name} has shape ${shape.join("x")} and dtype ${manifest.native_dtype}; ${ordered[0].name} has ${expected.join("x")} ${first.native_dtype}. Select acquisitions with matching native geometry and dtype.`);
      }
      const profile = manifest.tilts[0], firstProfile = first.tilts[0];
      if (profile.frames !== firstProfile.frames || profile.scale !== firstProfile.scale) {
        throw new Error(`Count-ANS file ${ordered[index].name} uses block_frames=${profile.frames}, scale=${profile.scale}; ${ordered[0].name} uses block_frames=${firstProfile.frames}, scale=${firstProfile.scale}. Re-encode the series with the same block_frames and scale before loading it together.`);
      }
    }
    sources.push(source); manifests.push(manifest);
  }
  const detectorPixels = manifests[0].tilts[0].K;
  requireANS(badPixels.every(index => Number.isInteger(index) && index >= 0 && index < detectorPixels), `badPixels must contain detector indices from 0 to ${detectorPixels - 1}`);
  const combined: RansManifest = {
    ...manifests[0],
    bad_pixels: [...new Set(badPixels)],
    source_metadata: ordered.length === 1 ? manifests[0].source_metadata : {
      acquisitions: ordered.map((file, index) => ({ file: file.name, metadata: manifests[index].source_metadata })),
    },
    tilts: manifests.map((manifest, index) => {
      const tilt = manifest.tilts[0]; const prefix = `acquisition-${index}/`;
      return { ...tilt, tilt: index, payload_url: prefix + tilt.payload_url,
        models: tilt.models.map(model => ({ ...model, colmeta_url: prefix + model.colmeta_url,
          entries_url: prefix + model.entries_url, lut_url: prefix + model.lut_url })),
      };
    }),
  };
  return { mode: "local-folder", async read(name, start, end) {
    if (name === "manifest.json") return new TextEncoder().encode(JSON.stringify(combined)).buffer;
    const namespaced = /^acquisition-(\d+)\/(.+)$/.exec(name);
    if (namespaced) return sources[Number(namespaced[1])].read(namespaced[2], start, end);
    const offsets = /^t(\d+)-offsets-(\d+)\.u32$/.exec(name);
    requireANS(offsets, `unknown series table ${name}`);
    return sources[Number(offsets[1])].read(`t0-offsets-${offsets[2]}.u32`);
  } };
}
