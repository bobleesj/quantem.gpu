/** Authenticated integer QEM admission; detector measurements remain encoded. */
import tables from "../../formats/qem/qem-rans-tables-v1.json";
import type { RansByteSource } from "./rans-source";
import type { RansManifest } from "./rans";

/** Byte access to one .qem acquisition: a local File or a file served beside the viewer. */
export interface QemByteFile {
  name: string;
  size: number;
  slice(start?: number, end?: number): { arrayBuffer(): Promise<ArrayBuffer> };
}

type Span = { offset: number; count: number };
type Chunk = { first: number; scans: number; arrays: Span[] };
type Header = {
  container: string;
  container_version: number;
  codec: string;
  profile: string;
  dtype: string;
  shape: number[];
  interval: number;
  bytes: number;
  sha256: string[];
  valid: string;
  chunks: Chunk[];
  metadata: Record<string, unknown>;
  scientific_metadata: Record<string, unknown>;
};
// Incremental SHA-256 of the authenticated JSON header. Payload chunks are
// verified with WebCrypto, which needs one whole ArrayBuffer per digest.
export class SectionSHA256 {
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

export function requireQem(condition: unknown, detail: string): asserts condition {
  if (!condition) throw new Error(`Invalid QEM source: ${detail}. Select an intact .qem acquisition.`);
}
export function parseManifest(text: string): unknown {
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    throw new Error("Invalid QEM JSON metadata; recopy or re-export the original acquisition.");
  }
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
      requireQem(!keys.has(key), `duplicate manifest field ${key}`); keys.add(key);
    }
  }
  return value;
}

const digest = (bytes: Uint8Array) => {
  const hash = new SectionSHA256();
  hash.update(bytes);
  return hash.finish();
};

/** Open .qem files served beside the viewer by a Range-capable HTTP server.
 * Every request bypasses the browser HTTP cache: admission authenticates every
 * byte it reads, and caching multi-gigabyte ranges only adds disk writes.
 */
export async function qemHttpFiles(base: string, names: string[]): Promise<QemByteFile[]> {
  return Promise.all(names.map(async name => {
    // Plain file names only: a path could reach files outside the served folder.
    requireQem(name.length > 0 && !/[\\/]/.test(name) && name !== "." && name !== "..", "invalid QEM file name");
    const url = new URL(encodeURIComponent(name), new URL(base, location.href));
    const header = await fetch(url, { method: "HEAD", cache: "no-store" });
    const size = Number(header.headers.get("Content-Length"));
    requireQem(header.ok && Number.isSafeInteger(size) && size >= 56, `cannot open ${name}; keep the data file beside the viewer`);
    return { name, size, slice(start = 0, end = size) {
      return { async arrayBuffer() {
        const response = await fetch(url, { cache: "no-store", headers: { Range: `bytes=${start}-${end - 1}` } });
        requireQem(response.status === 206, `the server ignored a byte range of ${name}; serve the folder with a Range-capable server`);
        const bytes = await response.arrayBuffer();
        requireQem(bytes.byteLength === end - start, `truncated range in ${name}`);
        return bytes;
      } };
    } };
  }));
}

/** Admit one .qem count acquisition without expanding detector counts. */
export async function qemFileSource(
  file: QemByteFile,
  onStatus: (text: string) => void = () => {},
): Promise<RansByteSource> {
  requireQem(file.size >= 56, "truncated envelope");
  const prefix = new Uint8Array(await file.slice(0, 56).arrayBuffer());
  requireQem(
    new TextDecoder().decode(prefix.subarray(0, 8)) === "QEMDATA1",
    "unsupported container; re-export the original acquisition as .qem",
  );
  const view = new DataView(prefix.buffer);
  const length = Number(view.getBigUint64(8, true)),
    body = Number(view.getBigUint64(16, true));
  requireQem(
    Number.isSafeInteger(length) &&
      length > 0 &&
      length <= 16 << 20 &&
      body === length + 56 &&
      body <= file.size,
    "invalid envelope bounds",
  );
  const json = new Uint8Array(await file.slice(56, body).arrayBuffer());
  requireQem(
    digest(json) ===
      [...prefix.subarray(24)]
        .map((value) => value.toString(16).padStart(2, "0"))
        .join(""),
    "header checksum mismatch",
  );
  const header = parseManifest(new TextDecoder().decode(json)) as Header;
  requireQem(
    header && typeof header === "object" && !Array.isArray(header),
    "header must be a JSON object",
  );
  requireQem(
    header.container === "quantem.qem" && header.container_version === 1,
    "unsupported container version",
  );
  requireQem(
    header.codec === "runtime-column-rans-spatial-v2" &&
      header.profile === header.codec &&
      ["uint8", "uint16"].includes(header.dtype),
    "this browser supports integer QEM only; open float32 QEM in the native application or a Python GPU session",
  );
  requireQem(
    Array.isArray(header.shape) &&
      header.shape.length === 4 &&
      header.shape.every((value) => Number.isSafeInteger(value) && value > 0),
    "invalid four-dimensional geometry",
  );
  requireQem(
    header.metadata && typeof header.metadata === "object" && !Array.isArray(header.metadata),
    "metadata must be an object",
  );
  const scientific = header.scientific_metadata;
  requireQem(
    scientific &&
      [
        "quantem.scientific-metadata/1",
        "quantem.scientific-metadata/2",
      ].includes(String(scientific.schema)),
    "unsupported scientific metadata schema",
  );
  const axes = scientific.axes as { name: string; size: number }[];
  const names = ["scan_row", "scan_column", "detector_row", "detector_column"];
  requireQem(
    Array.isArray(axes) &&
      axes.length === 4 &&
      axes.every(
        (axis, index) =>
          axis && axis.name === names[index] && axis.size === header.shape[index],
      ),
    "scientific axes disagree with stored geometry",
  );
  for (const section of ["calibration_overrides", "electron_microscope"]) {
    const quantities = scientific[section] ?? {};
    requireQem(
      quantities &&
        typeof quantities === "object" &&
        !Array.isArray(quantities),
      `${section} must contain named quantities`,
    );
    requireQem(
      !Object.keys(quantities).some((key) =>
        /(?:pixel_size|reciprocal_pixel_size)_[xy]$/.test(key),
      ),
      "retired x/y calibration names; re-export with row/column metadata",
    );
  }
  requireQem(
    header.interval === 512 &&
      Number.isSafeInteger(header.bytes) &&
      body + header.bytes === file.size,
    "invalid interval or payload bounds",
  );
  const [rows, cols, detRows, detCols] = header.shape,
    scans = rows * cols,
    K = detRows * detCols;
  requireQem(
    Number.isSafeInteger(scans) &&
      scans < 2 ** 32 &&
      K < 2 ** 24 &&
      K * (header.dtype === "uint8" ? 255 : 65535) < 2 ** 32,
    "geometry exceeds browser integer-product capacity; use the native GPU application",
  );
  const chunkBytes = 64 << 20;
  requireQem(
    Array.isArray(header.sha256) &&
      header.sha256.length === Math.ceil(header.bytes / chunkBytes),
    "missing payload checksums",
  );
  for (let index = 0; index < header.sha256.length; index++) {
    onStatus(`Verifying .qem ${index + 1}/${header.sha256.length}`);
    const begin = index * chunkBytes;
    const end = Math.min(header.bytes, (index + 1) * chunkBytes);
    // QEM hashes independent 64 MiB chunks. WebCrypto uses the platform's
    // SHA-256 implementation without a JavaScript loop over every byte.
    // Keep verification bounded to one chunk and retain every integrity check.
    const bytes = await file.slice(body + begin, body + end).arrayBuffer();
    const hash = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
    const actual = Array.from(hash, value => value.toString(16).padStart(2, "0")).join("");
    requireQem(actual === header.sha256[index], "payload checksum mismatch");
  }
  requireQem(
    typeof header.valid === "string" &&
      /^[0-9a-f]*$/.test(header.valid) &&
      header.valid.length === Math.ceil(K / 8) * 2,
    "invalid detector validity mask",
  );
  const badPixels: number[] = [];
  for (let k = 0; k < K; k++)
    if (
      !(
        parseInt(header.valid.slice((k >> 3) * 2, (k >> 3) * 2 + 2), 16) &
        (128 >> (k & 7))
      )
    )
      badPixels.push(k);
  const entries = new Uint32Array(64 * 33 * 2);
  tables.frequencies.forEach((frequencies, model) => {
    let cumulative = 0;
    frequencies.forEach((frequency, symbol) => {
      const at = (model * 33 + symbol) * 2;
      entries[at] = (cumulative << 16) | symbol;
      entries[at + 1] = frequency;
      cumulative += frequency;
    });
    requireQem(cumulative === 1024, "invalid fixed probability table");
  });
  const blockMeta: {
    index: number;
    bytes: number;
    model: number;
    byte_start: number;
    byte_end: number;
    frames: number;
  }[] = [];
  const offsets: Uint32Array<ArrayBuffer>[] = [],
    columns: Uint32Array<ArrayBuffer>[] = [];
  let nextScan = 0,
    previousEnd = 0;
  requireQem(
    Array.isArray(header.chunks) && header.chunks.length > 0,
    "missing encoded chunks",
  );
  for (const chunk of header.chunks) {
    requireQem(
      chunk.first === nextScan &&
        Number.isSafeInteger(chunk.scans) &&
        chunk.scans > 0 &&
        chunk.first + chunk.scans <= scans &&
        (chunk.first + chunk.scans === scans || chunk.scans % 512 === 0),
      "noncontiguous or unaligned scan chunks",
    );
    requireQem(
      Array.isArray(chunk.arrays) && chunk.arrays.length === 6,
      "missing typed chunk arrays",
    );
    const widths = [1, 4, 1, 4, 8, 1];
    chunk.arrays.forEach((span, index) => {
      requireQem(
        Number.isSafeInteger(span.offset) &&
          Number.isSafeInteger(span.count) &&
          span.count >= 0 &&
          span.offset === Math.ceil(previousEnd / 8) * 8 &&
          span.offset + span.count * widths[index] <= header.bytes,
        "invalid chunk array bounds",
      );
      previousEnd = span.offset + span.count * widths[index];
    });
    const [payload, offsetSpan, modelSpan] = chunk.arrays,
      blocks = Math.ceil(chunk.scans / 512);
    requireQem(
      offsetSpan.count === blocks * K + 1 && modelSpan.count === blocks * K,
      "invalid stream table dimensions",
    );
    const read = (span: Span, size: number) =>
      file
        .slice(body + span.offset, body + span.offset + span.count * size)
        .arrayBuffer();
    const local = new Uint32Array(await read(offsetSpan, 4)),
      models = new Uint8Array(await read(modelSpan, 1));
    requireQem(
      local[0] === 0 && local[local.length - 1] === payload.count,
      "stream table does not partition payload",
    );
    for (let block = 0; block < blocks; block++) {
      const first = block * K,
        start = local[first],
        end = local[first + K],
        frames = Math.min(512, chunk.scans - block * 512);
      const relative = new Uint32Array(K + 1),
        metadata = new Uint32Array(K * 3);
      for (let k = 0; k < K; k++) {
        const model = models[first + k],
          size = local[first + k + 1] - local[first + k];
        requireQem(
          local[first + k + 1] >= local[first + k] &&
            (model < 64
              ? size >= 4
              : model === 252
                ? size % 2 === 0 && size <= frames * 2
                : model === 253
                  ? size === 0
                  : model === 254
                    ? size === frames * 2
                    : model === 255 && size === 2),
          "invalid encoded stream mode or length",
        );
        metadata.set(
          model < 64
            ? [model * 33, (model + 1) * 33, 2]
            : [
                0,
                0,
                model === 252 ? 5 : model === 253 ? 3 : model === 254 ? 1 : 4,
              ],
          k * 3,
        );
        relative[k] = local[first + k] - start;
      }
      relative[K] = end - start;
      const index = blockMeta.length;
      blockMeta.push({
        index,
        bytes: end - start,
        model: index,
        byte_start: body + payload.offset + start,
        byte_end: body + payload.offset + end,
        frames,
      });
      offsets.push(relative);
      columns.push(metadata);
    }
    nextScan += chunk.scans;
  }
  requireQem(
    nextScan === scans && previousEnd === header.bytes,
    "incomplete scan coverage or undeclared payload",
  );
  const mapped: RansManifest = {
    scan_shape: [rows, cols],
    detector_shape: [detRows, detCols],
    native_dtype: header.dtype as "uint8" | "uint16",
    bad_pixels: badPixels,
    source_metadata: {
      ...header.metadata,
      scientific_metadata: header.scientific_metadata,
    },
    tilts: [
      {
        tilt: 0,
        K,
        frames: 512,
        blocks: blockMeta.length,
        scale: 10,
        model_frames: 512,
        binary_lookup: true,
        payload_url: "payload",
        blocks_meta: blockMeta,
        models: blockMeta.map((block) => ({
          index: block.index,
          symbols: 64 * 33,
          colmeta_url: `columns-${block.index}`,
          entries_url: "entries",
          lut_url: "lookup",
        })),
      },
    ],
  };
  return {
    mode: "local-folder",
    async read(name, start, end) {
      if (name === "manifest.json")
        return new TextEncoder().encode(JSON.stringify(mapped)).buffer;
      if (name === "entries") return entries.buffer;
      if (name === "lookup") return new ArrayBuffer(4);
      if (name === "payload") {
        requireQem(
          start !== undefined &&
            end !== undefined &&
            blockMeta.some(
              (block) =>
                start >= block.byte_start &&
                end <= block.byte_end &&
                end >= start,
            ),
          "invalid payload request",
        );
        return file.slice(start, end).arrayBuffer();
      }
      const column = /^columns-(\d+)$/.exec(name);
      if (column) return columns[Number(column[1])].buffer;
      const offset = /^t0-offsets-(\d+)\.u32$/.exec(name);
      requireQem(
        offset && offsets[Number(offset[1])],
        "invalid stream table request",
      );
      return offsets[Number(offset[1])].buffer;
    },
  };
}

/** Join compatible .qem acquisitions into one series without decoding counts or changing their order. */
export async function qemFilesSource(files: ArrayLike<QemByteFile>, onStatus: (text: string) => void = () => {}, badPixels: number[] = []): Promise<RansByteSource> {
  const ordered = Array.from(files);
  requireQem(ordered.length > 0, "select at least one .qem file");
  const sources: RansByteSource[] = [];
  const manifests: RansManifest[] = [];
  for (let index = 0; index < ordered.length; index++) {
    const source = await qemFileSource(ordered[index], text => onStatus(`${index + 1}/${ordered.length} ${ordered[index].name}: ${text}`));
    const manifest = JSON.parse(new TextDecoder().decode(await source.read("manifest.json"))) as RansManifest;
    if (index > 0) {
      const first = manifests[0];
      requireQem(JSON.stringify(manifest.bad_pixels) === JSON.stringify(first.bad_pixels), "series detector validity masks differ; open each acquisition separately");
      const shape = [...manifest.scan_shape!, ...manifest.detector_shape!];
      const expected = [...first.scan_shape!, ...first.detector_shape!];
      if (JSON.stringify(shape) !== JSON.stringify(expected) || manifest.native_dtype !== first.native_dtype) {
        throw new Error(`QEM file ${ordered[index].name} has shape ${shape.join("x")} and dtype ${manifest.native_dtype}; ${ordered[0].name} has ${expected.join("x")} ${first.native_dtype}. Select acquisitions with matching native geometry and dtype.`);
      }
      const profile = manifest.tilts[0], firstProfile = first.tilts[0];
      if (profile.frames !== firstProfile.frames || profile.scale !== firstProfile.scale) {
        throw new Error(`QEM file ${ordered[index].name} uses block_frames=${profile.frames}, scale=${profile.scale}; ${ordered[0].name} uses block_frames=${firstProfile.frames}, scale=${firstProfile.scale}. Re-encode the series with the same block_frames and scale before loading it together.`);
      }
    }
    sources.push(source); manifests.push(manifest);
  }
  const detectorPixels = manifests[0].tilts[0].K;
  requireQem(badPixels.every(index => Number.isInteger(index) && index >= 0 && index < detectorPixels), `badPixels must contain detector indices from 0 to ${detectorPixels - 1}`);
  const combined: RansManifest = {
    ...manifests[0],
    bad_pixels: [...new Set([...(manifests[0].bad_pixels ?? []), ...badPixels])],
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
    requireQem(offsets, `unknown series table ${name}`);
    return sources[Number(offsets[1])].read(`t0-offsets-${offsets[2]}.u32`);
  } };
}
