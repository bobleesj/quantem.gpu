/** Authenticated integer QEM admission; detector measurements remain encoded. */
import tables from "../../formats/qem/qem-rans-tables-v1.json";
import { payloadGroupLimit, type RansByteSource } from "./rans-source";
import type { RansManifest } from "./rans";

/** Byte access to one .qem acquisition: a local File or a file served beside the viewer. */
export interface QemByteFile {
  name: string;
  size: number;
  slice(start?: number, end?: number): { arrayBuffer(): Promise<ArrayBuffer> };
  /** Consecutive chunkBytes-sized pieces of [start, end), read in order from one stream;
   * `cancel` ends the download itself. */
  chunks?(start: number, end: number, chunkBytes: number, cancel?: AbortSignal): AsyncGenerator<ArrayBuffer, void, unknown>;
  /** Return a chunk the caller has finished with; the stream may refill it. */
  recycleChunk?(buffer: ArrayBuffer): void;
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
 * `signal` aborts every request these files make.
 */
export async function qemHttpFiles(base: string, names: string[], signal?: AbortSignal): Promise<QemByteFile[]> {
  return Promise.all(names.map(async name => {
    // Plain file names only: a path could reach files outside the served folder.
    requireQem(name.length > 0 && !/[\\/]/.test(name) && name !== "." && name !== "..", "invalid QEM file name");
    const url = new URL(encodeURIComponent(name), new URL(base, location.href));
    const header = await fetch(url, { method: "HEAD", cache: "no-store", signal });
    const size = Number(header.headers.get("Content-Length"));
    requireQem(header.ok && Number.isSafeInteger(size) && size >= 56, `cannot open ${name}; keep the data file beside the viewer`);
    // Finished 64 MiB chunks carry later ones instead of a new allocation per
    // chunk. At most four wait here, and none outlives the stream that read them.
    const reusable: ArrayBuffer[] = [];
    let streaming = false;
    const acquire = (wanted: number) => {
      const index = reusable.findIndex(buffer => buffer.byteLength === wanted);
      return new Uint8Array(index < 0 ? new ArrayBuffer(wanted) : reusable.splice(index, 1)[0]);
    };
    return { name, size, async *chunks(start: number, end: number, chunkBytes: number, cancel?: AbortSignal) {
      // One ranged response carries every authentication chunk instead of one
      // request per 64 MiB; each chunk is still hashed on its own.
      const stop = cancel && signal ? AbortSignal.any([signal, cancel]) : cancel ?? signal;
      const response = await fetch(url, { cache: "no-store", headers: { Range: `bytes=${start}-${end - 1}` }, signal: stop });
      // A server that ignores the range would otherwise go on sending the whole file.
      if (response.status !== 206) await response.body?.cancel();
      requireQem(response.status === 206 && response.body, `the server ignored a byte range of ${name}; serve the folder with a Range-capable server`);
      // A byte stream fills each chunk in place (BYOB); any other stream is copied from its parts.
      let reader: ReadableStreamBYOBReader | ReadableStreamDefaultReader<Uint8Array>;
      let byob = true;
      try { reader = response.body.getReader({ mode: "byob" }); }
      catch { reader = response.body.getReader(); byob = false; }
      let remaining = end - start;
      streaming = true;
      try {
        if (byob) {
          // The `min` read option (Chrome 125, Node 22) is missing from TypeScript 5.9's DOM types.
          const byteReader = reader as unknown as { read(view: Uint8Array, options: { min: number }): Promise<ReadableStreamReadResult<Uint8Array<ArrayBuffer>>> };
          while (remaining > 0) {
            const wanted = Math.min(chunkBytes, remaining);
            let chunk = acquire(wanted), filled = 0;
            while (filled < wanted) {
              const part = await byteReader.read(chunk.subarray(filled), { min: wanted - filled });
              requireQem(part.value && part.value.byteLength > 0, `truncated stream in ${name}`);
              filled += part.value.byteLength;
              remaining -= part.value.byteLength;
              // Every BYOB read transfers the buffer it fills; continue in the returned one.
              chunk = new Uint8Array(part.value.buffer, 0, wanted);
            }
            yield chunk.buffer;
          }
          const tail = await byteReader.read(new Uint8Array(1), { min: 1 });
          requireQem(tail.done && !tail.value?.byteLength, `oversized stream in ${name}`);
          return;
        }
        const defaultReader = reader as ReadableStreamDefaultReader<Uint8Array>;
        let chunk = acquire(Math.min(chunkBytes, remaining)), filled = 0;
        while (remaining > 0) {
          const part = await defaultReader.read();
          requireQem(!part.done, `truncated stream in ${name}`);
          for (let at = 0; at < part.value.length;) {
            requireQem(remaining > 0, `oversized stream in ${name}`);
            const count = Math.min(chunk.length - filled, part.value.length - at);
            chunk.set(part.value.subarray(at, at + count), filled);
            at += count; filled += count; remaining -= count;
            if (filled === chunk.length) {
              yield chunk.buffer;
              filled = 0;
              if (remaining > 0) chunk = acquire(Math.min(chunkBytes, remaining));
            }
          }
        }
        requireQem((await defaultReader.read()).done, `oversized stream in ${name}`);
      } finally {
        // Stops the download when admission ends early, on success or failure.
        streaming = false;
        reusable.length = 0;
        await reader.cancel();
        reader.releaseLock();
      }
    }, recycleChunk(buffer: ArrayBuffer) {
      // A BYOB read transfers a reused buffer: the caller must not keep any view of it.
      if (streaming && reusable.length < 4) reusable.push(buffer);
    }, slice(start = 0, end = size) {
      return { async arrayBuffer() {
        const response = await fetch(url, { cache: "no-store", headers: { Range: `bytes=${start}-${end - 1}` }, signal });
        if (response.status !== 206) await response.body?.cancel();
        requireQem(response.status === 206, `the server ignored a byte range of ${name}; serve the folder with a Range-capable server`);
        const bytes = await response.arrayBuffer();
        requireQem(bytes.byteLength === end - start, `truncated range in ${name}`);
        return bytes;
      } };
    } };
  }));
}

// QEM hashes its payload in independent 64 MiB chunks.
const chunkBytes = 64 << 20;

type QemHeader = { header: Header; body: number; badPixels: number[] };

/** Authenticate and check one .qem header, chunk layout and validity mask included, without reading its payload.
 * A series can then be rejected before any payload is read or staged on the GPU.
 */
async function readQemHeader(file: QemByteFile, signal?: AbortSignal): Promise<QemHeader> {
  signal?.throwIfAborted();
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
  requireQem(
    Array.isArray(header.sha256) &&
      header.sha256.length === Math.ceil(header.bytes / chunkBytes),
    "missing payload checksums",
  );
  requireQem(
    Array.isArray(header.chunks) && header.chunks.length > 0,
    "missing encoded chunks",
  );
  // The authenticated header fixes every chunk span before any payload is
  // read, so admission only ever copies declared bytes.
  const widths = [1, 4, 1, 4, 8, 1];
  let nextScan = 0,
    previousEnd = 0;
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
    const blocks = Math.ceil(chunk.scans / 512);
    requireQem(
      chunk.arrays[1].count === blocks * K + 1 && chunk.arrays[2].count === blocks * K,
      "invalid stream table dimensions",
    );
    nextScan += chunk.scans;
  }
  requireQem(
    nextScan === scans && previousEnd === header.bytes,
    "incomplete scan coverage or undeclared payload",
  );
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
  return { header, body, badPixels };
}

/** Admit one .qem count acquisition without expanding detector counts. */
export async function qemFileSource(
  file: QemByteFile,
  onStatus: (text: string) => void = () => {},
  device?: GPUDevice,
  signal?: AbortSignal,
): Promise<RansByteSource> {
  return admitQemFile(file, await readQemHeader(file, signal), onStatus, device, signal);
}

/** Authenticate and stage the payload described by an already checked header. */
async function admitQemFile(
  file: QemByteFile,
  { header, body, badPixels }: QemHeader,
  onStatus: (text: string) => void,
  device?: GPUDevice,
  signal?: AbortSignal,
): Promise<RansByteSource> {
  const [rows, cols, detRows, detCols] = header.shape,
    K = detRows * detCols;
  // Offset and model tables are copied out of authenticated chunks, never re-read.
  const tableCopies = header.chunks.flatMap(chunk => [
    { offset: chunk.arrays[1].offset, target: new Uint8Array(chunk.arrays[1].count * 4) },
    { offset: chunk.arrays[2].offset, target: new Uint8Array(chunk.arrays[2].count) },
  ]);
  // With a device, stream bytes are copied into resident GPU groups as their
  // chunks authenticate: the payload is read once and only hashed bytes are
  // ever decoded. A chunk larger than one group is split across groups on a
  // word boundary; the blocks a split cuts are copied whole on the GPU once
  // the authenticated offset tables locate them.
  type PayloadGroup = { size: number; end: number; buffer?: GPUBuffer; mapped?: Uint8Array };
  type Region = { start: number; end: number; offset: number; group: PayloadGroup };
  const groups: PayloadGroup[] = [];
  const regions: Region[] = [];
  const groupLimit = device ? payloadGroupLimit(device) : 0;
  if (device) {
    for (const chunk of header.chunks) {
      let { offset: start, count } = chunk.arrays[0];
      let group: PayloadGroup | undefined = groups[groups.length - 1];
      let offset = group ? Math.ceil(group.size / 4) * 4 : 0;
      if (group && offset + count > groupLimit) group = undefined;
      for (;;) {
        if (!group) { group = { size: 0, end: 0 }; groups.push(group); offset = 0; }
        const piece = count > groupLimit - offset ? Math.floor((groupLimit - offset) / 4) * 4 : count;
        group.size = offset + piece;
        // An empty region must not delay unmapping the group's last copied bytes.
        if (piece) group.end = start + piece;
        regions.push({ start, end: start + piece, offset, group });
        start += piece; count -= piece;
        if (!count) break;
        group = undefined;
      }
    }
  }
  requireQem(globalThis.crypto?.subtle, "checksum verification requires HTTPS or localhost; serve the viewer securely");
  // WebCrypto uses the platform's SHA-256 implementation without a JavaScript
  // loop over every byte. A served file streams the chunks through one response.
  // Cancellation, or any failure, ends that download instead of waiting for reads in flight.
  const stopping = new AbortController();
  const cancel = signal ? AbortSignal.any([signal, stopping.signal]) : stopping.signal;
  const stream = file.chunks?.(body, file.size, chunkBytes, cancel);
  const verifiedChunk = async (index: number) => {
    const begin = index * chunkBytes;
    const end = Math.min(header.bytes, (index + 1) * chunkBytes);
    cancel.throwIfAborted();
    // Request before the first await: queued stream reads resolve in call order.
    const bytes = stream
      ? await stream.next().then(part => part.done ? new ArrayBuffer(0) : part.value)
      : await file.slice(body + begin, body + end).arrayBuffer();
    requireQem(bytes.byteLength === end - begin, "truncated authentication chunk");
    const hash = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
    const actual = Array.from(hash, value => value.toString(16).padStart(2, "0")).join("");
    requireQem(actual === header.sha256[index], "payload checksum mismatch");
    return new Uint8Array(bytes);
  };
  // Four reads and digests stay in flight, so reading overlaps hashing while
  // at most four 64 MiB chunks are held.
  type Verified = { bytes: Uint8Array<ArrayBuffer> } | { error: unknown };
  const pending: Promise<Verified>[] = [];
  let next = 0;
  const enqueue = () => {
    const index = next++;
    pending.push(verifiedChunk(index).then(bytes => ({ bytes }), error => ({ error })));
  };
  for (let depth = 0; depth < 4 && next < header.sha256.length; depth++) enqueue();
  try {
    try {
      for (let index = 0; index < header.sha256.length; index++) {
        onStatus(`Verifying .qem ${index + 1}/${header.sha256.length}`);
        signal?.throwIfAborted();
        const result = await pending.shift()!;
        if ("error" in result) throw result.error;
        const bytes = result.bytes, start = index * chunkBytes, end = start + bytes.length;
        for (const region of regions) {
          const first = Math.max(start, region.start), last = Math.min(end, region.end);
          if (last <= first) continue;
          const group = region.group;
          if (!group.buffer) {
            group.buffer = device!.createBuffer({ size: Math.ceil(group.size / 4) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, mappedAtCreation: true });
            group.mapped = new Uint8Array(group.buffer.getMappedRange());
          }
          group.mapped!.set(bytes.subarray(first - start, last - start), region.offset + first - region.start);
          if (last === group.end) { group.buffer.unmap(); group.mapped = undefined; }
        }
        for (const { offset, target } of tableCopies) {
          const first = Math.max(start, offset), last = Math.min(end, offset + target.length);
          if (last > first) target.set(bytes.subarray(first - start, last - start), first - offset);
        }
        // Hashed and copied: nothing reads this chunk again, so its buffer may carry a later one.
        file.recycleChunk?.(bytes.buffer);
        if (next < header.sha256.length) enqueue();
      }
      // Resume the stream once more: its check for an oversized response runs after the last chunk.
      if (stream) requireQem((await stream.next()).done, "unexpected trailing authentication chunk");
    } finally {
      // The stream can only close after its queued reads settle; a failure must
      // also wait for them so no chunk arrives after the groups are released.
      // Ending the download first makes those reads settle at once.
      stopping.abort();
      await Promise.all(pending);
      await stream?.return(undefined);
    }
    for (const group of groups) {
      // A group whose chunks hold no stream bytes still needs a bindable buffer.
      group.buffer ??= device!.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      if (group.mapped) { group.buffer.unmap(); group.mapped = undefined; }
    }
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
    for (const [chunkIndex, chunk] of header.chunks.entries()) {
      const payload = chunk.arrays[0],
        blocks = Math.ceil(chunk.scans / 512);
      const local = new Uint32Array(tableCopies[chunkIndex * 2].target.buffer),
        models = tableCopies[chunkIndex * 2 + 1].target;
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
    }
    // Copy each block cut by a group split, whole and from authenticated bytes,
    // so the decoder binds every block in one buffer. Word-aligned extents keep
    // the GPU copies valid; the decoder skips the unaligned head through its
    // byte offset and never reads past the block's end.
    const cut = !device ? [] : blockMeta.filter(block => !regions.some(region => block.byte_start - body >= region.start && block.byte_end - body <= region.end));
    if (cut.length) {
      const copies: PayloadGroup[] = [], copied: Region[] = [];
      for (const block of cut) {
        const start = Math.floor((block.byte_start - body) / 4) * 4, end = Math.ceil((block.byte_end - body) / 4) * 4;
        requireQem(end - start <= groupLimit, "encoded block exceeds GPU buffer limits; use the native GPU application");
        let group: PayloadGroup | undefined = copies[copies.length - 1];
        if (!group || group.size + end - start > groupLimit) { group = { size: 0, end: 0 }; copies.push(group); }
        copied.push({ start, end, offset: group.size, group });
        group.size += end - start;
      }
      for (const group of copies) {
        group.buffer = device!.createBuffer({ size: group.size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        groups.push(group);
      }
      const encoder = device!.createCommandEncoder();
      for (const target of copied) {
        for (const source of regions) {
          // A staged region's last word is padding when its length is not a whole number of words.
          const first = Math.max(target.start, source.start), last = Math.min(target.end, source.start + Math.ceil((source.end - source.start) / 4) * 4);
          if (last > first) encoder.copyBufferToBuffer(source.group.buffer!, source.offset + first - source.start, target.group.buffer!, target.offset + first - target.start, last - first);
        }
      }
      device!.queue.submit([encoder.finish()]);
      regions.push(...copied);
    }
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
      residentPayload(name, start, end) {
        // Staged regions come first, so only a cut block resolves to its copy.
        const region = regions.find(region => start - body >= region.start && end - body <= region.end);
        requireQem(name === "payload" && region && end >= start, "invalid resident payload range");
        return { buffer: region.group.buffer!, offset: region.offset + start - body - region.start };
      },
      dispose() { for (const group of groups) group.buffer?.destroy(); },
      async read(name, start, end) {
        if (name === "manifest.json")
          return new TextEncoder().encode(JSON.stringify(mapped)).buffer;
        if (name === "entries") return entries.buffer;
        if (name === "lookup") return new ArrayBuffer(4);
        // Payload bytes are decoded only from the authenticated GPU groups; reading
        // the file again would bypass its checksums.
        requireQem(name !== "payload", "payload bytes stay in authenticated GPU groups; load the file with a WebGPU device");
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
  } catch (error) {
    // Staged groups belong to this admission until it returns a source.
    for (const group of groups) group.buffer?.destroy();
    throw error;
  }
}

/** Join compatible .qem acquisitions into one series without decoding counts or changing their order. */
export async function qemFilesSource(files: ArrayLike<QemByteFile>, onStatus: (text: string) => void = () => {}, badPixels: number[] = [], device?: GPUDevice, signal?: AbortSignal): Promise<RansByteSource> {
  const ordered = Array.from(files);
  requireQem(ordered.length > 0, "select at least one .qem file");
  // Compatibility follows from the authenticated headers, so a mismatched
  // series is rejected before any payload is read or staged on the GPU.
  // Each header is read once: its checksums authenticate the payload admitted below.
  const headers: QemHeader[] = [];
  for (const file of ordered) headers.push(await readQemHeader(file, signal));
  const first = headers[0].header;
  headers.forEach(({ header, badPixels: invalid }, index) => {
    if (JSON.stringify(header.shape) !== JSON.stringify(first.shape) || header.dtype !== first.dtype) {
      throw new Error(`QEM file ${ordered[index].name} has shape ${header.shape.join("x")} and dtype ${header.dtype}; ${ordered[0].name} has ${first.shape.join("x")} ${first.dtype}. Select acquisitions with matching native geometry and dtype.`);
    }
    requireQem(JSON.stringify(invalid) === JSON.stringify(headers[0].badPixels), "series detector validity masks differ; open each acquisition separately");
  });
  const detectorPixels = first.shape[2] * first.shape[3];
  requireQem(badPixels.every(index => Number.isInteger(index) && index >= 0 && index < detectorPixels), `badPixels must contain detector indices from 0 to ${detectorPixels - 1}`);
  const sources: RansByteSource[] = [];
  const manifests: RansManifest[] = [];
  try {
    for (let index = 0; index < ordered.length; index++) {
      const source = await admitQemFile(ordered[index], headers[index], text => onStatus(`${index + 1}/${ordered.length} ${ordered[index].name}: ${text}`), device, signal);
      // Owned from here, so a later failure releases its staged storage.
      sources.push(source);
      manifests.push(JSON.parse(new TextDecoder().decode(await source.read("manifest.json"))) as RansManifest);
    }
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
    return { mode: "local-folder",
      residentPayload(name, start, end) {
        const namespaced = /^acquisition-(\d+)\/(.+)$/.exec(name);
        requireQem(namespaced, "invalid resident acquisition name");
        return sources[Number(namespaced[1])].residentPayload!(namespaced[2], start, end);
      },
      dispose() { for (const source of sources) source.dispose!(); },
      async read(name, start, end) {
        if (name === "manifest.json") return new TextEncoder().encode(JSON.stringify(combined)).buffer;
        const namespaced = /^acquisition-(\d+)\/(.+)$/.exec(name);
        if (namespaced) return sources[Number(namespaced[1])].read(namespaced[2], start, end);
        const offsets = /^t(\d+)-offsets-(\d+)\.u32$/.exec(name);
        requireQem(offsets, `unknown series table ${name}`);
        return sources[Number(offsets[1])].read(`t0-offsets-${offsets[2]}.u32`);
      } };
  } catch (error) {
    for (const source of sources) source.dispose!();
    throw error;
  }
}
