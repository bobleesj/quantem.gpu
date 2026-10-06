/** Byte-exact acquisition of a served or user-granted rANS export. */
export interface RansDirectoryHandle {
  getDirectoryHandle(name: string): Promise<RansDirectoryHandle>;
  getFileHandle(name: string): Promise<{ getFile(): Promise<File> }>;
}

export interface RansByteSource {
  mode: "http" | "local-folder";
  read(name: string, start?: number, end?: number): Promise<ArrayBuffer>;
}

function localParts(name: string): string[] {
  const parts = name.split("/");
  if (!name || parts.some(p => !p || p === "." || p === ".." || /[:\\?#]/.test(p))) {
    throw new Error(`Invalid rANS relative file path: ${name}`);
  }
  return parts;
}

export function ransHttpSource(baseUrl: string): RansByteSource {
  const base = baseUrl.endsWith("/") ? baseUrl : baseUrl + "/";
  return { mode: "http", async read(name, start, end) {
    const ranged = start !== undefined && end !== undefined;
    const response = await fetch(base + name, ranged
      ? { headers: { Range: `bytes=${start}-${end - 1}` } } : undefined);
    if (!response.ok || (ranged && response.status !== 206)) {
      throw new Error(`${name}: HTTP ${response.status}; ranged payloads require a Range-capable server`);
    }
    const bytes = await response.arrayBuffer();
    if (ranged && bytes.byteLength !== end - start) throw new Error(`${name}: payload range length mismatch`);
    return bytes;
  } };
}

/** Accept the export root or its rans directory; never fall back to HTTP. */
export async function ransLocalSource(directory: RansDirectoryHandle): Promise<RansByteSource> {
  let root = directory;
  try { await root.getFileHandle("manifest.json"); }
  catch (error) {
    if ((error as { name?: string }).name !== "NotFoundError") throw error;
    root = await directory.getDirectoryHandle("rans");
    await root.getFileHandle("manifest.json");
  }
  const files = new Map<string, Promise<File>>();
  return { mode: "local-folder", async read(name, start, end) {
    const parts = localParts(name);
    let pending = files.get(name);
    if (!pending) {
      pending = (async () => {
        let dir = root;
        for (const part of parts.slice(0, -1)) dir = await dir.getDirectoryHandle(part);
        return (await dir.getFileHandle(parts[parts.length - 1])).getFile();
      })();
      files.set(name, pending);
    }
    const file = await pending;
    const lo = start ?? 0, hi = end ?? file.size;
    if (!Number.isSafeInteger(lo) || !Number.isSafeInteger(hi) || lo < 0 || hi < lo || hi > file.size) {
      throw new Error(`${name}: invalid byte range ${lo}:${hi} for ${file.size}-byte file`);
    }
    const bytes = await file.slice(lo, hi).arrayBuffer();
    if (bytes.byteLength !== hi - lo) throw new Error(`${name}: short local file read`);
    return bytes;
  } };
}

/** Directory-input fallback for browsers exposing File.webkitRelativePath. */
export async function ransLocalFilesSource(files: ArrayLike<File>): Promise<RansByteSource> {
  const byPath = new Map<string, File>();
  for (let i = 0; i < files.length; i++) {
    const file = files[i];
    const path = file.webkitRelativePath || file.name;
    localParts(path);
    byPath.set(path, file);
  }
  const manifests = [...byPath.keys()].filter(path => path === "manifest.json" || path.endsWith("/manifest.json"));
  const preferred = manifests.filter(path => path.endsWith("/rans/manifest.json") || path === "rans/manifest.json");
  const candidates = preferred.length ? preferred : manifests;
  if (candidates.length !== 1) throw new Error("Select one rANS export folder containing its manifest.json and payload files");
  const prefix = candidates[0].slice(0, -"manifest.json".length);
  const directory = (base: string): RansDirectoryHandle => ({
    async getDirectoryHandle(name) { return directory(base + name + "/"); },
    async getFileHandle(name) {
      const file = byPath.get(base + name);
      if (!file) throw new DOMException(`Missing local rANS file: ${base + name}`, "NotFoundError");
      return { async getFile() { return file; } };
    },
  });
  return ransLocalSource(directory(prefix));
}

/** Per-read elapsed durations overlap; wait time counts exposed await intervals. */
export interface RansPayloadProfile {
  payloadReadMs: number;
  payloadReadWaitMs: number;
  payloadStageMs: number;
  payloadChunks: number;
}

/** Copy an exact payload range with at most four retained 32 MiB reads.
 *
 * The mapped destination belongs to the caller. Reads may complete out of order,
 * but copies preserve file order. A finished copy releases its read before the
 * next request starts, keeping payload staging bounded to 128 MiB. On failure,
 * outstanding reads settle before returning; no deferred copy can touch a freed
 * destination. File reads are not abortable through RansByteSource.
 */
export async function copyRansPayload(
  source: RansByteSource, name: string, start: number, bytes: number,
  target: Uint8Array, offset: number, profile: RansPayloadProfile,
): Promise<void> {
  const chunkBytes = 32 * 1024 * 1024;
  const depth = 4;
  type ReadResult = { ok: true; raw: Uint8Array } | { ok: false; error: unknown };
  let failed = false;
  const readChunk = async (position: number): Promise<ReadResult> => {
    const size = Math.min(chunkBytes, bytes - position);
    const began = performance.now();
    try {
      const raw = new Uint8Array(await source.read(name, start + position, start + position + size));
      if (raw.byteLength !== size) throw new Error(`${name}: expected ${size} payload bytes, received ${raw.byteLength}`);
      return { ok: true, raw };
    } catch (error) {
      failed = true;
      return { ok: false, error };
    } finally {
      profile.payloadReadMs += performance.now() - began;
      profile.payloadChunks++;
    }
  };
  const pending: Promise<ReadResult>[] = [];
  let nextPosition = 0;
  const queueNext = () => {
    pending.push(readChunk(nextPosition));
    nextPosition += chunkBytes;
  };
  // Keep the current read/result inside this completed call, rather than
  // retaining it in the outer loop while refilling the next read slot.
  const copyNext = async (next: Promise<ReadResult>, position: number): Promise<void> => {
    const waitBegan = performance.now();
    const result = await next;
    profile.payloadReadWaitMs += performance.now() - waitBegan;
    if (!result.ok) throw result.error;
    const stageBegan = performance.now();
    target.set(result.raw, offset + position);
    profile.payloadStageMs += performance.now() - stageBegan;
  };
  for (let i = 0; i < depth && nextPosition < bytes && !failed; i++) queueNext();
  try {
    for (let position = 0; position < bytes; position += chunkBytes) {
      await copyNext(pending.shift()!, position);
      if (nextPosition < bytes && !failed) queueNext();
    }
  } finally {
    // Convert rejections into values at read launch, so an out-of-order failure
    // cannot become an unhandled rejection while waiting for an earlier chunk.
    await Promise.all(pending);
    pending.length = 0;
  }
}
