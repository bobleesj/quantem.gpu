/** Verified group storage owned by admission until a resident set takes it over. */
export interface RansPreparedGroup {
  checkpoints: GPUBuffer;
  tables: GPUBuffer;
  offsets: GPUBuffer;
  entriesWord: number;
}

/** Byte reader of one .qem series as the resident set loads it (qem-source.ts builds it). */
export interface RansByteSource {
  mode: "http" | "local-folder";
  read(name: string, start?: number, end?: number): Promise<ArrayBuffer>;
  /** Where an authenticated payload range lies on the GPU. A .qem source admitted with a
   * device stages every block and throws for any other range; a source that answers
   * undefined sends that block to the per-block upload path. */
  residentPayload?(name: string, start: number, end: number): { buffer: GPUBuffer; offset: number } | undefined;
  /** Exact checkpoints built while later authenticated payload groups loaded. */
  residentCheckpoints?(payload: GPUBuffer): RansPreparedGroup | undefined;
  /** Release staged GPU storage; a loaded resident set owns it instead. */
  dispose?(): void;
}

/** Largest mapped GPU payload group. Browsers refuse mapped allocations far below
 * the advertised device limits (a 4.29 GB mapping failed on Apple Metal WebGPU);
 * several groups of at most 256 MiB still keep the whole series resident.
 */
export function payloadGroupLimit(device: GPUDevice): number {
  return Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize, 256 * 1024 * 1024);
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
