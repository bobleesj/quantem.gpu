import type { AuthenticatedRead } from "./source112-worker-reader";
/** Internal ordered, authenticated disk prefetch. No GPU access or detached work. */
export interface Source112PrefetchProfile {
  payloadReadMs: number;
  payloadReadWaitMs: number;
  payloadHashMs: number;
  peakHostPrefetchBytes: number;
}
interface RecordBytes { record_bytes: number; sha256: string; chunk: number; }
const HOST_BUDGET = 2 * 1024 * 1024 * 1024;
const PREFETCH_RECORDS = 8;
const MAX_RECORD_BYTES = 256 * 1024 * 1024;
const digest = async (raw: ArrayBuffer) => Array.from(new Uint8Array(
  await crypto.subtle.digest('SHA-256', raw)), n => n.toString(16).padStart(2, '0')).join('');

/** The reservation includes two copies per record: the returned bytes and a
 * possible WebCrypto input snapshot. Browser/driver internal allocations are
 * not measurable here. Up to eight queued records read and authenticate in
 * parallel, within two GiB including the currently yielded record. Admission
 * remains ordered and never receives bytes before their digest passes. */
export async function* authenticatedRecords<T extends RecordBytes>(
  records: readonly T[], read: (record: T) => Promise<ArrayBuffer | AuthenticatedRead>,
  profile: Source112PrefetchProfile, signal?: AbortSignal,
): AsyncGenerator<{ record: T; raw: ArrayBuffer }> {
  // Reject before any disk operation; never silently exceed the host budget.
  for (const record of records) {
    if (!Number.isSafeInteger(record.record_bytes) || record.record_bytes <= 0 || record.record_bytes > MAX_RECORD_BYTES)
      throw new Error(`Source112 record ${record.chunk} exceeds the bounded authenticated read budget (256 MiB per record).`);
  }
  type Outcome = { raw: ArrayBuffer; error?: never } | { error: unknown; raw?: never };
  let reserved = 0;
  const start = (record: T): Promise<Outcome> => {
    signal?.throwIfAborted();
    reserved += record.record_bytes * 2;
    profile.peakHostPrefetchBytes = Math.max(profile.peakHostPrefetchBytes, reserved);
    // Attach the rejection handler immediately, including for speculative reads.
    return (async () => {
      const began = performance.now();
      const response = await read(record);
      const measured = response instanceof ArrayBuffer ? null : response;
      const raw = measured ? measured.raw : response as ArrayBuffer;
      profile.payloadReadMs += measured ? measured.readMs : performance.now() - began;
      signal?.throwIfAborted();
      if (!(raw instanceof ArrayBuffer) || raw.byteLength !== record.record_bytes)
        throw new Error(`Short source112 record ${record.chunk}.`);
      const hashBegan = performance.now();
      let actual: string;
      try { actual = measured ? measured.digest : await digest(raw); }
      finally {
        profile.payloadHashMs += measured ? measured.hashMs : performance.now() - hashBegan;
        reserved -= record.record_bytes; // The digest snapshot is no longer live.
      }
      signal?.throwIfAborted();
      if (actual !== record.sha256)
        throw new Error(`Source112 record ${record.chunk} failed its preserved SHA-256 digest.`);
      return { raw };
    })().catch(error => ({ error }));
  };
  const pending = new Map<number, Promise<Outcome>>();
  let next = 0;
  const fill = () => {
    while (next < records.length && pending.size < PREFETCH_RECORDS
      && reserved + records[next].record_bytes * 2 <= HOST_BUDGET) {
      pending.set(next, start(records[next]));
      next++;
    }
  };
  try {
    for (let index = 0; index < records.length; index++) {
      const record = records[index];
      fill();
      const current = pending.get(index);
      if (!current) throw new Error('Authenticated source read exceeds the reserved host budget.');
      const wait = performance.now();
      const outcome = await current;
      profile.payloadReadWaitMs += performance.now() - wait;
      pending.delete(index);
      if ('error' in outcome) throw outcome.error;
      signal?.throwIfAborted();
      fill();
      yield { record, raw: outcome.raw };
      reserved -= record.record_bytes;
      signal?.throwIfAborted();
    }
  } finally {
    // Blob reads and digest cannot be cancelled. Drain their handled outcome
    // before owner cleanup completes; never leak an unhandled rejection.
    await Promise.all(pending.values());
    reserved = 0;
  }
}
