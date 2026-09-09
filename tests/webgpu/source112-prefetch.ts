import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { authenticatedRecords } from '../../src/quantem/gpu/detector/compute/webgpu/source112-prefetch';
const bytes = Uint8Array.of(1, 2, 3, 4);
const sha256 = createHash('sha256').update(bytes).digest('hex');
const records = Array.from({ length: 3 }, (_, chunk) => ({ chunk, record_bytes: 4, sha256 }));
const profile = () => ({ payloadReadMs: 0, payloadReadWaitMs: 0, payloadHashMs: 0, peakHostPrefetchBytes: 0 });
const tick = () => new Promise(resolve => setTimeout(resolve, 0));

test('reads ahead during ordered admission and authenticates every record', async () => {
  const events: string[] = [], stats = profile();
  for await (const { record, raw } of authenticatedRecords(records, async r => {
    events.push(`read${r.chunk}`); return bytes.slice().buffer;
  }, stats)) {
    events.push(`upload${record.chunk}`);
    assert.deepEqual(new Uint8Array(raw), bytes);
    await tick(); // Mock queue completion while the next read/digest progresses.
  }
  assert.ok(events.indexOf('read1') < events.indexOf('upload0'));
  assert.deepEqual(events.filter(e => e.startsWith('upload')), ['upload0', 'upload1', 'upload2']);
  assert.equal(stats.peakHostPrefetchBytes, 24);
  assert.ok(stats.payloadHashMs >= 0);
});

test('corrupt prefetched data is never admitted and rejects in record order', async () => {
  const uploaded: number[] = [];
  await assert.rejects(async () => {
    for await (const { record } of authenticatedRecords(records, async r =>
      r.chunk === 1 ? new Uint8Array(4).buffer : bytes.slice().buffer, profile())) {
      uploaded.push(record.chunk); await tick();
    }
  }, /record 1 failed.*SHA-256/);
  assert.deepEqual(uploaded, [0]);
});

test('consumer GPU failure drains a pending read and preserves original error', async () => {
  let release!: () => void, drained = false;
  const pending = new Promise<ArrayBuffer>((_, reject) => { release = () => { drained = true; reject(Error('disk failure')); }; });
  let finished = false;
  const run = (async () => {
    for await (const _ of authenticatedRecords(records, async r => r.chunk ? pending : bytes.slice().buffer, profile()))
      throw Error('GPU admission failure');
  })().finally(() => { finished = true; });
  const checked = assert.rejects(run, /GPU admission failure/);
  while (!release) await tick();
  await tick(); assert.equal(finished, false);
  release(); await checked; assert.equal(drained, true);
});

test('abort drains read ahead and never uploads its result', async () => {
  const controller = new AbortController();
  let reads = 0, uploads = 0;
  await assert.rejects(async () => {
    for await (const _ of authenticatedRecords(records, async () => {
      reads++; await tick(); return bytes.slice().buffer;
    }, profile(), controller.signal)) {
      uploads++; controller.abort();
    }
  }, { name: 'AbortError' });
  assert.equal(uploads, 1); assert.equal(reads, 3);
});

test('oversized records reject before any disk read', async () => {
  let reads = 0;
  await assert.rejects(async () => {
    for await (const _ of authenticatedRecords([{ ...records[0], record_bytes: 256 * 1024 ** 2 + 1 }],
      async () => { reads++; return bytes.buffer; }, profile())) { /* no admission */ }
  }, /256 MiB/);
  assert.equal(reads, 0);
});

test('short reads reject before hashing or GPU admission', async () => {
  await assert.rejects(async () => {
    for await (const _ of authenticatedRecords(records, async () => new ArrayBuffer(1), profile()))
      assert.fail('Must not admit short bytes');
  }, /Short source112 record 0/);
});

test('budget bounds concurrent records including raw bytes and digest snapshots', async () => {
  // Mock a large Blob result without allocating/reading scientific-sized data.
  // WebCrypto hashes the actual four backing bytes; the reservation sees256MiB.
  class BudgetBytes extends ArrayBuffer {
    get byteLength() { return 256 * 1024 ** 2; }
  }
  const input = Array.from({length: 6}, (_, chunk) => ({ ...records[0], chunk, record_bytes: 256 * 1024 ** 2 }));
  const stats = profile(), events: string[] = [];
  for await (const { record } of authenticatedRecords(input, async r => {
    events.push(`read${r.chunk}`);
    const raw = new BudgetBytes(4); new Uint8Array(raw).set(bytes); return raw;
  }, stats)) {
    events.push(`upload${record.chunk}`); await tick();
  }
  assert.deepEqual(events.filter(e => e.startsWith('upload')), input.map(r => `upload${r.chunk}`));
  assert.ok(events.indexOf('read1') < events.indexOf('upload0'));
  assert.ok(events.indexOf('read2') > events.indexOf('read1'));
  assert.equal(stats.peakHostPrefetchBytes, 2 * 1024 * 1024 ** 2);
});


test('large records remain authenticated and ordered across small neighbors', async () => {
  const sizes = [4, 128 * 1024 ** 2 + 4, 4, 256 * 1024 ** 2, 4];
  const input = sizes.map((record_bytes, chunk) => ({ chunk, record_bytes, sha256 }));
  const stats = profile(), events: string[] = [];
  for await (const { record } of authenticatedRecords(input, async r => {
    events.push(`read${r.chunk}`);
    class LogicalBytes extends ArrayBuffer { get byteLength() { return r.record_bytes; } }
    const raw = new LogicalBytes(4); new Uint8Array(raw).set(bytes); return raw;
  }, stats)) { events.push(`upload${record.chunk}`); await tick(); }
  assert.deepEqual(events.filter(e => e.startsWith('upload')), sizes.map((_, i) => `upload${i}`));
  assert.ok(stats.peakHostPrefetchBytes <= 2 * 1024 * 1024 ** 2);
});

test('large serial record corruption and abort never admit its bytes', async () => {
  for (const abort of [false, true]) {
    const controller = new AbortController();
    const input = [{ ...records[0], record_bytes: 256 * 1024 ** 2 }];
    class LogicalBytes extends ArrayBuffer { get byteLength() { return input[0].record_bytes; } }
    await assert.rejects(async () => {
      for await (const _ of authenticatedRecords(input, async () => {
        if (abort) controller.abort();
        return new LogicalBytes(4);
      }, profile(), controller.signal)) assert.fail('Must not admit unauthenticated bytes');
    }, abort ? { name: 'AbortError' } : /SHA-256/);
  }
});

test('out-of-order disk completions stay authenticated and ordered with eight queued reads', async () => {
  const input = Array.from({length: 9}, (_, chunk) => ({chunk, record_bytes: 4, sha256}));
  const releases = new Map<number, () => void>(), reads: number[] = [], uploaded: number[] = [];
  const run = (async () => {
    for await (const {record, raw} of authenticatedRecords(input, record => {
      reads.push(record.chunk);
      return new Promise<ArrayBuffer>(resolve => releases.set(record.chunk, () => resolve(bytes.slice().buffer)));
    }, profile())) {
      assert.deepEqual(new Uint8Array(raw), bytes);
      uploaded.push(record.chunk);
    }
  })();
  await tick();
  assert.deepEqual(reads, [0, 1, 2, 3, 4, 5, 6, 7]);
  for (const i of [7, 6, 5, 4, 3, 2, 1]) releases.get(i)!();
  await tick(); assert.deepEqual(uploaded, []);
  releases.get(0)!();
  while (uploaded.length < input.length) {
    for (const [i, release] of releases) { if (i >= 8) release(); }
    await tick();
  }
  await run;
  assert.deepEqual(uploaded, input.map(r => r.chunk));
});

test('worker authentication is checked before ordered admission without a second hash', async () => {
  const stats = profile();
  let uploads = 0;
  for await (const _ of authenticatedRecords(records, async () => ({
    raw: bytes.slice().buffer, digest: sha256, readMs: 2, hashMs: 3,
  }), stats)) uploads++;
  assert.equal(uploads, 3);
  assert.equal(stats.payloadReadMs, 6);
  assert.equal(stats.payloadHashMs, 9);
  await assert.rejects(async () => {
    for await (const _ of authenticatedRecords(records, async () => ({
      raw: bytes.slice().buffer, digest: 'bad', readMs: 2, hashMs: 3,
    }), profile())) assert.fail('Unauthenticated worker bytes escaped');
  }, /SHA-256/);
});
