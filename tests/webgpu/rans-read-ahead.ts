/** CPU-only byte ordering, staging bounds and failure ownership tests. */
import assert from "node:assert/strict";
import test from "node:test";
import { copyRansPayload, type RansByteSource, type RansPayloadProfile } from "../../src/quantem/gpu/detector/compute/webgpu/rans-source";

const CHUNK = 32 * 1024 * 1024;
const tick = () => new Promise<void>(resolve => setImmediate(resolve));
const profile = (): RansPayloadProfile => ({ payloadReadMs: 0, payloadReadWaitMs: 0, payloadStageMs: 0, payloadChunks: 0 });

function controlledSource(start = 37) {
  const requests: { start: number; end: number }[] = [];
  const pending = new Map<number, { resolve: (value: ArrayBuffer) => void; reject: (reason: unknown) => void }>();
  let live = 0, peak = 0;
  const source: RansByteSource = { mode: "local-folder", read(name, first, end) {
    assert.equal(name, "payload.bin");
    assert.equal(first, start + requests.length * CHUNK);
    assert.ok(end! - first! <= CHUNK);
    const index = requests.length;
    requests.push({ start: first!, end: end! });
    live++; peak = Math.max(peak, live);
    return new Promise((resolve, reject) => pending.set(index, { resolve, reject }));
  } };
  return {
    source, requests,
    copied() { live--; },
    get peak() { return peak; },
    complete(index: number, short = false) {
      const request = pending.get(index)!; pending.delete(index);
      const size = requests[index].end - requests[index].start - Number(short);
      request.resolve(new Uint8Array(size).fill(index + 1).buffer);
    },
    reject(index: number, error: Error) {
      const request = pending.get(index)!; pending.delete(index); request.reject(error);
    },
  };
}

test("four retained reads preserve out-of-order completions, offsets and the exact tail", async () => {
  const control = controlledSource();
  const bytes = CHUNK * 6 + 17;
  const target = new Uint8Array(bytes + 11).fill(255);
  const nativeSet = target.set.bind(target);
  const copies: number[] = [];
  target.set = (raw, offset) => {
    copies.push(offset!);
    nativeSet(raw, offset);
    control.copied();
  };
  const timings = profile();
  const loading = copyRansPayload(control.source, "payload.bin", 37, bytes, target, 5, timings);
  assert.equal(control.requests.length, 4);
  control.complete(3); control.complete(1); control.complete(2);
  await tick();
  assert.equal(copies.length, 0);
  assert.equal(control.requests.length, 4);
  control.complete(0);
  await tick();
  assert.equal(control.requests.length, 7);
  assert.equal(control.peak, 4);
  control.complete(6); control.complete(5); control.complete(4);
  await loading;
  assert.deepEqual(copies, Array.from({ length: 7 }, (_, index) => 5 + index * CHUNK));
  for (let index = 0; index < 7; index++) {
    const first = 5 + index * CHUNK;
    const size = Math.min(CHUNK, bytes - index * CHUNK);
    assert.ok(Buffer.from(target.buffer, target.byteOffset + first, size).equals(Buffer.alloc(size, index + 1)));
  }
  assert.ok(target.subarray(0, 5).every(value => value === 255));
  assert.ok(target.subarray(5 + bytes).every(value => value === 255));
  assert.equal(timings.payloadChunks, 7);
  assert.ok(timings.payloadReadMs >= 0 && timings.payloadReadWaitMs >= 0);
  assert.ok(timings.payloadStageMs >= 0);
});

test("a later failed read is handled immediately and drained before rejection", async () => {
  const control = controlledSource();
  const target = new Uint8Array(CHUNK * 5);
  const nativeSet = target.set.bind(target);
  let copies = 0, settled = false;
  target.set = (raw, offset) => { copies++; nativeSet(raw, offset); control.copied(); };
  const failure = new Error("source unavailable");
  const loading = copyRansPayload(control.source, "payload.bin", 37, target.length, target, 0, profile());
  const rejected = assert.rejects(loading, error => error === failure).then(() => { settled = true; });
  control.reject(1, failure);
  await tick();
  control.complete(0);
  await tick();
  assert.equal(control.requests.length, 4);
  assert.equal(settled, false);
  assert.equal(copies, 1);
  control.complete(3); control.complete(2);
  await rejected;
  assert.equal(copies, 1);
});

test("short reads and destination-copy failures reject without publishing remaining chunks", async () => {
  for (const short of [true, false]) {
    const control = controlledSource();
    const target = new Uint8Array(1);
    const loading = copyRansPayload(control.source, "payload.bin", 37, CHUNK * 2, target, 0, profile());
    const rejected = assert.rejects(loading, short ? /expected .* payload bytes/ : RangeError);
    control.complete(0, short);
    await tick();
    control.complete(1);
    await rejected;
    assert.equal(control.requests.length, 2);
  }
});
