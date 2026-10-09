/** CPU-only checks of streamed .qem reads: BYOB and default readers, exact bounds, pipelined verification. */
import assert from "node:assert/strict";
import test from "node:test";
import { qemFileSource, qemHttpFiles, type QemByteFile } from "../../src/quantem/gpu/detector/webgpu/qem-source";
import { syntheticQem } from "./qem-synthetic";
import { fakeDevice } from "./fake-gpu";

Object.assign(globalThis, { location: { href: "http://localhost/" } });
const tick = () => new Promise(resolve => setImmediate(resolve));
const pattern = (length: number) => Uint8Array.from({ length }, (_, index) => (index * 73 + 19) & 255);

/** Serve `bytes`. Ranged GETs stream fragmented parts from a byte stream (BYOB) or a default stream;
 * `extra` bytes are appended to (positive) or removed from (negative) every ranged response.
 * Parts are 1 to 11 bytes plus `partBytes`, so they never align with chunk boundaries. */
function serve(bytes: Uint8Array, byob: boolean, extra = 0, partBytes = 0) {
  const log = { ranges: [] as string[], byobRequests: 0, cancelled: 0 };
  globalThis.fetch = (async (_url: URL, init: RequestInit = {}) => {
    if (init.method === "HEAD") return new Response(null, { headers: { "Content-Length": String(bytes.length) } });
    const range = (init.headers as Record<string, string>).Range;
    log.ranges.push(range);
    const [first, last] = /bytes=(\d+)-(\d+)/.exec(range)!.slice(1).map(Number);
    const body = new Uint8Array(Math.max(0, last + 1 - first + extra));
    body.set(bytes.subarray(first, first + body.length));
    let at = 0;
    const stream = new ReadableStream({
      ...(byob ? { type: "bytes" as const } : {}),
      pull(controller: ReadableByteStreamController | ReadableStreamDefaultController<Uint8Array>) {
        const request = (controller as ReadableByteStreamController).byobRequest;
        if (at === body.length) { controller.close(); request?.respond(0); return; }
        const size = Math.min(partBytes + 1 + at % 11, body.length - at);
        if (request) {
          log.byobRequests++;
          const view = request.view!, count = Math.min(size, view.byteLength);
          new Uint8Array(view.buffer, view.byteOffset, count).set(body.subarray(at, at + count));
          at += count; request.respond(count);
        } else { controller.enqueue(body.slice(at, at + size)); at += size; }
      },
      cancel() { log.cancelled++; },
    });
    return { status: 206, body: stream, arrayBuffer: async () => body.slice().buffer } as unknown as Response;
  }) as typeof fetch;
  return log;
}

async function collect(file: QemByteFile, length: number, chunkBytes: number, queued: number): Promise<Uint8Array[]> {
  const stream = file.chunks!(0, length, chunkBytes);
  const pending: Promise<IteratorResult<ArrayBuffer, void>>[] = [];
  const chunks: Uint8Array[] = [];
  for (let index = 0; index < queued; index++) pending.push(stream.next());
  for (;;) {
    const part = await pending.shift()!;
    if (part.done) break;
    chunks.push(new Uint8Array(part.value.slice(0)));
    pending.push(stream.next());
  }
  await Promise.all(pending);
  return chunks;
}

test("fragmented BYOB and default streams arrive as exact chunks, read singly or four queued", async () => {
  const bytes = pattern(16 * 13 + 7);
  for (const byob of [true, false]) {
    for (const queued of [1, 4]) {
      const log = serve(bytes, byob);
      const [file] = await qemHttpFiles("/", ["a.qem"]);
      const chunks = await collect(file, bytes.length, 16, queued);
      assert.deepEqual(chunks.map(chunk => chunk.length), [...Array(13).fill(16), 7]);
      assert.deepEqual(Buffer.concat(chunks), Buffer.from(bytes));
      assert.equal(log.byobRequests > 0, byob, "byte streams are filled in place");
      assert.deepEqual(log.ranges, [`bytes=0-${bytes.length - 1}`]);
    }
  }
});

test("short and long responses are rejected with both readers", async () => {
  const bytes = pattern(16 * 5 + 3);
  for (const byob of [true, false]) {
    for (const [extra, message] of [[-1, /truncated stream/], [1, /oversized stream/]] as const) {
      serve(bytes, byob, extra);
      const [file] = await qemHttpFiles("/", ["a.qem"]);
      await assert.rejects(collect(file, bytes.length, 16, 4), message);
    }
  }
});

test("stopping after the first chunk cancels the download", async () => {
  const bytes = pattern(16 * 5);
  for (const byob of [true, false]) {
    const log = serve(bytes, byob);
    const [file] = await qemHttpFiles("/", ["a.qem"]);
    const stream = file.chunks!(0, bytes.length, 16);
    await stream.next();
    await stream.return(undefined);
    assert.equal(log.cancelled, 1);
  }
});

test("admission verifies every 64 MiB chunk from one ranged response", async () => {
  const bytes = syntheticQem([[40 << 20], [40 << 20]]);
  const body = Number(new DataView(bytes.buffer).getBigUint64(16, true));
  for (const byob of [true, false]) {
    const log = serve(bytes, byob, 0, 1 << 20);
    const [file] = await qemHttpFiles("/", ["two-chunks.qem"]);
    await qemFileSource(file);
    const streamed = `bytes=${body}-${bytes.length - 1}`;
    assert.equal(log.ranges.filter(range => range === streamed).length, 1);
    for (const range of log.ranges.filter(range => range !== streamed)) {
      const [first, last] = /bytes=(\d+)-(\d+)/.exec(range)!.slice(1).map(Number);
      assert.ok(last - first < 1 << 20, `no separate chunk-sized request: ${range}`);
    }
    const corrupted = bytes.slice();
    corrupted[corrupted.length - 1000] ^= 1;
    serve(corrupted, byob, 0, 1 << 20);
    const [changed] = await qemHttpFiles("/", ["two-chunks.qem"]);
    await assert.rejects(qemFileSource(changed), /payload checksum mismatch/);
  }
});

test("local reads keep four authentication chunks in flight", async () => {
  const bytes = syntheticQem([[200 << 20], [60 << 20]]);
  const waiting: (() => void)[] = [];
  let inFlight = 0, peak = 0;
  const file: QemByteFile = { name: "five-chunks.qem", size: bytes.length, slice(start = 0, end = bytes.length) {
    return { async arrayBuffer() {
      const copy = bytes.slice(start, end).buffer;
      if (end - start < 1 << 20) return copy;
      inFlight++; peak = Math.max(peak, inFlight);
      await new Promise<void>(resolve => waiting.push(resolve));
      inFlight--;
      return copy;
    } };
  } };
  let settled = false;
  const admitted = qemFileSource(file).finally(() => { settled = true; });
  while (!settled) { await tick(); waiting.shift()?.(); }
  await admitted;
  assert.equal(peak, 4);
});

test("a recycled chunk carries a later chunk with both readers", async () => {
  const bytes = pattern(16 * 6);
  for (const byob of [true, false]) {
    serve(bytes, byob);
    const [file] = await qemHttpFiles("/", ["a.qem"]);
    const stream = file.chunks!(0, bytes.length, 16);
    const first = (await stream.next()).value as ArrayBuffer;
    assert.deepEqual(new Uint8Array(first), bytes.subarray(0, 16));
    file.recycleChunk!(first);
    const second = (await stream.next()).value as ArrayBuffer;
    assert.deepEqual(new Uint8Array(second), bytes.subarray(16, 32));
    // BYOB transfers the reused buffer, detaching the caller's old reference.
    if (byob) assert.equal(first.byteLength, 0);
    else assert.equal(second, first);
    await stream.return(undefined);
  }
});

test("at most four recycled chunks wait for reuse", async () => {
  const bytes = pattern(16 * 8);
  serve(bytes, false);
  const [file] = await qemHttpFiles("/", ["a.qem"]);
  const stream = file.chunks!(0, bytes.length, 16);
  await stream.next();
  const offered = Array.from({ length: 6 }, () => new ArrayBuffer(16));
  offered.forEach(buffer => file.recycleChunk!(buffer));
  const later: ArrayBuffer[] = [];
  for await (const chunk of stream) later.push(chunk);
  assert.equal(later.length, 7);
  assert.deepEqual(later.map(chunk => offered.indexOf(chunk)), [0, 1, 2, 3, -1, -1, -1]);
});

test("recycled chunks never outlive a completed or failed stream", async () => {
  const bytes = pattern(16 * 4);
  for (const extra of [0, -1]) {
    serve(bytes, false, extra);
    const [file] = await qemHttpFiles("/", ["a.qem"]);
    const stream = file.chunks!(0, bytes.length, 16);
    const kept = [(await stream.next()).value as ArrayBuffer, new ArrayBuffer(16)];
    kept.forEach(buffer => file.recycleChunk!(buffer));
    await (async () => { for await (const chunk of stream) kept.push(chunk); })().catch(() => {});
    kept.forEach(buffer => file.recycleChunk!(buffer));
    serve(bytes, false);
    const reread = await collect(file, bytes.length, 16, 1);
    assert.deepEqual(Buffer.concat(reread), Buffer.from(bytes));
    const fresh = file.chunks!(0, bytes.length, 16);
    for await (const chunk of fresh) assert.ok(!kept.includes(chunk), "a buffer from an earlier stream was reused");
  }
});

test("admission recycles a chunk only after its checksum matched", async () => {
  const bytes = syntheticQem([[40 << 20], [40 << 20]]);
  const digest = crypto.subtle.digest.bind(crypto.subtle);
  const digested = new Set<unknown>();
  crypto.subtle.digest = async (algorithm: AlgorithmIdentifier, data: BufferSource) => {
    const hash = await digest(algorithm, data);
    digested.add(data);
    return hash;
  };
  try {
    for (const corrupt of [false, true]) {
      const served = bytes.slice();
      if (corrupt) served[served.length - 1000] ^= 1;
      serve(served, false, 0, 1 << 20);
      const [file] = await qemHttpFiles("/", ["two-chunks.qem"]);
      const recycled: ArrayBuffer[] = [];
      const recycle = file.recycleChunk!.bind(file);
      file.recycleChunk = buffer => {
        assert.ok(digested.has(buffer), "recycled before its digest finished");
        recycled.push(buffer);
        recycle(buffer);
      };
      const admitted = qemFileSource(file);
      if (corrupt) await assert.rejects(admitted, /payload checksum mismatch/);
      else await admitted;
      assert.equal(recycled.length, corrupt ? 1 : 2, "the chunk that failed authentication is never recycled");
    }
  } finally { crypto.subtle.digest = digest; }
});

/** Serve `bytes`; the payload stream stops after `deliver` bytes and only an abort of its request ends it. */
function stalling(bytes: Uint8Array, deliver: number) {
  const log = { aborted: false, afterAbort: 0 };
  globalThis.fetch = (async (_url: URL, init: RequestInit = {}) => {
    if (log.aborted) log.afterAbort++;
    if (init.method === "HEAD") return new Response(null, { headers: { "Content-Length": String(bytes.length) } });
    const [first, last] = /bytes=(\d+)-(\d+)/.exec((init.headers as Record<string, string>).Range)!.slice(1).map(Number);
    const body = bytes.slice(first, last + 1);
    if (body.length < 1 << 20) return new Response(body, { status: 206 });
    let at = 0;
    const stream = new ReadableStream<Uint8Array>({
      start(controller) { init.signal?.addEventListener("abort", () => { log.aborted = true; controller.error(init.signal!.reason); }); },
      pull(controller) {
        if (at >= deliver) return new Promise(() => {});
        controller.enqueue(body.slice(at, at + (1 << 20)));
        at += 1 << 20;
      },
    });
    return { status: 206, body: stream } as unknown as Response;
  }) as typeof fetch;
  return log;
}
const within = <T>(promise: Promise<T>, milliseconds: number) => Promise.race([promise,
  new Promise<never>((_, reject) => setTimeout(() => reject(new Error("still waiting on the stalled download")), milliseconds).unref())]);

test("cancelling admission ends the download at once and releases staged buffers", async () => {
  const bytes = syntheticQem([[40 << 20], [40 << 20]]);
  const log = stalling(bytes, 70 << 20);
  const gpu = fakeDevice();
  const controller = new AbortController();
  const [file] = await qemHttpFiles("/", ["stalled.qem"], controller.signal);
  const admitted = qemFileSource(file, text => { if (text.startsWith("Verifying .qem 2/")) controller.abort(); }, gpu.device, controller.signal);
  await assert.rejects(within(admitted, 2000), { name: "AbortError" });
  assert.ok(log.aborted, "the payload request itself was aborted");
  assert.equal(log.afterAbort, 0, "nothing is requested after cancellation");
  assert.ok(gpu.buffers.length > 0 && gpu.buffers.every(buffer => buffer.destroyed));
});

test("a failed admission ends its download instead of waiting for reads in flight", async () => {
  const bytes = syntheticQem([[40 << 20], [40 << 20]]);
  const body = Number(new DataView(bytes.buffer).getBigUint64(16, true));
  bytes[body + 1000] ^= 1;
  const log = stalling(bytes, 70 << 20);
  const [file] = await qemHttpFiles("/", ["corrupt.qem"]);
  await assert.rejects(within(qemFileSource(file, () => {}, fakeDevice().device), 2000), /payload checksum mismatch/);
  assert.ok(log.aborted);
});

test("a server that ignores the range stops sending once the response is refused", async () => {
  const bytes = pattern(1 << 16);
  let cancelled = 0;
  globalThis.fetch = (async (_url: URL, init: RequestInit = {}) => {
    if (init.method === "HEAD") return new Response(null, { headers: { "Content-Length": String(bytes.length) } });
    const body = new ReadableStream<Uint8Array>({ pull(controller) { controller.enqueue(bytes.slice(0, 1024)); }, cancel() { cancelled++; } });
    return { status: 200, body } as unknown as Response;
  }) as typeof fetch;
  const [file] = await qemHttpFiles("/", ["whole.qem"]);
  await assert.rejects(file.slice(0, 56).arrayBuffer(), /ignored a byte range/);
  await assert.rejects(file.chunks!(0, bytes.length, 16).next(), /ignored a byte range/);
  assert.equal(cancelled, 2);
});
