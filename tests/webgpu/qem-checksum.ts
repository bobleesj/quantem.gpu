/** Transfer ownership, fallback and failure contracts for the QEM digest pool. */
import assert from "node:assert/strict";
import test from "node:test";
import { createQemHasher } from "../../src/quantem/gpu/detector/webgpu/qem-checksum";
import { qemFileSource } from "../../src/quantem/gpu/detector/webgpu/qem-source";
import { syntheticQem } from "./qem-synthetic";
import { fakeDevice } from "./fake-gpu";

const nativeWorker = globalThis.Worker;
let mode = "normal";
const created: TestWorker[] = [];
class TestWorker {
  onmessage?: (event: { data: unknown }) => void;
  onerror?: (event: { preventDefault(): void }) => void;
  onmessageerror?: () => void;
  terminated = false;
  constructor() {
    if (mode === "constructor-blocked") throw new Error("CSP blocks workers");
    created.push(this);
    queueMicrotask(() => {
      if (mode === "blocked") this.onerror?.({ preventDefault() {} });
      else if (mode !== "starting") this.onmessage?.({ data: { ready: true } });
    });
  }
  postMessage(data: { id: number; bytes: ArrayBuffer }, transfer: Transferable[]) {
    const received = structuredClone(data, { transfer });
    if (mode === "pending") return;
    if (mode === "crash") { queueMicrotask(() => this.onerror?.({ preventDefault() {} })); return; }
    if (mode === "message-error") { queueMicrotask(() => this.onmessageerror?.()); return; }
    void crypto.subtle.digest("SHA-256", received.bytes).then(hash => {
      const response = structuredClone({ ...received, hash }, { transfer: [received.bytes, hash] });
      setTimeout(() => { if (!this.terminated) this.onmessage?.({ data: response }); }, data.id % 2 ? 0 : 5);
    });
  }
  terminate() { this.terminated = true; }
}
function setup(value = "normal") {
  mode = value; created.length = 0;
  globalThis.Worker = TestWorker as unknown as typeof Worker;
  return new AbortController();
}
const tick = () => new Promise(resolve => setImmediate(resolve));

test("four transferred chunks return exact bytes and hashes even out of order", async () => {
  const controller = setup();
  const pool = createQemHasher(20, controller.signal);
  try {
    const inputs = Array.from({ length: 4 }, (_, i) => new Uint8Array(1024).fill(i + 1));
    const expected = await Promise.all(inputs.map(bytes => crypto.subtle.digest("SHA-256", bytes)));
    const results = await Promise.all(inputs.map(bytes => pool.digest(bytes.buffer)));
    assert.equal(created.length, 4);
    for (let i = 0; i < inputs.length; i++) {
      assert.equal(inputs[i].byteLength, 0, "ownership transferred, not cloned");
      assert.deepEqual(new Uint8Array(results[i].bytes), new Uint8Array(1024).fill(i + 1));
      assert.deepEqual(results[i].hash, expected[i]);
    }
  } finally { pool.dispose(); globalThis.Worker = nativeWorker; }
  assert.ok(created.every(worker => worker.terminated));
});

test("worker startup restrictions retain the verified WebCrypto fallback", async () => {
  for (const value of ["constructor-blocked", "blocked", "absent", "small"]) {
    const controller = setup(value);
    if (value === "absent") globalThis.Worker = undefined as unknown as typeof Worker;
    const pool = createQemHasher(value === "small" ? 1 : 4, controller.signal);
    try {
      const bytes = new Uint8Array([1, 2, 3]).buffer;
      const expected = await crypto.subtle.digest("SHA-256", bytes);
      const result = await pool.digest(bytes);
      assert.equal(bytes.byteLength, 3, "fallback still owns its input");
      assert.deepEqual(result.hash, expected);
      assert.deepEqual(result.bytes, bytes);
    } finally { pool.dispose(); globalThis.Worker = nativeWorker; }
    assert.ok(created.every(worker => worker.terminated));
  }
});

test("cancellation settles startup and transferred jobs without retaining workers", async () => {
  for (const value of ["starting", "pending"]) {
    const controller = setup(value), pool = createQemHasher(4, controller.signal);
    const job = pool.digest(new ArrayBuffer(64));
    const rejected = assert.rejects(job, /cancelled/);
    await tick();
    controller.abort(new Error("cancelled"));
    await rejected;
    pool.dispose(); globalThis.Worker = nativeWorker;
    assert.ok(created.every(worker => worker.terminated));
  }
});

test("a runtime worker failure rejects verification rather than admitting bytes", async () => {
  for (const value of ["crash", "message-error"]) {
    const controller = setup(value), pool = createQemHasher(4, controller.signal);
    try {
      const jobs = Array.from({ length: 4 }, () => pool.digest(new ArrayBuffer(64)));
      const results = await Promise.allSettled(jobs);
      assert.ok(results.every(result => result.status === "rejected"));
      assert.ok(created.every(worker => worker.terminated));
      await assert.rejects(pool.digest(new ArrayBuffer(1)), /worker/);
    } finally { pool.dispose(); globalThis.Worker = nativeWorker; }
  }
});

test("worker verification rejects a damaged acquisition and releases staged bytes", async () => {
  const bytes = syntheticQem([[40 << 20], [40 << 20]]);
  for (const corrupt of [false, true]) {
    setup();
    const gpu = fakeDevice();
    if (corrupt) bytes[bytes.length - 1000] ^= 1;
    try {
      const admitted = qemFileSource(new File([bytes], "acquisition.qem"), () => {}, gpu.device);
      if (corrupt) await assert.rejects(admitted, /payload checksum mismatch/);
      else (await admitted).dispose?.();
      assert.equal(created.length, 2, "both authentication chunks use workers");
      assert.ok(created.every(worker => worker.terminated));
      assert.ok(gpu.buffers.length > 0 && gpu.buffers.every(buffer => buffer.destroyed));
    } finally { globalThis.Worker = nativeWorker; }
  }
});
