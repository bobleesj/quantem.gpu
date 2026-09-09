import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {resolveObjectURL} from 'node:buffer';
import {Worker as Thread} from 'node:worker_threads';
import {Source112WorkerReader} from '../../src/quantem/gpu/detector/compute/webgpu/source112-worker-reader';

// Run the exact browser worker program on real Node threads, with Blob and
// transferable ArrayBuffer support. Only the event/constructor surface is adapted.
class BrowserWorker {
  onmessage?: (event: {data: unknown}) => void;
  onerror?: (event: {message: string}) => void;
  onmessageerror?: () => void;
  thread?: Thread;
  closed = false;
  ready: Promise<void>;
  constructor(url: string) {
    this.ready = resolveObjectURL(url)!.text().then(source => {
      if (this.closed) return;
      this.thread = new Thread(`const {parentPort}=require('node:worker_threads');
        global.crypto=require('node:crypto').webcrypto;
        global.self={postMessage:(data,transfer)=>parentPort.postMessage(data,transfer)};
        ${source}
        parentPort.on('message',data=>self.onmessage({data}));`, {eval: true});
      this.thread.on('message', data => this.onmessage?.({data}));
      this.thread.on('error', error => this.onerror?.({message: error.message}));
    });
  }
  postMessage(blob: Blob) { void this.ready.then(() => this.thread?.postMessage(blob)); }
  terminate() { this.closed = true; void this.thread?.terminate(); }
}

test('real worker reads and authenticates original bytes, then reuses bounded lanes', async () => {
  const original = globalThis.Worker;
  globalThis.Worker = BrowserWorker as unknown as typeof Worker;
  const reader = new Source112WorkerReader();
  try {
    for (let repeat = 0; repeat < 2; repeat++) {
      const pending = Array.from({length: 8}, (_, i) => {
        const raw = Uint8Array.of(repeat, i, 255, 4);
        return reader.read(new Blob([raw])).then(result => {
          assert.deepEqual(new Uint8Array(result.raw), raw);
          assert.equal(result.digest, createHash('sha256').update(raw).digest('hex'));
          assert.ok(result.readMs >= 0 && result.hashMs >= 0);
        });
      });
      assert.throws(() => reader.read(new Blob()), /eight-record prefetch limit/);
      const replies = await Promise.all(pending);
      assert.equal(replies.length, 8);
    }
  } finally { reader.dispose(); globalThis.Worker = original; }
});

test('abort rejects all pending reads and future work without leaked workers', async () => {
  const original = globalThis.Worker;
  globalThis.Worker = BrowserWorker as unknown as typeof Worker;
  const abort = new AbortController(), reader = new Source112WorkerReader(abort.signal);
  try {
    const checks = Array.from({length: 8}, () => assert.rejects(
      reader.read(new Blob([new Uint8Array(1024)])), /stop admission/));
    abort.abort(Error('stop admission'));
    await Promise.all(checks);
    assert.throws(() => reader.read(new Blob()), /stop admission/);
  } finally { reader.dispose(); globalThis.Worker = original; }
});
