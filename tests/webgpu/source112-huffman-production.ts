/** CPU-only metadata and lifecycle tests for the explicit production option. */
import assert from 'node:assert/strict';
import test from 'node:test';
import { Source112ResidentSet } from '../../src/quantem/gpu/detector/compute/webgpu/source112';
import { source112HuffmanBooks, encodingBook } from '../../src/quantem/gpu/detector/compute/webgpu/source112-huffman-books';
const decoding = new Uint32Array(81 * 1024);
for (const model of [78, 79]) for (let i = 0; i < 1024; i++) decoding[model * 1024 + i] = [0, 1, 2, 4095][i % 4];
Object.assign(globalThis, { GPUBufferUsage: { STORAGE: 1, COPY_SRC: 2, COPY_DST: 4, UNIFORM: 8 }, GPUShaderStage: { COMPUTE: 1 } });

test('canonical metadata ties are deterministic and complete', () => {
  const books = source112HuffmanBooks(decoding);
  for (const book of books) assert.deepEqual(book.entries, [0, 1, 2, 4095].map((symbol, i) => ({ symbol, bits: 2, canonical_msb_code: i })));
  const data = encodingBook(books);
  for (let model = 0; model < 2; model++) for (let symbol = 0; symbol < 4096; symbol++) {
    const entry = data[model * 4096 + symbol], lookup = data[8192 + model * 1024 + (entry & 1023)];
    assert.equal(lookup & 4095, entry & 16384 ? 4095 : symbol);
    assert.equal(lookup >>> 12, (entry >>> 10) & 15);
  }
  assert.deepEqual(source112HuffmanBooks(decoding), books);
  assert.throws(() => source112HuffmanBooks(new Uint32Array(1)), /complete authenticated/);
  assert.throws(() => encodingBook(source112HuffmanBooks(new Uint32Array(81 * 1024))), /Invalid|Missing/);
});

function deferred() { let resolve!: () => void; const promise = new Promise<void>(r => { resolve = r; }); return { promise, resolve }; }
function mock({ compileGate, fenceFailure = false, rejectPop = false }: { compileGate?: ReturnType<typeof deferred>; fenceFailure?: boolean; rejectPop?: boolean } = {}) {
  let scopes = rejectPop ? 1 : 0, destroys = 0, firstPop = true; const buffers = [{ size: 128, destroy() { destroys++; } }];
  const device = {
    lost: new Promise(() => {}),
    limits: { maxBufferSize: 2 ** 31, maxStorageBufferBindingSize: 2 ** 31 },
    pushErrorScope() { scopes++; }, async popErrorScope() { scopes--; if (rejectPop && firstPop) { firstPop = false; throw Error('injected pop rejection'); } return null; },
    createShaderModule() { return { async getCompilationInfo() { await compileGate?.promise; return { messages: rejectPop ? [] : [{ type: 'error', lineNum: 1, linePos: 1, message: 'injected compile error' }] }; } }; },
    createBindGroupLayout() { return {}; }, createPipelineLayout() { return {}; }, async createComputePipelineAsync() { return {}; },
    queue: { async onSubmittedWorkDone() { if (fenceFailure) throw Error('injected cleanup fence'); } },
  };
  // Tests construct a mock owner; production uses its actual private constructor.
  const source = Object.assign(Object.create(Source112ResidentSet.prototype), {
    device, disposed: false, preparing: false, representation: 'tans', restartPreparing: false,
    owned: buffers, groups: [], previousMask: null, profile: { residentBytes: 128 },
    displayBuffers: new Map(), convertBindings: new Map(), displayViews: new Map(),
  });
  return { source, get scopes() { return scopes; }, get destroys() { return destroys; } };
}

test('invalid representation rejects before file access or allocation', async () => {
  await assert.rejects(Source112ResidentSet.loadFiles({} as GPUDevice, [], undefined, undefined, { representation: 'invalid' as any }), /Select Source112 representation/);
});

test('preparation blocks science without replacing owner methods and closes on compilation failure', async () => {
  const gate = deferred(), m = mock({ compileGate: gate });
  const originalCheck = m.source.check, originalIntegrate = m.source.integrate;
  const pending = m.source.prepareHuffman64(decoding, () => {});
  assert.equal(m.source.check, originalCheck); assert.equal(m.source.integrate, originalIntegrate);
  assert.throws(() => m.source.check(), /preparing/);
  await assert.rejects(m.source.prepareHuffman64(decoding, () => {}), /preparing/);
  assert.equal(m.source.disposed, false);
  gate.resolve(); await assert.rejects(pending, /injected compile error/);
  assert.equal(m.source.disposed, true); assert.equal(m.destroys, 1); assert.equal(m.scopes, 0);
});

test('abort before preparation performs owned cleanup without GPU work', async () => {
  const m = mock(), abort = new AbortController(); abort.abort(Error('injected abort'));
  await assert.rejects(m.source.prepareHuffman64(decoding, () => {}, abort.signal), /injected abort/);
  assert.equal(m.source.disposed, true); assert.equal(m.destroys, 1); assert.equal(m.scopes, 0);
});

test('cleanup fence rejection preserves the original preparation failure', async () => {
  const m = mock({ fenceFailure: true });
  await assert.rejects(m.source.prepareHuffman64(decoding, () => {}), /injected compile error/);
  assert.equal(m.destroys, 1); assert.equal(m.scopes, 0);
});

test('completed Huffman generation rejects legacy cache mutation', async () => {
  const m = mock(); m.source.representation = 'huffman64';
  assert.equal(m.source.storageRepresentation, 'huffman64');
  await assert.rejects(m.source.prepareDenseRestarts(), /Reload/);
  await assert.rejects(m.source.releaseDenseRestarts(), /Reload/);
  assert.throws(() => m.source.setDenseRestartsEnabled(false), /Reload/);
  assert.equal(m.source.disposed, false);
});

 test('rejected pop consumes only the preparation scope, preserving outer scope', async () => {
  const m = mock({ rejectPop: true });
  await assert.rejects(m.source.prepareHuffman64(decoding, () => {}), /injected pop rejection/);
  assert.equal(m.destroys, 1); assert.equal(m.scopes, 1, 'Caller sentinel scope remains owned by caller');
 });
