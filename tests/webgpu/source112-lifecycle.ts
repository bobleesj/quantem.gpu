/** CPU-only admission and display-resource regression checks; no GPU required. */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import test from 'node:test';
import { Source112ResidentSet } from '../../src/quantem/gpu/detector/compute/webgpu/source112';

Object.assign(globalThis, {
  GPUBufferUsage: { STORAGE: 1, COPY_SRC: 2, COPY_DST: 4, UNIFORM: 8, MAP_WRITE: 32 },
});
const Q = 192 * 192;
const N = 512 * 512;
const digest = (data: Uint8Array) => createHash('sha256').update(data).digest('hex');

function metadataFixture() {
  const arrays = {
    dense_columns: new Uint32Array(Array.from({ length: 17466 }, (_, i) => i)),
    sparse_columns: new Uint32Array(Array.from({ length: 19398 }, (_, i) => i + 17466)),
    model_ids: new Uint8Array(264 * Q),
    decoding: new Uint32Array(81 * 1024),
    hardware: new Uint8Array(Q),
    valid: new Uint8Array(Q).fill(1),
  };
  const globals: Record<string, any> = {};
  const files: File[] = [];
  for (const [name, values] of Object.entries(arrays)) {
    const raw = new Uint8Array(values.buffer);
    const shape = name === 'model_ids' ? [264, Q] : name === 'decoding' ? [81, 1024] : [values.length];
    globals[name] = {
      file: `${name}.bin`, dtype: values instanceof Uint32Array ? '<u4' : '|u1',
      shape, nbytes: raw.byteLength, sha256: digest(raw),
    };
    files.push(new File([raw], globals[name].file));
  }
  let end = 0;
  const components = [['dense', 4], ['dense_offsets', 628776], ['sparse', 16], ['sparse_offsets', 698332]].map(([name, nbytes]) => {
    const offset = Math.ceil(end / 64) * 64;
    end = offset + Number(nbytes);
    return { name, nbytes, offset, dtype: '<u4' };
  });
  const recordBytes = Math.ceil(end / 4096) * 4096;
  const manifest = {
    format: 'source112-tans1024-pair-v1', dtype: '<u2', shape: [66, 512, 512, 192, 192], globals,
    layout: {
      files: [{ name: 'payload.bin', nbytes: 1056 * recordBytes }],
      chunks: Array.from({ length: 1056 }, (_, chunk) => ({
        chunk, acquisition: Math.floor(chunk / 16), first_scan: chunk % 16 * 16384,
        scan_count: 16384, shard: 0, file_offset: chunk * recordBytes,
        record_bytes: recordBytes, sha256: '0'.repeat(64), components: structuredClone(components),
      })),
    },
  };
  // A metadata-only virtual file: tests must stop before attempting source reads.
  files.push({ name: 'payload.bin', size: manifest.layout.files[0].nbytes,
    slice() { throw new Error('Unexpected scientific payload read in CPU test.'); },
  } as unknown as File);
  return { manifest, files };
}
const fixture = metadataFixture();
function grantedFiles(manifest = fixture.manifest, omitPayload = false) {
  return [new File([JSON.stringify(manifest)], 'manifest.json'),
    ...fixture.files.filter(file => !omitPayload || file.name !== 'payload.bin')];
}

function allocationFailureDevice(failAt: number) {
  const state = { calls: 0, destroyed: 0, scopes: 0 };
  const device = {
    limits: { maxStorageBufferBindingSize: 2 ** 31, maxBufferSize: 2 ** 31 },
    pushErrorScope() { state.scopes++; },
    async popErrorScope() { state.scopes--; return null; },
    createBuffer() {
      state.calls++;
      if (state.calls === failAt) throw new Error('Synthetic allocation failure.');
      return { destroy() { state.destroyed++; } };
    },
  } as unknown as GPUDevice;
  return { device, state };
}

test('malformed folder metadata rejects before allocation or payload reads', async () => {
  const mutations: [string, (manifest: any) => void, boolean?][] = [
    ['negative address', m => { m.layout.chunks[0].file_offset = -4; }],
    ['unsafe address', m => { m.layout.chunks[0].file_offset = Number.MAX_SAFE_INTEGER + 1; }],
    ['duplicate components', m => { m.layout.chunks[0].components[1] = m.layout.chunks[0].components[0]; }],
    ['wrong seek size', m => { m.layout.chunks[0].components[1].nbytes -= 4; }],
    ['overlapping components', m => { m.layout.chunks[0].components[1].offset = 0; }],
    ['overlapping records', m => { m.layout.chunks[1].file_offset = 0; }],
    ['missing global', m => { delete m.globals.decoding; }],
    ['missing detector validity', m => { delete m.globals.valid; }],
    ['wrong dtype', m => { m.globals.decoding.dtype = '<f4'; }],
    ['wrong shape', m => { m.globals.decoding.shape = [1]; }],
    ['wrong digest', m => { m.globals.decoding.sha256 = '0'.repeat(64); }],
    ['missing payload', () => {}, true],
  ];
  for (const [name, mutate, omit] of mutations) {
    const manifest = structuredClone(fixture.manifest); mutate(manifest);
    const { device, state } = allocationFailureDevice(1);
    await assert.rejects(Source112ResidentSet.loadFiles(device, grantedFiles(manifest, omit)), undefined, name);
    assert.equal(state.calls, 0, `${name}: no allocation`);
    assert.equal(state.scopes, 0, `${name}: no leaked error scope`);
  }
});

test('allocation failure releases every partially owned buffer and error scope', async () => {
  for (const failAt of [1, 2, 3]) {
    const { device, state } = allocationFailureDevice(failAt);
    await assert.rejects(Source112ResidentSet.loadFiles(device, grantedFiles()), /Synthetic allocation failure/);
    assert.equal(state.calls, failAt);
    assert.equal(state.destroyed, failAt - 1);
    assert.equal(state.scopes, 0);
  }
});

test('aborted admission owns no device resources', async () => {
  const { device, state } = allocationFailureDevice(1);
  const controller = new AbortController(); controller.abort();
  await assert.rejects(Source112ResidentSet.loadFiles(device, grantedFiles(), undefined, controller.signal), { name: 'AbortError' });
  assert.equal(state.calls, 0); assert.equal(state.scopes, 0);
});

test('all-panel display updates reuse aligned uniforms and preserve ownership', () => {
  const counts = { buffers: 0, binds: 0, writes: 0, passes: 0, dispatches: 0, destroyed: 0 };
  let uniformSnapshot = new ArrayBuffer(0);
  const device = {
    limits: { minUniformBufferOffsetAlignment: 256 },
    createBuffer({ size }: { size: number }) {
      counts.buffers++; return { size, destroy() { counts.destroyed++; } };
    },
    createShaderModule() { return {}; },
    createComputePipeline() { return { getBindGroupLayout() { return {}; } }; },
    createBindGroup(spec: any) {
      counts.binds++;
      assert.equal(spec.entries[2].resource.offset % 256, 0);
      assert.equal(spec.entries[2].resource.size, 16);
      return spec;
    },
    createCommandEncoder() {
      return {
        beginComputePass() {
          counts.passes++;
          return { setPipeline() {}, setBindGroup() {}, end() {},
            dispatchWorkgroups(size: number) { assert.equal(size, N / 256); counts.dispatches++; } };
        },
        finish() { return {}; },
      };
    },
    queue: {
      writeBuffer(_buffer: unknown, _offset: number, data: ArrayBuffer) {
        counts.writes++; uniformSnapshot = data.slice(0);
      },
      submit() {},
    },
  };
  // Reproduce the lifecycle of an existing set without allocating its source.
  const source = Object.create(Source112ResidentSet.prototype);
  Object.assign(source, { acquisitionCount: 66, device, owned: [], profile: { residentBytes: 93_664_812_200, peakResidentBytes: 93_686_345_812 },
    displayBuffers: new Map(), outputBuffer: {}, groups: [], disposed: false });
  const indices = Array.from({ length: 66 }, (_, index) => index);
  const displays = source.imageBuffersF32(indices);
  // Reproduce the real post-migration peak overtaken by 66 lazy display images
  // plus one aligned uniform slab; no source or GPU allocation is needed here.
  assert.equal(source.profile.residentBytes, 93_734_035_112);
  assert.equal(source.profile.peakResidentBytes, source.profile.residentBytes);
  assert.equal(counts.buffers, 67); assert.equal(counts.binds, 66);
  source.normalizeDisplayBuffers(displays, 123);
  source.normalizeDisplayBuffers(displays, 7);
  assert.equal(counts.buffers, 67); assert.equal(counts.binds, 66);
  assert.equal(source.profile.peakResidentBytes, 93_734_035_112, 'Reused display buffers do not increase peak');
  assert.equal(counts.writes, 3); assert.equal(counts.passes, 3); assert.equal(counts.dispatches, 198);
  for (const index of indices) {
    const word = index * 256 / 4;
    assert.equal(new Uint32Array(uniformSnapshot)[word], index * N);
    assert.equal(new Uint32Array(uniformSnapshot)[word + 1], N);
    assert.equal(new Float32Array(uniformSnapshot)[word + 2], 7);
  }
  assert.deepEqual(source.imageBuffersF32([65, 0, 65]), [displays[65], displays[0], displays[65]]);
  const beforeInvalid = { ...counts };
  assert.throws(() => source.imageBuffersF32([0, 66]));
  assert.throws(() => source.normalizeDisplayBuffers([{}], 2));
  assert.throws(() => source.normalizeDisplayBuffers(displays, 0));
  assert.deepEqual(counts, beforeInvalid);
  source.destroy(); source.destroy();
  assert.equal(counts.destroyed, 67);
  assert.throws(() => source.imageBuffersF32([0]));
});

test('full masks and deltas use the dense sum pipeline; patterns retain original pipelines', async () => {
  Object.assign(globalThis, { GPUMapMode: { READ: 1, WRITE: 2 } });
  const seen: unknown[] = [], modes: number[] = [];
  let allocations = 0;
  const device = {
    createBuffer({ size }: { size: number }) {
      allocations++;
      return { async mapAsync() {}, getMappedRange() { return new ArrayBuffer(size); }, destroy() {} };
    },
    createBindGroup() { return {}; },
    createCommandEncoder() {
      return { clearBuffer() {}, copyBufferToBuffer() {}, finish() { return {}; },
        beginComputePass() { return { setPipeline(pipeline: unknown) { seen.push(pipeline); }, setBindGroup() {}, dispatchWorkgroups() {}, end() {} }; } };
    },
    queue: { writeBuffer(_buffer: unknown, _offset: number, data: Uint32Array) { if (data.length === 16) modes.push(data[5]); }, submit() {} },
  };
  const dense = {}, sparse = {}, sum = {};
  const source = Object.create(Source112ResidentSet.prototype);
  Object.assign(source, { device, disposed: false, owned: [], previousMask: null,
    columns: [new Uint32Array([0]), new Uint32Array([1])], selected: [{}, {}],
    pipelines: [dense, sparse], denseSumPipeline: sum, outputBuffer: {}, errorBuffer: {},
    globals: [{}, {}, {}, {}], layout: {},
    groups: [{ records: [{ chunk: 0 }], params: [{}, {}], bind: [{}, {}], payload: {}, descriptors: {} }] });
  const mask = new Uint32Array(Q); mask[0] = mask[1] = 1;
  assert.equal(source.integrate(mask).full, true);
  mask[0] = mask[1] = 0;
  assert.equal(source.integrate(mask).removed, 2);
  assert.deepEqual(seen, [sum, sparse, sum, sparse]);
  assert.equal(allocations, 0, 'drag submissions allocate no device buffers');
  await source.pattern(0, 0);
  assert.deepEqual(seen, [sum, sparse, sum, sparse, dense, sparse]);
  assert.deepEqual(modes, [0, 0, 0, 0, 1, 1]);
});

test('sum shader or pipeline compilation failure releases admission resources before payload reads', async () => {
  Object.assign(globalThis, { GPUShaderStage: { COMPUTE: 1 } });
  for (const failure of ['shader', 'pipeline']) {
    let allocations = 0, destroyed = 0, scopes = 0;
    const device = {
      limits: { maxStorageBufferBindingSize: 2 ** 31, maxBufferSize: 2 ** 31 },
      pushErrorScope() { scopes++; }, async popErrorScope() { scopes--; return null; },
      createBuffer({ size }: { size: number }) {
        allocations++; return { getMappedRange() { return new ArrayBuffer(size); }, unmap() {}, destroy() { destroyed++; } };
      },
      createBindGroupLayout() { return {}; }, createPipelineLayout() { return {}; },
      createShaderModule({ code }: { code: string }) {
        const sumOnly = !code.includes('p.mode');
        return { sumOnly, async getCompilationInfo() {
          return { messages: failure === 'shader' && sumOnly ? [{ type: 'error', message: 'Synthetic sum shader failure' }] : [] };
        } };
      },
      async createComputePipelineAsync({ compute }: any) {
        if (failure === 'pipeline' && compute.module.sumOnly) throw new Error('Synthetic sum pipeline failure');
        return {};
      },
    } as unknown as GPUDevice;
    await assert.rejects(Source112ResidentSet.loadFiles(device, grantedFiles()), /Synthetic sum/);
    assert.ok(allocations > 0); assert.equal(destroyed, allocations); assert.equal(scopes, 0);
  }
});

test('complete admission prepares and enables checked cache before ready; failures release everything', async () => {
  Object.assign(GPUBufferUsage, { MAP_READ: 16 });
  Object.assign(globalThis, { GPUMapMode: { READ: 1, WRITE: 2 }, GPUShaderStage: { COMPUTE: 1 } });
  const manifest = structuredClone(fixture.manifest);
  const recordBytes = manifest.layout.chunks[0].record_bytes;
  const raw = new Uint8Array(recordBytes);
  const recordDigest = digest(raw);
  for (const record of manifest.layout.chunks) record.sha256 = recordDigest;
  const payload = { name: 'payload.bin', size: manifest.layout.files[0].nbytes,
    slice(first: number, end: number) {
      assert.equal(end - first, recordBytes);
      return { size: raw.byteLength, async arrayBuffer() { return raw.buffer; } };
    },
  } as unknown as File;
  const files = [new File([JSON.stringify(manifest)], 'manifest.json'),
    ...fixture.files.filter(file => file.name !== 'payload.bin'), payload];
  for (const failure of ['none', 'allocation', 'corruption', 'abort']) {
    const allocated: { size: number; destroyed: boolean }[] = [];
    let scopes = 0, preparing = false;
    const statuses: string[] = [];
    const controller = new AbortController();
    const device = {
      lost: new Promise(() => {}),
      // Virtual buffers avoid allocating payload/cache memory. A large mock limit
      // keeps this short encoded fixture in one group while testing admission order.
      limits: { maxStorageBufferBindingSize: 16 * 2 ** 30, maxBufferSize: 16 * 2 ** 30 },
      pushErrorScope() { scopes++; }, async popErrorScope() { scopes--; return null; },
      queue: { writeBuffer() {}, submit() {}, async onSubmittedWorkDone() {} },
      createBuffer({ size, usage }: { size: number; usage: number }) {
        if (preparing && size > 5 * 2 ** 30 && failure === 'allocation') throw new Error('Synthetic cache allocation failure');
        const buffer = { size, destroyed: false,
          getMappedRange() {
            if (usage & GPUBufferUsage.MAP_READ) return new Uint32Array([failure === 'corruption' ? 4 : 0]).buffer;
            assert.ok((usage & GPUBufferUsage.MAP_WRITE) ? size === 32 * 2 ** 20 : size < 20 * 2 ** 20, 'Map only bounded staging or authenticated metadata');
            return new ArrayBuffer(size);
          },
          async mapAsync() {}, unmap() {}, destroy() { buffer.destroyed = true; },
        };
        allocated.push(buffer); return buffer;
      },
      createBindGroupLayout() { return {}; }, createPipelineLayout() { return {}; }, createBindGroup() { return {}; },
      createShaderModule() { return { async getCompilationInfo() { return { messages: [] }; } }; },
      async createComputePipelineAsync() { return {}; },
      createCommandEncoder() { return { clearBuffer() {}, copyBufferToBuffer() {}, finish() {},
        beginComputePass() { return { setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() {}, end() {} }; } }; },
    } as unknown as GPUDevice;
    const admission = Source112ResidentSet.loadFiles(device, files, text => {
      statuses.push(text);
      if (text.startsWith('Preparing exact')) { preparing = true; assert.ok(allocated.filter(b => b.size === 32 * 2 ** 20).every(b => b.destroyed), 'Mapped staging released before cache allocation'); }
      if (failure === 'abort' && text.startsWith('Preparing exact') && text.includes('group')) controller.abort();
    }, controller.signal);
    if (failure === 'none') {
      const source = await admission;
      assert.equal(source.denseRestartStatus().enabled, true);
      assert.equal(source.profile.restartCacheBytes, 9_443_427_840);
      assert.ok(source.profile.restartPreparationMs > 0);
      assert.ok(source.readyMs >= source.profile.restartPreparationMs);
      assert.equal(source.profile.records, 1056);
      assert.ok(statuses.some(text => text.includes('restart cache') && text.includes('group')));
      source.destroy();
      assert.equal(source.denseRestartStatus().profile, null);
      assert.equal(source.profile.restartCacheBytes, 0);
    } else {
      await assert.rejects(admission, failure === 'abort' ? { name: 'AbortError' } : failure === 'allocation' ? /cache allocation failure/ : /decoder status 4/);
    }
    assert.equal(scopes, 0, `${failure}: error scopes closed`);
    assert.ok(allocated.every(buffer => buffer.destroyed), `${failure}: all payload, cache and temporary buffers released`);
  }
});

test('explicit Huffman load skips legacy cache and waits for preparation before exposure', async () => {
  const manifest = structuredClone(fixture.manifest);
  const bytes = manifest.layout.chunks[0].record_bytes, raw = new Uint8Array(bytes);
  for (const record of manifest.layout.chunks) record.sha256 = digest(raw);
  const payload = { name: 'payload.bin', size: manifest.layout.files[0].nbytes,
    slice() { return { size: raw.byteLength, async arrayBuffer() { return raw.buffer; } }; },
  } as unknown as File;
  const files = [new File([JSON.stringify(manifest)], 'manifest.json'), ...fixture.files.filter(f => f.name !== 'payload.bin'), payload];
  let finish!: () => void, started!: () => void;
  const ready = new Promise<void>(r => { finish = r; }), entered = new Promise<void>(r => { started = r; });
  const proto = Source112ResidentSet.prototype as any, original = proto.prepareHuffman64, originalCompact = proto.prepareCompactHuffman;
  let finishCompact!: () => void, enteredCompact!: () => void;
  const compactReady = new Promise<void>(r => { finishCompact = r; }), compactEntered = new Promise<void>(r => { enteredCompact = r; });
  let preparingSource: any, scopes = 0, productBinds = 0;
  const allocations: { size: number; destroyed: boolean }[] = [];
  const device = {
    lost: new Promise(() => {}),
    createCommandEncoder() { return { copyBufferToBuffer() {}, finish() {} }; },
    limits: { maxStorageBufferBindingSize: 16 * 2 ** 30, maxBufferSize: 16 * 2 ** 30 },
    pushErrorScope() { scopes++; }, async popErrorScope() { scopes--; return null; },
    queue: { writeBuffer() {}, submit() {}, async onSubmittedWorkDone() {} },
    createBuffer({ size, usage }: { size: number; usage: number }) {
      const b = { size, destroyed: false, getMappedRange() { assert.ok((usage & GPUBufferUsage.MAP_WRITE) ? size === 32 * 2 ** 20 : size < 20 * 2 ** 20); return new ArrayBuffer(size); }, async mapAsync() {}, unmap() {}, destroy() { b.destroyed = true; } };
      allocations.push(b); return b;
    },
    createBindGroupLayout() { return {}; }, createPipelineLayout() { return {}; },
    createBindGroup() { productBinds++; return {}; },
    createShaderModule() { return { async getCompilationInfo() { return { messages: [] }; } }; },
    async createComputePipelineAsync() { return {}; },
  } as unknown as GPUDevice;
  try {
    proto.prepareHuffman64 = async function(table: Uint32Array) {
      preparingSource = this;
      assert.equal(table.length, 81 * 1024); assert.equal(this.restartOriginals, undefined);
      assert.equal(this.profile.restartCacheBytes, 0); assert.equal(productBinds, 0);
      assert.ok(this.groups.every((g: any) => g.bind.length === 0));
      started(); await ready; this.representation = 'huffman64';
    };
    proto.prepareCompactHuffman = async function() { assert.equal(this.representation, 'huffman64'); enteredCompact(); await compactReady; this.profile.restartCacheLayout = 'huffman64-compact'; };
    let exposed = false;
    const pending = Source112ResidentSet.loadFiles(device, files, undefined, undefined, { representation: 'huffman64' }).then(s => { exposed = true; return s; });
    await entered; await Promise.resolve(); assert.equal(exposed, false);
    assert.equal(preparingSource.profile.readyMs, 0); finish();
    await compactEntered; await Promise.resolve(); assert.equal(exposed, false); assert.equal(preparingSource.profile.readyMs, 0); finishCompact();
    const source = await pending; assert.equal(source.storageRepresentation, 'huffman64'); assert.ok(source.readyMs > 0);
    source.destroy(); assert.equal(scopes, 0); assert.ok(allocations.every(b => b.destroyed));
  } finally { proto.prepareHuffman64 = original; proto.prepareCompactHuffman = originalCompact; finish?.(); finishCompact?.(); }
});

 test('owned allocation peak counts actual aligned bytes and never falls after release', () => {
  const source = Object.assign(Object.create(Source112ResidentSet.prototype), {
    acquisitionCount: 66,
    device: { createBuffer({ size }: { size: number }) { return { size, destroy() {} }; } },
    owned: [], profile: { residentBytes: 100, peakResidentBytes: 103 },
  });
  const first = source.buffer(1, GPUBufferUsage.STORAGE);
  assert.equal(first.size, 4); assert.equal(source.profile.residentBytes, 104);
  assert.equal(source.profile.peakResidentBytes, 104);
  source.owned.pop(); source.profile.residentBytes -= first.size; first.destroy();
  source.recordResidentPeak(); assert.equal(source.profile.peakResidentBytes, 104);
  source.recordResidentPeak(72); assert.equal(source.profile.peakResidentBytes, 172);
  source.buffer(4, GPUBufferUsage.STORAGE); assert.equal(source.profile.peakResidentBytes, 172);
 });
