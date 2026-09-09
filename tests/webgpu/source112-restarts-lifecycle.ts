/// <reference types="@webgpu/types" />
import { strict as assert } from 'node:assert';
import { Source112ResidentSet } from '../../src/quantem/gpu/detector/compute/webgpu/source112';

// No adapter, GPU execution, payload allocation or browser is involved.
Object.assign(globalThis, { GPUBufferUsage: { STORAGE: 1, UNIFORM: 2, COPY_SRC: 4, COPY_DST: 8, MAP_READ: 16 }, GPUShaderStage: { COMPUTE: 1 }, GPUMapMode: { READ: 1 } });
class Buffer {
  destroyed = false;
  constructor(readonly size: number) {}
  destroy() { this.destroyed = true; }
  async mapAsync() {}
  getMappedRange() { return new ArrayBuffer(4); }
  unmap() {}
}
function fixture(failPop = 0) {
  const allocations: Buffer[] = [], bindings: unknown[] = [], selectedPipelines: unknown[] = [];
  let pops = 0, scopeDepth = 0;
  const device = {
    limits: { maxBufferSize: 2 ** 30, maxStorageBufferBindingSize: 2 ** 30 },
    pushErrorScope() { scopeDepth++; },
    async popErrorScope() { scopeDepth--; return ++pops === failPop ? { message: 'simulated allocation failure' } : null; },
    queue: { async onSubmittedWorkDone() {}, writeBuffer() {}, submit() {} },
    createBuffer({ size }: { size: number }) { const buffer = new Buffer(size); allocations.push(buffer); return buffer; },
    createBindGroupLayout() { return {}; }, createPipelineLayout() { return {}; },
    createShaderModule() { return { async getCompilationInfo() { return { messages: [] }; } }; },
    async createComputePipelineAsync() { return {}; },
    createBindGroup(descriptor: unknown) { const group = { descriptor }; bindings.push(group); return group; },
    createCommandEncoder() { return { clearBuffer() {}, copyBufferToBuffer() {}, finish() {},
      beginComputePass() { return { setPipeline(pipeline: unknown) { selectedPipelines.push(pipeline); }, setBindGroup() {}, dispatchWorkgroups() {}, end() {} }; } }; },
  };
  const old = [new Buffer(48), new Buffer(48)];
  const groups = old.map((descriptors, i) => ({ descriptors, bind: [{ old: i, dense: true }, { old: i, dense: false }],
    payload: new Buffer(512), params: [new Buffer(64), new Buffer(64)], records: [{ chunk: i, acquisition: 0, first_scan: i * 16384, record_bytes: 512,
      components: ['dense', 'dense_offsets', 'sparse', 'sparse_offsets'].map((name, index) => ({ name, offset: index * 128, nbytes: 128 })) }] }));
  const set = Object.assign(Object.create(Source112ResidentSet.prototype), { device, groups, columns: [new Uint32Array([0, 1]), new Uint32Array([2])],
    globals: Array.from({ length: 4 }, () => new Buffer(4)), selected: [new Buffer(8), new Buffer(4)], outputBuffer: new Buffer(4), errorBuffer: new Buffer(4),
    layout: {}, pipelines: [{ baseline: "dense" }, { baseline: "sparse" }], owned: old.slice(), profile: { residentBytes: 96 } }) as Source112ResidentSet;
  return { set, groups, old, allocations, selectedPipelines, scopeDepth: () => scopeDepth };
}

async function main() {
  {
    const f = fixture(); const originalBindings = f.groups.map(group => group.bind);
    const profile = await f.set.prepareDenseRestarts();
    assert.equal(profile.checkpointBytes, 2 * 32 * 2 * 3 * 4);
    assert.equal(profile.offsetBytes, 2 * 32 * 2 * 4);
    assert.equal(profile.additionalBytes, profile.checkpointBytes + profile.offsetBytes + 96);
    assert.equal(f.set.denseRestartStatus().enabled, false);
    assert.ok(f.old.every(buffer => !buffer.destroyed));
    f.set.setDenseRestartsEnabled(true);
    assert.equal(f.set.denseRestartStatus().enabled, true);
    const caches = f.groups.map(group => group.descriptors);
    await f.set.releaseDenseRestarts();
    assert.deepEqual(f.groups.map(group => group.descriptors), f.old);
    assert.deepEqual(f.groups.map(group => group.bind), originalBindings);
    assert.ok(caches.every(buffer => buffer.destroyed));
    assert.equal(f.set.profile.residentBytes, 96);
    assert.equal(f.scopeDepth(), 0);
    assert.throws(() => f.set.setDenseRestartsEnabled(true), /Prepare dense restarts/);
  }
  {
    const f = fixture(5); const originalBindings = f.groups.map(group => group.bind);
    await assert.rejects(f.set.prepareDenseRestarts(), /simulated allocation failure/);
    assert.deepEqual(f.groups.map(group => group.descriptors), f.old);
    assert.deepEqual(f.groups.map(group => group.bind), originalBindings);
    assert.ok(f.allocations.every(buffer => buffer.destroyed));
    assert.ok(f.old.every(buffer => !buffer.destroyed));
    assert.equal(f.set.profile.residentBytes, 96); assert.equal(f.scopeDepth(), 0);
  }
  {
    const f = fixture(), abort = new AbortController();
    await assert.rejects(f.set.prepareDenseRestarts(() => abort.abort(), abort.signal), /abort/i);
    assert.deepEqual(f.groups.map(group => group.descriptors), f.old);
    assert.ok(f.allocations.every(buffer => buffer.destroyed)); assert.equal(f.scopeDepth(), 0);
  }
  {
    const f = fixture();
    const internal = f.set as unknown as { pipelines: unknown[]; denseSumPipeline: unknown; restartEnabled: boolean };
    const original = internal.pipelines.slice();
    const control = { specialized: 'sum-only' }; internal.denseSumPipeline = control;
    f.set.integrate(new Uint32Array(192 * 192).fill(1));
    assert.ok(f.selectedPipelines.includes(control));
    assert.deepEqual(internal.pipelines, original, 'Pattern/gather pipelines must never be replaced by sum-only control');
    internal.restartEnabled = true;
    await f.set.prepareDenseRestarts();
    f.set.setDenseRestartsEnabled(true);
    f.set.integrate(new Uint32Array(192 * 192));
    assert.deepEqual(internal.pipelines, original);
    await f.set.releaseDenseRestarts();
  }
  {
    const f = fixture();
    await assert.rejects(f.set.prepareDenseRestarts(() => f.set.destroy()), /closed/);
    assert.ok(f.allocations.every(buffer => buffer.destroyed));
    assert.ok(f.old.every(buffer => buffer.destroyed));
    assert.equal(f.scopeDepth(), 0);
    assert.equal(f.set.denseRestartStatus().enabled, false);
    assert.equal(f.set.denseRestartStatus().profile, null);
  }
  console.log('5 restart lifecycle cases passed without GPU execution');
}
void main();
