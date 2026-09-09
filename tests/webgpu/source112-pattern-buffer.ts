import assert from 'node:assert/strict';
import test from 'node:test';
import { Source112ResidentSet } from '../../src/quantem/gpu/detector/compute/webgpu/source112';
Object.assign(globalThis, {GPUBufferUsage: {STORAGE: 1, COPY_SRC: 2, COPY_DST: 4, UNIFORM: 8}});
function fixture(failAt = 0) {
  let calls = 0;
  const allocated: any[] = [], clears: any[] = [];
  const device = {
    createBuffer({size}: any) {
      if (++calls === failAt) throw Error('allocation failed');
      const b = {size, destroyed: 0, destroy() { this.destroyed++; }, getMappedRange() {return new ArrayBuffer(size);}, unmap() {}};
      allocated.push(b); return b;
    },
    createShaderModule() {return {};},
    createComputePipeline() {return {getBindGroupLayout() {return {};}};},
    createBindGroup() {return {};},
    createCommandEncoder() {return {clearBuffer(...args: any[]) {clears.push(args);}, finish() {return {};},
      beginComputePass() {return {setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() {}, end() {}};}};},
    queue: {writeBuffer() {}, submit() {}},
  };
  const source = Object.create(Source112ResidentSet.prototype);
  Object.assign(source, {device, acquisitionCount: 1, disposed: false, owned: [],
    profile: {residentBytes: 0, peakResidentBytes: 0}, badPx: [3, 100], errorBuffer: {},
    columns: [new Uint32Array([0]), new Uint32Array([1])], selected: [{}, {}], pipelines: [{}, {}],
    globals: [{}, {}, {}, {}], layout: {},
    patternDisplays: new Map(), patternConvertBindings: new Map(),
    groups: [{records: [{chunk: 0}], params: [{}, {}], payload: {}, descriptors: {}}]});
  return {source, allocated, clears};
}
test('resident DP reuses storage, clears sparse counts, masks display, and releases only owned buffers', () => {
  const {source, allocated, clears} = fixture();
  const a = source.patternBuffer(0, 0), b = source.patternBuffer(0, 1);
  assert.equal(a, b); assert.equal(allocated.length, 3);
  assert.equal(clears.filter(c => c.length === 1).length, 2);
  assert.deepEqual(clears.filter(c => c.length > 1).map(c => c.slice(1)), [[12,4],[400,4],[12,4],[400,4]]);
  source.destroy(); source.destroy();
  assert.ok(allocated.every(b => b.destroyed === 1));
  assert.throws(() => source.patternBuffer(0, 0), /closed/);
});
test('failed display initialization rolls back and can retry without leaking', () => {
  for (const failAt of [1,2]) {
    const {source, allocated} = fixture(failAt);
    assert.throws(() => source.patternBuffer(0,0), /allocation failed/);
    assert.equal(source.profile.residentBytes, 0);
    assert.ok(allocated.every(b => b.destroyed === 1));
    assert.ok(source.patternBuffer(0,1));
    source.destroy(); assert.ok(allocated.every(b => b.destroyed === 1));
  }
});
test('invalid native indices reject before allocation', () => {
  const {source, allocated} = fixture();
  for (const [acq,scan] of [[-1,0],[1,0],[0,-1],[0,262144],[0,0.5]]) assert.throws(() => source.patternBuffer(acq,scan));
  assert.equal(allocated.length, 0);
});

test('distinct acquisitions retain distinct display buffers', () => {
  const {source} = fixture();
  source.acquisitionCount = 2;
  source.groups[0].records.push({chunk: 16});
  const a = source.patternBuffer(0,0), b = source.patternBuffer(1,0);
  assert.notEqual(a,b);
  assert.equal(source.patternBuffer(0,1), a);
  source.destroy();
});
