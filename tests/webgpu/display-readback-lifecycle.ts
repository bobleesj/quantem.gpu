import test from 'node:test';
import assert from 'node:assert/strict';
import { GPUColormapEngine } from '../../src/quantem/gpu/display/webgpu/colormaps';
Object.assign(globalThis, {GPUBufferUsage: {COPY_DST: 1, MAP_READ: 2}, GPUMapMode: {READ: 1}});
for (const failure of ['allocation', 'mapping']) test(`readback ${failure} failure releases scratch without destroying source`, async () => {
  const reads: any[] = [];
  const source = {destroy() {throw Error('Borrowed source must survive');}};
  const device = {
    createCommandEncoder() {return {copyBufferToBuffer() {},finish() {return {};}};},
    createBuffer() {
      if (failure === 'allocation' && reads.length === 1) throw Error('allocation failed');
      const b = {mapState:'unmapped',destroyed:0,
        async mapAsync() {if (failure === 'mapping') throw Error('mapping failed');},
        destroy() {this.destroyed++;}};
      reads.push(b);return b;
    },queue:{submit() {}},
  };
  const engine = new GPUColormapEngine(device as unknown as GPUDevice);
  (engine as any).slots = [0,1].map(()=>({dataKind:'f32',count:4,dataBuffer:source}));
  await assert.rejects(engine.readDataSlots([0,1]), new RegExp(failure+' failed'));
  assert.equal(reads.length,failure==='allocation'?1:2);
  assert.ok(reads.every(b=>b.destroyed===1));
});
