import test from 'node:test';
import assert from 'node:assert/strict';
import {GPUColormapEngine} from '../../src/quantem/gpu/display/webgpu/colormaps';
Object.assign(globalThis, {GPUBufferUsage: {STORAGE: 1, COPY_SRC: 2, COPY_DST: 4, UNIFORM: 8, MAP_READ: 16}});
function setup() {
  const buffers: Array<GPUBuffer & {destroyed: number}> = [];
  const device = {createBuffer({size}: {size: number}) {
    const buffer = {size, destroyed: 0, destroy() {this.destroyed++;}} as unknown as GPUBuffer & {destroyed: number};
    buffers.push(buffer); return buffer;
  }, queue: {async onSubmittedWorkDone() {}}} as unknown as GPUDevice;
  return {device, engine: new GPUColormapEngine(device), buffers};
}
const drained = async () => {await new Promise(resolve => setTimeout(resolve, 0));};
test('selection and reorder release display resources without destroying shared source buffers', async () => {
  const {device, engine, buffers} = setup();
  const first = device.createBuffer({size: 1024, usage: 1}), second = device.createBuffer({size: 1024, usage: 1});
  engine.borrowBuffer(0, first, 16, 16);
  engine.borrowBuffer(100, first, 16, 16);
  engine.borrowBuffer(101, second, 16, 16);
  // Selected inspection changes and the comparison grid swaps the same source images.
  engine.borrowBuffer(0, second, 16, 16);
  engine.borrowBuffer(100, second, 16, 16);
  engine.borrowBuffer(101, first, 16, 16);
  await drained(); engine.releaseSlot(100); engine.destroy(); await drained();
  assert.equal(buffers[0].destroyed, 0); assert.equal(buffers[1].destroyed, 0);
  assert(buffers.slice(2).every(buffer => buffer.destroyed === 1));
  first.destroy(); second.destroy(); assert(buffers.every(buffer => buffer.destroyed === 1));
});
test('adopted computation buffers still transfer ownership and release exactly once', async () => {
  const {device, engine, buffers} = setup();
  const first = device.createBuffer({size: 1024, usage: 1}), second = device.createBuffer({size: 1024, usage: 1});
  engine.adoptBuffer(0, first, 16, 16); engine.adoptBuffer(0, second, 16, 16);
  await drained(); engine.destroy(); await drained();
  assert(buffers.every(buffer => buffer.destroyed === 1));
});
test('reshaping the same owned buffer retires only old display resources', async () => {
  const {device, engine, buffers} = setup();
  const source = device.createBuffer({size: 1024, usage: 1});
  engine.adoptBuffer(0, source, 16, 16); engine.adoptBuffer(0, source, 32, 8);
  await drained(); assert.equal(buffers[0].destroyed, 0);
  engine.destroy(); await drained(); assert(buffers.every(buffer => buffer.destroyed === 1));
});
