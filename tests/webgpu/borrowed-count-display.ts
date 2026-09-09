/** CPU-only borrowed-view admission, binding, and ownership checks. */
import assert from 'node:assert/strict';
import test from 'node:test';
import { GPUColormapEngine } from '../../src/quantem/gpu/display/webgpu/colormaps';
import { validateUint32ImageView, type Uint32ImageView } from '../../src/quantem/gpu/display/webgpu/borrowed-image';

Object.assign(globalThis, {GPUBufferUsage: {STORAGE: 1, COPY_SRC: 2, COPY_DST: 4, UNIFORM: 8, MAP_READ: 16}});
Object.defineProperty(globalThis, 'navigator', {configurable: true, value: {gpu: {getPreferredCanvasFormat: () => 'bgra8unorm'}}});
function fixture() {
  const shaders: string[] = [], bindings: any[] = [], writes: any[] = [];
  let allocations = 0;
  const buffer = (desc: any) => ({...desc, destroyed: false, destroy() {this.destroyed = true;}});
  const pass = () => ({setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() {}, setViewport() {}, setScissorRect() {}, draw() {}, end() {}});
  const pipeline = () => ({getBindGroupLayout: () => ({})});
  const device: any = {
    limits: {minStorageBufferOffsetAlignment: 256, maxStorageBufferBindingSize: 1 << 28},
    queue: {writeBuffer(...args: any[]) {writes.push(args);}, submit() {}, onSubmittedWorkDone: async () => {}},
    createBuffer(desc: any) {allocations++; return buffer(desc);},
    createShaderModule({code}: any) {shaders.push(code); return {};},
    createComputePipeline: pipeline, createRenderPipeline: pipeline,
    createBindGroup(desc: any) {bindings.push(desc);return {};},
    createCommandEncoder: () => ({beginComputePass: pass, beginRenderPass: pass, finish() {return {};}}),
  };
  const raw = buffer({size: 3 * 4096 * 4, usage: 1});
  const view: Uint32ImageView = {device, buffer: raw as unknown as GPUBuffer, byteOffset: 4096 * 4, count: 4096, divisor: 2472};
  return {device, raw, view, shaders, bindings, writes, allocations: () => allocations};
}

test('reject incompatible count views before renderer allocation or submission', () => {
  const f = fixture();
  validateUint32ImageView(f.view, f.device, 4096);
  for (const changed of [{device: {}}, {byteOffset: 4}, {byteOffset: -256}, {count: 4095}, {divisor: 0}, {divisor: Infinity}, {divisor: 1e-100}, {byteOffset: Number.MAX_SAFE_INTEGER}, {byteOffset: 3 * 4096 * 4}]) {
    assert.throws(() => validateUint32ImageView({...f.view,...changed} as Uint32ImageView, f.device, 4096));
  }
  assert.equal(f.allocations(), 0);
});

test('shared integer means borrow aligned slices and leave float ownership intact', () => {
  const f = fixture(), engine = new GPUColormapEngine(f.device);
  const float = f.device.createBuffer({size: 4096 * 4, usage: 1});
  engine.adoptBuffer(0, float, 64, 64);
  engine.uploadLUT('gray', Uint8Array.from({length: 768},(_,i)=>Math.floor(i/3)));
  const context = {getCurrentTexture: () => ({createView: () => ({})})} as unknown as GPUCanvasContext;
  const render = (counts?: ReadonlyMap<number, Uint32ImageView>) => engine.renderSlotsDirectWithGpuRangeToCanvas(
    [0], [{x: 0, y: 0, width: 64, height: 64}], context, 1, 99, true, {width: 64, height: 64, bgRgb: 0, counts});
  const before = f.allocations();
  assert.throws(() => render(new Map([[0,{...f.view, device: {} as GPUDevice}]])));
  assert.equal(f.allocations(), before);
  assert.equal(render(new Map([[0,f.view]])), 1);
  const borrowed = f.bindings.flatMap(b => b.entries).filter(e => e.resource.buffer === f.raw);
  assert.equal(borrowed.length, 2); // range partials plus fragment
  assert(borrowed.every(e => e.resource.offset === 16384 && e.resource.size === 16384));
  assert(!f.writes.some(write => write[0] === f.raw));
  assert(f.shaders.some(code => code.includes('f32(data[at]) / bitcast<f32>(params._pad2)')));
  assert(f.shaders.some(code => code.includes('f32(data[index]) / range_in._p0')));
  const parameterWrites = f.writes.filter(write => write[2] instanceof Uint32Array && write[2].length === 8);
  assert.equal(new Float32Array(parameterWrites.at(-1)[2].buffer)[7], 2472);
  assert.equal(render(), 1); // unchanged float fallback after integer repaint
  engine.destroy();
  assert.equal(float.destroyed, true);
  assert.equal(f.raw.destroyed, false);
});

test('source image views preserve acquisition offsets and reject a disposed source', async () => {
  const {Source112ResidentSet} = await import('../../src/quantem/gpu/detector/compute/webgpu/source112');
  const source: any = Object.create(Source112ResidentSet.prototype);
  const f = fixture(); source.device = f.device; source.acquisitionCount = 66;
  // Constructor-free fixture checks the view admission without allocating source data.
  Object.defineProperty(source, 'output', {value: f.raw});
  source.disposed = false;
  const views = source.imageViewsU32([65, 0], 2472);
  assert.deepEqual(views.map((v: Uint32ImageView) => v.byteOffset), [65 * 262144 * 4, 0]);
  assert(views.every((v: Uint32ImageView) => v.buffer === f.raw && v.device === f.device && v.divisor === 2472));
  assert.throws(() => source.imageViewsU32([-1], 1));
  assert.throws(() => source.imageViewsU32([66], 1));
  assert.throws(() => source.imageViewsU32([0], 0));
  source.disposed = true;
  assert.throws(() => source.imageViewsU32([0], 1), /closed/);
  assert.equal(f.allocations(), 0);
});
