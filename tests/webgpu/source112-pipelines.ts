import assert from 'node:assert/strict';
import test from 'node:test';
import {source112Layout, source112Pipeline} from '../../src/quantem/gpu/detector/compute/webgpu/source112-pipelines';
Object.assign(globalThis, {GPUShaderStage: {COMPUTE: 1}});
function device() {
  const calls = {modules: 0, pipelines: 0, layouts: [] as GPUBindGroupLayoutDescriptor[], rejectShader: false, rejectPipeline: false};
  const gpu = {
    createBindGroupLayout(description: GPUBindGroupLayoutDescriptor) {calls.layouts.push(description); return {description};},
    createPipelineLayout(description: GPUPipelineLayoutDescriptor) {return {description};},
    createShaderModule() {
      calls.modules++;
      return {async getCompilationInfo() {
        return {messages: calls.rejectShader ? [{type: 'error', message: 'injected WGSL error'}] : []};
      }};
    },
    async createComputePipelineAsync(description: GPUComputePipelineDescriptor) {
      calls.pipelines++;
      if (calls.rejectPipeline) throw Error('injected pipeline rejection');
      return {description};
    },
    createBuffer() {throw Error('Program reuse must never allocate acquisition buffers');},
  } as unknown as GPUDevice;
  return {gpu, calls};
}
test('many acquisitions share immutable programs, while separate GPUs compile independently', async () => {
  const a = device(), b = device();
  const compiled = await Promise.all(Array.from({length: 66}, () => source112Pipeline(a.gpu, 'sum', 'main')));
  assert(compiled.every(pipeline => pipeline === compiled[0]));
  assert.equal(a.calls.modules, 1); assert.equal(a.calls.pipelines, 1);
  assert.equal(source112Layout(a.gpu), source112Layout(a.gpu));
  assert.notEqual(await source112Pipeline(b.gpu, 'sum', 'main'), compiled[0]);
  assert.notEqual(source112Layout(a.gpu), source112Layout(b.gpu));
  assert.equal(b.calls.modules, 1); assert.equal(b.calls.pipelines, 1);
});
test('scratch and scientific bindings remain distinct, with eight storage buffers and one uniform', async () => {
  const {gpu, calls} = device();
  const a = await source112Pipeline(gpu, 'multi', 'first');
  const b = await source112Pipeline(gpu, 'multi', 'second');
  const c = await source112Pipeline(gpu, 'multi', 'first', 5);
  assert.notEqual(a, b); assert.notEqual(a, c);
  assert.equal(calls.modules, 1); assert.equal(calls.pipelines, 3);
  assert.notEqual(source112Layout(gpu, 5), source112Layout(gpu, 6));
  for (const description of calls.layouts) {
    const entries = Array.from(description.entries);
    assert.equal(entries.length, 9);
    assert.equal(entries.filter(entry => entry.buffer?.type === 'uniform').length, 1);
    assert.equal(entries[8].buffer?.type, 'uniform');
  }
});
test('reopening after a compilation rejection does not reuse a failed program', async () => {
  const {gpu, calls} = device();
  calls.rejectShader = true;
  await assert.rejects(source112Pipeline(gpu, 'bad', 'main'), /injected WGSL/);
  calls.rejectShader = false; calls.rejectPipeline = true;
  await assert.rejects(source112Pipeline(gpu, 'bad', 'main'), /pipeline rejection/);
  calls.rejectPipeline = false;
  await source112Pipeline(gpu, 'bad', 'main');
  assert.equal(calls.modules, 3); assert.equal(calls.pipelines, 2);
  assert.equal(calls.layouts.length, 3, "Failed layouts must not poison a later folder reopen");
});
