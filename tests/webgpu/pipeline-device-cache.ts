/** CPU-only check that decoder pipelines are cached per device, not only by WGSL source. */
import assert from 'node:assert/strict';
import test from 'node:test';
import { cachedPipeline } from '../../src/quantem/gpu/io/hdf5/webgpu/bslz4';

function fakeDevice() {
  let compiled = 0;
  const device: any = {
    createShaderModule: ({code}: {code: string}) => ({code}),
    createComputePipeline(desc: any) { compiled++; return {device, code: desc.compute.module.code}; },
  };
  return {device: device as GPUDevice, compiled: () => compiled};
}

test('a replacement device after device loss compiles its own pipelines', () => {
  const lost = fakeDevice(), replacement = fakeDevice();
  const first = cachedPipeline(lost.device, 'fn main() {}') as any;
  assert.equal(cachedPipeline(lost.device, 'fn main() {}'), first);
  assert.equal(lost.compiled(), 1);
  const second = cachedPipeline(replacement.device, 'fn main() {}') as any;
  assert.notEqual(second, first);
  assert.equal(second.device, replacement.device);
  assert.equal(replacement.compiled(), 1);
  assert.notEqual(cachedPipeline(replacement.device, 'fn other() {}'), second);
});
