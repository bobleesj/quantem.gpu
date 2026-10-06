/// <reference types="@webgpu/types" />
import { GPUColormapEngine } from '../../src/quantem/gpu/display/webgpu/colormaps';

/** Real-adapter regression for rounded means, including near exponent boundaries. */
export async function runResidentPatternMeanParity(device: GPUDevice) {
  const engine = new GPUColormapEngine(device);
  const values = Array.from({length: 4097}, (_, i) => i);
  for (let exponent = 0; exponent < 24; exponent++) {
    for (let delta = -4; delta <= 4; delta++) {
      const value = 2 ** exponent + delta;
      if (value >= 0 && value <= 16777215) values.push(value);
    }
  }
  const source = Float32Array.from(values), zero = new Float32Array(values.length);
  let checked = 0;
  try {
    for (let index = 1; index <= 66; index++) engine.uploadData(index, index === 1 ? source : zero, values.length, 1);
    for (let count = 1; count <= 66; count++) {
      if (!engine.averageResidentSlotsInto(0, Array.from({length: count}, (_, i) => i + 1))) throw Error('Mean dispatch failed');
      const [actual] = await engine.readDataSlots([0]);
      if (!actual) throw Error('Missing mean');
      for (let i = 0; i < values.length; i++) {
        const expected = Math.fround(values[i] / count);
        if (actual[i] !== expected) throw Error(`Mean mismatch: sum=${values[i]}, count=${count}, actual=${actual[i]}, expected=${expected}`);
        checked++;
      }
    }
    return {checked, allExact: true};
  } finally { await device.queue.onSubmittedWorkDone(); engine.destroy(); }
}
