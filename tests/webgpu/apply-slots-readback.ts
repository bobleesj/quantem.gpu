/// <reference types="@webgpu/types" />
import { GPUColormapEngine } from '../../src/quantem/gpu/display/webgpu/colormaps';

/** Raw RGBA readback of adopted slots and of slots uploaded with spare RGBA capacity, on a physical adapter.

An adopted slot used to keep a 16-byte read buffer, so its readback copy failed validation, and a slot whose
rgbaCapacityHint exceeded its pixel count mapped more bytes than the returned image and threw RangeError.
*/
export async function runApplySlotsReadback(device: GPUDevice) {
  const engine = new GPUColormapEngine(device);
  const gray = new Uint8Array(256 * 3).map((_, i) => Math.floor(i / 3));
  engine.uploadLUT('gray', gray);
  const values = Float32Array.from({length: 4 * 5}, (_, i) => i);
  const adopted = device.createBuffer({size: values.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC});
  device.queue.writeBuffer(adopted, 0, values);
  device.pushErrorScope('validation');
  try {
    engine.uploadData(0, values, 4, 5);
    engine.uploadData(1, values, 4, 5, 64);
    engine.adoptBuffer(2, adopted, 4, 5, 'borrowed');
    const results = await engine.applySlots([0, 1, 2], [0, 1, 2].map(() => ({vmin: 0, vmax: 19})));
    const error = await device.popErrorScope();
    if (error) throw Error(`Validation error: ${error.message}`);
    if (results.length !== 3) throw Error(`Expected three slots, got ${results.length}`);
    const [reference] = results;
    for (const {idx, rgba} of results) {
      if (rgba.length !== values.length * 4) throw Error(`Slot ${idx} returned ${rgba.length} bytes`);
      for (let i = 0; i < rgba.length; i++) {
        if (rgba[i] !== reference.rgba[i]) throw Error(`Slot ${idx} byte ${i}: ${rgba[i]} != ${reference.rgba[i]}`);
      }
    }
    if (reference.rgba[0] !== 0 || reference.rgba[(values.length - 1) * 4] !== 255) throw Error('Gray ramp ends are wrong');
    return {slots: results.length, bytes: reference.rgba.length, allEqual: true};
  } finally {
    await device.queue.onSubmittedWorkDone();
    engine.destroy();
    adopted.destroy();
  }
}
