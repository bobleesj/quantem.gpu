/// <reference types="@webgpu/types" />
import { GPUColormapEngine } from "../../src/quantem/gpu/display/webgpu/colormaps";

type Slot = { dataBuffer: GPUBuffer; rangeBuffer: GPUBuffer; rangePartialsBuffer: GPUBuffer | null };
type Internals = {
  slots: Slot[];
  lutBuffer: GPUBuffer;
  rangeRegionPipeline: GPUComputePipeline;
  colormapRangePipeline: GPUComputePipeline;
  ensureColormapRangePipeline(): void;
};

/** Compare parallel ranges and rendered pixels with the retained legacy GPU path. */
export async function runParallelRangeParity(device: GPUDevice): Promise<Record<string, number | boolean>> {
  device.pushErrorScope("validation");
  const engine = new GPUColormapEngine(device);
  const internals = engine as unknown as Internals;
  const width = 512, height = 512;
  const lut = Uint8Array.from({ length: 768 }, (_, index) => Math.floor(index / 3));
  engine.uploadLUT("parity-gray", lut);
  const results: Record<string, number | boolean> = {};
  let persistentScratch: GPUBuffer | null = null;
  try {
    const cases = ["signed", "signed-log", "nonfinite", "constant", "negative-log", "extreme", "subregion"];
    for (const name of cases) {
      const log = name.endsWith("log");
      const data = Float32Array.from({ length: width * height }, (_, index) => ((index * 37) % 10007) - 5003);
      if (name === "nonfinite") data.set(Float32Array.from(data, (_, index) => [NaN, Infinity, -Infinity][index % 3]));
      else if (name === "constant") data.fill(-7);
      else if (name === "negative-log") data.set(Float32Array.from(data, value => -Math.abs(value)));
      else if (name === "extreme") { data[117] = -2e38; data[201033] = 2e38; }
      if (name !== "constant" && name !== "nonfinite") { data[0] = NaN; data[8193] = Infinity; data[130111] = -Infinity; }
      const region = name === "subregion" ? { x: 3, y: 7, width: 503, height: 491 } : { x: 0, y: 0, width, height };
      engine.uploadData(0, data, width, height);
      engine.computeRangeRegion(0, region, log);
      internals.ensureColormapRangePipeline();
      const slot = internals.slots[0];
      if (persistentScratch && persistentScratch !== slot.rangePartialsBuffer) throw new Error("Range scratch was reallocated for an unchanged slot");
      persistentScratch = slot.rangePartialsBuffer;
      const reference = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      const regionParams = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(regionParams, 0, new Uint32Array([region.x, region.y, region.width, region.height, width, +log, 0, 0]));
      const count = region.width * region.height;
      const rgba = [0, 1].map(() => device.createBuffer({ size: count * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC }));
      const mapped = device.createBuffer({ size: 32 + count * 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const colorParams = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const colorBytes = new ArrayBuffer(32), colorU32 = new Uint32Array(colorBytes), colorF32 = new Float32Array(colorBytes);
      colorU32.set([region.width, region.height, 0, 0, +log, region.x, region.y, width]);
      colorF32[2] = 7; colorF32[3] = 91;
      device.queue.writeBuffer(colorParams, 0, colorBytes);
      try {
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        const legacy = internals.rangeRegionPipeline;
        pass.setPipeline(legacy);
        pass.setBindGroup(0, device.createBindGroup({ layout: legacy.getBindGroupLayout(0), entries: [
          { binding: 0, resource: { buffer: slot.dataBuffer } }, { binding: 1, resource: { buffer: regionParams } }, { binding: 2, resource: { buffer: reference } },
        ] }));
        pass.dispatchWorkgroups(1);
        const color = internals.colormapRangePipeline;
        pass.setPipeline(color);
        for (let index = 0; index < 2; index++) {
          pass.setBindGroup(0, device.createBindGroup({ layout: color.getBindGroupLayout(0), entries: [
            { binding: 0, resource: { buffer: colorParams } }, { binding: 1, resource: { buffer: slot.dataBuffer } },
            { binding: 2, resource: { buffer: internals.lutBuffer } }, { binding: 3, resource: { buffer: rgba[index] } },
            { binding: 4, resource: { buffer: index ? reference : slot.rangeBuffer } },
          ] }));
          pass.dispatchWorkgroups(Math.ceil(region.width / 16), Math.ceil(region.height / 16));
        }
        pass.end();
        encoder.copyBufferToBuffer(slot.rangeBuffer, 0, mapped, 0, 16);
        encoder.copyBufferToBuffer(reference, 0, mapped, 16, 16);
        encoder.copyBufferToBuffer(rgba[0], 0, mapped, 32, count * 4);
        encoder.copyBufferToBuffer(rgba[1], 0, mapped, 32 + count * 4, count * 4);
        device.queue.submit([encoder.finish()]);
        await mapped.mapAsync(GPUMapMode.READ);
        const mappedBytes = mapped.getMappedRange();
        const ranges = new Float32Array(mappedBytes, 0, 8);
        for (let i = 0; i < 4; i++) if (ranges[i] !== ranges[4 + i]) throw new Error(`${name} range ${i}: ${ranges[i]} != ${ranges[4 + i]}`);
        const pixels = new Uint32Array(mappedBytes, 32, count * 2);
        for (let i = 0; i < count; i++) if (pixels[i] !== pixels[count + i]) throw new Error(`${name} normalized pixel ${i} differs`);
        results[name] = true;
      } finally { for (const buffer of [reference, regionParams, colorParams, mapped, ...rgba]) buffer.destroy(); }
    }
    // Seven independent slots share one encoder, matching the compare grid.
    const batched = device.createCommandEncoder();
    const readback = device.createBuffer({ size: 7 * 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    for (let slot = 0; slot < 7; slot++) {
      engine.uploadData(slot, new Float32Array(width * height).fill(slot + 1), width, height);
      engine.recordComputeRangeRegion(batched, slot);
      batched.copyBufferToBuffer(internals.slots[slot].rangeBuffer, 0, readback, slot * 16, 16);
    }
    device.queue.submit([batched.finish()]);
    // The convenience call also releases the recorded parameter buffers after
    // submission, as the production render wrapper does.
    engine.computeRangeRegion(0);
    try {
      await readback.mapAsync(GPUMapMode.READ);
      const ranges = new Float32Array(readback.getMappedRange());
      for (let slot = 0; slot < 7; slot++) {
        if (ranges[slot * 4] !== slot + 1 || ranges[slot * 4 + 1] !== slot + 1) throw new Error(`Batched slot ${slot} range differs`);
      }
    } finally { readback.destroy(); }
    results.seven_slots_exact = true;
    results.scratch_reused = true;
    results.all_passed = true;
    return results;
  } finally {
    engine.destroy();
    const validation = await device.popErrorScope();
    if (validation) throw new Error(validation.message);
  }
}
