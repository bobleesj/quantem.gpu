/// <reference types="@webgpu/types" />
import { GPUColormapEngine, type Uint32ImageView } from '../../src/quantem/gpu/display/webgpu/colormaps';

/** Real-device display regression: exact uint32 means before range/log/smoothing.
 * Run on an identified hardware adapter. No performance assertions are made.
 */
export async function runBorrowedCountDisplayParity(device: GPUDevice) {
  const engine = new GPUColormapEngine(device), stride = 4096, width = 128, height = 64;
  const raw = device.createBuffer({size: stride * 2 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC});
  const values = Uint32Array.from({length: stride * 2}, (_, i) => [0, 1, 65535, 16777217, 0xfffffffe, 0xffffffff][i % 6]);
  device.queue.writeBuffer(raw, 0, values);
  const params = device.createBuffer({size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST});
  const module = device.createShaderModule({code: `
struct P { count:u32, divisor:f32, pad:vec2u }
@group(0) @binding(0) var<storage,read> counts:array<u32>;
@group(0) @binding(1) var<storage,read_write> means:array<f32>;
@group(0) @binding(2) var<uniform> p:P;
@compute @workgroup_size(256) fn convert(@builtin(global_invocation_id) id:vec3u) {
 if(id.x<p.count){means[id.x]=f32(counts[id.x])/p.divisor;}
}`});
  const pipeline = device.createComputePipeline({layout: 'auto', compute: {module, entryPoint: 'convert'}});
  const dimensions = [32, 64], floats = dimensions.map(size => device.createBuffer({size: size * size * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC}));
  floats.forEach((buffer, i) => engine.adoptBuffer(i, buffer, dimensions[i], dimensions[i]));
  engine.uploadLUT('count-parity-gray', Uint8Array.from({length: 768}, (_, i) => Math.floor(i / 3)));
  const texture = device.createTexture({size: [width, height], format: navigator.gpu.getPreferredCanvasFormat(), usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC});
  const read = device.createBuffer({size: width * height * 4 + 32, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});
  const rawRead = device.createBuffer({size: values.byteLength, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});
  const context = {getCurrentTexture: () => texture} as unknown as GPUCanvasContext;
  const rectangles = dimensions.map((_, i) => ({x: i * 64, y: 0, width: 64, height: 64}));
  const results: {divisor: number; log: boolean; smooth: boolean; rgbaBytes: number; rangeBytes: number}[] = [];
  device.pushErrorScope('validation');
  try {
    for (const divisor of [1, 3, 2472]) {
      for (let i = 0; i < 2; i++) {
        const count = dimensions[i] ** 2, settings = new Uint32Array([count, 0, 0, 0]);
        new Float32Array(settings.buffer)[1] = divisor;
        device.queue.writeBuffer(params, 0, settings);
        const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, device.createBindGroup({layout: pipeline.getBindGroupLayout(0), entries: [
          {binding: 0, resource: {buffer: raw, offset: i * stride * 4, size: count * 4}},
          {binding: 1, resource: {buffer: floats[i]}}, {binding: 2, resource: {buffer: params}},
        ]}));
        pass.dispatchWorkgroups(Math.ceil(count / 256)); pass.end(); device.queue.submit([encoder.finish()]);
      }
      for (const log of [false, true]) for (const smooth of [false, true]) {
        const capture = async (counts?: ReadonlyMap<number, Uint32ImageView>) => {
          const painted = engine.renderSlotsDirectWithGpuRangeToCanvas([0, 1], rectangles, context, 3, 94, log, {width, height, bgRgb: 0, smooth, counts});
          if (painted !== 2) throw new Error('Both native images must render.');
          const encoder = device.createCommandEncoder();
          encoder.copyTextureToBuffer({texture}, {buffer: read, bytesPerRow: width * 4}, [width, height]);
          // Internal range layout is checked as bytes: vmin/vmax, excluding the divisor metadata.
          const slots = (engine as unknown as {slots: {rangeBuffer: GPUBuffer}[]}).slots;
          for (let i = 0; i < 2; i++) encoder.copyBufferToBuffer(slots[i].rangeBuffer, 0, read, width * height * 4 + i * 16, 8);
          device.queue.submit([encoder.finish()]); await read.mapAsync(GPUMapMode.READ);
          try {return new Uint8Array(read.getMappedRange()).slice();} finally {read.unmap();}
        };
        const expected = await capture();
        const views = new Map(dimensions.map((size, i) => [i, {device, buffer: raw, byteOffset: i * stride * 4, count: size * size, divisor}]));
        const actual = await capture(views);
        for (let i = 0; i < actual.length; i++) if (actual[i] !== expected[i]) throw new Error(`Mean display mismatch at byte ${i}, divisor=${divisor}, log=${log}, smooth=${smooth}`);
        results.push({divisor, log, smooth, rgbaBytes: width * height * 4, rangeBytes: 16});
      }
    }
    const encoder = device.createCommandEncoder(); encoder.copyBufferToBuffer(raw, 0, rawRead, 0, values.byteLength); device.queue.submit([encoder.finish()]);
    await rawRead.mapAsync(GPUMapMode.READ);
    try {
      const actual = new Uint32Array(rawRead.getMappedRange());
      if (actual.some((value, i) => value !== values[i])) throw new Error('Display changed source-owned counts.');
    } finally {rawRead.unmap();}
    return {allExact: true, results, unchangedIntegerCounts: values.length};
  } finally {
    await device.queue.onSubmittedWorkDone();
    engine.destroy(); raw.destroy(); params.destroy(); texture.destroy(); read.destroy(); rawRead.destroy();
    const error = await device.popErrorScope(); if (error) throw new Error(error.message);
  }
}
