/// <reference types="@webgpu/types" />
/** Uint8 scientific-display decoding shared by browser widgets. */

import { requireHardwareGPUDevice } from "../../device/webgpu";

/** Decode uint8 display samples using ``value * (high - low) / 255 + low``. */
export function dequantizeUint8(
  values: Uint8Array,
  low: number,
  high: number,
  output: Float32Array = new Float32Array(values.length),
): Float32Array {
  if (output.length < values.length) throw new Error("dequantizeUint8 output is shorter than input");
  const finiteLow = Number.isFinite(low) ? low : 0;
  const finiteHigh = Number.isFinite(high) ? high : finiteLow;
  const scale = finiteHigh > finiteLow ? (finiteHigh - finiteLow) / 255 : 0;
  for (let index = 0; index < values.length; index++) {
    output[index] = values[index] * scale + finiteLow;
  }
  return output;
}

const DEQUANTIZE_UINT8_WGSL = /* wgsl */ `
struct Params {
  count: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
  low: f32,
  scale: f32,
  _pad3: f32,
  _pad4: f32,
};
@group(0) @binding(0) var<storage, read> packed: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= p.count) { return; }
  let word = packed[index >> 2u];
  let shift = (index & 3u) * 8u;
  let encoded = (word >> shift) & 255u;
  output[index] = f32(encoded) * p.scale + p.low;
}
`;

function packUint8(values: Uint8Array): Uint32Array {
  const packed = new Uint32Array(Math.max(1, Math.ceil(values.length / 4)));
  for (let index = 0; index < values.length; index++) {
    packed[index >> 2] |= values[index] << ((index & 3) * 8);
  }
  return packed;
}

/** Decode uint8 scientific values on a real hardware WebGPU adapter. */
export async function dequantizeUint8WebGPU(
  values: Uint8Array,
  low: number,
  high: number,
): Promise<Float32Array> {
  if (values.length === 0) return new Float32Array(0);
  const finiteLow = Number.isFinite(low) ? low : 0;
  const finiteHigh = Number.isFinite(high) ? high : finiteLow;
  const scale = finiteHigh > finiteLow ? (finiteHigh - finiteLow) / 255 : 0;
  const packed = packUint8(values);
  const device = await requireHardwareGPUDevice("Uint8 scientific-value decoding");
  const sourceBuffer = device.createBuffer({
    size: packed.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const outputBuffer = device.createBuffer({
    size: values.length * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const paramsBuffer = device.createBuffer({
    size: 32,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const readBuffer = device.createBuffer({
    size: values.length * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  try {
    const params = new ArrayBuffer(32);
    new Uint32Array(params)[0] = values.length;
    const paramsFloats = new Float32Array(params);
    paramsFloats[4] = finiteLow;
    paramsFloats[5] = scale;
    device.queue.writeBuffer(
      sourceBuffer,
      0,
      packed.buffer as ArrayBuffer,
      packed.byteOffset,
      packed.byteLength,
    );
    device.queue.writeBuffer(paramsBuffer, 0, params);
    const module = device.createShaderModule({ code: DEQUANTIZE_UINT8_WGSL });
    const pipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: sourceBuffer } },
        { binding: 1, resource: { buffer: outputBuffer } },
        { binding: 2, resource: { buffer: paramsBuffer } },
      ],
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(values.length / 256));
    pass.end();
    encoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, values.length * 4);
    device.queue.submit([encoder.finish()]);
    await readBuffer.mapAsync(GPUMapMode.READ);
    const output = new Float32Array(readBuffer.getMappedRange().slice(0));
    readBuffer.unmap();
    return output;
  } finally {
    sourceBuffer.destroy();
    outputBuffer.destroy();
    paramsBuffer.destroy();
    readBuffer.destroy();
  }
}
