/// <reference types="@webgpu/types" />

import { requireHardwareGPUDevice } from "../../device/webgpu";
import { fft2d, nextPow2, requireWebGPUFFT } from "./fft";

export type FrequencyFilterMode = "none" | "lowpass" | "highpass" | "bandpass";

export interface FrequencyFilterOptions {
  mode: FrequencyFilterMode | string;
  cutoff?: number;
  center?: number;
  width?: number;
  edge?: number;
}

export type FrequencyFilterBackend = "off" | "WebGPU" | "unavailable";
let lastFrequencyFilterBackend: FrequencyFilterBackend = "off";

export function getFrequencyFilterBackend(): FrequencyFilterBackend {
  return lastFrequencyFilterBackend;
}

export function normalizeFrequencyFilterMode(mode: string): FrequencyFilterMode {
  const value = String(mode ?? "none").trim().toLowerCase().replace(/[ _-]/g, "");
  if (value === "low" || value === "lowpass") return "lowpass";
  if (value === "high" || value === "highpass") return "highpass";
  if (value === "band" || value === "bandpass") return "bandpass";
  return "none";
}

export function frequencyFilterActive(mode: string): boolean {
  return normalizeFrequencyFilterMode(mode) !== "none";
}

function clamp01(value: number, fallback: number): number {
  return Math.max(0, Math.min(1, Number.isFinite(value) ? value : fallback));
}

/** Smooth radial mask where radius is normalized to Nyquist (0..1). */
export function frequencyMaskValue(radius: number, options: FrequencyFilterOptions): number {
  const mode = normalizeFrequencyFilterMode(options.mode);
  if (mode === "none") return 1;
  const edge = Math.max(0.002, clamp01(Number(options.edge), 0.035));
  const sigmoid = (x: number) => 1 / (1 + Math.exp(-x / edge));
  if (mode === "lowpass") return 1 - sigmoid(radius - clamp01(Number(options.cutoff), 0.25));
  if (mode === "highpass") return sigmoid(radius - clamp01(Number(options.cutoff), 0.08));
  const center = clamp01(Number(options.center), 0.3);
  const half = Math.max(edge, clamp01(Number(options.width), 0.12) / 2);
  return sigmoid(radius - Math.max(0, center - half)) * (1 - sigmoid(radius - Math.min(1, center + half)));
}

function applyMask(
  real: Float32Array,
  imag: Float32Array,
  width: number,
  height: number,
  options: FrequencyFilterOptions,
): void {
  const nx = Math.max(1, width / 2);
  const ny = Math.max(1, height / 2);
  for (let y = 0; y < height; y++) {
    const fy = Math.min(y, height - y) / ny;
    for (let x = 0; x < width; x++) {
      const fx = Math.min(x, width - x) / nx;
      const radius = Math.min(1, Math.hypot(fx, fy));
      const mask = frequencyMaskValue(radius, options);
      const idx = y * width + x;
      real[idx] *= mask;
      imag[idx] *= mask;
    }
  }
}

const FREQUENCY_MASK_SHADER = /* wgsl */ `
struct Params {
  width: u32,
  height: u32,
  mode: u32,
  _pad: u32,
  cutoff: f32,
  center: f32,
  band_width: f32,
  edge: f32,
}
@group(0) @binding(0) var<storage, read_write> complex_data: array<vec2<f32>>;
@group(0) @binding(1) var<uniform> p: Params;

fn sigmoid(value: f32) -> f32 {
  return 1.0 / (1.0 + exp(-value / max(0.002, p.edge)));
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= p.width || gid.y >= p.height) { return; }
  let nx = max(1.0, f32(p.width) * 0.5);
  let ny = max(1.0, f32(p.height) * 0.5);
  let fx = f32(min(gid.x, p.width - gid.x)) / nx;
  let fy = f32(min(gid.y, p.height - gid.y)) / ny;
  let radius = min(1.0, length(vec2<f32>(fx, fy)));
  var mask = 1.0;
  if (p.mode == 1u) {
    mask = 1.0 - sigmoid(radius - p.cutoff);
  } else if (p.mode == 2u) {
    mask = sigmoid(radius - p.cutoff);
  } else if (p.mode == 3u) {
    let half_width = max(p.edge, p.band_width * 0.5);
    mask = sigmoid(radius - max(0.0, p.center - half_width))
      * (1.0 - sigmoid(radius - min(1.0, p.center + half_width)));
  }
  let index = gid.y * p.width + gid.x;
  complex_data[index] = complex_data[index] * mask;
}
`;

async function applyFrequencyFilterWebGPUResident(
  data: Float32Array,
  width: number,
  height: number,
  options: FrequencyFilterOptions,
): Promise<Float32Array> {
  if (width < 1 || height < 1 || data.length !== width * height) {
    throw new Error(
      `Frequency filtering expects width*height values; got ${width}x${height} and ${data.length}.`,
    );
  }
  const device = await requireHardwareGPUDevice("Frequency filtering");
  const fft = await requireWebGPUFFT("Frequency filtering");
  const paddedWidth = nextPow2(width);
  const paddedHeight = nextPow2(height);
  const paddedCount = paddedWidth * paddedHeight;
  const complex = new Float32Array(paddedCount * 2);
  for (let row = 0; row < height; row++) {
    for (let column = 0; column < width; column++) {
      complex[2 * (row * paddedWidth + column)] = data[row * width + column];
    }
  }
  const dataBuffer = device.createBuffer({
    size: complex.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const paramsBuffer = device.createBuffer({
    size: 32,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  try {
    device.queue.writeBuffer(dataBuffer, 0, complex);
    await fft.fft2DResident(dataBuffer, paddedWidth, paddedHeight, false);
    const params = new ArrayBuffer(32);
    const u32 = new Uint32Array(params);
    const f32 = new Float32Array(params);
    const mode = normalizeFrequencyFilterMode(options.mode);
    u32[0] = paddedWidth;
    u32[1] = paddedHeight;
    u32[2] = mode === "lowpass" ? 1 : mode === "highpass" ? 2 : mode === "bandpass" ? 3 : 0;
    f32[4] = clamp01(Number(options.cutoff), mode === "lowpass" ? 0.25 : 0.08);
    f32[5] = clamp01(Number(options.center), 0.3);
    f32[6] = clamp01(Number(options.width), 0.12);
    f32[7] = Math.max(0.002, clamp01(Number(options.edge), 0.035));
    device.queue.writeBuffer(paramsBuffer, 0, params);
    const module = device.createShaderModule({ code: FREQUENCY_MASK_SHADER });
    const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: dataBuffer } },
        { binding: 1, resource: { buffer: paramsBuffer } },
      ],
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(paddedWidth / 16), Math.ceil(paddedHeight / 16));
    pass.end();
    device.queue.submit([encoder.finish()]);
    await fft.fft2DResident(dataBuffer, paddedWidth, paddedHeight, true);

    const readBuffer = device.createBuffer({
      size: complex.byteLength,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const readEncoder = device.createCommandEncoder();
    readEncoder.copyBufferToBuffer(dataBuffer, 0, readBuffer, 0, complex.byteLength);
    device.queue.submit([readEncoder.finish()]);
    await readBuffer.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(readBuffer.getMappedRange().slice(0));
    readBuffer.unmap();
    readBuffer.destroy();
    const output = new Float32Array(width * height);
    for (let row = 0; row < height; row++) {
      for (let column = 0; column < width; column++) {
        output[row * width + column] = result[2 * (row * paddedWidth + column)];
      }
    }
    return output;
  } finally {
    dataBuffer.destroy();
    paramsBuffer.destroy();
  }
}

/** Deterministic CPU parity oracle. Production callers use WebGPU below. */
export function applyFrequencyFilterCPU(
  data: Float32Array,
  width: number,
  height: number,
  options: FrequencyFilterOptions,
): Float32Array {
  if (!frequencyFilterActive(options.mode)) return Float32Array.from(data);
  const real = Float32Array.from(data);
  const imag = new Float32Array(real.length);
  fft2d(real, imag, width, height, false);
  applyMask(real, imag, width, height, options);
  fft2d(real, imag, width, height, true);
  return real;
}

/**
 * Browser-side frequency filter. The padded complex plane stays resident from
 * forward FFT through the radial mask and inverse FFT. Missing WebGPU is an
 * explicit unsupported path; the CPU implementation above is a parity oracle.
 */
export async function applyFrequencyFilterBrowser(
  data: Float32Array,
  width: number,
  height: number,
  options: FrequencyFilterOptions,
): Promise<Float32Array> {
  if (!frequencyFilterActive(options.mode)) {
    lastFrequencyFilterBackend = "off";
    return Float32Array.from(data);
  }
  try {
    const output = await applyFrequencyFilterWebGPUResident(data, width, height, options);
    lastFrequencyFilterBackend = "WebGPU";
    return output;
  } catch (error) {
    lastFrequencyFilterBackend = "unavailable";
    throw error;
  }
}

export function formatFrequencyFilterBanner(options: FrequencyFilterOptions, unit = "Nyquist"): string {
  const mode = normalizeFrequencyFilterMode(options.mode);
  if (mode === "none") return "";
  if (mode === "bandpass") {
    return `Filter: Band-pass center ${clamp01(Number(options.center), 0.3).toFixed(3)}, width ${clamp01(Number(options.width), 0.12).toFixed(3)} ${unit} (view only; raw counts unchanged)`;
  }
  const label = mode === "lowpass" ? "Low-pass" : "High-pass";
  return `Filter: ${label} cutoff ${clamp01(Number(options.cutoff), mode === "lowpass" ? 0.25 : 0.08).toFixed(3)} ${unit} (view only; raw counts unchanged)`;
}
