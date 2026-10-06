import type { WebGPUFFTConfig } from "./common";
import { FFT128 } from "./fft128";
import { FFT256 } from "./fft256";
import { FFT512 } from "./fft512";
import { FFT1024 } from "./fft1024";

export type SupportedSsbSize = WebGPUFFTConfig["size"];

export const WEBGPU_FFT_CONFIGS: ReadonlyMap<number, WebGPUFFTConfig> = new Map([
  [FFT128.size, FFT128],
  [FFT256.size, FFT256],
  [FFT512.size, FFT512],
  [FFT1024.size, FFT1024],
]);

export const SUPPORTED_SSB_SIZES = [128, 256, 512, 1024] as const;

export function getWebGPUFFTConfig(size: number): WebGPUFFTConfig | null {
  return WEBGPU_FFT_CONFIGS.get(size) ?? null;
}
