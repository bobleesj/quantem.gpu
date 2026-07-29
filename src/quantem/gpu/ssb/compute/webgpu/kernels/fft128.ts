import type { WebGPUFFTConfig } from "./common";

export const FFT128: WebGPUFFTConfig = {
  size: 128,
  workgroupSize: 128,
  specialized: true,
};
