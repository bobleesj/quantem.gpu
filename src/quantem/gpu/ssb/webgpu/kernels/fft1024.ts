import type { WebGPUFFTConfig } from "./common";

export const FFT1024: WebGPUFFTConfig = {
  size: 1024,
  workgroupSize: 256,
  specialized: true,
};
