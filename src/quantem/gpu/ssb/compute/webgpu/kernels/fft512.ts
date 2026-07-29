import type { WebGPUFFTConfig } from "./common";

export const FFT512: WebGPUFFTConfig = {
  size: 512,
  workgroupSize: 256,
  specialized: true,
};
