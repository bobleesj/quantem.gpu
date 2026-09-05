import type { WebGPUFFTConfig } from "./common";

export const FFT256: WebGPUFFTConfig = {
  size: 256,
  workgroupSize: 256,
  specialized: true,
};
