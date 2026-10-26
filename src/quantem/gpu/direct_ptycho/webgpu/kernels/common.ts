/// <reference types="@webgpu/types" />

export interface WebGPUFFTConfig {
  readonly size: 128 | 256 | 512 | 1024;
  readonly workgroupSize: number;
  readonly specialized: boolean;
}
