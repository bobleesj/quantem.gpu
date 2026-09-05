/** Consumer imports only. Kernels remain owned by their scientific domains. */
export * from "../io/backends/webgpu/dense";
export * from "../io/backends/webgpu/packed";
export {
  DetectorCompute,
  buildFullDetectorMask,
} from "../detector/backends/webgpu/backend";
export {
  GPUColormapEngine,
  createGPUColormapEngine,
} from "../display/backends/webgpu/colormaps";
