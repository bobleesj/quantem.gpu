/** Consumer imports only. Kernels remain owned by their scientific domains. */
export * from "../io/hdf5/webgpu/dense";
export {
  DetectorCompute,
  buildFullDetectorMask,
} from "../detector/webgpu/backend";
export {
  GPUColormapEngine,
  createGPUColormapEngine,
} from "../display/webgpu/colormaps";
