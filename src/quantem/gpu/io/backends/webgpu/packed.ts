/** Lossless-packed IO entry point; authentication and kernels have one owner. */
export {
  CompactH5SessionQualificationCache,
  WebGPUCompactH5ResidentSource,
  loadCompactH5WebGPU,
  parseCompactH5Index,
  probeCompactH5HttpSource,
} from "./compact-h5";
export type {
  CompactH5ByteSource,
  CompactH5DetectorCalibration,
  CompactH5Index,
  CompactH5PreparedDetectorProduct,
  CompactH5PreparedDetectorProducts,
  CompactH5PreparedDpcMoments,
  CompactH5ResidentReceipt,
  CompactH5ShardIndex,
  CompactH5Source,
  CompactH5TrustedQualificationV1,
  WebGPUCompactH5DetectorMetrics,
  WebGPUCompactH5ExactMomentSnapshot,
  WebGPUCompactH5LoadProfile,
} from "./compact-h5";
