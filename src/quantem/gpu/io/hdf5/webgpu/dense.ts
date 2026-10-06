/** Dense IO entry point. The implementation and legacy imports stay in local-h5. */
export {
  loadLocalH5Master,
  loadShow4DSTEMLocalH5MaskedSum as loadLocalH5MaskedSum,
  setShow4DSTEMLocalFiles as setLocalFiles,
  clearShow4DSTEMLocalFiles as clearLocalFiles,
} from "./local-h5";
export type {
  DataRepresentation,
  LocalH5GpuChunk,
  LocalH5LoadOptions,
  LocalH5LoadProfile,
  LocalH5LoadResult,
  LocalH5MaskedSumOptions,
  LocalH5MaskedSumProfile,
  LocalH5MaskedSumResult,
} from "./local-h5";
