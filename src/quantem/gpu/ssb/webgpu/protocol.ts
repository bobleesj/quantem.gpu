export interface WebGPUOptimizationOptions {
  readonly trials: number;
  readonly refinement: "nelder-mead" | null;
  readonly seed: number;
}

export interface SSBPrecision {
  readonly real_dtype: "float32";
  readonly complex_dtype: "complex64";
}

export interface SSBOptimizationResult {
  readonly aberrations: Readonly<Record<string, number>>;
  readonly loss: number;
  readonly trials: number;
}

export interface WebGPUReconstructionOptions {
  readonly upsamplingFactor?: 1 | 2 | 4 | 8;
  readonly phaseEstimator?: "phase_of_mean" | "mean_phase";
  /** No effect: every count is the strided subset `numBrightFieldPixels` selects. */
  readonly preview?: boolean;
  readonly numBrightFieldPixels?: number;
  readonly computeLoss?: boolean;
  readonly scanDetectorRotationDeg?: number;
  /** Higher orders by name: magnitudes in Angstrom (`C21_A`, `C30_A`, ...), azimuths in degrees (`phi21_deg`, ...). */
  readonly higherOrder?: Record<string, number>;
  /** Thick, tilted crystal: tilt in mrad (scan frame, [row, col]), depth spread in Angstrom (C10 unit). Absent or 0 = thin SSB. */
  readonly sample?: WebGPUSampleOptions;
}

export interface WebGPUSampleOptions {
  readonly sampleTiltMrad: readonly [number, number];
  readonly depthSpreadA: number;
}

export interface SSBProtocol<Result> {
  readonly backend: "webgpu";
  readonly precision: SSBPrecision;
  prepare(): Promise<void>;
  fit(
    options: WebGPUOptimizationOptions,
  ): Promise<SSBOptimizationResult>;
  reconstruct(
    c10A: number,
    c12A: number,
    phi12Rad: number,
    options?: WebGPUReconstructionOptions,
  ): Promise<Result>;
  close(): void;
}
