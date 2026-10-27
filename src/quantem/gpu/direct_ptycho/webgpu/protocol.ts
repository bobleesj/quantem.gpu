export interface WebGPUOptimizationOptions {
  /** Number of search trials, as Python's `num_trials`. */
  readonly numTrials: number;
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
  readonly numTrials: number;
}

export interface WebGPUReconstructionOptions {
  readonly upsampling?: 1 | 2 | 4 | 8;
  readonly phaseEstimator?: "phase_of_mean" | "mean_phase";
  /** No effect: every count is the strided subset `numBrightFieldPixels` selects. */
  readonly preview?: boolean;
  readonly numBrightFieldPixels?: number;
  readonly computeLoss?: boolean;
  /** Cancel a diagnostic between GPU chunks; never return a partial result. */
  readonly signal?: AbortSignal;
  readonly scanDetectorRotationDeg?: number;
  /** Higher orders by name: magnitudes in Angstrom (`C21_A`, `C30_A`, ...), azimuths in degrees (`phi21_deg`, ...).
   * A key outside that set throws. */
  readonly higherOrder?: Record<string, number>;
  /** Thick, tilted crystal: tilt in mrad (scan frame, [row, col]), depth spread in Angstrom (C10 unit). Absent or 0 = thin SSB. */
  readonly sample?: WebGPUSampleOptions;
}

export interface WebGPUSampleOptions {
  /** Omitted: no tilt. */
  readonly sampleTiltMrad?: readonly [number, number];
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
