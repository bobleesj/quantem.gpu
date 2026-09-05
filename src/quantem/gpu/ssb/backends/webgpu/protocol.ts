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
  readonly preview?: boolean;
  readonly bfCount?: number;
  readonly computeLoss?: boolean;
  readonly rotationDeg?: number;
  readonly higherOrder?: Record<string, number>;
}

export interface SSBProtocol<Result> {
  readonly backend: "webgpu";
  readonly precision: SSBPrecision;
  prepare(): Promise<void>;
  fit(
    options: WebGPUOptimizationOptions,
  ): Promise<SSBOptimizationResult>;
  reconstruct(
    c10: number,
    c12: number,
    phi12: number,
    options?: WebGPUReconstructionOptions,
  ): Promise<Result>;
  close(): void;
}
