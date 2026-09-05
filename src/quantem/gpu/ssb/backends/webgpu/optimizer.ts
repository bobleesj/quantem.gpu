import type {
  SSBOptimizationResult,
  WebGPUOptimizationOptions,
} from "./protocol";


export async function fit(
  _options: WebGPUOptimizationOptions,
): Promise<SSBOptimizationResult> {
  throw new Error(
    "WebGPU SSB aberration optimization is not implemented. "
    + "Run the exact 200-trial plus Nelder-Mead workflow on CUDA or MPS; "
    + "the browser backend never substitutes a reduced objective.",
  );
}
