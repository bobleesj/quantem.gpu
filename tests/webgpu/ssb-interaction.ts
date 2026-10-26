/// <reference types="@webgpu/types" />
import { WebGPUDirectPtychoBackend } from "../../src/quantem/gpu/direct_ptycho/webgpu/backend";

/** A stopped diagnostic must leave the next complete reconstruction usable. */
export async function runSsbInteractionParity(base: string): Promise<Record<string, number>> {
  const calibration = await (await fetch(base + "ssb-calibration.json")).text();
  const cal = JSON.parse(calibration);
  const plane = cal.g_shape[1] * cal.g_shape[2];
  const source = (url: string) => ({ json: JSON.stringify({ kind: "bf_columns", url,
    dtype: "uint16", encoding: "uint16", num_bright_field_pixels: cal.num_bright_field_pixels,
    plane, scan_shape: cal.g_shape.slice(1), bytes_per_bright_field_pixel: plane * 2, bits_per_value: 16 }) });
  const engine = new WebGPUDirectPtychoBackend(calibration, source(base + "ssb-columns.u16"));
  const options = { computeLoss: true, phaseEstimator: "mean_phase" as const,
    sample: { sampleTiltMrad: [2.2, -3.1] as [number, number], depthSpreadA: 140 } };
  try {
    const expected = await engine.reconstruct(-120, 35, 0.6, options);
    const yielded = await engine.reconstruct(-120, 35, 0.6,
      { ...options, signal: new AbortController().signal });
    if (yielded.loss !== expected.loss || yielded.phase.some((v, i) => v !== expected.phase[i])) {
      throw new Error("Yielding between diagnostic chunks changed the complete result");
    }
    const controller = new AbortController();
    const cancelled = engine.reconstruct(-120, 35, 0.6, { ...options, signal: controller.signal });
    setTimeout(() => controller.abort(), 0);
    try { await cancelled; throw new Error("Cancelled diagnostic returned a result"); }
    catch (error) { if (!(error instanceof DOMException) || error.name !== "AbortError") throw error; }
    const restored = await engine.reconstruct(-120, 35, 0.6, options);
    if (restored.loss !== expected.loss || restored.phase.some((v, i) => v !== expected.phase[i])) {
      throw new Error("Cancelled diagnostic damaged the following reconstruction");
    }
  } finally { engine.close(); }
  try { await engine.reconstruct(-120, 35, 0.6, options); throw new Error("Closed backend reconstructed"); }
  catch (error) { if (!(error instanceof Error) || !error.message.includes("SSB backend is closed")) throw error; }

  // A constant scan has only DC. Zero insertion must not turn replicas of
  // that mean into atomic-looking contrast on an upsampled output grid.
  const constant = new Uint16Array(plane * cal.num_bright_field_pixels);
  for (let bf = 0; bf < cal.num_bright_field_pixels; bf++) constant.fill(100 + bf, bf * plane, (bf + 1) * plane);
  const url = URL.createObjectURL(new Blob([constant.buffer]));
  const flat = new WebGPUDirectPtychoBackend(calibration, source(url));
  let maximum = 0;
  try {
    for (const factor of [1, 2, 4, 8] as const) for (const depth of [0, 180]) {
      const result = await flat.reconstruct(-80, 40, -0.7, { computeLoss: false,
        upsamplingFactor: factor, sample: { sampleTiltMrad: [-9, 5], depthSpreadA: depth } });
      for (const phase of result.phase) maximum = Math.max(maximum, Math.abs(phase));
      if (!(maximum < 1e-7)) throw new Error(`Constant scan acquired phase contrast: ${maximum}`);
    }
  } finally { flat.close(); URL.revokeObjectURL(url); }
  return { constant_scan_max_phase: maximum, diagnostic_parity: 0 };
}
