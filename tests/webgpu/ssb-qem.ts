/// <reference types="@webgpu/types" />
import { WebGPUSSBBackend, setSSBLocalFiles } from "../../src/quantem/gpu/ssb/webgpu/backend";
import type { WebGPUReconstructionOptions } from "../../src/quantem/gpu/ssb/webgpu/protocol";

/** The exported-folder consumer must receive the same phase from streamed QEM and raw BF columns. */
export async function runSsbQemParity(base: string): Promise<Record<string, number>> {
  const calibration = await (await fetch(base + "ssb-calibration.json")).text();
  const cal = JSON.parse(calibration);
  const plane = cal.g_shape[1] * cal.g_shape[2];
  const qemName = "ssb columns.qem";
  const qemURL = new URL(qemName + "?admission=stream-check", base).href;
  const source = (url: string, encoding: "uint16" | "qem") => ({
    kind: "bf_columns" as const,
    json: JSON.stringify({ kind: "bf_columns", url, dtype: "uint16", encoding,
      num_bright_field_pixels: cal.num_bright_field_pixels, plane,
      scan_shape: cal.g_shape.slice(1), bytes_per_bright_field_pixel: plane * 2, bits_per_value: 16 }),
  });
  const modes: WebGPUReconstructionOptions[] = [
    { computeLoss: true, phaseEstimator: "mean_phase" },
    { computeLoss: true, phaseEstimator: "phase_of_mean", upsamplingFactor: 2,
      sample: { sampleTiltMrad: [2.2, -3.1], depthSpreadA: 140 } },
  ];
  const raw = new WebGPUSSBBackend(calibration, source(base + "ssb-columns.u16", "uint16"));
  const expected: { phase: Float32Array; loss: number | null }[] = [];
  try {
    for (const options of modes) {
      const result = await raw.reconstruct(-120, 35, 0.6, options);
      if (!result.phase.some(value => value !== result.phase[0])) {
        throw new Error("The raw BF reference must have phase contrast");
      }
      expected.push({ phase: new Float32Array(result.phase), loss: result.loss });
    }
  } finally { raw.close(); }
  const results: Record<string, number> = { bright_field_pixels: cal.num_bright_field_pixels };
  async function compare(engine: WebGPUSSBBackend, transport: string): Promise<void> {
    for (const [index, options] of modes.entries()) {
      const actual = await engine.reconstruct(-120, 35, 0.6, options);
      const before = new Uint32Array(expected[index].phase.buffer);
      const after = new Uint32Array(actual.phase.buffer, actual.phase.byteOffset, actual.phase.length);
      if (after.length !== before.length || after.some((value, i) => value !== before[i])) {
        throw new Error(`${transport}: ${index ? "tilt/depth" : "regular"} SSB phase changed`);
      }
      if (!Number.isFinite(actual.loss) || actual.loss !== expected[index].loss
          || actual.phase.some(value => !Number.isFinite(value))) {
        throw new Error(`${transport}: SSB loss or finite phase changed`);
      }
      results[`${transport}_${index ? "tilt_depth" : "regular"}_max_difference`] = 0;
    }
  }
  function readOnce(ranges: number[][], size: number): void {
    let end = 0;
    for (const [first, stop] of ranges.sort((a, b) => a[0] - b[0])) {
      if (first !== end) throw new Error("SSB skipped or reread QEM source bytes");
      end = stop;
    }
    if (end !== size) throw new Error("SSB did not authenticate the complete QEM");
  }
  const fetchSource = globalThis.fetch;
  const requests: { method: string; range: string | null }[] = [];
  globalThis.fetch = async (input, init) => {
    if (String(input) === qemURL) {
      requests.push({ method: init?.method ?? "GET", range: new Headers(init?.headers).get("Range") });
    }
    return fetchSource(input, init);
  };
  const http = new WebGPUSSBBackend(calibration, source(qemURL, "qem"));
  try {
    await http.prepare();
    await compare(http, "http");
  } finally { globalThis.fetch = fetchSource; http.close(); }
  if (requests.filter(request => request.method === "HEAD").length !== 1
      || requests.some(request => request.method === "GET" && !request.range)) {
    throw new Error("SSB buffered the full HTTP QEM instead of using authenticated ranges");
  }
  const bytes = await (await fetch(qemURL)).arrayBuffer();
  readOnce(requests.filter(request => request.range).map(request => {
    const [first, last] = /bytes=(\d+)-(\d+)/.exec(request.range!)!.slice(1).map(Number);
    return [first, last + 1];
  }), bytes.byteLength);
  const localReads: number[][] = [];
  class SelectedFile extends File {
    async arrayBuffer(): Promise<ArrayBuffer> { throw new Error("SSB read the whole selected QEM at once"); }
    slice(first = 0, end = this.size): Blob {
      localReads.push([first, end]);
      return super.slice(first, end);
    }
  }
  setSSBLocalFiles([new SelectedFile([bytes], qemName)]);
  const local = new WebGPUSSBBackend(calibration, source(qemName, "qem"));
  try {
    await compare(local, "local");
    readOnce(localReads, bytes.byteLength);
  } finally { local.close(); }
  return { ...results, qem_bytes: bytes.byteLength, http_reads: requests.length, local_reads: localReads.length };
}
