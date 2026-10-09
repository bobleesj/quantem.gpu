/// <reference types="@webgpu/types" />
/** Browser parity workflow. Bundle with esbuild, then call runRansProductParity(device).
 * The caller supplies an authenticated hardware device; no adapter is selected here.
 */
import { RansResidentSet, type RansManifest } from "../../src/quantem/gpu/detector/webgpu/rans";
import { DetectorCompute } from "../../src/quantem/gpu/detector/webgpu/backend";


type Fixture = { manifest: RansManifest; files: Map<string, Uint8Array>; counts: Uint16Array[] };
function fixture(frames: number, blocks: number, K: number, tilts: number, saturated = false): Fixture {
  const files = new Map<string, Uint8Array>();
  const counts: Uint16Array[] = [];
  const manifest: RansManifest = { tilts: [], bad_pixels: saturated ? [] : [K - 1] };
  const put = (name: string, data: ArrayBufferView) => files.set(name, new Uint8Array(data.buffer, data.byteOffset, data.byteLength).slice());
  for (let tilt = 0; tilt < tilts; tilt++) {
    const stack = new Uint16Array(frames * blocks * K); counts.push(stack);
    const literal = new Uint8Array(K).fill(1); if (!saturated) literal[0] = 0;
    const entries = new Uint32Array([0, 16384, (16384 << 16) | 7, 16384]);
    const lookup = new Uint8Array(K * 256); lookup.fill(1, 128, 256);
    put(`t${tilt}-ctx-0.u32`, new Uint32Array(K + 1));
    put(`t${tilt}-literal-0.u8`, literal);
    put(`t${tilt}-entries-0.u32`, entries);
    put(`t${tilt}-lut-0.u8`, lookup);
    const blocksMeta = [];
    for (let block = 0; block < blocks; block++) {
      const payload: number[] = []; const offsets = new Uint32Array(K + 1);
      for (let k = 0; k < K; k++) {
        offsets[k] = payload.length;
        const column: number[] = [];
        for (let frame = 0; frame < frames; frame++) {
          const scan = block * frames + frame;
          const value = saturated ? 65535 : k === K - 1 ? 65535 : k === 0 ? ((scan + tilt) % 2) * 7 : (scan % 17 === 0 ? 0 : (scan * (k + 1) + tilt * 7 + k) % 31);
          stack[scan * K + k] = value; column.push(value);
        }
        if (literal[k]) {
          for (const value of column) payload.push(value & 255, value >>> 8);
        } else {
          // Frozen two-symbol byte-rANS model: half of the 32768 slots each.
          let state = 8388608; const emitted: number[] = [];
          for (let frame = frames - 1; frame >= 0; frame--) {
            const cumulative = column[frame] ? 16384 : 0;
            while (state >= 1073741824) { emitted.push(state & 255); state = Math.floor(state / 256); }
            state = Math.floor(state / 16384) * 32768 + state % 16384 + cumulative;
          }
          payload.push(state & 255, (state >>> 8) & 255, (state >>> 16) & 255, state >>> 24, ...emitted.reverse());
        }
      }
      offsets[K] = payload.length;
      put(`t${tilt}-payload-${String(block).padStart(2, "0")}.bin`, new Uint8Array(payload));
      put(`t${tilt}-offsets-${String(block).padStart(2, "0")}.u32`, offsets);
      blocksMeta.push({ index: block, bytes: payload.length, model: 0 });
    }
    manifest.tilts.push({ tilt, K, frames, blocks, scale: 15, model_frames: frames, blocks_meta: blocksMeta, models: [{ index: 0, symbols: 2 }] });
  }
  files.set("manifest.json", new TextEncoder().encode(JSON.stringify(manifest)));
  return { manifest, files, counts };
}
async function loadFixture(device: GPUDevice, data: Fixture): Promise<RansResidentSet> {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (input, init) => {
    const name = String(input).replace("https://rans-parity.invalid/", "");
    const bytes = data.files.get(name);
    if (!bytes) return originalFetch(input, init);
    const range = new Headers(init?.headers).get("Range");
    if (range) {
      const match = /^bytes=(\d+)-(\d+)$/.exec(range)!;
      return new Response(bytes.slice(Number(match[1]), Number(match[2]) + 1).buffer, { status: 206 });
    }
    return new Response(bytes.slice().buffer);
  };
  try { return await RansResidentSet.load(device, "https://rans-parity.invalid/"); }
  finally { globalThis.fetch = originalFetch; }
}
function compare(label: string, actual: Float32Array, expected: Float32Array, tolerance = 0): number {
  if (actual.length !== expected.length) throw new Error(`${label}: lengths differ`);
  let maxError = 0;
  for (let index = 0; index < actual.length; index++) {
    const error = Math.abs(actual[index] - expected[index]);
    if (!Number.isFinite(error) || error > tolerance) throw new Error(`${label}[${index}]: ${actual[index]} != ${expected[index]}, tolerance ${tolerance}`);
    maxError = Math.max(maxError, error);
  }
  return maxError;
}

/** Test real scientist products against dense GPU and exact integer references. */
async function runProducts(device: GPUDevice): Promise<Record<string, number | boolean>> {
  const results: Record<string, number | boolean> = {};
  const data = fixture(512, 2, 12, 2);
  const set = await loadFixture(device, data);
  try {
    for (let tilt = 0; tilt < 2; tilt++) {
      const compute = set.computes[tilt];
      const raw = data.counts[tilt];
      const denseBuffer = device.createBuffer({ size: raw.byteLength, usage: GPUBufferUsage.STORAGE, mappedAtCreation: true });
      new Uint16Array(denseBuffer.getMappedRange()).set(raw); denseBuffer.unmap();
      const dense = DetectorCompute.fromGpuChunks(device, [{ buffer: denseBuffer, startScan: 0, nScan: 1024 }], 1024, 12, 0);
      dense.badPx = new Uint32Array([11]);
      try {
        for (const scan of [0, 255, 256, 511, 512, 1023]) {
          const expected = Float32Array.from(raw.subarray(scan * 12, (scan + 1) * 12)); expected[11] = 0;
          compare(`pattern tilt ${tilt} scan ${scan}`, await compute.frameAt(scan), expected);
        }
        const roi = Uint32Array.from({ length: 1024 }, (_, scan) => scan % 3 === 0 ? 1 : 0);
        for (const mean of [false, true]) {
          compare(`ROI ${tilt} mean=${mean}`, await compute.reduceFrames(roi, mean), await dense.reduceFrames(roi, mean));
        }
        const mask = new Uint32Array(12).fill(1); mask[2] = 0;
        compare(`sum ${tilt}`, await compute.maskedSum(mask), await dense.maskedSum(mask));
        const com = await compute.maskedCoM(mask, 4); const baseline = await dense.maskedCoM(mask, 4);
        results[`com_row_${tilt}`] = compare("CoM row", com.comY, baseline.comY, 1e-6);
        results[`com_col_${tilt}`] = compare("CoM col", com.comX, baseline.comX, 1e-6);
        for (const component of ["row", "col"] as const) {
          results[`dpc_${component}_${tilt}`] = compare(`DPC ${component}`, await compute.maskedDpc(mask, 4, component), await dense.maskedDpc(mask, 4, component), 1e-6);
        }
        results[`magnitude_${tilt}`] = compare("DPC magnitude", await compute.maskedDpcMagnitude(mask, 4), await dense.maskedDpcMagnitude(mask, 4), 1e-6);
        for (const [rotation, transpose] of [[0, false], [37, true]] as const) {
          results[`idpc_${tilt}_${rotation}`] = compare("iDPC", await compute.maskedIDpc(mask, 4, 32, 32, rotation, transpose), await dense.maskedIDpc(mask, 4, 32, 32, rotation, transpose), 1e-6);
        }
        const singlePixel = new Uint32Array(12); singlePixel[5] = 1;
        const point = await compute.maskedCoM(singlePixel, 4);
        compare("single-pixel CoM row", point.comY, Float32Array.from(raw.filter((_, i) => i % 12 === 5), (value) => value ? 1 : 0));
        compare("single-pixel CoM col", point.comX, Float32Array.from(raw.filter((_, i) => i % 12 === 5), (value) => value ? 1 : 0));
        const empty = new Uint32Array(12);
        compare("empty iDPC", await compute.maskedIDpc(empty, 4, 32, 32), new Float32Array(1024));
      } finally { dense.dispose(); }
    }
  } finally { set.dispose(); }
  // Full ROI crosses dispatch limits and 32-bit sums: exact sum then divide.
  const large = await loadFixture(device, fixture(131072, 1, 4, 1, true));
  try {
    const mask = new Uint32Array(131072).fill(1);
    compare("wide ROI sum", await large.computes[0].reduceFrames(mask, false), new Float32Array(4).fill(Math.fround(131072 * 65535)));
    compare("wide ROI mean", await large.computes[0].reduceFrames(mask), new Float32Array(4).fill(65535));
    results.large_roi_exact = true;
  } finally { large.dispose(); }
  // Weighted column moments exceed 32 bits although every count is uint16.
  const wide = await loadFixture(device, fixture(256, 1, 512, 1, true));
  try {
    const com = await wide.computes[0].maskedCoM(new Uint32Array(512).fill(1), 512);
    compare("wide moment row", com.comY, new Float32Array(256));
    compare("wide moment col", com.comX, new Float32Array(256).fill(255.5));
    results.wide_moments_exact = true;
  } finally { wide.dispose(); }
  await device.queue.onSubmittedWorkDone();
  results.all_passed = true;
  return results;
}

/** Run the workflow and always balance the caller device's validation scope. */
export async function runRansProductParity(device: GPUDevice): Promise<Record<string, number | boolean>> {
  device.pushErrorScope("validation");
  try { return await runProducts(device); }
  finally {
    const validation = await device.popErrorScope();
    if (validation) throw new Error(validation.message);
  }
}
