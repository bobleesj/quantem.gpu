/// <reference types="@webgpu/types" />
/** Browser parity workflow. Bundle with esbuild, then call runRansProductParity(device).
 * The caller supplies an authenticated hardware device; no adapter is selected here.
 */
import { RansResidentSet, ransMaskedSumBuffersBatch, ransMaskedSumDeltaBuffersBatch, type RansManifest } from "../../src/quantem/gpu/detector/compute/webgpu/rans";
import { DetectorCompute } from "../../src/quantem/gpu/detector/compute/webgpu/backend";

import { GPUColormapEngine } from "../../src/quantem/gpu/display/webgpu/colormaps";

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

/** Bound mean-intensity display differences against stock float32(sum / mask area).
 * Raw resident integer sums and default single-view sum buffers stay unchanged.
 */
export async function runRansDisplayNormalizationParity(device: GPUDevice): Promise<Record<string, boolean | number>> {
  device.pushErrorScope("validation");
  const results: Record<string, boolean | number> = { max_preview_ulps: 0, max_linear_range_ulps: 0, max_log_range_ulps: 0, max_rgba_channel_error: 0 };
  const compareUlps = (label: string, actual: Float32Array, expected: Float32Array, allowed: number): number => {
    if (actual.length !== expected.length) throw new Error(`${label}: lengths differ`);
    const a = new Uint32Array(actual.buffer, actual.byteOffset, actual.length);
    const b = new Uint32Array(expected.buffer, expected.byteOffset, expected.length);
    let max = 0;
    for (let index = 0; index < a.length; index++) {
      if (!Number.isFinite(actual[index]) || !Number.isFinite(expected[index]) || actual[index] < 0 || expected[index] < 0) throw new Error(`${label}: invalid nonnegative preview`);
      const ulps = Math.abs(a[index] - b[index]);
      if (ulps > allowed) throw new Error(`${label}[${index}]: ${actual[index]} != ${expected[index]} (${ulps} ULP, allowed ${allowed})`);
      max = Math.max(max, ulps);
    }
    return max;
  };
  const readWords = async (buffer: GPUBuffer, count: number): Promise<Uint32Array> => {
    const target = device.createBuffer({ size: count * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    try {
      const encoder = device.createCommandEncoder(); encoder.copyBufferToBuffer(buffer, 0, target, 0, count * 4);
      device.queue.submit([encoder.finish()]); await target.mapAsync(GPUMapMode.READ);
      return new Uint32Array(target.getMappedRange().slice(0));
    } finally { target.destroy(); }
  };
  try {
    for (const highCounts of [false, true]) {
      const K = highCounts ? 512 : 12, tilts = 7, scans = 256;
      const data = fixture(scans, 1, K, tilts, highCounts);
      const set = await loadFixture(device, data);
      const engine = new GPUColormapEngine(device);
      engine.uploadLUT("normalization-parity-gray", Uint8Array.from({ length: 768 }, (_, index) => Math.floor(index / 3)));
      const canonical = (set as unknown as { images: GPUBuffer }).images;
      let previousMask: Uint32Array | null = null, buffers: GPUBuffer[] = [];
      try {
        // A non-power-of-two divisor and sums just beyond float32's exact-int
        // boundary distinguish division from rounded reciprocal multiplication.
        for (const area of highCounts ? [257, 511] : [4, 9]) {
          const mask = Uint32Array.from({ length: K }, (_, index) => +(index < area));
          if (!previousMask) buffers = ransMaskedSumBuffersBatch(set.computes, mask).buffers;
          else {
            const added = Uint32Array.from(mask, (value, index) => +(value && !previousMask![index]));
            const removed = Uint32Array.from(mask, (value, index) => +(!value && previousMask![index]));
            const refreshed = ransMaskedSumDeltaBuffersBatch(set.computes, added, removed, buffers).buffers;
            if (refreshed.some((buffer, index) => buffer !== buffers[index])) throw new Error("Delta replaced persistent display buffers");
          }
          const expectedCounts = data.counts.map(raw => Uint32Array.from({ length: scans }, (_, scan) => {
            let sum = 0;
            for (let k = 0; k < K; k++) if (mask[k] && !(data.manifest.bad_pixels ?? []).includes(k)) sum += raw[scan * K + k];
            return sum;
          }));
          for (let tilt = 0; tilt < tilts; tilt++) {
            compare("default sum copy", new Float32Array((await readWords(buffers[tilt], scans)).buffer), Float32Array.from(expectedCounts[tilt]));
          }
          set.normalizeDisplayBuffers(buffers, area);
          const stored = await readWords(canonical, scans * tilts);
          for (let tilt = 0; tilt < tilts; tilt++) {
            for (let scan = 0; scan < scans; scan++) if (stored[tilt * scans + scan] !== expectedCounts[tilt][scan]) throw new Error("Display normalization changed exact resident counts");
            const expected = Float32Array.from(expectedCounts[tilt], sum => Math.fround(sum) / area);
            const previewUlps = compareUlps("normalized copy", new Float32Array((await readWords(buffers[tilt], scans)).buffer), expected, 1);
            results.max_preview_ulps = Math.max(Number(results.max_preview_ulps), previewUlps);
            compare("single view remains summed", await set.readImage(tilt), Float32Array.from(expectedCounts[tilt]));
            engine.adoptBuffer(tilt * 2, buffers[tilt], 16, 16);
            engine.uploadData(tilt * 2 + 1, expected, 16, 16);
            for (const log of [false, true]) {
              engine.computeRangeRegion(tilt * 2, undefined, log);
              engine.computeRangeRegion(tilt * 2 + 1, undefined, log);
              const display = engine as unknown as {
                slots: { rangeBuffer: GPUBuffer; dataBuffer: GPUBuffer }[];
                lutBuffer: GPUBuffer; colormapRangePipeline: GPUComputePipeline;
                ensureColormapRangePipeline(): void;
              };
              const actualRange = await readWords(display.slots[tilt * 2].rangeBuffer, 4);
              const expectedRange = await readWords(display.slots[tilt * 2 + 1].rangeBuffer, 4);
              const rangeUlps = compareUlps(`${log ? "log" : "linear"} range`, new Float32Array(actualRange.buffer, 0, 2), new Float32Array(expectedRange.buffer, 0, 2), log ? 2 : 1);
              const key = log ? "max_log_range_ulps" : "max_linear_range_ulps";
              results[key] = Math.max(Number(results[key]), rangeUlps);
              // Use the stock range-driven colormap pipeline for both inputs.
              // Grayscale makes a one-bin quantization shift exactly one 8-bit
              // channel level, avoiding a palette's unrelated adjacent-bin jumps.
              display.ensureColormapRangePipeline();
              const params = [0, 1].map(() => {
                const buffer = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM, mappedAtCreation: true });
                const u32 = new Uint32Array(buffer.getMappedRange());
                u32.set([16, 16, 0, 0, +log, 0, 0, 16]);
                const f32 = new Float32Array(u32.buffer); f32[2] = 0; f32[3] = 100;
                buffer.unmap(); return buffer;
              });
              const rgba = [0, 1].map(() => device.createBuffer({ size: scans * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC }));
              try {
                const encoder = device.createCommandEncoder(); const pass = encoder.beginComputePass();
                const pipeline = display.colormapRangePipeline; pass.setPipeline(pipeline);
                for (let index = 0; index < 2; index++) {
                  const slot = display.slots[tilt * 2 + index];
                  pass.setBindGroup(0, device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
                    { binding: 0, resource: { buffer: params[index] } }, { binding: 1, resource: { buffer: slot.dataBuffer } },
                    { binding: 2, resource: { buffer: display.lutBuffer } }, { binding: 3, resource: { buffer: rgba[index] } },
                    { binding: 4, resource: { buffer: slot.rangeBuffer } },
                  ] }));
                  pass.dispatchWorkgroups(1, 1);
                }
                pass.end(); device.queue.submit([encoder.finish()]);
                const actualRgba = new Uint8Array((await readWords(rgba[0], scans)).buffer);
                const referenceRgba = new Uint8Array((await readWords(rgba[1], scans)).buffer);
                for (let channel = 0; channel < actualRgba.length; channel++) {
                  const error = Math.abs(actualRgba[channel] - referenceRgba[channel]);
                  if (error > 1) throw new Error(`Normalized ${log ? "log" : "linear"} RGBA channel ${channel} differs by ${error}/255`);
                  results.max_rgba_channel_error = Math.max(Number(results.max_rgba_channel_error), error);
                }
              } finally { params.forEach(buffer => buffer.destroy()); rgba.forEach(buffer => buffer.destroy()); }
            }
          }
          previousMask = mask;
          results[`${highCounts ? "high_counts" : "mixed"}_area_${area}_within_display_bounds`] = true;
        }
        for (const area of [-1, 0.5, NaN, Infinity, K + 1]) {
          let rejected = false; try { set.normalizeDisplayBuffers(buffers, area); } catch { rejected = true; }
          if (!rejected) throw new Error("Invalid mask area accepted");
        }
        let rejected = false; try { set.normalizeDisplayBuffers([canonical], 2); } catch { rejected = true; }
        if (!rejected) throw new Error("Canonical source accepted as a display copy");
      } finally { engine.destroy(); set.dispose(); }
    }
    results.canonical_integer_sums_exact = true;
    results.default_sum_buffers_exact = true;
    results.all_passed = true;
    return results;
  } finally {
    const validation = await device.popErrorScope();
    if (validation) throw new Error(validation.message);
  }
}
