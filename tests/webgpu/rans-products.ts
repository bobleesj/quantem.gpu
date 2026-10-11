/// <reference types="@webgpu/types" />
/** Browser parity workflow. Bundle with esbuild, then call runRansProductParity(device, fixturesURL).
 * The caller supplies an authenticated hardware device; no adapter is selected here. The .qem
 * fixtures come from make_qem_browser_fixture.generate.
 */
import { RansResidentSet } from "../../src/quantem/gpu/detector/webgpu/rans";
import { DetectorCompute } from "../../src/quantem/gpu/detector/webgpu/backend";

async function loadQem(device: GPUDevice, base: string, name: string): Promise<RansResidentSet> {
  const file = new File([await (await fetch(base + name)).arrayBuffer()], name);
  return RansResidentSet.loadQemFile(device, file);
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
async function runProducts(device: GPUDevice, base: string): Promise<Record<string, number | boolean>> {
  const results: Record<string, number | boolean> = {};
  for (let tilt = 0; tilt < 2; tilt++) {
    const set = await loadQem(device, base, `products-${tilt}.qem`);
    try {
      const compute = set.computes[0];
      const raw = new Uint16Array(await (await fetch(base + `products-${tilt}.bin`)).arrayBuffer());
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
        results[`com_row_${tilt}`] = compare("CoM row", com.comY, baseline.comY);
        results[`com_col_${tilt}`] = compare("CoM col", com.comX, baseline.comX);
        for (const component of ["row", "col"] as const) {
          results[`dpc_${component}_${tilt}`] = compare(`DPC ${component}`, await compute.maskedDpc(mask, 4, component), await dense.maskedDpc(mask, 4, component));
        }
        for (const [rotation, transpose] of [[0, false], [37, true]] as const) {
          results[`idpc_${tilt}_${rotation}`] = compare("iDPC", await compute.maskedIDpc(mask, 4, 32, 32, rotation, transpose), await dense.maskedIDpc(mask, 4, 32, 32, rotation, transpose));
        }
        const singlePixel = new Uint32Array(12); singlePixel[5] = 1;
        const point = await compute.maskedCoM(singlePixel, 4);
        compare("single-pixel CoM row", point.comY, Float32Array.from(raw.filter((_, i) => i % 12 === 5), (value) => value ? 1 : 0));
        compare("single-pixel CoM col", point.comX, Float32Array.from(raw.filter((_, i) => i % 12 === 5), (value) => value ? 1 : 0));
        const empty = new Uint32Array(12);
        compare("empty iDPC", await compute.maskedIDpc(empty, 4, 32, 32), new Float32Array(1024));
      } finally { dense.dispose(); }
    } finally { set.dispose(); }
  }
  // Full ROI crosses dispatch limits and 32-bit sums: exact sum then divide.
  const large = await loadQem(device, base, "saturated-roi.qem");
  try {
    const mask = new Uint32Array(131072).fill(1);
    compare("wide ROI sum", await large.computes[0].reduceFrames(mask, false), new Float32Array(4).fill(Math.fround(131072 * 65535)));
    compare("wide ROI mean", await large.computes[0].reduceFrames(mask), new Float32Array(4).fill(65535));
    results.large_roi_exact = true;
  } finally { large.dispose(); }
  // Weighted column moments exceed 32 bits although every count is uint16.
  const wide = await loadQem(device, base, "saturated-columns.qem");
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
export async function runRansProductParity(device: GPUDevice, baseURL: string): Promise<Record<string, number | boolean>> {
  const base = baseURL.endsWith("/") ? baseURL : baseURL + "/";
  device.pushErrorScope("validation");
  try { return await runProducts(device, base); }
  finally {
    const validation = await device.popErrorScope();
    if (validation) throw new Error(validation.message);
  }
}
