import { WebGPUCompactH5ResidentSource } from "../../../src/quantem/gpu/io/backends/webgpu/compact-h5";

/** Compare a moving detector against exact counts and an independent rebase. */
export async function verifyDetectorTrajectory(
  resident: WebGPUCompactH5ResidentSource,
  masks: Uint32Array[],
  valueAt: (scan: number, detector: number) => number,
) {
  const excluded = new Set(resident.metadata.excludedDetectorPixels);
  const results = [];
  for (const mask of masks) {
    const metrics = await resident.updateVirtualDetector(mask);
    const observed = await resident.virtualDetectorValues();
    const expected = new Uint32Array(observed.length);
    for (let scan = 0; scan < expected.length; scan++) {
      let sum = 0;
      for (let pixel = 0; pixel < mask.length; pixel++) {
        if (mask[pixel] && !excluded.has(pixel)) sum += valueAt(scan, pixel);
      }
      if (sum > 0xffffffff) throw new Error("Oracle exceeds the admitted u32 sum contract");
      expected[scan] = sum;
    }
    let mismatches = 0;
    for (let scan = 0; scan < expected.length; scan++) {
      if (observed[scan] !== expected[scan]) mismatches++;
    }
    await resident.rebaseVirtualDetector(mask);
    const rebased = await resident.virtualDetectorValues();
    let rebaseMismatches = 0;
    for (let scan = 0; scan < expected.length; scan++) {
      if (rebased[scan] !== observed[scan]) rebaseMismatches++;
    }
    results.push({ mode: metrics.mode, changed: metrics.changedDetectorPixels,
      mismatches, rebaseMismatches, scans: expected.length });
  }
  return results;
}

/** Compare rapidly queued shared-buffer updates against the final exact mask. */
export async function verifyQueuedDetectorBatches(
  residents: WebGPUCompactH5ResidentSource[],
  masks: Uint32Array[],
  valueAt: (scan: number, detector: number) => number,
) {
  for (const mask of masks) {
    WebGPUCompactH5ResidentSource.maskedSumDisplayBuffersBatch(residents, mask);
  }
  await Promise.all(residents.map(resident => resident.quiesce()));
  const mask = masks[masks.length - 1];
  return Promise.all(residents.map(async resident => {
    const observed = await resident.virtualDetectorValues();
    const excluded = new Set(resident.metadata.excludedDetectorPixels);
    let mismatches = 0;
    for (let scan = 0; scan < observed.length; scan++) {
      let expected = 0;
      for (let pixel = 0; pixel < mask.length; pixel++) {
        if (mask[pixel] && !excluded.has(pixel)) expected += valueAt(scan, pixel);
      }
      if (observed[scan] !== expected) mismatches++;
    }
    return { scans: observed.length, mismatches, ownedBufferBytes: resident.ownedBufferBytes };
  }));
}
