/// <reference types="@webgpu/types" />
import { RansResidentSet } from "../../src/quantem/gpu/detector/compute/webgpu/rans";

type Case = { name: string; shape: number[]; dtype: "uint8" | "uint16"; block_frames: number };
function exact(label: string, actual: Float32Array, expected: Float32Array): void {
  if (actual.length !== expected.length) throw new Error(`${label}: dimensions differ`);
  for (let index = 0; index < actual.length; index++) {
    if (actual[index] !== expected[index]) throw new Error(`${label}[${index}]: ${actual[index]} != ${expected[index]}`);
  }
}

/** Decode public io.save output and compare native patterns/products exactly. */
export async function runCountANSBrowserParity(device: GPUDevice, baseURL: string): Promise<Record<string, boolean>> {
  const base = baseURL.endsWith("/") ? baseURL : baseURL + "/";
  const cases = await (await fetch(base + "cases.json")).json() as Case[];
  const results: Record<string, boolean> = {};
  device.pushErrorScope("validation");
  try {
    for (const item of cases) {
      const file = new File([await (await fetch(base + item.name + ".ans")).arrayBuffer()], item.name + ".ans");
      const rawBytes = await (await fetch(base + item.name + ".bin")).arrayBuffer();
      const counts = item.dtype === "uint8" ? new Uint8Array(rawBytes) : new Uint16Array(rawBytes);
      const scans = item.shape[0] * item.shape[1], K = item.shape[2] * item.shape[3];
      const source = await RansResidentSet.loadCountANS(device, file);
      try {
        if (JSON.stringify(source.shape) !== JSON.stringify(item.shape) || source.scanCount !== scans || source.nativeDtype !== item.dtype) throw new Error("Source geometry/dtype changed");
        const compute = source.computes[0];
        const positions = [...new Set([0, 16, 17, 255, 256, 511, 512, item.block_frames - 1, item.block_frames, scans - 1].filter(index => index < scans))];
        for (const scan of positions) exact(`${item.name} pattern ${scan}`, await compute.frameAt(scan), Float32Array.from(counts.subarray(scan * K, (scan + 1) * K)));
        const masks = [new Uint32Array(K).fill(1), Uint32Array.from({ length: K }, (_, k) => k % 2), new Uint32Array(K), Uint32Array.from({ length: K }, (_, k) => k === 0 ? 1 : 0)];
        for (const mask of masks) {
          const expected = new Float32Array(scans);
          for (let scan = 0; scan < scans; scan++) for (let k = 0; k < K; k++) if (mask[k]) expected[scan] += counts[scan * K + k];
          exact(`${item.name} full/delta mask`, await compute.maskedSum(mask), expected);
        }
        const scanMask = Uint32Array.from({ length: scans }, (_, scan) => scan % 3 === 0 ? 1 : 0);
        const selected = scanMask.reduce((sum, value) => sum + value, 0);
        const sums = new Float64Array(K);
        for (let scan = 0; scan < scans; scan++) if (scanMask[scan]) for (let k = 0; k < K; k++) sums[k] += counts[scan * K + k];
        exact(`${item.name} ROI sum`, await compute.reduceFrames(scanMask, false), Float32Array.from(sums));
        exact(`${item.name} ROI mean`, await compute.reduceFrames(scanMask, true), Float32Array.from(sums, value => value / selected));
        results[item.name] = true;
      } finally { source.dispose(); }
      // The package checksum must fail before corrupted payloads reach a decoder.
      const bytes = new Uint8Array(await file.arrayBuffer()); bytes[65536] ^= 1;
      let rejected = false;
      try { await RansResidentSet.loadCountANS(device, new File([bytes], "corrupt.ans")); }
      catch (error) { rejected = String(error).includes("payload checksum mismatch"); }
      if (!rejected) throw new Error("Modified payload was not rejected by checksum admission");
    }
    results.payload_corruption_rejected = true;
    results.all_passed = true;
    return results;
  } finally {
    const validation = await device.popErrorScope();
    if (validation) throw new Error(validation.message);
  }
}
