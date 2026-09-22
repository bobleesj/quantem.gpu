/// <reference types="@webgpu/types" />
import { RansResidentSet } from "../../src/quantem/gpu/detector/compute/webgpu/rans";
import { countAnsFileSource } from "../../src/quantem/gpu/detector/backends/webgpu/count-ans";

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
      const file = new File([await (await fetch(base + item.name + ".qem")).arrayBuffer()], item.name + ".qem");
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
      const bytes = new Uint8Array(await file.arrayBuffer()); bytes[bytes.length - 1] ^= 1;
      let rejected = false;
      try { await RansResidentSet.loadCountANS(device, new File([bytes], "corrupt.qem")); }
      catch (error) { rejected = String(error).includes("payload checksum mismatch"); }
      if (!rejected) throw new Error("Modified payload was not rejected by checksum admission");
    }
    results.payload_corruption_rejected = true;
    const floating = new File([await (await fetch(base + "float.qem")).arrayBuffer()], "float.qem");
    let floatRejected = false;
    try { await RansResidentSet.loadCountANS(device, floating); }
    catch (error) { floatRejected = String(error).includes("integer QEM only"); }
    if (!floatRejected) throw new Error("Unsupported float QEM was not rejected with a corrective message");
    results.float_profile_rejected = true;
    const original = new Uint8Array(await (await fetch(base + cases[0].name + ".qem")).arrayBuffer());
    const originalBody = Number(new DataView(original.buffer).getBigUint64(16, true));
    const headerText = new TextDecoder().decode(original.subarray(56, originalBody));
    for (const [field, reason] of [
      ["axes", "axes disagree"],
      ["bounds", "chunk array bounds"],
      ["calibration", "retired x/y"],
    ]) {
      const header = JSON.parse(headerText);
      if (field === "axes") header.scientific_metadata.axes[0].name = "x";
      if (field === "bounds") header.chunks[0].arrays[0].offset = 7;
      if (field === "calibration") header.scientific_metadata.calibration_overrides["imaging_system/reciprocal_pixel_size_x"] = {value: 1, unit: "mrad"};
      const encoded = new TextEncoder().encode(JSON.stringify(header));
      const changed = new Uint8Array(56 + encoded.length + original.length - originalBody);
      changed.set(new TextEncoder().encode("QEMDATA1"));
      const prefix = new DataView(changed.buffer);
      prefix.setBigUint64(8, BigInt(encoded.length), true);
      prefix.setBigUint64(16, BigInt(56 + encoded.length), true);
      changed.set(new Uint8Array(await crypto.subtle.digest("SHA-256", encoded)), 24);
      changed.set(encoded, 56);
      changed.set(original.subarray(originalBody), 56 + encoded.length);
      let rejected = false;
      try { await countAnsFileSource(new File([changed], "invalid.qem")); }
      catch (error) { rejected = String(error).includes(reason); }
      if (!rejected) throw new Error(`Authenticated invalid ${field} was not rejected before GPU upload`);
    }
    results.semantic_metadata_and_bounds_rejected = true;
    results.all_passed = true;
    return results;
  } finally {
    const validation = await device.popErrorScope();
    if (validation) throw new Error(validation.message);
  }
}
