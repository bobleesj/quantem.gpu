/// <reference types="@webgpu/types" />
import { RansResidentSet } from "../../src/quantem/gpu/detector/compute/webgpu/rans";
import { DetectorCompute } from "../../src/quantem/gpu/detector/compute/webgpu/backend";

/** Ordered public-encoder files must drive one exact batched resident series. */
export async function runCountANSSeriesParity(device: GPUDevice, baseURL: string): Promise<Record<string, boolean>> {
  const base = baseURL.endsWith("/") ? baseURL : baseURL + "/";
  const read = async (name: string) => (await fetch(base + name)).arrayBuffer();
  const file = async (name: string) => new File([await read(name + ".ans")], name + ".ans");
  // Deliberately reverse fixture order. The caller owns acquisition ordering.
  const names = ["uint16-series-second", "uint16-model512-tail"];
  const files = await Promise.all(names.map(file));
  const originals = await Promise.all(names.map(async name => new Uint16Array(await read(name + ".bin"))));
  device.pushErrorScope("validation");
  const results: Record<string, boolean> = {};
  try {
    const source = await RansResidentSet.loadCountANSFiles(device, files);
    let buffers: GPUBuffer[] = [];
    try {
      if (source.computes.length !== 2 || source.T !== 2 || source.computes.some(compute => compute.set !== source)) throw new Error("Not all acquisitions share one resident set");
      const metadata = source.sourceMetadata.acquisitions as { file: string }[];
      if (metadata.map(item => item.file).join() !== files.map(item => item.name).join()) throw new Error("Acquisition metadata order changed");
      const scans = source.scanCount, K = source.K;
      for (let tilt = 0; tilt < 2; tilt++) {
        for (const scan of [0, 511, 4095, 4096, scans - 1]) {
          const actual = await source.computes[tilt].frameAt(scan);
          for (let k = 0; k < K; k++) if (actual[k] !== originals[tilt][scan * K + k]) throw new Error(`Acquisition order/pattern mismatch ${tilt}:${scan}:${k}`);
        }
      }
      const computes = source.computes as unknown as DetectorCompute[];
      const mask = new Uint32Array([1, 1, 0, 0]);
      buffers = DetectorCompute.maskedSumBuffersBatch(computes, mask).buffers;
      const verify = async () => {
        const readback = device.createBuffer({ size: scans * 2 * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        const encoder = device.createCommandEncoder();
        for (let tilt = 0; tilt < 2; tilt++) encoder.copyBufferToBuffer(buffers[tilt], 0, readback, tilt * scans * 4, scans * 4);
        device.queue.submit([encoder.finish()]);
        try {
          await readback.mapAsync(GPUMapMode.READ);
          const actual = new Float32Array(readback.getMappedRange());
          for (let tilt = 0; tilt < 2; tilt++) for (let scan = 0; scan < scans; scan++) {
            let expected = 0;
            for (let k = 0; k < K; k++) if (mask[k]) expected += originals[tilt][scan * K + k];
            if (actual[tilt * scans + scan] !== expected) throw new Error(`Batched mask differs ${tilt}:${scan}`);
          }
        } finally { readback.destroy(); }
      };
      await verify();
      for (const next of [new Uint32Array([0, 1, 1, 0]), new Uint32Array([1, 0, 1, 1])]) {
        const added = new Uint32Array(K), removed = new Uint32Array(K);
        for (let k = 0; k < K; k++) { added[k] = next[k] && !mask[k] ? 1 : 0; removed[k] = mask[k] && !next[k] ? 1 : 0; }
        const previous = buffers;
        buffers = DetectorCompute.maskedSumDeltaBuffersBatch(computes, buffers, added, removed).buffers;
        mask.set(next); await verify();
        for (const buffer of previous) if (!buffers.includes(buffer)) buffer.destroy();
      }
      results.ordered_patterns_exact = true; results.single_set_batch_delta_exact = true;
    } finally { buffers.forEach(buffer => buffer.destroy()); source.dispose(); }
    const maskedSource = await RansResidentSet.loadCountANSFiles(device, files, () => {}, [2, 2]);
    try {
      if (maskedSource.badPx.length !== 1 || maskedSource.badPx[0] !== 2) throw new Error("Bad-pixel indices were not deduplicated");
      for (let tilt = 0; tilt < 2; tilt++) {
        if ((await maskedSource.pattern(tilt, 0))[2] !== 65535) throw new Error("Raw sentinel count was changed");
        if ((await maskedSource.computes[tilt].frameAt(0))[2] !== 0) throw new Error("Bad pixel was not excluded in every acquisition");
        const image = await maskedSource.computes[tilt].maskedSum(new Uint32Array(4).fill(1));
        for (let scan = 0; scan < maskedSource.scanCount; scan++) {
          const expected = originals[tilt][scan * 4] + originals[tilt][scan * 4 + 1] + originals[tilt][scan * 4 + 3];
          if (image[scan] !== expected) throw new Error("Bad-pixel mask did not reach every acquisition product");
        }
      }
      results.global_bad_pixels_exact = true;
    } finally { maskedSource.dispose(); }
    // Verify that admission rejects mismatches before creating any GPU buffer.
    const guardedDevice = new Proxy(device, { get(target, key) {
      if (key === "createBuffer") return () => { throw new Error("Mismatch reached GPU upload"); };
      const value = Reflect.get(target, key, target);
      return typeof value === "function" ? value.bind(target) : value;
    } });
    for (const [name, reason] of [["uint8-block17-tail", "matching native geometry and dtype"], ["uint8-dtype-mismatch", "matching native geometry and dtype"], ["uint16-profile-mismatch", "same block_frames and scale"]]) {
      let rejected = false;
      try { await RansResidentSet.loadCountANSFiles(guardedDevice, [files[0], await file(name)]); }
      catch (error) { rejected = String(error).includes(reason); }
      if (!rejected) throw new Error(`Series mismatch ${name} was not rejected before upload`);
    }
    let badPixelsRejected = false;
    try { await RansResidentSet.loadCountANSFiles(guardedDevice, files, () => {}, [4]); }
    catch (error) { badPixelsRejected = String(error).includes("badPixels must contain detector indices"); }
    if (!badPixelsRejected) throw new Error("Out-of-range bad pixel reached GPU upload");
    results.mismatches_rejected_before_upload = true;
    results.all_passed = true;
    return results;
  } finally {
    const validation = await device.popErrorScope();
    if (validation) throw new Error(validation.message);
  }
}
