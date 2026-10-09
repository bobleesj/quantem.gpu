/// <reference types="@webgpu/types" />
import { RansResidentSet } from "../../src/quantem/gpu/detector/webgpu/rans";
import { DetectorCompute } from "../../src/quantem/gpu/detector/webgpu/backend";
import { RansResidentSeries } from "../../src/quantem/gpu/detector/webgpu/rans-series";

/** Copy a lent count view and compare it with the exact image of the same acquisition. */
async function borrowedViewExact(set: RansResidentSet, tilt: number): Promise<void> {
  const [view] = set.imageViewsU32([tilt], 1);
  const readback = set.device.createBuffer({ size: view.count * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const encoder = set.device.createCommandEncoder();
  encoder.copyBufferToBuffer(view.buffer, view.byteOffset, readback, 0, view.count * 4);
  set.device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const borrowed = new Uint32Array(readback.getMappedRange().slice(0));
  readback.destroy();
  const exact = await set.readImageU32(tilt);
  if (borrowed.some((value, scan) => value !== exact[scan])) throw new Error(`Borrowed count view of acquisition ${tilt} differs from its exact image`);
}

/** Ordered public-encoder files must drive one exact batched resident series. */
export async function runQemSeriesParity(device: GPUDevice, baseURL: string): Promise<Record<string, boolean>> {
  const base = baseURL.endsWith("/") ? baseURL : baseURL + "/";
  const read = async (name: string) => (await fetch(base + name)).arrayBuffer();
  const file = async (name: string) => new File([await read(name + ".qem")], name + ".qem");
  // Deliberately reverse fixture order. The caller owns acquisition ordering.
  const names = ["uint16-series-second", "uint16-qem-modes"];
  const files = await Promise.all(names.map(file));
  const originals = await Promise.all(names.map(async name => new Uint16Array(await read(name + ".bin"))));
  device.pushErrorScope("validation");
  const results: Record<string, boolean> = {};
  try {
    const source = await RansResidentSet.loadQemFiles(device, files);
    let buffers: GPUBuffer[] = [];
    try {
      if (source.computes.length !== 2 || source.T !== 2 || source.computes.some(compute => compute.set !== source)) throw new Error("Not all acquisitions share one resident set");
      const metadata = source.sourceMetadata.acquisitions as { file: string }[];
      if (metadata.map(item => item.file).join() !== files.map(item => item.name).join()) throw new Error("Acquisition metadata order changed");
      const scans = source.scanCount, K = source.K;
      for (let tilt = 0; tilt < 2; tilt++) {
        for (const scan of [0, 255, 256, 511, 512, scans - 1]) {
          const actual = await source.computes[tilt].frameAt(scan);
          for (let k = 0; k < K; k++) if (actual[k] !== originals[tilt][scan * K + k]) throw new Error(`Acquisition order/pattern mismatch ${tilt}:${scan}:${k}`);
        }
      }
      const computes = source.computes as unknown as DetectorCompute[];
      const mask = new Uint32Array([1, 1, 0, 0, 1, 0]);
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
      for (const next of [new Uint32Array([0, 1, 1, 0, 0, 1]), new Uint32Array([1, 0, 1, 1, 1, 0])]) {
        const added = new Uint32Array(K), removed = new Uint32Array(K);
        for (let k = 0; k < K; k++) { added[k] = next[k] && !mask[k] ? 1 : 0; removed[k] = mask[k] && !next[k] ? 1 : 0; }
        const previous = buffers;
        buffers = DetectorCompute.maskedSumDeltaBuffersBatch(computes, buffers, added, removed).buffers;
        mask.set(next); await verify();
        for (const buffer of previous) if (!buffers.includes(buffer)) buffer.destroy();
      }
      // Borrowed views address the same canonical counts readImageU32 snapshots;
      // the second acquisition of this 561-scan set starts off a binding boundary.
      await borrowedViewExact(source, 0);
      let refused = false;
      try { source.imageViewsU32([1], 1); } catch (error) { refused = String(error).includes("cannot be lent"); }
      if (!refused) throw new Error("An unbindable count view was lent");
      results.ordered_patterns_exact = true; results.single_set_batch_delta_exact = true; results.borrowed_views_exact = true;
    } finally { buffers.forEach(buffer => buffer.destroy()); source.dispose(); }
    // One resident set per acquisition: the compare-grid batch must integrate each set's own image.
    const series = await RansResidentSeries.load(device, files, () => {}, () => {}, new AbortController().signal);
    let seriesBuffers: GPUBuffer[] = [];
    try {
      await series.completion;
      if (series.loadedAcquisitions !== 2 || series.computes[0].set === series.computes[1].set) throw new Error("Series acquisitions were not loaded separately");
      const computes = series.computes as unknown as DetectorCompute[];
      const scans = series.computes[0].scanCount, K = series.computes[0].detSize;
      const exact = async (mask: Uint32Array) => {
        for (let tilt = 0; tilt < 2; tilt++) {
          const readback = device.createBuffer({ size: scans * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
          const encoder = device.createCommandEncoder();
          encoder.copyBufferToBuffer(seriesBuffers[tilt], 0, readback, 0, scans * 4);
          device.queue.submit([encoder.finish()]);
          await readback.mapAsync(GPUMapMode.READ);
          const actual = new Float32Array(readback.getMappedRange().slice(0));
          readback.destroy();
          for (let scan = 0; scan < scans; scan++) {
            let expected = 0;
            for (let k = 0; k < K; k++) if (mask[k]) expected += originals[tilt][scan * K + k];
            if (actual[scan] !== expected) throw new Error(`Series panel ${tilt} differs at scan ${scan}`);
          }
        }
      };
      const mask = new Uint32Array([1, 1, 0, 0, 1, 0]);
      seriesBuffers = DetectorCompute.maskedSumBuffersBatch(computes, mask).buffers;
      await exact(mask);
      const next = new Uint32Array([0, 1, 1, 0, 0, 1]);
      const added = next.map((value, k) => value && !mask[k] ? 1 : 0), removed = next.map((value, k) => mask[k] && !value ? 1 : 0);
      seriesBuffers = DetectorCompute.maskedSumDeltaBuffersBatch(computes, seriesBuffers, added, removed).buffers;
      await exact(next);
      for (let tilt = 0; tilt < 2; tilt++) await borrowedViewExact(series.computes[tilt].set, 0);
      results.per_acquisition_series_batch_exact = true;
    } finally { seriesBuffers.forEach(buffer => buffer.destroy()); series.dispose(); }
    const maskedSource = await RansResidentSet.loadQemFiles(device, files, () => {}, [1, 1]);
    try {
      if (maskedSource.badPx.length !== 1 || maskedSource.badPx[0] !== 1) throw new Error("Bad-pixel indices were not deduplicated");
      for (let tilt = 0; tilt < 2; tilt++) {
        if ((await maskedSource.pattern(tilt, 0))[1] !== 65535) throw new Error("Raw sentinel count was changed");
        if ((await maskedSource.computes[tilt].frameAt(0))[1] !== 0) throw new Error("Bad pixel was not excluded in every acquisition");
        const image = await maskedSource.computes[tilt].maskedSum(new Uint32Array(6).fill(1));
        for (let scan = 0; scan < maskedSource.scanCount; scan++) {
          let expected = 0;
          for (let k = 0; k < 6; k++) if (k !== 1) expected += originals[tilt][scan * 6 + k];
          if (image[scan] !== expected) throw new Error("Bad-pixel mask did not reach every acquisition product");
        }
      }
      results.global_bad_pixels_exact = true;
    } finally { maskedSource.dispose(); }
    const savedMask = await RansResidentSet.loadQemFile(device, await file("validity-mismatch"));
    try {
      if (savedMask.badPx.length !== 1 || savedMask.badPx[0] !== 1) throw new Error("Saved QEM validity mask was lost");
      if ((await savedMask.pattern(0, 0))[1] !== 65535) throw new Error("Saved mask changed original counts");
      if ((await savedMask.computes[0].frameAt(0))[1] !== 0) throw new Error("Saved mask did not reach displayed DP");
      if (!(savedMask.sourceMetadata.scientific_metadata as { schema?: string })?.schema) throw new Error("Scientific metadata was lost");
      results.saved_validity_and_metadata_preserved = true;
    } finally { savedMask.dispose(); }
    // Verify that admission rejects mismatches before creating any GPU buffer.
    const guardedDevice = new Proxy(device, { get(target, key) {
      if (key === "createBuffer") return () => { throw new Error("Mismatch reached GPU upload"); };
      const value = Reflect.get(target, key, target);
      return typeof value === "function" ? value.bind(target) : value;
    } });
    for (const [name, reason] of [["shape-mismatch", "matching native geometry and dtype"], ["uint8-qem-modes", "matching native geometry and dtype"], ["validity-mismatch", "validity masks differ"]]) {
      let rejected = false;
      try { await RansResidentSet.loadQemFiles(guardedDevice, [files[0], await file(name)]); }
      catch (error) { rejected = String(error).includes(reason); }
      if (!rejected) throw new Error(`Series mismatch ${name} was not rejected before upload`);
    }
    let badPixelsRejected = false;
    try { await RansResidentSet.loadQemFiles(guardedDevice, files, () => {}, [6]); }
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
