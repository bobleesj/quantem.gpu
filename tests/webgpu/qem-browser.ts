/// <reference types="@webgpu/types" />
import { RansResidentSet } from "../../src/quantem/gpu/detector/webgpu/rans";
import { qemFileSource } from "../../src/quantem/gpu/detector/webgpu/qem-source";

type Case = { name: string; shape: number[]; dtype: "uint8" | "uint16"; block_frames: number };
function exact(label: string, actual: Float32Array, expected: Float32Array): void {
  if (actual.length !== expected.length) throw new Error(`${label}: dimensions differ`);
  for (let index = 0; index < actual.length; index++) {
    if (actual[index] !== expected[index]) throw new Error(`${label}[${index}]: ${actual[index]} != ${expected[index]}`);
  }
}

/** Decode public io.save output and compare native patterns/products exactly. */
export async function runQemBrowserParity(device: GPUDevice, baseURL: string): Promise<Record<string, boolean>> {
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
      const source = await RansResidentSet.loadQemFile(device, file);
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
        // SSB reads BF columns as complex64, including a reordered subset.
        const columns = Uint32Array.from([K - 1, 0, 2]);
        const output = device.createBuffer({size: columns.length * scans * 8,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC});
        const readback = device.createBuffer({size: output.size,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});
        try {
          await source.columnsComplex(0, columns, output);
          const encoder = device.createCommandEncoder();
          encoder.copyBufferToBuffer(output, 0, readback, 0, output.size);
          device.queue.submit([encoder.finish()]);
          await readback.mapAsync(GPUMapMode.READ);
          const actual = new Float32Array(readback.getMappedRange());
          for (let c = 0; c < columns.length; c++) for (let scan = 0; scan < scans; scan++) {
            if (actual[(c * scans + scan) * 2] !== counts[scan * K + columns[c]]
              || actual[(c * scans + scan) * 2 + 1] !== 0) throw new Error("SSB complex BF columns changed counts/order");
          }
        } finally { output.destroy(); readback.destroy(); }
        results[item.name] = true;
      } finally { source.dispose(); }
      // The package checksum must fail before corrupted payloads reach a decoder.
      const bytes = new Uint8Array(await file.arrayBuffer()); bytes[bytes.length - 1] ^= 1;
      let rejected = false;
      try { await RansResidentSet.loadQemFile(device, new File([bytes], "corrupt.qem")); }
      catch (error) { rejected = String(error).includes("payload checksum mismatch"); }
      if (!rejected) throw new Error("Modified payload was not rejected by checksum admission");
    }
    results.payload_corruption_rejected = true;
    // Groups smaller than a chunk on the real device: the split staged payload and
    // the GPU copy of the block a split cuts must decode exactly.
    for (const item of cases.filter(item => item.name.endsWith("-qem-blocks"))) {
      const bytes = new Uint8Array(await (await fetch(base + item.name + ".qem")).arrayBuffer());
      const body = 56 + Number(new DataView(bytes.buffer).getBigUint64(8, true));
      const header = JSON.parse(new TextDecoder().decode(bytes.subarray(56, body)));
      const K = item.shape[2] * item.shape[3], scans = item.shape[0] * item.shape[1];
      let largestBlock = 0, largestChunk = 0;
      for (const chunk of header.chunks) {
        const start = body + chunk.arrays[1].offset;
        const table = new Uint32Array(bytes.slice(start, start + chunk.arrays[1].count * 4).buffer);
        for (let block = 0; (block + 1) * K < table.length; block++) largestBlock = Math.max(largestBlock, table[(block + 1) * K] - table[block * K]);
        largestChunk = Math.max(largestChunk, chunk.arrays[0].count);
      }
      // One word above the largest block: a split then falls inside a block, never on its start.
      const limit = Math.ceil(largestBlock / 4) * 4 + 4;
      if (limit >= largestChunk) throw new Error(`${item.name} has no chunk larger than its group limit`);
      const limits = { maxStorageBufferBindingSize: limit, maxBufferSize: limit,
        maxComputeWorkgroupsPerDimension: device.limits.maxComputeWorkgroupsPerDimension,
        minStorageBufferOffsetAlignment: device.limits.minStorageBufferOffsetAlignment };
      // Count copies out of staged groups (mapped at creation, copy sources) to prove a block was cut.
      const staged = new Set<GPUBuffer>();
      let copiedBlocks = 0;
      const smallGroups = new Proxy(device, { get(target, key) {
        if (key === "limits") return limits;
        if (key === "createBuffer") return (descriptor: GPUBufferDescriptor) => {
          const buffer = target.createBuffer(descriptor);
          if (descriptor.mappedAtCreation && descriptor.usage & GPUBufferUsage.COPY_SRC) staged.add(buffer);
          return buffer;
        };
        if (key === "createCommandEncoder") return (descriptor?: GPUCommandEncoderDescriptor) => {
          const encoder = target.createCommandEncoder(descriptor);
          const copy = encoder.copyBufferToBuffer.bind(encoder) as (...args: unknown[]) => void;
          (encoder as unknown as { copyBufferToBuffer: (...args: unknown[]) => void }).copyBufferToBuffer = (source, ...rest) => {
            if (staged.has(source as GPUBuffer)) copiedBlocks++;
            copy(source, ...rest);
          };
          return encoder;
        };
        const value = Reflect.get(target, key, target);
        return typeof value === "function" ? value.bind(target) : value;
      } });
      const rawBytes = await (await fetch(base + item.name + ".bin")).arrayBuffer();
      const counts = item.dtype === "uint8" ? new Uint8Array(rawBytes) : new Uint16Array(rawBytes);
      const source = await RansResidentSet.loadQemFile(smallGroups, new File([bytes], item.name + ".qem"));
      try {
        if (!copiedBlocks) throw new Error(`${item.name}: no block was cut and copied at a ${limit}-byte group limit`);
        for (const scan of [0, 255, 256, 511, 512, scans - 1]) exact(`${item.name} split pattern ${scan}`, await source.computes[0].frameAt(scan), Float32Array.from(counts.subarray(scan * K, (scan + 1) * K)));
        for (const mask of [new Uint32Array(K).fill(1), Uint32Array.from({ length: K }, (_, k) => k % 2)]) {
          const expected = new Float32Array(scans);
          for (let scan = 0; scan < scans; scan++) for (let k = 0; k < K; k++) if (mask[k]) expected[scan] += counts[scan * K + k];
          exact(`${item.name} split mask`, await source.computes[0].maskedSum(mask), expected);
        }
      } finally { source.dispose(); }
    }
    results.split_groups_exact = true;
    const floating = new File([await (await fetch(base + "float.qem")).arrayBuffer()], "float.qem");
    let floatRejected = false;
    try { await RansResidentSet.loadQemFile(device, floating); }
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
      try { await qemFileSource(new File([changed], "invalid.qem")); }
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
