/// <reference types="@webgpu/types" />
import { decodeBslz4Batch } from "../../src/quantem/gpu/io/hdf5/webgpu/bslz4";
import { readH5Volume } from "../../src/quantem/gpu/io/hdf5/webgpu/h5reader";
import {
  loadShow4DSTEMLocalH5MaskedSum,
  loadShow4DSTEMLocalH5Master,
  setShow4DSTEMLocalFiles,
} from "../../src/quantem/gpu/io/hdf5/webgpu/local-h5";
import { DetectorCompute } from "../../src/quantem/gpu/detector/webgpu/backend";
import { annulusMask, diskMask } from "../../src/quantem/gpu/detector/webgpu/geometry";

type Case = { name: string; dtype: "uint8" | "uint16" | "uint32" | "float32"; frames: number; detector: [number, number]; bad_pixels: number[] };
type Values = Uint8Array | Uint16Array | Uint32Array | Float32Array;

function fail(message: string): never { throw new Error(message); }

async function readBack(device: GPUDevice, buffer: GPUBuffer, bytes: number): Promise<ArrayBuffer> {
  const readback = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, readback, 0, bytes);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const out = readback.getMappedRange().slice(0);
  readback.destroy();
  return out;
}

/** Exact masked sum of the raw counts, bad pixels excluded, rounded once to float32. */
function expectedSums(counts: Values, frames: number, mask: Uint32Array, bad: Set<number>): Float32Array {
  const detSize = mask.length;
  const out = new Float32Array(frames);
  for (let frame = 0; frame < frames; frame++) {
    let sum = 0;
    for (let pixel = 0; pixel < detSize; pixel++) if (mask[pixel] && !bad.has(pixel)) sum += counts[frame * detSize + pixel];
    out[frame] = sum;
  }
  return out;
}

function exact(label: string, actual: Float32Array, expected: Float32Array): void {
  if (actual.length !== expected.length) fail(`${label}: ${actual.length} values, expected ${expected.length}`);
  for (let i = 0; i < actual.length; i++) {
    if (actual[i] !== expected[i]) fail(`${label}[${i}]: ${actual[i]} != ${expected[i]}`);
  }
}

type LoadedCase = {
  item: Case; counts: Values; bad: Set<number>; files: File[];
  rows: number; cols: number; detSize: number; scanRows: number; scanCols: number; masks: Record<string, Uint32Array>;
};

async function loadCases(baseURL: string): Promise<LoadedCase[]> {
  const base = baseURL.endsWith("/") ? baseURL : baseURL + "/";
  const cases = await (await fetch(base + "h5-cases.json")).json() as Case[];
  return Promise.all(cases.map(async (item) => {
    const [rows, cols] = item.detector;
    const detSize = rows * cols;
    const raw = await (await fetch(base + item.name + ".bin")).arrayBuffer();
    const names = [`${item.name}_master.h5`, `${item.name}_data_000001.h5`, `${item.name}_data_000002.h5`];
    const files = await Promise.all(names.map(async (name) => new File([await (await fetch(base + name)).arrayBuffer()], name)));
    const scanCols = item.frames % 20 === 0 ? 20 : 10;
    return {
      item, counts: { uint8: Uint8Array, uint16: Uint16Array, uint32: Uint32Array, float32: Float32Array }[item.dtype].from(new { uint8: Uint8Array, uint16: Uint16Array, uint32: Uint32Array, float32: Float32Array }[item.dtype](raw)), bad: new Set(item.bad_pixels), files,
      rows, cols, detSize, scanRows: item.frames / scanCols, scanCols,
      masks: {
        full: new Uint32Array(detSize).fill(1),
        disk: diskMask(rows, cols, rows / 2, cols / 2, rows / 5),
        annulus: annulusMask(rows, cols, rows / 2, cols / 2, rows / 8, rows / 2),
        center: Uint32Array.from({ length: detSize }, (_, k) => (k === (rows / 2) * cols + cols / 2 ? 1 : 0)),
        pair: Uint32Array.from({ length: detSize }, (_, k) => (k === 3 * cols + 4 || k === (rows - 2) * cols + cols - 3 ? 1 : 0)),
      },
    };
  }));
}

/** Full-stack bitshuffle-LZ4 decode on every kernel: native width is lossless, uint8 output
saturates at 255 and the low-8 decode (count-audited browse data) keeps the low byte. float32
values compare as bit patterns. */
export async function runDenseDecodeParity(baseURL: string): Promise<Record<string, number>> {
  const checked: Record<string, number> = {};
  const globals = globalThis as Record<string, unknown>;
  const plans = {
    uint16: [["uint16", "native"], ["uint8", "clip"], ["uint8", "low8"]],
    uint8: [["uint8", "native"], ["uint8", "low8"]],
    uint32: [["uint32", "native"], ["uint8", "clip"], ["uint8", "low8"]],
    float32: [["float32", "native"]],
  } as const;
  for (const { item, counts, files, detSize } of await loadCases(baseURL)) {
    const specs = [];
    let start = 0;
    for (const file of files.slice(1)) {
      const volume = readH5Volume(await file.arrayBuffer(), file.name);
      for (const chunk of volume.chunks) { specs.push({ ...chunk, startScan: start }); start += chunk.nFrames; }
    }
    const want = new Uint32Array(item.dtype === "float32" ? new Float32Array(counts).buffer : Uint32Array.from(counts).buffer);
    let count = 0;
    for (const [dtype, kind] of plans[item.dtype]) {
      globals.__BSLZ4_LOW8_ONLY = kind === "low8";
      const decoded = await decodeBslz4Batch(specs, dtype, item.dtype, 4);
      delete globals.__BSLZ4_LOW8_ONLY;
      if (!decoded) fail("WebGPU unavailable");
      const bytesPerPixel = { uint8: 1, uint16: 2, uint32: 4, float32: 4 }[dtype];
      for (let i = 0; i < decoded.buffers.length; i++) {
        const frames = specs[i].nFrames, offset = specs[i].startScan * detSize;
        const bytes = await readBack(decoded.device, decoded.buffers[i], Math.ceil(frames * detSize * bytesPerPixel / 4) * 4);
        const values = { 1: Uint8Array, 2: Uint16Array, 4: Uint32Array }[bytesPerPixel];
        const actual = new values(bytes, 0, frames * detSize);
        for (let k = 0; k < frames * detSize; k++) {
          const value = want[offset + k];
          const expected = kind === "native" ? value : kind === "clip" ? Math.min(255, value) : value & 255;
          if (actual[k] !== expected) fail(`${item.name} ${dtype} ${kind} decode frame ${Math.floor(k / detSize)} pixel ${k % detSize}: ${actual[k]} != ${expected}`);
        }
        count += frames * detSize;
        decoded.buffers[i].destroy();
      }
    }
    checked[item.name] = count;
  }
  return checked;
}

/** Product-first virtual images, the widget's first image before the stack is resident.

Through the widget's entry point, with the pixel and the group-mask kernel and a scan region.
The fixtures hold counts above 255, so a decoder that keeps only the low 8 bit planes fails.
*/
export async function runProductFirstParity(baseURL: string): Promise<Record<string, number>> {
  const checked: Record<string, number> = {};
  const globals = globalThis as Record<string, unknown>;
  for (const { item, counts, bad, files, masks, scanRows, scanCols } of await loadCases(baseURL)) {
    setShow4DSTEMLocalFiles(files);
    if (item.dtype === "uint32" || item.dtype === "float32") {
      // The product-first sum reads uint8 and uint16 counts; wider sources load the resident stack.
      const refused = await loadShow4DSTEMLocalH5MaskedSum(`${item.name}_master.h5`, { scanRows, scanCols, mask: masks.disk }).then(() => false, () => true);
      if (!refused) fail(`${item.name}: product-first accepted a ${item.dtype} source`);
      checked[item.name] = 0;
      continue;
    }
    let count = 0;
    for (const groupMask of [false, true]) {
      globals.__QT_BSLZ4_MASKED_SUM_GROUPMASK = groupMask;
      try {
        for (const [maskName, mask] of Object.entries(masks)) {
          const product = await loadShow4DSTEMLocalH5MaskedSum(`${item.name}_master.h5`, { scanRows, scanCols, mask });
          if (!product) fail(`${item.name}: no product-first result`);
          const actual = new Float32Array(await readBack(product.device, product.buffer, item.frames * 4));
          product.buffer.destroy();
          exact(`${item.name} ${groupMask ? "groupmask" : "pixel"} ${maskName} product-first`, actual, expectedSums(counts, item.frames, mask, bad));
          count += item.frames;
        }
        const region = [1, scanRows - 1, 2, scanCols - 3] as const;
        const product = await loadShow4DSTEMLocalH5MaskedSum(`${item.name}_master.h5`, { scanRows, scanCols, mask: masks.disk, scanRegion: region });
        if (!product) fail(`${item.name}: no scan-region product`);
        const cropCols = region[3] - region[2];
        const n = (region[1] - region[0]) * cropCols;
        const actual = new Float32Array(await readBack(product.device, product.buffer, n * 4));
        product.buffer.destroy();
        const full = expectedSums(counts, item.frames, masks.disk, bad);
        exact(`${item.name} ${groupMask ? "groupmask" : "pixel"} scan-region product-first`, actual,
          Float32Array.from({ length: n }, (_, i) => full[(region[0] + Math.floor(i / cropCols)) * scanCols + region[2] + (i % cropCols)]));
        count += n;
      } finally {
        delete globals.__QT_BSLZ4_MASKED_SUM_GROUPMASK;
      }
    }
    checked[item.name] = count;
  }
  return checked;
}

/** Virtual images and centers of mass of the resident stack the widget builds from the same files.

Every mask, down to one and two pixels, must sum exactly the selected counts, and the center of
mass must equal the exact integer moments divided once.
*/
export async function runResidentParity(baseURL: string): Promise<Record<string, number>> {
  const checked: Record<string, number> = {};
  for (const { item, counts, bad, files, masks, scanRows, scanCols, cols, detSize } of await loadCases(baseURL)) {
    if (item.dtype === "float32") continue;   // float32 sums in float32; the integer cases are exact
    setShow4DSTEMLocalFiles(files);
    const resident = await loadShow4DSTEMLocalH5Master(`${item.name}_master.h5`, { scanRows, scanCols, decodeDtype: "native" });
    if (!resident) fail(`${item.name}: no resident load`);
    const compute = DetectorCompute.fromGpuChunks(resident.device, resident.chunks, resident.scanCount, resident.detSize, resident.mode);
    compute.badPx = resident.badPixels;
    let count = 0;
    for (const [maskName, mask] of Object.entries(masks)) {
      exact(`${item.name} resident ${maskName}`, await compute.maskedSum(mask), expectedSums(counts, item.frames, mask, bad));
      const com = await compute.maskedCoM(mask, cols);
      const expectedRow = new Float32Array(item.frames), expectedCol = new Float32Array(item.frames);
      for (let frame = 0; frame < item.frames; frame++) {
        let weight = 0, rowMoment = 0, colMoment = 0;
        for (let pixel = 0; pixel < detSize; pixel++) {
          if (!mask[pixel] || bad.has(pixel)) continue;
          const value = counts[frame * detSize + pixel];
          weight += value; rowMoment += Math.floor(pixel / cols) * value; colMoment += (pixel % cols) * value;
        }
        expectedRow[frame] = weight > 0 ? rowMoment / weight : 0;
        expectedCol[frame] = weight > 0 ? colMoment / weight : 0;
      }
      exact(`${item.name} resident ${maskName} CoM row`, com.comY, expectedRow);
      exact(`${item.name} resident ${maskName} CoM col`, com.comX, expectedCol);
      count += 3 * item.frames;
    }
    compute.dispose();
    checked[item.name] = count;
  }
  return checked;
}

/** Dense virtual images and centers of mass keep integer counts exact, including sums above 2^24,
and float32 data sums its values.

A uint16 stack near the 65535 maximum gives full-detector totals near 2^28, where float32
accumulation drops counts. The direct sum and the complement path a large aperture takes
(total minus the excluded pixels) must equal the exact integer sum rounded once to float32, and
the center of mass the exact moments divided once. The shared device has subgroups; `device`
(requested without features) runs the shared-memory reductions that adapters without subgroups,
such as Windows D3D feature level 11, compile.
*/
export async function runDenseVirtualImageParity(device: GPUDevice): Promise<Record<string, number>> {
  const rows = 64, cols = 64, detSize = rows * cols, scanCount = 96;
  const values = new Uint16Array(scanCount * detSize);
  for (let i = 0; i < values.length; i++) values[i] = (i % 7 === 0) ? 65535 : 65535 - ((i * 7919) % 4093);
  const bad = new Set([5, 4000]);
  const sums = (mask: Uint32Array) => expectedSums(values, scanCount, mask, bad);
  const masks: Record<string, Uint32Array> = {
    full: new Uint32Array(detSize).fill(1),
    annulus: annulusMask(rows, cols, 32, 32, 6, 60),
    disk: diskMask(rows, cols, 32, 32, 9),
    pixel: Uint32Array.from({ length: detSize }, (_, k) => (k === 33 * cols + 31 ? 1 : 0)),
  };
  const shared = await DetectorCompute.createChunked([{ bytes: new Uint8Array(values.buffer), startScan: 0, nScan: scanCount }], scanCount, detSize, 0);
  if (!shared) fail("WebGPU unavailable");
  const chunk = (first: number, stop: number) => {
    const buffer = device.createBuffer({ size: (stop - first) * detSize * 2, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(buffer, 0, values, first * detSize, (stop - first) * detSize);
    return { buffer, startScan: first, nScan: stop - first };
  };
  const plain = DetectorCompute.fromGpuChunks(device, [chunk(0, 40), chunk(40, scanCount)], scanCount, detSize, 0);
  const checked: Record<string, number> = {};
  if (device.features.has("subgroups")) fail("the shared-memory reductions need a device requested without subgroups");
  const sharedLabel = shared.getDevice().features.has("subgroups") ? "subgroups" : "shared device without subgroups";
  for (const [label, compute] of [[sharedLabel, shared], ["shared-memory", plain]] as const) {
    compute.badPx = Uint32Array.from(bad);
    let count = 0;
    for (const [name, mask] of Object.entries(masks)) {
      const expected = sums(mask);
      exact(`${label} virtual image ${name} readback`, await compute.maskedSum(mask), expected);
      const buffer = compute.maskedSumBuffer(mask).buffer;
      exact(`${label} virtual image ${name} resident`, new Float32Array(await readBack(compute.getDevice(), buffer, scanCount * 4)), expected);
      buffer.destroy();
      const com = await compute.maskedCoM(mask, cols);
      const expectedRow = new Float32Array(scanCount), expectedCol = new Float32Array(scanCount);
      for (let scan = 0; scan < scanCount; scan++) {
        let weight = 0, rowMoment = 0, colMoment = 0;
        for (let pixel = 0; pixel < detSize; pixel++) {
          if (!mask[pixel] || bad.has(pixel)) continue;
          const value = values[scan * detSize + pixel];
          weight += value; rowMoment += Math.floor(pixel / cols) * value; colMoment += (pixel % cols) * value;
        }
        expectedRow[scan] = rowMoment / weight; expectedCol[scan] = colMoment / weight;
      }
      exact(`${label} ${name} CoM row`, com.comY, expectedRow);
      exact(`${label} ${name} CoM col`, com.comX, expectedCol);
      count += 4 * scanCount;
    }
    // A drag adds the exact delta of the changed pixels to the previous image; below 2^24 that is exact.
    const from = diskMask(rows, cols, 32, 32, 3), to = diskMask(rows, cols, 33, 31, 3);
    const previous = DetectorCompute.maskedSumBuffersBatch([compute], from).buffers;
    const added = Uint32Array.from(to, (v, k) => (v && !from[k] ? 1 : 0));
    const removed = Uint32Array.from(to, (v, k) => (!v && from[k] ? 1 : 0));
    const delta = DetectorCompute.maskedSumDeltaBuffersBatch([compute], previous, added, removed);
    exact(`${label} virtual image drag delta`, new Float32Array(await readBack(compute.getDevice(), delta.buffers[0], scanCount * 4)), sums(to));
    count += scanCount;
    compute.dispose();
    checked[label] = count;
  }

  // float32 data: quarter-integer values keep every virtual-image sum exact in float32; the
  // center of mass accumulates float32 moments, so it matches float64 to rounding.
  const floats = Float32Array.from({ length: scanCount * detSize }, (_, i) => ((i * 37) % 200) * 0.25);
  const floatShared = await DetectorCompute.createChunked([{ bytes: new Uint8Array(floats.buffer), startScan: 0, nScan: scanCount }], scanCount, detSize, 2);
  if (!floatShared) fail("WebGPU unavailable");
  const floatChunk = device.createBuffer({ size: floats.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(floatChunk, 0, floats);
  const floatPlain = DetectorCompute.fromGpuChunks(device, [{ buffer: floatChunk, startScan: 0, nScan: scanCount }], scanCount, detSize, 2);
  for (const [label, compute] of [[sharedLabel, floatShared], ["shared-memory", floatPlain]] as const) {
    let count = 0;
    for (const [name, mask] of Object.entries(masks)) {
      const expected = expectedSums(floats, scanCount, mask, new Set());
      exact(`${label} float32 virtual image ${name}`, await compute.maskedSum(mask), expected);
      const buffer = compute.maskedSumBuffer(mask).buffer;
      exact(`${label} float32 virtual image ${name} resident`, new Float32Array(await readBack(compute.getDevice(), buffer, scanCount * 4)), expected);
      buffer.destroy();
      const com = await compute.maskedCoM(mask, cols);
      for (let scan = 0; scan < scanCount; scan++) {
        let weight = 0, rowMoment = 0, colMoment = 0;
        for (let pixel = 0; pixel < detSize; pixel++) {
          if (!mask[pixel]) continue;
          const value = floats[scan * detSize + pixel];
          weight += value; rowMoment += Math.floor(pixel / cols) * value; colMoment += (pixel % cols) * value;
        }
        const row = weight > 0 ? rowMoment / weight : 0, col = weight > 0 ? colMoment / weight : 0;
        if (!(Math.abs(com.comY[scan] - row) <= 1e-5 * Math.max(1, Math.abs(row)) && Math.abs(com.comX[scan] - col) <= 1e-5 * Math.max(1, Math.abs(col)))) {
          fail(`${label} float32 ${name} CoM[${scan}]: (${com.comY[scan]}, ${com.comX[scan]}) != (${row}, ${col})`);
        }
      }
      count += 4 * scanCount;
    }
    compute.dispose();
    checked[`${label} float32`] = count;
  }
  return checked;
}
