import {
  binDetectorChunksExact,
  planExactDetectorBinStorage,
} from "./binning.js";

const resultNode = document.querySelector("#result");

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function percentile(values, fraction) {
  const ordered = [...values].sort((a, b) => a - b);
  const index = Math.ceil(fraction * ordered.length) - 1;
  return ordered[Math.max(0, index)];
}

function pack(values, dtype) {
  if (dtype === "uint8") {
    const words = new Uint32Array(Math.ceil(values.length / 4));
    values.forEach((value, index) => {
      words[index >> 2] |= (value & 0xff) << ((index & 3) * 8);
    });
    return words;
  }
  if (dtype === "uint16") {
    const words = new Uint32Array(Math.ceil(values.length / 2));
    values.forEach((value, index) => {
      words[index >> 1] |= (value & 0xffff) << ((index & 1) * 16);
    });
    return words;
  }
  return Uint32Array.from(values);
}

function unpack(words, dtype, count) {
  const values = new Uint32Array(count);
  for (let index = 0; index < count; index++) {
    if (dtype === "uint8") {
      values[index] = (words[index >> 2] >> ((index & 3) * 8)) & 0xff;
    } else if (dtype === "uint16") {
      values[index] = (words[index >> 1] >> ((index & 1) * 16)) & 0xffff;
    } else {
      values[index] = words[index];
    }
  }
  return values;
}

function expectedBin(values, scans, rows, columns, bin, badPixels) {
  const bad = new Set(badPixels);
  const outputRows = rows / bin;
  const outputColumns = columns / bin;
  const output = new Uint32Array(scans * outputRows * outputColumns);
  for (let scan = 0; scan < scans; scan++) {
    for (let outputRow = 0; outputRow < outputRows; outputRow++) {
      for (let outputColumn = 0; outputColumn < outputColumns; outputColumn++) {
        let sum = 0;
        for (let binRow = 0; binRow < bin; binRow++) {
          for (let binColumn = 0; binColumn < bin; binColumn++) {
            const detectorRow = outputRow * bin + binRow;
            const detectorColumn = outputColumn * bin + binColumn;
            const detectorIndex = detectorRow * columns + detectorColumn;
            if (!bad.has(detectorIndex)) {
              sum += values[scan * rows * columns + detectorIndex];
            }
          }
        }
        output[(scan * outputRows + outputRow) * outputColumns + outputColumn] = sum;
      }
    }
  }
  return output;
}

function createSource(device, values, dtype) {
  const packed = pack(values, dtype);
  const buffer = device.createBuffer({
    size: Math.max(4, packed.byteLength),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  device.queue.writeBuffer(buffer, 0, packed);
  return buffer;
}

async function readOutput(device, buffer, bytes, dtype, count) {
  const readback = device.createBuffer({
    size: Math.max(4, bytes),
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, readback, 0, Math.max(4, bytes));
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const words = new Uint32Array(readback.getMappedRange().slice(0));
  const values = unpack(words, dtype, count);
  readback.unmap();
  readback.destroy();
  return values;
}

function assertArrayEqual(actual, expected, label) {
  assert(actual.length === expected.length, `${label}: length ${actual.length} != ${expected.length}`);
  for (let index = 0; index < actual.length; index++) {
    if (actual[index] !== expected[index]) {
      throw new Error(`${label}: index ${index}: ${actual[index]} != ${expected[index]}`);
    }
  }
}

async function runExactCase(device, spec) {
  const sources = spec.values.map((values) => createSource(device, values, spec.sourceDtype));
  const wallStart = performance.now();
  const result = await binDetectorChunksExact({
    device,
    sources,
    sourceDtype: spec.sourceDtype,
    scanCounts: spec.scanCounts,
    sourceDetectorRows: spec.rows,
    sourceDetectorColumns: spec.columns,
    detectorBin: spec.bin,
    badPixels: new Uint32Array(spec.badPixels),
  });
  const wallMs = performance.now() - wallStart;
  assert(result.residentDtype === spec.expectedDtype, `${spec.label}: resident dtype ${result.residentDtype}`);
  assert(result.sourceBuffersMutated === (spec.badPixels.length > 0), `${spec.label}: source mutation flag`);
  const outputRows = spec.rows / spec.bin;
  const outputColumns = spec.columns / spec.bin;
  for (let chunk = 0; chunk < sources.length; chunk++) {
    const count = spec.scanCounts[chunk] * outputRows * outputColumns;
    const bytes = result.residentDtype === "uint16" ? Math.ceil(count / 2) * 4 : count * 4;
    const actual = await readOutput(device, result.buffers[chunk], bytes, result.residentDtype, count);
    const expected = expectedBin(
      spec.values[chunk],
      spec.scanCounts[chunk],
      spec.rows,
      spec.columns,
      spec.bin,
      spec.badPixels,
    );
    assertArrayEqual(actual, expected, `${spec.label} chunk ${chunk}`);
  }
  result.buffers.forEach((buffer) => buffer.destroy());
  sources.forEach((buffer) => buffer.destroy());
  return { label: spec.label, wallMs, gpuSubmitWaitMs: result.elapsedMs, residentBytes: result.residentBytes };
}

async function main() {
  assert(navigator.gpu, "WebGPU unavailable");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  assert(adapter, "No WebGPU adapter");
  const adapterInfo = adapter.info || {};
  const adapterText = JSON.stringify(adapterInfo).toLowerCase();
  assert(!adapterText.includes("swiftshader") && !adapterText.includes("software"), `Software adapter: ${adapterText}`);
  const device = await adapter.requestDevice();

  const plans = {
    uint8Bin2: planExactDetectorBinStorage("uint8", 2),
    uint16Bin2: planExactDetectorBinStorage("uint16", 2),
  };
  assert(plans.uint8Bin2.residentDtype === "uint16", "uint8 bin2 must widen to uint16");
  assert(plans.uint16Bin2.residentDtype === "uint32", "uint16 bin2 must widen to uint32");
  let uint32Reject = false;
  try { planExactDetectorBinStorage("uint32", 2); } catch { uint32Reject = true; }
  assert(uint32Reject, "uint32 bin2 must fail closed without uint64-pair support");

  const uint8Values = [2, 1].map((scans, chunk) => Uint32Array.from(
    { length: scans * 6 * 6 },
    (_, index) => (index * 17 + chunk * 23) % 251,
  ));
  const uint16Values = [1, 2].map((scans, chunk) => Uint32Array.from(
    { length: scans * 8 * 8 },
    (_, index) => 50000 + ((index * 97 + chunk * 31) % 15000),
  ));
  const exactCases = [];
  exactCases.push(await runExactCase(device, {
    label: "uint8-to-uint16 odd-packed multiple-chunk bad-pixel",
    sourceDtype: "uint8",
    expectedDtype: "uint16",
    scanCounts: [2, 1],
    values: uint8Values,
    rows: 6,
    columns: 6,
    bin: 2,
    badPixels: [0, 17, 35],
  }));
  exactCases.push(await runExactCase(device, {
    label: "uint16-to-uint32 multiple-chunk",
    sourceDtype: "uint16",
    expectedDtype: "uint32",
    scanCounts: [1, 2],
    values: uint16Values,
    rows: 8,
    columns: 8,
    bin: 2,
    badPixels: [],
  }));

  const repeatWalls = [];
  const repeatGpu = [];
  for (let repeat = 0; repeat < 12; repeat++) {
    const trial = await runExactCase(device, {
      label: `warm-repeat-${repeat}`,
      sourceDtype: "uint16",
      expectedDtype: "uint32",
      scanCounts: [2],
      values: [uint16Values[1]],
      rows: 8,
      columns: 8,
      bin: 2,
      badPixels: [],
    });
    repeatWalls.push(trial.wallMs);
    repeatGpu.push(trial.gpuSubmitWaitMs);
  }

  const dispatchScans = 300000;
  const dispatchValues = new Uint32Array(dispatchScans * 8 * 8);
  dispatchValues.fill(1);
  const dispatchSources = [createSource(device, dispatchValues, "uint16")];
  const dispatchStart = performance.now();
  const dispatchResult = await binDetectorChunksExact({
    device,
    sources: dispatchSources,
    sourceDtype: "uint16",
    scanCounts: [dispatchScans],
    sourceDetectorRows: 8,
    sourceDetectorColumns: 8,
    detectorBin: 2,
  });
  const dispatchWallMs = performance.now() - dispatchStart;
  const dispatchCount = dispatchScans * 4 * 4;
  const dispatchReadbackStart = performance.now();
  const dispatchActual = await readOutput(
    device,
    dispatchResult.buffers[0],
    dispatchCount * 4,
    "uint32",
    dispatchCount,
  );
  let dispatchChecksum = 0;
  let dispatchMismatchCount = 0;
  for (let index = 0; index < dispatchActual.length; index++) {
    dispatchChecksum += dispatchActual[index];
    if (dispatchActual[index] !== 4) dispatchMismatchCount += 1;
  }
  const dispatchReadbackValidationMs = performance.now() - dispatchReadbackStart;
  assert(dispatchMismatchCount === 0, `tiled dispatch mismatches: ${dispatchMismatchCount}`);
  assert(dispatchChecksum === dispatchCount * 4, `tiled dispatch checksum: ${dispatchChecksum}`);
  dispatchResult.buffers.forEach((buffer) => buffer.destroy());
  dispatchSources.forEach((buffer) => buffer.destroy());
  device.destroy();

  return {
    status: "passed",
    adapterInfo,
    softwareAdapter: false,
    plans,
    exactCases,
    repeat: {
      samples: repeatWalls.length,
      wallP50Ms: percentile(repeatWalls, 0.5),
      wallP95Ms: percentile(repeatWalls, 0.95),
      wallMaxMs: Math.max(...repeatWalls),
      gpuSubmitWaitP50Ms: percentile(repeatGpu, 0.5),
      gpuSubmitWaitP95Ms: percentile(repeatGpu, 0.95),
      gpuSubmitWaitMaxMs: Math.max(...repeatGpu),
    },
    tiledDispatch: {
      scans: dispatchScans,
      sourceValues: dispatchValues.length,
      outputValues: dispatchCount,
      outputBytes: dispatchResult.residentBytes,
      wallMs: dispatchWallMs,
      gpuSubmitWaitMs: dispatchResult.elapsedMs,
      readbackValidationMs: dispatchReadbackValidationMs,
      mismatchCount: dispatchMismatchCount,
      checksum: dispatchChecksum,
      expectedChecksum: dispatchCount * 4,
    },
  };
}

try {
  const result = await main();
  window.__qgpuResult = result;
  window.__qgpuDone = true;
  resultNode.textContent = JSON.stringify(result, null, 2);
} catch (error) {
  const result = { status: "failed", error: String(error?.stack || error) };
  window.__qgpuResult = result;
  window.__qgpuDone = true;
  resultNode.textContent = JSON.stringify(result, null, 2);
}
