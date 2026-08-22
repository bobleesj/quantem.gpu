// src/quantem/gpu/detector/compute/webgpu/binning.ts
var UINT16_MAX = 65535;
var UINT32_MAX = 4294967295;
function requirePositiveInteger(value, name) {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new Error(`${name} must be a positive safe integer; got ${value}.`);
  }
  return value;
}
function requireUniformU32(value, name) {
  if (!Number.isSafeInteger(value) || value < 0 || value > UINT32_MAX) {
    throw new Error(`${name} must fit one WebGPU u32 uniform; got ${value}.`);
  }
  return value;
}
function modeForDtype(dtype) {
  if (dtype === "uint16") return 0;
  if (dtype === "uint8") return 1;
  if (dtype === "float32") return 2;
  return 3;
}
function bytesForDtype(dtype) {
  if (dtype === "uint8") return 1;
  if (dtype === "uint16") return 2;
  return 4;
}
function planExactDetectorBinStorage(sourceDtype, detectorBin) {
  const bin = requirePositiveInteger(detectorBin, "detectorBin");
  if (bin > 16) {
    throw new Error(`detectorBin must be between 1 and 16; got ${bin}.`);
  }
  const sourceMode = modeForDtype(sourceDtype);
  if (sourceDtype === "float32") {
    return {
      sourceDtype,
      residentDtype: "float32",
      sourceMode,
      residentMode: 2,
      detectorBin: bin,
      bytesPerValue: 4,
      maximumBinnedCount: null
    };
  }
  const maximumSourceCount = sourceDtype === "uint8" ? 255 : sourceDtype === "uint16" ? UINT16_MAX : UINT32_MAX;
  const maximumBinnedCount = maximumSourceCount * bin * bin;
  if (maximumBinnedCount > UINT32_MAX) {
    throw new Error(
      `Exact ${sourceDtype} detector bin ${bin} can reach ${maximumBinnedCount}, which exceeds uint32. WebGPU integer binning refuses a float32 fallback; use an audited narrower working dtype or a future uint64-pair kernel.`
    );
  }
  const residentDtype = maximumBinnedCount <= 255 && bin === 1 ? "uint8" : maximumBinnedCount <= UINT16_MAX ? "uint16" : "uint32";
  return {
    sourceDtype,
    residentDtype,
    sourceMode,
    residentMode: modeForDtype(residentDtype),
    detectorBin: bin,
    bytesPerValue: bytesForDtype(residentDtype),
    maximumBinnedCount
  };
}
var ZERO_BAD_PIXELS_WGSL = `
@group(0) @binding(0) var<storage,read_write> src: array<atomic<u32>>;
@group(0) @binding(1) var<storage,read> badPixels: array<u32>;
@group(0) @binding(2) var<uniform> cfg: vec4<u32>;
// cfg: totalJobs, detectorSize, nBadPixels, sourceMode
@group(0) @binding(3) var<uniform> cfg2: vec4<u32>;
// cfg2: dispatchStride, unused, unused, unused
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let job = gid.y * cfg2.x + gid.x;
  if (job >= cfg.x) { return; }
  let scan = job / cfg.z;
  let badPixel = badPixels[job - scan * cfg.z];
  let valueIndex = scan * cfg.y + badPixel;
  var wordIndex = valueIndex;
  var clearMask = 0u;
  if (cfg.w == 1u) {
    wordIndex = valueIndex >> 2u;
    clearMask = ~(255u << ((valueIndex & 3u) * 8u));
  } else if (cfg.w == 0u) {
    wordIndex = valueIndex >> 1u;
    clearMask = ~(65535u << ((valueIndex & 1u) * 16u));
  }
  atomicAnd(&src[wordIndex], clearMask);
}`;
var DETECTOR_BIN_INTEGER_COMMON_WGSL = `
@group(0) @binding(0) var<storage,read> src: array<u32>;
@group(0) @binding(1) var<storage,read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> cfg: vec4<u32>;
// cfg: totalOutputValues, outputDetectorSize, sourceDetectorSize, sourceMode
@group(0) @binding(3) var<uniform> cfg2: vec4<u32>;
// cfg2: sourceRows, sourceColumns, detectorBin, dispatchStride

fn sampleInteger(index: u32) -> u32 {
  if (cfg.w == 3u) { return src[index]; }
  if (cfg.w == 1u) {
    let word = src[index >> 2u];
    return (word >> ((index & 3u) * 8u)) & 255u;
  }
  let word = src[index >> 1u];
  return select(word >> 16u, word & 65535u, (index & 1u) == 0u);
}

fn exactBin(outputIndex: u32) -> u32 {
  let scan = outputIndex / cfg.y;
  let outputPixel = outputIndex - scan * cfg.y;
  let outputColumns = cfg2.y / cfg2.z;
  let outputRow = outputPixel / outputColumns;
  let outputColumn = outputPixel - outputRow * outputColumns;
  var sum = 0u;
  for (var binRow = 0u; binRow < cfg2.z; binRow = binRow + 1u) {
    for (var binColumn = 0u; binColumn < cfg2.z; binColumn = binColumn + 1u) {
      let sourceRow = outputRow * cfg2.z + binRow;
      let sourceColumn = outputColumn * cfg2.z + binColumn;
      let sourcePixel = sourceRow * cfg2.y + sourceColumn;
      sum = sum + sampleInteger(scan * cfg.z + sourcePixel);
    }
  }
  return sum;
}`;
var DETECTOR_BIN_UINT16_WGSL = `${DETECTOR_BIN_INTEGER_COMMON_WGSL}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let wordIndex = gid.y * cfg2.w + gid.x;
  let first = wordIndex * 2u;
  if (first >= cfg.x) { return; }
  let low = exactBin(first);
  var high = 0u;
  if (first + 1u < cfg.x) { high = exactBin(first + 1u); }
  dst[wordIndex] = (low & 65535u) | ((high & 65535u) << 16u);
}`;
var DETECTOR_BIN_UINT32_WGSL = `${DETECTOR_BIN_INTEGER_COMMON_WGSL}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let outputIndex = gid.y * cfg2.w + gid.x;
  if (outputIndex >= cfg.x) { return; }
  dst[outputIndex] = exactBin(outputIndex);
}`;
var DETECTOR_BIN_FLOAT32_WGSL = `
@group(0) @binding(0) var<storage,read> src: array<u32>;
@group(0) @binding(1) var<storage,read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> cfg: vec4<u32>;
// cfg: totalOutputValues, outputDetectorSize, sourceDetectorSize, unused
@group(0) @binding(3) var<uniform> cfg2: vec4<u32>;
// cfg2: sourceRows, sourceColumns, detectorBin, dispatchStride

fn binnedValue(outputIndex: u32) -> f32 {
  let scan = outputIndex / cfg.y;
  let outputPixel = outputIndex - scan * cfg.y;
  let outputColumns = cfg2.y / cfg2.z;
  let outputRow = outputPixel / outputColumns;
  let outputColumn = outputPixel - outputRow * outputColumns;
  var sum = 0.0;
  for (var binRow = 0u; binRow < cfg2.z; binRow = binRow + 1u) {
    for (var binColumn = 0u; binColumn < cfg2.z; binColumn = binColumn + 1u) {
      let sourceRow = outputRow * cfg2.z + binRow;
      let sourceColumn = outputColumn * cfg2.z + binColumn;
      let sourcePixel = sourceRow * cfg2.y + sourceColumn;
      sum = sum + bitcast<f32>(src[scan * cfg.z + sourcePixel]);
    }
  }
  return sum;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let outputIndex = gid.y * cfg2.w + gid.x;
  if (outputIndex >= cfg.x) { return; }
  dst[outputIndex] = bitcast<u32>(binnedValue(outputIndex));
}`;
var devicePipelines = /* @__PURE__ */ new WeakMap();
function pipelinesFor(device) {
  let pipelines = devicePipelines.get(device);
  if (!pipelines) {
    pipelines = {};
    devicePipelines.set(device, pipelines);
  }
  return pipelines;
}
function getZeroBadPixelsPipe(device) {
  const pipelines = pipelinesFor(device);
  if (!pipelines.zeroBadPixels) {
    pipelines.zeroBadPixels = device.createComputePipeline({
      layout: "auto",
      compute: {
        module: device.createShaderModule({ code: ZERO_BAD_PIXELS_WGSL }),
        entryPoint: "main"
      }
    });
  }
  return pipelines.zeroBadPixels;
}
function getDetectorBinPipe(device, residentDtype) {
  const pipelines = pipelinesFor(device);
  if (residentDtype === "uint16") {
    pipelines.uint16 ||= device.createComputePipeline({
      layout: "auto",
      compute: { module: device.createShaderModule({ code: DETECTOR_BIN_UINT16_WGSL }), entryPoint: "main" }
    });
    return pipelines.uint16;
  }
  if (residentDtype === "uint32") {
    pipelines.uint32 ||= device.createComputePipeline({
      layout: "auto",
      compute: { module: device.createShaderModule({ code: DETECTOR_BIN_UINT32_WGSL }), entryPoint: "main" }
    });
    return pipelines.uint32;
  }
  pipelines.float32 ||= device.createComputePipeline({
    layout: "auto",
    compute: { module: device.createShaderModule({ code: DETECTOR_BIN_FLOAT32_WGSL }), entryPoint: "main" }
  });
  return pipelines.float32;
}
function uniform(device, values) {
  const array = new Uint32Array(values.map((value, index) => requireUniformU32(value, `uniform[${index}]`)));
  const buffer = device.createBuffer({
    size: Math.max(16, array.byteLength),
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  });
  device.queue.writeBuffer(buffer, 0, array.buffer, array.byteOffset, array.byteLength);
  return buffer;
}
function validateBadPixels(badPixels, detectorSize) {
  for (let i = 0; i < badPixels.length; i++) {
    const index = badPixels[i];
    if (index >= detectorSize) {
      throw new Error(`badPixels[${i}]=${index} is outside detector size ${detectorSize}.`);
    }
  }
}
function packedStorageBytes(dtype, valueCount) {
  if (dtype === "uint8") return Math.ceil(valueCount / 4) * 4;
  if (dtype === "uint16") return Math.ceil(valueCount / 2) * 4;
  return valueCount * 4;
}
async function binDetectorChunksExact(request) {
  const {
    device,
    sources,
    sourceDtype,
    scanCounts,
    sourceDetectorRows,
    sourceDetectorColumns,
    detectorBin,
    badPixels = new Uint32Array(0)
  } = request;
  if (sources.length !== scanCounts.length) {
    throw new Error(`sources has ${sources.length} chunks but scanCounts has ${scanCounts.length}.`);
  }
  if (sources.length === 0) {
    throw new Error("sources must contain at least one decoded detector chunk.");
  }
  const rows = requirePositiveInteger(sourceDetectorRows, "sourceDetectorRows");
  const columns = requirePositiveInteger(sourceDetectorColumns, "sourceDetectorColumns");
  const bin = requirePositiveInteger(detectorBin, "detectorBin");
  if (bin === 1) {
    throw new Error("binDetectorChunksExact requires detectorBin greater than one; retain the source chunks for detectorBin=1.");
  }
  if (rows % bin !== 0 || columns % bin !== 0) {
    throw new Error(`Detector shape ${rows}x${columns} is not divisible by detectorBin=${bin}.`);
  }
  const plan = planExactDetectorBinStorage(sourceDtype, bin);
  const detectorRows = rows / bin;
  const detectorColumns = columns / bin;
  const detectorSize = detectorRows * detectorColumns;
  const sourceDetectorSize = rows * columns;
  requireUniformU32(detectorSize, "output detector size");
  requireUniformU32(sourceDetectorSize, "source detector size");
  validateBadPixels(badPixels, sourceDetectorSize);
  const totalOutputValues = scanCounts.map((count, chunk) => {
    const scans = requirePositiveInteger(count, `scanCounts[${chunk}]`);
    const sourceValueCount = requireUniformU32(
      scans * sourceDetectorSize,
      `chunk ${chunk} source value count`
    );
    const minimumBytes = packedStorageBytes(sourceDtype, sourceValueCount);
    if (sources[chunk].size < minimumBytes) {
      throw new Error(
        `Chunk ${chunk} buffer has ${sources[chunk].size} bytes but ${minimumBytes} are required for ${scans}x${rows}x${columns} ${sourceDtype} values.`
      );
    }
    return requireUniformU32(scans * detectorSize, `chunk ${chunk} output value count`);
  });
  const outputBytes = totalOutputValues.map((count) => {
    if (plan.residentDtype === "uint16") return Math.ceil(count / 2) * 4;
    return count * 4;
  });
  const outputs = [];
  const temporaries = [];
  const maxWorkgroups = Math.max(1, Number(device.limits.maxComputeWorkgroupsPerDimension || 65535));
  try {
    for (const size of outputBytes) {
      outputs.push(device.createBuffer({
        size: Math.max(4, size),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
      }));
    }
    const zeroBindGroups = [];
    let zeroPipe = null;
    if (badPixels.length > 0) {
      const badPixelBuffer = device.createBuffer({
        size: badPixels.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
      });
      temporaries.push(badPixelBuffer);
      device.queue.writeBuffer(
        badPixelBuffer,
        0,
        badPixels.buffer,
        badPixels.byteOffset,
        badPixels.byteLength
      );
      zeroPipe = getZeroBadPixelsPipe(device);
      for (let chunk = 0; chunk < sources.length; chunk++) {
        const totalJobs = requireUniformU32(
          scanCounts[chunk] * badPixels.length,
          `chunk ${chunk} bad-pixel clear job count`
        );
        const workgroups = Math.ceil(totalJobs / 64);
        const workgroupsX = Math.min(workgroups, maxWorkgroups);
        const workgroupsY = Math.ceil(workgroups / maxWorkgroups);
        if (workgroupsY > maxWorkgroups) {
          throw new Error(
            `Chunk ${chunk} bad-pixel clear requires ${workgroupsX}x${workgroupsY} workgroups, exceeding the device limit ${maxWorkgroups}.`
          );
        }
        const dispatchStride = workgroupsX * 64;
        const config = uniform(device, [totalJobs, sourceDetectorSize, badPixels.length, plan.sourceMode]);
        const config2 = uniform(device, [dispatchStride, 0, 0, 0]);
        temporaries.push(config, config2);
        zeroBindGroups.push({
          bindGroup: device.createBindGroup({
            layout: zeroPipe.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: sources[chunk] } },
              { binding: 1, resource: { buffer: badPixelBuffer } },
              { binding: 2, resource: { buffer: config } },
              { binding: 3, resource: { buffer: config2 } }
            ]
          }),
          workgroupsX,
          workgroupsY
        });
      }
    }
    const binPipe = getDetectorBinPipe(device, plan.residentDtype);
    const binJobs = [];
    for (let chunk = 0; chunk < sources.length; chunk++) {
      const invocationCount = plan.residentDtype === "uint16" ? Math.ceil(totalOutputValues[chunk] / 2) : totalOutputValues[chunk];
      const workgroups = Math.ceil(invocationCount / 64);
      const workgroupsX = Math.min(workgroups, maxWorkgroups);
      const workgroupsY = Math.ceil(workgroups / maxWorkgroups);
      if (workgroupsY > maxWorkgroups) {
        throw new Error(
          `Chunk ${chunk} requires ${workgroupsX}x${workgroupsY} WebGPU workgroups, exceeding the device limit ${maxWorkgroups}.`
        );
      }
      const dispatchStride = workgroupsX * 64;
      const config = uniform(device, [
        totalOutputValues[chunk],
        detectorSize,
        sourceDetectorSize,
        plan.sourceMode
      ]);
      const config2 = uniform(device, [rows, columns, bin, dispatchStride]);
      temporaries.push(config, config2);
      binJobs.push({
        bindGroup: device.createBindGroup({
          layout: binPipe.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: sources[chunk] } },
            { binding: 1, resource: { buffer: outputs[chunk] } },
            { binding: 2, resource: { buffer: config } },
            { binding: 3, resource: { buffer: config2 } }
          ]
        }),
        workgroupsX,
        workgroupsY
      });
    }
    const encoder = device.createCommandEncoder();
    if (zeroPipe && zeroBindGroups.length > 0) {
      const zeroPass = encoder.beginComputePass();
      zeroPass.setPipeline(zeroPipe);
      for (const job of zeroBindGroups) {
        zeroPass.setBindGroup(0, job.bindGroup);
        zeroPass.dispatchWorkgroups(job.workgroupsX, job.workgroupsY);
      }
      zeroPass.end();
    }
    const binPass = encoder.beginComputePass();
    binPass.setPipeline(binPipe);
    for (const job of binJobs) {
      binPass.setBindGroup(0, job.bindGroup);
      binPass.dispatchWorkgroups(job.workgroupsX, job.workgroupsY);
    }
    binPass.end();
    const start = performance.now();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    const elapsedMs = performance.now() - start;
    temporaries.forEach((buffer) => buffer.destroy());
    return {
      buffers: outputs,
      detectorRows,
      detectorColumns,
      detectorSize,
      residentDtype: plan.residentDtype,
      residentMode: plan.residentMode,
      residentBytes: outputBytes.reduce((total, value) => total + value, 0),
      maximumBinnedCount: plan.maximumBinnedCount,
      sourceBuffersMutated: badPixels.length > 0,
      elapsedMs
    };
  } catch (error) {
    temporaries.forEach((buffer) => buffer.destroy());
    outputs.forEach((buffer) => buffer.destroy());
    throw error;
  }
}
export {
  binDetectorChunksExact,
  planExactDetectorBinStorage
};
