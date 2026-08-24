/// Byte-exact WebGPU quarter turns of scan-major scientific data.

export type ScanQuarterTurns = 0 | 1 | 2 | 3;

export interface ScanQuarterTurnOptions {
  scanRows: number;
  scanColumns: number;
  wordsPerScan: number;
  quarterTurns: ScanQuarterTurns;
  destination?: GPUBuffer;
}

export interface RotatedScanBuffer {
  buffer: GPUBuffer;
  scanRows: number;
  scanColumns: number;
}

const WORKGROUP_SIZE = 256;
const pipelineByDevice = new WeakMap<GPUDevice, Promise<GPUComputePipeline>>();

export const SCAN_QUARTER_TURN_WGSL = /* wgsl */ `
struct Parameters {
  sourceRows: u32,
  sourceColumns: u32,
  wordsPerScan: u32,
  quarterTurns: u32,
  outputRows: u32,
  outputColumns: u32,
  totalWords: u32,
  groupColumns: u32,
}

@group(0) @binding(0) var<storage, read> source: array<u32>;
@group(0) @binding(1) var<storage, read_write> destination: array<u32>;
@group(0) @binding(2) var<uniform> parameters: Parameters;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn rotateScanQuarterTurn(
  @builtin(workgroup_id) workgroup: vec3<u32>,
  @builtin(local_invocation_id) local: vec3<u32>,
) {
  let group = workgroup.y * parameters.groupColumns + workgroup.x;
  let outputWord = group * ${WORKGROUP_SIZE}u + local.x;
  if (outputWord >= parameters.totalWords) { return; }

  let outputScan = outputWord / parameters.wordsPerScan;
  let wordInScan = outputWord % parameters.wordsPerScan;
  let outputRow = outputScan / parameters.outputColumns;
  let outputColumn = outputScan % parameters.outputColumns;
  var sourceRow = outputRow;
  var sourceColumn = outputColumn;
  switch parameters.quarterTurns {
    case 1u: {
      sourceRow = outputColumn;
      sourceColumn = parameters.sourceColumns - 1u - outputRow;
    }
    case 2u: {
      sourceRow = parameters.sourceRows - 1u - outputRow;
      sourceColumn = parameters.sourceColumns - 1u - outputColumn;
    }
    case 3u: {
      sourceRow = parameters.sourceRows - 1u - outputColumn;
      sourceColumn = outputRow;
    }
    default: {}
  }
  let sourceScan = sourceRow * parameters.sourceColumns + sourceColumn;
  destination[outputWord] = source[
    sourceScan * parameters.wordsPerScan + wordInScan
  ];
}
`;

function positiveInteger(value: number, name: string): number {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new Error(`${name} must be a positive safe integer; got ${value}.`);
  }
  return value;
}

export function scanQuarterTurnOutputShape(
  scanRows: number,
  scanColumns: number,
  quarterTurns: ScanQuarterTurns,
): [number, number] {
  return quarterTurns % 2 === 0
    ? [scanRows, scanColumns]
    : [scanColumns, scanRows];
}

/** Return the source scan index for one rotated output coordinate. */
export function scanQuarterTurnSourceIndex(
  outputRow: number,
  outputColumn: number,
  scanRows: number,
  scanColumns: number,
  quarterTurns: ScanQuarterTurns,
): number {
  let sourceRow = outputRow;
  let sourceColumn = outputColumn;
  if (quarterTurns === 1) {
    sourceRow = outputColumn;
    sourceColumn = scanColumns - 1 - outputRow;
  } else if (quarterTurns === 2) {
    sourceRow = scanRows - 1 - outputRow;
    sourceColumn = scanColumns - 1 - outputColumn;
  } else if (quarterTurns === 3) {
    sourceRow = scanRows - 1 - outputColumn;
    sourceColumn = outputRow;
  }
  return sourceRow * scanColumns + sourceColumn;
}

async function pipeline(device: GPUDevice): Promise<GPUComputePipeline> {
  let retained = pipelineByDevice.get(device);
  if (retained == null) {
    retained = device.createComputePipelineAsync({
      layout: "auto",
      compute: {
        module: device.createShaderModule({ code: SCAN_QUARTER_TURN_WGSL }),
        entryPoint: "rotateScanQuarterTurn",
      },
    });
    pipelineByDevice.set(device, retained);
  }
  return retained;
}

/**
 * Rotate only the scan plane while preserving every 32-bit payload word.
 *
 * Detector samples are treated as opaque words, so packed integer counts and
 * floating-point bit patterns are copied without conversion. Reuse
 * `destination` for repeated rotations to avoid allocation during interaction.
 * Positive quarter turns are counterclockwise in the displayed scan frame.
 */
export async function rotateScanQuarterTurnWebGPU(
  device: GPUDevice,
  source: GPUBuffer,
  options: ScanQuarterTurnOptions,
): Promise<RotatedScanBuffer> {
  const scanRows = positiveInteger(options.scanRows, "scanRows");
  const scanColumns = positiveInteger(options.scanColumns, "scanColumns");
  const wordsPerScan = positiveInteger(options.wordsPerScan, "wordsPerScan");
  const quarterTurns = options.quarterTurns;
  if (![0, 1, 2, 3].includes(quarterTurns)) {
    throw new Error(`quarterTurns must be 0, 1, 2, or 3; got ${quarterTurns}.`);
  }
  const [outputRows, outputColumns] = scanQuarterTurnOutputShape(
    scanRows,
    scanColumns,
    quarterTurns,
  );
  const totalWords = outputRows * outputColumns * wordsPerScan;
  if (!Number.isSafeInteger(totalWords) || totalWords > 0xffffffff) {
    throw new Error(
      `Rotated scan needs ${totalWords} words; WebGPU supports at most 4294967295.`,
    );
  }
  const byteLength = totalWords * Uint32Array.BYTES_PER_ELEMENT;
  if (source.size < byteLength) {
    throw new Error(
      `source must contain at least ${byteLength} bytes; got ${source.size}.`,
    );
  }
  const destination = options.destination ?? device.createBuffer({
    size: byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  if (destination === source) {
    throw new Error("destination must differ from source for an exact scan rotation.");
  }
  if (destination.size < byteLength) {
    throw new Error(
      `destination must contain at least ${byteLength} bytes; got ${destination.size}.`,
    );
  }

  const groups = Math.ceil(totalWords / WORKGROUP_SIZE);
  const groupColumns = Math.min(
    groups,
    device.limits.maxComputeWorkgroupsPerDimension,
  );
  const groupRows = Math.ceil(groups / groupColumns);
  if (groupRows > device.limits.maxComputeWorkgroupsPerDimension) {
    throw new Error(
      `Rotated scan needs a ${groupColumns} by ${groupRows} dispatch, which exceeds this WebGPU device limit.`,
    );
  }
  const parameters = new Uint32Array([
    scanRows,
    scanColumns,
    wordsPerScan,
    quarterTurns,
    outputRows,
    outputColumns,
    totalWords,
    groupColumns,
  ]);
  const parameterBuffer = device.createBuffer({
    size: parameters.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(parameterBuffer, 0, parameters);
  const retainedPipeline = await pipeline(device);
  const bindGroup = device.createBindGroup({
    layout: retainedPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: source } },
      { binding: 1, resource: { buffer: destination } },
      { binding: 2, resource: { buffer: parameterBuffer } },
    ],
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(retainedPipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(groupColumns, groupRows);
  pass.end();
  device.queue.submit([encoder.finish()]);
  return { buffer: destination, scanRows: outputRows, scanColumns: outputColumns };
}
