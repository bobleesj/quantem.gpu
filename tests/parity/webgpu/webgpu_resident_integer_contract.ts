/** Real compact-v1 WebGPU runner over the shared frozen uint16 contract.
 * Bundle this module, then pass the two compact files prepared by the host test.
 * Importing or compiling it alone does not execute or qualify a WebGPU device.
 */
import fixture from "../fixtures/resident_integer_products_v1.json";
import {
  loadCompactH5WebGPU,
  parseCompactH5Index,
  type CompactH5Index,
  type WebGPUCompactH5ResidentSource,
} from "../../../src/quantem/gpu/io/backends/webgpu/compact-h5";

export const unsupportedAdapterCases = {
  denseExactIntegerProduct: "DetectorCompute.maskedSum returns float32, not an exact integer product.",
  duplicateSelectedFrameSum: "Compact reduceFrames takes a binary scan mask and returns float32; it cannot preserve duplicate selections.",
  radiusConvenienceParity: "This adapter supplies frozen mask bytes, not a shared radius-edge convention.",
} as const;

function equal(actual: ArrayLike<number>, expected: ArrayLike<number>, label: string): void {
  if (actual.length !== expected.length) throw new Error(`${label}: wrong element count`);
  for (let index = 0; index < expected.length; index++) {
    if (actual[index] !== expected[index]) {
      throw new Error(`${label}[${index}]: ${actual[index]} != ${expected[index]}`);
    }
  }
}

async function rejects(operation: () => unknown | Promise<unknown>, label: string): Promise<void> {
  try { await operation(); } catch { return; }
  throw new Error(`${label}: expected rejection, but the operation succeeded`);
}

/** Host-callable admission check only; this is not a compute conformance result. */
export function checkFixtureMetadata(metadata: CompactH5Index, large = false): void {
  equal(metadata.shape, large ? fixture.large_sum.shape : fixture.shape, "working shape");
  if (metadata.schemaVersion !== 1 || metadata.workingDtype !== "uint16") {
    throw new Error("Frozen full-range uint16 cases require compact-v1 uint16, without narrowing.");
  }
  equal(metadata.excludedDetectorPixels,
    large ? [] : fixture.excluded_detector_flat_indices, "declared exclusions");
}

/** Parse the exact test sources without acquiring a device or running a kernel. */
export async function inspectFixtureSources(source: File, largeSource: File) {
  const metadata = await parseCompactH5Index(source);
  const largeMetadata = await parseCompactH5Index(largeSource);
  checkFixtureMetadata(metadata);
  checkFixtureMetadata(largeMetadata, true);
  return { shape: metadata.shape, largeShape: largeMetadata.shape,
    workingDtype: metadata.workingDtype, unsupportedAdapterCases };
}

async function verifyMasksAndFrames(resident: WebGPUCompactH5ResidentSource): Promise<void> {
  checkFixtureMetadata(resident.metadata);
  const masks = new Map(fixture.detector_masks.map(request => [request.id, request]));
  for (const name of fixture.request_order) {
    const request = masks.get(name)!;
    const metrics = await resident.updateVirtualDetector(Uint32Array.from(request.mask_u8));
    if (metrics.fftDispatchCount !== 0) throw new Error("Detector-only requests must not compute FFT.");
    const values = await resident.virtualDetectorValues();
    if (!(values instanceof Uint32Array)) throw new Error("Detector sums were converted to display floats.");
    equal(values, request.expected_sum_u64, name);
    for (let index = 0; index < fixture.selected_scan_row_columns.length; index++) {
      const [row, column] = fixture.selected_scan_row_columns[index];
      const selected = await resident.extractDiffraction(row, column);
      if (!(selected instanceof Uint32Array)) throw new Error("Selected DP must remain integer-valued.");
      equal(selected, fixture.expected_selected_frames_u16[index], `selected ${index}`);
      selected.fill(0);
      equal(await resident.extractDiffraction(row, column),
        fixture.expected_selected_frames_u16[index], "independent selected output");
    }
  }
  const full = masks.get("full")!;
  const recover = async () => {
    await resident.updateVirtualDetector(Uint32Array.from(full.mask_u8));
    equal(await resident.virtualDetectorValues(), full.expected_sum_u64, "recovery");
  };
  for (const [row, column] of fixture.invalid_selected_scan_row_columns) {
    await rejects(() => resident.extractDiffraction(row, column), "out-of-range selected DP");
    await recover();
  }
  for (const badMask of [new Uint32Array(1), new Uint32Array(full.mask_u8.length).fill(2)]) {
    await rejects(() => resident.updateVirtualDetector(badMask), "malformed detector mask");
    await recover();
  }
  // A rebase to an empty detector must clear the previously nonzero result.
  const empty = masks.get("empty")!;
  await resident.rebaseVirtualDetector(Uint32Array.from(empty.mask_u8));
  equal(await resident.virtualDetectorValues(), empty.expected_sum_u64, "empty rebase");
  await recover();
}

/** Load exact inputs, exercise real GPU operations, release, then reject stale use. */
export async function verifyWebGPUResidentIntegerContract(
  source: File, largeSource: File, device?: GPUDevice,
) {
  const resident = await loadCompactH5WebGPU(source, { device, integrity: "decoded-sha256" });
  try {
    await verifyMasksAndFrames(resident);
  } finally {
    await resident.quiesce();
    resident.releaseResidentStorage();
  }
  resident.releaseResidentStorage();
  await rejects(() => resident.extractDiffraction(0, 0), "selected DP after release");
  await rejects(() => resident.virtualDetectorValues(), "detector result after release");

  const large = await loadCompactH5WebGPU(largeSource, { device, integrity: "decoded-sha256" });
  try {
    checkFixtureMetadata(large.metadata, true);
    await large.updateVirtualDetector(new Uint32Array(large.detSize).fill(1));
    const values = await large.virtualDetectorValues();
    if (!(values instanceof Uint32Array)) throw new Error("Large sum must remain integer-valued.");
    equal(values, fixture.large_sum.expected_full_sum_u64, "full uint16 sum above float32 range");
  } finally {
    await large.quiesce();
    large.releaseResidentStorage();
  }
  return { contract: fixture.schema, representation: "compact-v1-uint16",
    unsupportedAdapterCases };
}
