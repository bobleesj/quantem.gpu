/// <reference types="@webgpu/types" />
import { RansResidentSet } from "../../src/quantem/gpu/detector/compute/webgpu/rans";

/** Exact quantitative snapshots from a public-encoder saturated uint16 source. */
export async function runRansIntegerReadbackParity(device: GPUDevice, fixtureURL: string): Promise<Record<string, boolean>> {
  const response = await fetch(fixtureURL);
  if (!response.ok) throw new Error(`Fixture fetch failed: ${response.status}`);
  const file = new File([await response.arrayBuffer()], "saturated.ans");
  device.pushErrorScope("validation");
  try {
    const source = await RansResidentSet.loadCountANS(device, file);
    let displays: GPUBuffer[] = [];
    try {
      if (source.scanCount !== 1 || source.K !== 65536) throw new Error("Expected one 256×256 saturated uint16 pattern");
      const mask = new Uint32Array(source.K);
      const select = (n: number) => { mask.fill(0); mask.fill(1, 0, n); source.computes[0].update(mask); };
      const check = async (n: number) => {
        const expected = n * 65535;
        const exact = await source.readImageU32(0);
        const preview = await source.readImage(0);
        if (exact[0] !== expected) throw new Error(`Integer sum ${exact[0]} != ${expected}`);
        if (preview[0] !== Math.fround(expected)) throw new Error("Float32 compatibility changed");
        if ((n === 257 || n === 65535) && preview[0] === exact[0]) throw new Error("Fixture did not exercise float32 precision loss");
      };
      select(257);
      const firstSnapshot = source.readImageU32(0);
      select(65535);
      if ((await firstSnapshot)[0] !== 257 * 65535) throw new Error("Readback did not snapshot GPU queue order");
      await check(65535);
      displays = source.imageBuffersF32([0]);
      source.normalizeDisplayBuffers(displays, 65535);
      await check(65535);
      for (const n of [65536, 257, 0]) { select(n); await check(n); }
      for (const tilt of [-1, 1, 0.5, NaN]) {
        let rejected = false;
        try { await source.readImageU32(tilt); } catch (error) { rejected = String(error).includes("Acquisition index"); }
        if (!rejected) throw new Error(`Invalid acquisition ${tilt} accepted`);
      }
      return { high_counts_exact: true, near_uint32_limit_exact: true, queue_snapshot_exact: true, normalized_display_preserves_counts: true, float_compatibility: true, invalid_acquisitions_rejected: true, all_passed: true };
    } finally { displays.forEach(buffer => buffer.destroy()); source.dispose(); }
  } finally {
    const validation = await device.popErrorScope();
    if (validation) throw new Error(validation.message);
  }
}
