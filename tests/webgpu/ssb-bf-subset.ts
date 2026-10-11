/// <reference types="@webgpu/types" />
import { WebGPUSSBBackend, setSSBLocalFiles } from "../../src/quantem/gpu/ssb/webgpu/backend";

/** A synthetic SSB folder: square n x n scan, a bright-field disk of `radius` detector pixels with
 * uint16 counts, and the calibration fields the browser engine reads. */
function syntheticFolder(n: number, radius: number) {
  const wavelength = 0.019687;   // 300 kV, Angstrom
  const angularSampling = 0.002; // rad per detector pixel
  const kSampling = angularSampling / wavelength;
  const kx: number[] = [], ky: number[] = [];
  for (let row = -radius; row <= radius; row++) {
    for (let col = -radius; col <= radius; col++) {
      if (row * row + col * col <= radius * radius) { kx.push(row * kSampling); ky.push(col * kSampling); }
    }
  }
  const numBf = kx.length, plane = n * n;
  const frequency = Array.from({ length: n }, (_, i) => (i < n / 2 ? i : i - n) / n);
  const cal = {
    num_bf: numBf, g_shape: [numBf, n, n], wavelength_A: wavelength, semiangle_rad: (radius + 0.25) * angularSampling,
    angular_sampling_rad: [angularSampling, angularSampling], dc_value: [plane * 3.0, 0], kx_bf: kx, ky_bf: ky,
    qx_1d: frequency, qy_1d: frequency, aperture_k: kx.map(() => 1), alpha_k2: kx.map(() => 0), cos2phi_k: kx.map(() => 1),
    sin2phi_k: kx.map(() => 0), rotation_angle_deg: 0, precision: { real_dtype: "float32", complex_dtype: "complex64" },
  };
  let state = 12345;
  const counts = Uint16Array.from({ length: numBf * plane }, () => { state = (state * 1103515245 + 12345) >>> 0; return state >>> 28; });
  const source = { kind: "bf_columns", url: "bf_columns.u16", dtype: "uint16", encoding: "uint16", numBf, plane, scanShape: [n, n], bytesPerBf: plane * 2, bitsPerValue: 16 };
  return { cal: JSON.stringify(cal), source: JSON.stringify(source), file: new File([counts], "bf_columns.u16"), numBf };
}

/** Largest wrapped phase difference (radians) between two images. */
function maxPhaseDifference(a: Float32Array, b: Float32Array): number {
  let worst = 0;
  for (let i = 0; i < a.length; i++) worst = Math.max(worst, Math.abs(Math.atan2(Math.sin(a[i] - b[i]), Math.cos(a[i] - b[i]))));
  return worst;
}

/** Lowering the bright-field count after a full-BF load must give the image of that BF subset.

The full load keeps G(q,k) of every BF pixel resident in chunks of 256; a reduced count reuses
them. The result must equal an engine prepared with only that subset, for the drag (phase of the
mean) and the committed (mean phase with loss) estimator, up to float32 summation order.
*/
export async function runSsbBfSubsetParity(): Promise<Record<string, number>> {
  const folder = syntheticFolder(128, 12);
  setSSBLocalFiles([folder.file]);
  const aberrations = [-120.0, 35.0, 0.6] as const;
  const full = new WebGPUSSBBackend(folder.cal, { kind: "bf_columns", json: folder.source });
  const out: Record<string, number> = { numBf: folder.numBf };
  try {
    await full.prepareBfCount(folder.numBf);
    await full.reconstruct(...aberrations, { bfCount: folder.numBf });
    for (const count of [100, 300]) {
      const subset = new WebGPUSSBBackend(folder.cal, { kind: "bf_columns", json: folder.source });
      try {
        await subset.prepareBfCount(count);
        for (const computeLoss of [false, true]) {
          const reused = await full.reconstruct(...aberrations, { bfCount: count, computeLoss });
          const fresh = await subset.reconstruct(...aberrations, { bfCount: count, computeLoss });
          const difference = maxPhaseDifference(reused.phase, fresh.phase);
          const label = `bf${count}${computeLoss ? "-loss" : ""}`;
          out[label] = difference;
          if (!(difference < 1e-6)) throw new Error(`${label}: phase after a full-BF load differs from a ${count}-BF engine by ${difference} rad`);
          if (computeLoss) {
            const lossDifference = Math.abs(reused.loss! - fresh.loss!) / Math.abs(fresh.loss!);
            out[`${label}-relative`] = lossDifference;
            if (!(lossDifference < 1e-7)) throw new Error(`${label}: loss ${reused.loss} != ${fresh.loss}`);
          }
        }
      } finally { subset.close(); }
    }
    const back = await full.reconstruct(...aberrations, { bfCount: folder.numBf, computeLoss: true });
    const reference = new WebGPUSSBBackend(folder.cal, { kind: "bf_columns", json: folder.source });
    try {
      await reference.prepareBfCount(folder.numBf);
      const again = await reference.reconstruct(...aberrations, { bfCount: folder.numBf, computeLoss: true });
      out.fullAgain = maxPhaseDifference(back.phase, again.phase);
      if (out.fullAgain !== 0 || back.loss !== again.loss) throw new Error("returning to the full BF count changed the image");
    } finally { reference.close(); }
  } finally { full.close(); }
  return out;
}
