/// <reference types="@webgpu/types" />
import { WebGPUDirectPtychoBackend, setSSBLocalFiles } from "../../src/quantem/gpu/direct_ptycho/webgpu/backend";

type Case = { C10_A: number; C12_A: number; phi12_deg: number; higherOrder: Record<string, number> };
type Reference = {
  cases: Record<string, Case>;
  order: [string, "phase_of_mean" | "mean_phase", "k-q" | "q-k"][];
  tolerance_rad: number;
};

/** Largest wrapped phase difference (radians) between two images. */
function maxPhaseDifference(a: Float32Array, b: Float32Array): number {
  let worst = 0;
  for (let i = 0; i < a.length; i++) worst = Math.max(worst, Math.abs(Math.atan2(Math.sin(a[i] - b[i]), Math.cos(a[i] - b[i]))));
  return worst;
}

/** Coma and trefoil must be corrected with the probe phase of the first sideband at k - q.

For a thin object G(q, k) ~ P(k - q) conj P(k) - P(k) conj P(k + q). The WGSL once evaluated the
first term at q - k, which equals k - q for C10, C12 and C30 but reverses the odd orders on that
sideband. make_qem_browser_fixture.generate_ssb_higher_order simulates a weak phase object under a
probe with C21, C23 and C30 and reconstructs it in float64 with either form. Every estimator and
case (the probe's own coefficients, none above C12, and the odd ones reversed) must match the k - q
reference to float32 precision; the q - k form is 0.03 rad away.
*/
export async function runSsbHigherOrderSign(base: string): Promise<Record<string, number | string>> {
  const text = async (name: string) => (await fetch(base + name)).text();
  const cal = await text("cal.json");
  const source = await text("source.json");
  const reference = JSON.parse(await text("reference.json")) as Reference;
  const phases = new Float32Array(await (await fetch(base + "reference.f32")).arrayBuffer());
  setSSBLocalFiles([new File([await (await fetch(base + "bf_columns.u16")).arrayBuffer()], "bf_columns.u16")]);
  const numBf = JSON.parse(cal).num_bright_field_pixels as number;
  const plane = phases.length / reference.order.length;
  const expected = (name: string, estimator: string, sign: string) => {
    const index = reference.order.findIndex(([n, e, s]) => n === name && e === estimator && s === sign);
    return index < 0 ? null : phases.subarray(index * plane, (index + 1) * plane);
  };
  const engine = new WebGPUDirectPtychoBackend(cal, { json: source });
  const out: Record<string, number | string> = { numBf };
  try {
    await engine.prepareBfCount(numBf);
    for (const [name, terms] of Object.entries(reference.cases)) {
      for (const estimator of ["phase_of_mean", "mean_phase"] as const) {
        const result = await engine.reconstruct(terms.C10_A, terms.C12_A, terms.phi12_deg * Math.PI / 180, {
          numBrightFieldPixels: numBf, phaseEstimator: estimator, computeLoss: false, higherOrder: terms.higherOrder,
        });
        if (result.softwareAdapter || /swiftshader/i.test(result.adapterInfo)) {
          throw new Error(`SSB ran on a software adapter (${result.adapterInfo}); its numbers are not a GPU result`);
        }
        out.adapter = result.adapterInfo;
        const label = `${name}|${estimator}`;
        const error = maxPhaseDifference(result.phase, expected(name, estimator, "k-q")!);
        out[label] = error;
        const reversedSign = expected(name, estimator, "q-k");
        if (reversedSign) out[`${label}|vs q-k`] = maxPhaseDifference(result.phase, reversedSign);
        if (!(error < reference.tolerance_rad)) {
          throw new Error(`${label}: phase differs from the float64 k - q reference by ${error} rad (limit ${reference.tolerance_rad})`
            + (reversedSign ? `, and from the q - k form by ${out[`${label}|vs q-k`]} rad` : ""));
        }
      }
    }
  } finally { engine.close(); }
  return out;
}
