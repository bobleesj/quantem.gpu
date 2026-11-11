/// <reference types="@webgpu/types" />
import { WebGPUDirectBackend, setDirectLocalFiles } from "../../src/quantem/gpu/direct_ptycho/webgpu/backend";

type Case = { C10_A: number; C12_A: number; phi12_deg: number; higherOrder: Record<string, number> };
type Reference = {
  cases: Record<string, Case>;
  order: [string, "k-q" | "q-k"][];
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
sideband. make_qem_browser_fixture.generate_direct_higher_order simulates a weak phase object under a
probe with C21, C23 and C30 and reconstructs it in float64 with either form. Every case (the
probe's own coefficients, none above C12, and the odd ones reversed) must match the k - q
reference to float32 precision; the q - k form is 0.03 rad away.
*/
export async function runDirectHigherOrderSign(base: string): Promise<Record<string, number | string>> {
  const text = async (name: string) => (await fetch(base + name)).text();
  const cal = await text("cal.json");
  const source = await text("source.json");
  const reference = JSON.parse(await text("reference.json")) as Reference;
  const phases = new Float32Array(await (await fetch(base + "reference.f32")).arrayBuffer());
  setDirectLocalFiles([new File([await (await fetch(base + "bf_columns.u16")).arrayBuffer()], "bf_columns.u16")]);
  const numBf = JSON.parse(cal).num_bright_field_pixels as number;
  const plane = phases.length / reference.order.length;
  const expected = (name: string, sign: string) => {
    const index = reference.order.findIndex(([n, s]) => n === name && s === sign);
    return index < 0 ? null : phases.subarray(index * plane, (index + 1) * plane);
  };
  const engine = new WebGPUDirectBackend(cal, { json: source });
  const out: Record<string, number | string> = { numBf };
  try {
    await engine.prepareBfCount(numBf);
    for (const [name, terms] of Object.entries(reference.cases)) {
      const result = await engine.reconstruct(terms.C10_A, terms.C12_A, terms.phi12_deg * Math.PI / 180, {
        numBrightFieldPixels: numBf, computeLoss: false, higherOrder: terms.higherOrder,
      });
      if (result.softwareAdapter || /swiftshader/i.test(result.adapterInfo)) {
        throw new Error(`Direct ptychography ran on a software adapter (${result.adapterInfo}); its numbers are not a GPU result`);
      }
      out.adapter = result.adapterInfo;
      const error = maxPhaseDifference(result.phase, expected(name, "k-q")!);
      out[name] = error;
      const reversedSign = expected(name, "q-k");
      if (reversedSign) out[`${name}|vs q-k`] = maxPhaseDifference(result.phase, reversedSign);
      if (!(error < reference.tolerance_rad)) {
        throw new Error(`${name}: phase differs from the float64 k - q reference by ${error} rad (limit ${reference.tolerance_rad})`
          + (reversedSign ? `, and from the q - k form by ${out[`${name}|vs q-k`]} rad` : ""));
      }
    }
  } finally { engine.close(); }
  return out;
}
