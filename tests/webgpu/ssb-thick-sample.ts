// CPU reference for the WebGPU SSB thick-sample correction (backend.ts gamma_mul with params.thickness > 0).
//
// Mirrors the WGSL line by line in float64 so it can be compared with the CUDA reference kernel
// (backends/cuda/engine.py _thick_correct_kernel) on the same (q, k) grid. run_ssb_thick_sample.py writes the
// fixture with CuPy and runs this file with Node (type stripping):
//
//   node tests/webgpu/ssb-thick-sample.ts <fixture.json>
//
// The fixture holds G, qx, qy, kx, ky per point, the scalar parameters, and the CUDA output. Exit code 1 on mismatch.

import { readFileSync } from "node:fs";

export interface ThickParams {
  wavelength: number; semiangle_rad: number; ang_y_rad: number; ang_x_rad: number;
  C10: number; C12: number; cos2phi12: number; sin2phi12: number; factor: number;
  thickness: number; tilt_row_rad: number; tilt_col_rad: number;
}

function geometry(p: ThickParams, dx: number, dy: number): [number, number, number, number] {
  const r2 = dx * dx + dy * dy;
  const r = Math.sqrt(r2);
  const alpha = r * p.wavelength;
  const invR2 = r2 > 1e-30 ? 1 / r2 : 0;
  const denomNum2 = (dx * p.ang_y_rad) ** 2 + (dy * p.ang_x_rad) ** 2;
  const invR = r > 1e-15 ? 1 / r : 0;
  const denom = Math.sqrt(denomNum2) * invR;
  const edge = denom > 1e-15 ? (p.semiangle_rad - alpha) / denom + 0.5 : 1;
  return [alpha * alpha, (dx * dx - dy * dy) * invR2, 2 * dx * dy * invR2, Math.min(1, Math.max(0, edge))];
}

/** Depth-average weights (w1, w2) of the two SSB terms: sinc(rate t / 2), rate from the defocus difference and the tilt. */
export function thickWeights(p: ThickParams, qx: number, qy: number, kx: number, ky: number): [number, number] {
  if (!(p.thickness > 0)) return [1, 1];
  const sinc = (x: number) => (Math.abs(x) < 1e-6 ? 1 : Math.sin(x) / x);
  const alphaK2 = geometry(p, kx, ky)[0];
  const alphaM2 = geometry(p, qx - kx, qy - ky)[0];
  const alphaP2 = geometry(p, qx + kx, qy + ky)[0];
  const shift = 2 * Math.PI * (qx * p.tilt_row_rad + qy * p.tilt_col_rad);
  const rate1 = -p.factor * (alphaM2 - alphaK2) - shift;
  const rate2 = p.factor * (alphaP2 - alphaK2) - shift;
  return [sinc(0.5 * rate1 * p.thickness), sinc(0.5 * rate2 * p.thickness)];
}

/** corrected = G conj(gamma / |gamma|), gamma = w1 P(q-k) conj P(k) - w2 conj P(q+k) P(k) (C10/C12 only). */
export function thickCorrect(p: ThickParams, G: [number, number], qx: number, qy: number, kx: number, ky: number): [number, number] {
  const chi = (g: [number, number, number, number]) => p.factor * g[0] * (p.C12 * (g[1] * p.cos2phi12 + g[2] * p.sin2phi12) + p.C10);
  const gk = geometry(p, kx, ky), gm = geometry(p, qx - kx, qy - ky), gp = geometry(p, qx + kx, qy + ky);
  const [w1, w2] = thickWeights(p, qx, qy, kx, ky);
  const a1 = w1 * gm[3] * gk[3], a2 = w2 * gp[3] * gk[3];
  const d1 = chi(gm) - chi(gk), d2 = chi(gp) - chi(gk);
  let re = a1 * Math.cos(d1) - a2 * Math.cos(d2);
  let im = -a1 * Math.sin(d1) - a2 * Math.sin(d2);
  const magSq = re * re + im * im;
  const inv = magSq > 1e-16 ? 1 / Math.sqrt(magSq) : 1e8;
  re *= inv; im *= inv;
  return [G[0] * re + G[1] * im, G[1] * re - G[0] * im];
}

interface Fixture {
  params: ThickParams;
  G: [number, number][]; qx: number[]; qy: number[]; kx: number[]; ky: number[];
  cuda: [number, number][];
}

function main(path: string): void {
  const fx = JSON.parse(readFileSync(path, "utf8")) as Fixture;
  let maxAbs = 0, maxRel = 0, worst = -1;
  const rels: number[] = [];
  for (let i = 0; i < fx.qx.length; i++) {
    const [re, im] = thickCorrect(fx.params, fx.G[i], fx.qx[i], fx.qy[i], fx.kx[i], fx.ky[i]);
    const [cre, cim] = fx.cuda[i];
    const diff = Math.hypot(re - cre, im - cim);
    const scale = Math.hypot(fx.G[i][0], fx.G[i][1]);
    if (diff > maxAbs) { maxAbs = diff; worst = i; }
    rels.push(diff / Math.max(scale, 1e-30));
    maxRel = Math.max(maxRel, rels[rels.length - 1]);
  }
  let minW = 1, maxW = 0;
  for (let i = 0; i < fx.qx.length; i++) {
    const [w1, w2] = thickWeights(fx.params, fx.qx[i], fx.qy[i], fx.kx[i], fx.ky[i]);
    minW = Math.min(minW, w1, w2); maxW = Math.max(maxW, w1, w2);
  }
  rels.sort((a, b) => a - b);
  const medianRel = rels[Math.floor(rels.length / 2)];
  // CUDA evaluates chi in float32 with fast __sincosf (phases up to ~50 rad at |q| = 2 / A), so the float32 floor at
  // thickness 0 is ~1e-4 relative; the depth weights must not add more than that order.
  const ok = maxRel < 1e-3 && medianRel < 1e-5;
  console.log(JSON.stringify({ points: fx.qx.length, thickness: fx.params.thickness, maxAbs, maxRel, medianRel, worst, weightRange: [minW, maxW], ok }));
  if (!ok) process.exit(1);
}

if (process.argv[2]) main(process.argv[2]);
