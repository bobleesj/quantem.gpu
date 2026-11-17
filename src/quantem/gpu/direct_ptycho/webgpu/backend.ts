/// <reference types="@webgpu/types" />

import { getGPUDevice, getGPUInfo, isSoftwareGPUAdapter } from "../../device/webgpu";
import {
  getWebGPUFFTConfig,
  SUPPORTED_DIRECT_SIZES,
  type SupportedDirectSize,
} from "./kernels";
import type {
  DirectOptimizationResult,
  WebGPUOptimizationOptions,
  WebGPUReconstructionOptions,
  WebGPUSampleOptions,
  DirectProtocol,
} from "./protocol";
import {phaseLayout, phaseNormalShader, solvePhase} from "./kernels/phase";
import { agreementGradient } from "./kernels/agreement";
import { fit } from "./optimizer";
import { RansResidentSet } from "../../detector/webgpu/rans";
import { qemHttpFile, type QemByteFile } from "../../detector/webgpu/qem-source";

const MAX_BF_WORKGROUPS_PER_SUBMIT = 256;
const MAX_GQK_CHUNK_BYTES = 1024 * 1024 * 1024;
// Resident G(q,k) budget. Caps how many BF pixels the session keeps resident
// so a full-BF drag on a big scan cannot device-lost a small GPU; the export
// raises it with cal.json gpu_memory_gb.
const GQK_GPU_BUDGET_BYTES = 4.5 * 1024 * 1024 * 1024;

function gqkBudgetBytes(cal: DirectCal): number {
  const raw = cal.gpu_memory_gb;
  if (Number.isFinite(raw) && Number(raw) > 0) return Number(raw) * 1024 * 1024 * 1024;
  return GQK_GPU_BUDGET_BYTES;
}
const REDUCE_BF_GROUP = 32;
const BF_COLUMN_UNPACK_WORKGROUP_X = 32;
const BF_COLUMN_UNPACK_WORKGROUP_Y = 8;

function supportedDirectSize(n: number): SupportedDirectSize | null {
  return getWebGPUFFTConfig(n)?.size ?? null;
}

// The Hermitian half-plane is lossless for the FFT of real detector counts and
// halves resident G(q,k) storage without changing float32 precision.
type GqkMode = "herm";
const GQK_MODE: GqkMode = "herm";
const GQK_BYTES_PER_VALUE = 8;   // complex64

function storedPlaneFor(n: number): number {
  return n * (n / 2 + 1);
}

// ---------------------------------------------------------------------------
// No-server local folder source. When the viewer is opened via file:// the
// browser cannot fetch() sibling files, so the user grants the folder once
// (File System Access picker, or a webkitdirectory input) and every read
// below routes through File.slice() instead of HTTP. Range reads stay
// byte-for-byte identical to the Range-server path.
// ---------------------------------------------------------------------------
type LocalFileResolver = (path: string) => Promise<File>;
let localSourceResolver: LocalFileResolver | null = null;

function normaliseSourcePath(path: string): string {
  return path.replace(/^[.][/]/, "").replace(/^[/]+/, "");
}

let localDirHandle: FileSystemDirectoryHandle | null = null;

export function setDirectLocalDirectory(handle: FileSystemDirectoryHandle): void {
  localDirHandle = handle;
  localSourceResolver = async (path: string) => {
    const parts = normaliseSourcePath(path).split("/").filter(Boolean);
    let dir: FileSystemDirectoryHandle = handle;
    for (const part of parts.slice(0, -1)) dir = await dir.getDirectoryHandle(part);
    const fileHandle = await dir.getFileHandle(parts[parts.length - 1]);
    return fileHandle.getFile();
  };
}

export function setDirectLocalFiles(files: ArrayLike<File>): void {
  const byPath = new Map<string, File>();
  const byName = new Map<string, File>();
  for (let i = 0; i < files.length; i++) {
    const file = files[i] as File & { webkitRelativePath?: string };
    const rel = String(file.webkitRelativePath || "");
    if (rel) {
      // webkitdirectory prefixes the picked folder name; strip it so paths
      // match the manifest-relative urls ("source/bf_columns.u16").
      byPath.set(rel.split("/").slice(1).join("/") || rel, file);
      byPath.set(rel, file);
    }
    byName.set(file.name, file);
  }
  localSourceResolver = async (path: string) => {
    const clean = normaliseSourcePath(path);
    const hit = byPath.get(clean) || byName.get(clean.split("/").pop() || clean);
    if (!hit) throw new Error(`local folder is missing ${clean}`);
    return hit;
  };
}

export function directNeedsLocalSource(): boolean {
  try {
    return globalThis.location?.protocol === "file:" && !localSourceResolver;
  } catch {
    return false;
  }
}

async function openQemFile(path: string): Promise<QemByteFile> {
  if (localSourceResolver) return localSourceResolver(path);
  return qemHttpFile(path);
}

async function readSourceBytes(url: string, byteOffset?: number, byteLength?: number): Promise<Uint8Array> {
  if (localSourceResolver) {
    const file = await localSourceResolver(url);
    const blob = byteOffset != null && byteLength != null
      ? file.slice(byteOffset, byteOffset + byteLength)
      : file;
    return new Uint8Array(await blob.arrayBuffer());
  }
  if (byteOffset != null && byteLength != null) {
    const res = await fetch(url, {
      headers: { Range: `bytes=${byteOffset}-${Math.max(byteOffset, byteOffset + byteLength - 1)}` },
    });
    if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
    return new Uint8Array(await res.arrayBuffer());
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return new Uint8Array(await res.arrayBuffer());
}

// ---------------------------------------------------------------------------
// Folder snapshot write-back. Saved review states (JPEG + snapshots.json) persist
// INSIDE the export folder so they survive relaunch from both the no-server
// double-click path (via the readwrite directory handle) and the CLI-served
// path (via PUT/DELETE handled by the exported project server). Writes are
// restricted to the snapshots/ subfolder on both transports.
// ---------------------------------------------------------------------------
export function directFolderWritable(): boolean {
  if (localDirHandle) return true;
  try {
    const proto = globalThis.location?.protocol || "";
    return proto === "http:" || proto === "https:";
  } catch {
    return false;
  }
}

function validateSnapshotWritePath(path: string, operation: "write" | "delete"): string {
  if (path === "snapshots/snapshots.json") {
    if (operation === "delete") throw new Error("folder deletes cannot remove snapshots/snapshots.json");
    return path.slice("snapshots/".length);
  }
  const name = path.slice("snapshots/".length);
  const isSnapshotImage = path.startsWith("snapshots/snapshot_")
    && !name.includes("/")
    && (name.endsWith(".jpg") || name.endsWith(".jpeg"));
  if (!isSnapshotImage) {
    throw new Error("folder writes are restricted to snapshots/snapshots.json and snapshots/snapshot_*.jpg");
  }
  return name;
}

async function snapshotsDirHandle(): Promise<FileSystemDirectoryHandle> {
  if (!localDirHandle) throw new Error("no local directory handle");
  return localDirHandle.getDirectoryHandle("snapshots", { create: true });
}

export async function writeDirectFolderFile(path: string, body: Blob | string): Promise<void> {
  const clean = normaliseSourcePath(path);
  const name = validateSnapshotWritePath(clean, "write");
  if (localDirHandle) {
    const dir = await snapshotsDirHandle();
    const handle = await dir.getFileHandle(name, { create: true });
    const writable = await handle.createWritable();
    await writable.write(body);
    await writable.close();
    return;
  }
  const res = await fetch(clean, { method: "PUT", body });
  if (!res.ok) throw new Error(`${clean}: HTTP ${res.status} (open this project with its .command launcher or the QuantEM CLI to enable snapshot writes)`);
}

export async function deleteDirectFolderFile(path: string): Promise<void> {
  const clean = normaliseSourcePath(path);
  const name = validateSnapshotWritePath(clean, "delete");
  if (localDirHandle) {
    const dir = await snapshotsDirHandle();
    await dir.removeEntry(name);
    return;
  }
  const res = await fetch(clean, { method: "DELETE" });
  if (!res.ok) throw new Error(`${clean}: HTTP ${res.status}`);
}

export async function readDirectFolderJson<T>(path: string): Promise<T | null> {
  try {
    const bytes = await readSourceBytes(path);
    return JSON.parse(new TextDecoder().decode(bytes)) as T;
  } catch {
    return null;
  }
}

export async function readDirectFolderBytes(path: string): Promise<Uint8Array> {
  return readSourceBytes(path);
}

/** The layout of a ShowPtycho folder's `snapshots/cal.json`, which this engine reads as written. */
export const BROWSER_CALIBRATION_SCHEMA = "quantem.direct-browser-calibration/4";

/** `snapshots/cal.json`: optics in kV, mrad and Angstrom; the bright-field reciprocal coordinates in the scan frame. */
type BrowserCalibration = {
  schema: string;
  gpu_memory_gb?: number;
  num_bright_field_pixels: number;
  g_shape: [number, number, number];
  wavelength_A: number;
  semiangle_mrad: number;
  detector_sampling_mrad: [number, number];
  dc_value: [number, number];
  detector_shape?: [number, number];
  bright_field_rows?: number[];
  bright_field_cols?: number[];
  bright_field_k_row: number[];
  bright_field_k_col: number[];
  scan_q_row: number[];
  scan_q_col: number[];
  aperture_k: number[];
  alpha_k2: number[];
  cos2phi_k: number[];
  sin2phi_k: number[];
  scan_detector_rotation_deg?: number;
  /** Detector row and column axes swapped relative to the scan. */
  scan_detector_rotation_flipped?: boolean;
  /** Display-only phase negation; never part of the reconstruction. */
  display_negate_phase?: boolean;
  precision: {
    real_dtype: "float32";
    complex_dtype: "complex64";
  };
};

/** The calibration with its angles in the radians the kernels evaluate in, converted once at entry. */
type DirectCal = BrowserCalibration & {
  semiangle_rad: number;
  angular_sampling_rad: [number, number];
};

/** The bright-field columns a ShowPtycho folder declares (`source.bf_columns` of its manifest). */
type BfColumnDirectSource = {
  kind: "bf_columns";
  url: string;
  dtype?: "uint8" | "uint16";
  encoding?: "uint8" | "uint16" | "qem";
  num_bright_field_pixels: number;
  plane: number;
  scan_shape?: [number, number];
  bytes_per_bright_field_pixel: number;
  bits_per_value?: number;
};

/** An unchanged acquisition; BF coordinates refer to the calibrated detector crop. */
type FullDetectorDirectSource = Omit<BfColumnDirectSource, "kind" | "encoding" | "scan_shape"> & {
  kind: "full_detector";
  encoding: "qem";
  scan_shape: [number, number];
  detector_shape: [number, number];
  detector_origin_px: [number, number];
  scan_detector_rotation_flipped: boolean;
  header_sha256: string;
  region?: [number, number, number, number];
  resident?: RansResidentSet;
};
type DetectorDirectSource = BfColumnDirectSource | FullDetectorDirectSource;

function fullDetectorColumns(cal: DirectCal, source: FullDetectorDirectSource): Uint32Array {
  const shape = source.detector_shape;
  const origin = source.detector_origin_px;
  const crop = cal.detector_shape;
  const rows = cal.bright_field_rows, cols = cal.bright_field_cols;
  const count = cal.num_bright_field_pixels;
  if (source.encoding !== "qem" || !["uint8", "uint16"].includes(source.dtype ?? "")
      || (source.region?.[2] ?? source.scan_shape?.[0]) !== cal.g_shape[1] || (source.region?.[3] ?? source.scan_shape?.[1]) !== cal.g_shape[2]
      || source.num_bright_field_pixels !== count
      || !shape || shape.length !== 2 || shape.some(value => !Number.isSafeInteger(value) || value <= 0)
      || shape[0] * shape[1] > 0xffffffff
      || !origin || origin.length !== 2 || origin.some(value => !Number.isSafeInteger(value) || value < 0)
      || typeof source.scan_detector_rotation_flipped !== "boolean"
      || !/^[0-9a-f]{64}$/.test(source.header_sha256 ?? "")
      || !crop || crop.length !== 2 || crop.some(value => !Number.isSafeInteger(value) || value <= 0)
      || rows?.length !== count || cols?.length !== count) {
    throw new Error("Full-detector QEM needs matching scan geometry, native count dtype and complete BF coordinates; export the review again.");
  }
  const sourceCrop = source.scan_detector_rotation_flipped ? [crop[1], crop[0]] : crop;
  if (sourceCrop.some((value, axis) => value + origin[axis] > shape[axis])) {
    throw new Error("The calibrated detector crop is outside the full QEM; export the review again.");
  }
  const columns = new Uint32Array(count);
  for (let index = 0; index < count; index++) {
    const row = rows[index], col = cols[index];
    if (!Number.isSafeInteger(row) || !Number.isSafeInteger(col)
        || row < 0 || row >= crop[0] || col < 0 || col >= crop[1]) {
      throw new Error("Direct ptychography BF coordinates must lie inside the calibrated detector crop.");
    }
    const sourceRow = origin[0] + (source.scan_detector_rotation_flipped ? col : row);
    const sourceCol = origin[1] + (source.scan_detector_rotation_flipped ? row : col);
    columns[index] = sourceRow * shape[1] + sourceCol;
  }
  if (new Set(columns).size !== count) {
    throw new Error("Direct ptychography BF coordinates contain duplicates; export a complete unique BF selection.");
  }
  return columns;
}

type DirectBuffers = {
  params: GPUBuffer;
  paramsChunks: GPUBuffer[];
  aberrations: GPUBuffer;
  gqkChunks: GPUBuffer[];
  gqkMode: GqkMode;
  gqkResidentBytes: number;
  chunkBfCounts: number[];
  chunkCapacity: number;
  stage: GPUBuffer;
  phaseStack: GPUBuffer;
  partialSum: GPUBuffer;
  partialSumSq: GPUBuffer;
  partialGroups: number;
  activeBfCount: number;
  nonzeroBfCount: number;
  activeSourceIndices: Uint32Array;
  phase: GPUBuffer;
  variance: GPUBuffer;
  bfGeom: GPUBuffer;
  bfTrig: GPUBuffer;
  qx: GPUBuffer;
  qy: GPUBuffer;
};

type DirectPipelines = {
  rows: GPUComputePipeline;
  cols: GPUComputePipeline;
  reducePartial: GPUComputePipeline;
  finalizeGroups: GPUComputePipeline;
  objSum: GPUComputePipeline;
  objFftRows: GPUComputePipeline;
  objFftCols: GPUComputePipeline;
  // thick-sample variants of the two passes that apply the direct ptychography correction (makeDirectShader(n, true))
  rowsThick: GPUComputePipeline;
  objSumThick: GPUComputePipeline;
};

type DirectBindGroups = {
  rows: GPUBindGroup[];
  cols: GPUBindGroup[];
  reducePartial: GPUBindGroup[];
  finalizeGroups: GPUBindGroup;
  objSum: GPUBindGroup[];
  objFftRows: GPUBindGroup;
  objFftCols: GPUBindGroup;
  rowsThick: GPUBindGroup[];
  objSumThick: GPUBindGroup[];
};

export type WebGPUDirectResult = {
  phase: Float32Array;
  width: number;
  height: number;
  gpuMs: number;
  numBrightFieldPixels: number;
  scanDetectorRotationDeg: number;
  loss: number | null;
  adapterInfo: string;
  softwareAdapter: boolean;
};

export type WebGPULoadProgress = {
  stage: "idle" | "device" | "pipeline" | "metadata" | "fetch" | "parse" | "decode" | "gather" | "fft" | "ready" | "error";
  message: string;
  detail?: string;
  current?: number;
  total?: number;
  percent?: number;
  activeBf?: number;
  totalBf?: number;
  sourceFrames?: number;
  elapsedMs?: number;
};

type WebGPUProgressHandler = (progress: WebGPULoadProgress) => void;

function formatF32(value: number): string {
  if (Math.abs(value) < 1e-12) return "0.0";
  if (Math.abs(value - 1) < 1e-12) return "1.0";
  if (Math.abs(value + 1) < 1e-12) return "-1.0";
  return value.toPrecision(9);
}

function makeFftConstants(n: number): string {
  const twiddles: string[] = [];
  for (let len = 2; len <= n; len *= 2) {
    const half = len / 2;
    for (let j = 0; j < half; j++) {
      const angle = 2 * Math.PI * j / len;
      twiddles.push(`vec2<f32>(${formatF32(Math.cos(angle))}, ${formatF32(Math.sin(angle))})`);
    }
  }
  const bitrev: number[] = [];
  const bits = Math.log2(n);
  for (let x = 0; x < n; x++) {
    let y = 0;
    for (let bit = 0; bit < bits; bit++) y = (y << 1) | ((x >> bit) & 1);
    bitrev.push(y);
  }
  return `
const FFT_TWIDDLES: array<vec2<f32>, ${twiddles.length}> = array<vec2<f32>, ${twiddles.length}>(
  ${twiddles.join(",\n  ")}
);
const FFT_BITREV: array<u32, ${n}> = array<u32, ${n}>(
  ${bitrev.map(value => `${value}u`).join(", ")}
);`;
}

function makeFftStages(n: number): string {
  const workgroupSize = Math.min(n, 256);
  const lines: string[] = ["  workgroupBarrier();"];
  for (let len = 2; len <= n; len *= 2) {
    const half = len / 2;
    const twiddleOffset = half - 1;
    lines.push(
      `  for (var butterfly = tid; butterfly < ${n / 2}u; butterfly = butterfly + ${workgroupSize}u) {`,
      `    let group = butterfly / ${half}u;`,
      `    let j = butterfly - group * ${half}u;`,
      `    let i0 = group * ${len}u + j;`,
      `    let i1 = i0 + ${half}u;`,
      `    let w = FFT_TWIDDLES[${twiddleOffset}u + j];`,
      "    let t = cmul(w, s[i1]);",
      `    let u = s[i0];`,
      `    s[i0] = u + t;`,
      `    s[i1] = u - t;`,
      "  }",
      "  workgroupBarrier();",
    );
  }
  return lines.join("\n");
}

// Thick-sample WGSL pieces. Compiled into a separate shader variant (makeDirectShader(n, true)) so the standard (thin) direct ptychography shader text,
// and therefore its compiled code and results, stay exactly as before; only directRows and directObjSum differ between variants.
const THICK_PARAMS_FIELDS = `
  thickness: f32,
  tilt_row_rad: f32,
  tilt_col_rad: f32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,`;
const THICK_GAMMA_SIGNATURE = "fn transfer(qx: f32, qy: f32, kx: f32, ky: f32, alpha_k2: f32, pk: vec2<f32>) -> vec2<f32> {";
const THIN_GAMMA_SIGNATURE = "fn transfer(qx: f32, qy: f32, kx: f32, ky: f32, pk: vec2<f32>) -> vec2<f32> {";
// Thick, tilted crystal (CUDA _thick_correct_kernel): each direct ptychography term is averaged over the sample depth t. Term 1 dephases at
// rate1 = -factor (alpha_m^2 - alpha_k^2) - 2 pi q.theta and term 2 at rate2 = +factor (alpha_p^2 - alpha_k^2) - 2 pi q.theta, so
// the depth average multiplies them by sinc(rate t / 2). Without it the two overlap terms of a thick sample partly cancel at
// frequencies where free-space propagation differs across the depth. Thickness 0 gives weights exactly 1.
const THICK_TERMS = `  var t1 = cmul(pm, pk_conj);
  var t2 = cmul(pp_conj, pk);
  if (params.thickness > 0.0) {
    let shift = 6.283185307 * (qx * params.tilt_row_rad + qy * params.tilt_col_rad);
    let rate1 = -params.factor * (m.x - alpha_k2) - shift;
    let rate2 = params.factor * (p.x - alpha_k2) - shift;
    t1 *= thick_sinc(0.5 * rate1 * params.thickness);
    t2 *= thick_sinc(0.5 * rate2 * params.thickness);
  }`;
const THIN_TERMS = `  let t1 = cmul(pm, pk_conj);
  let t2 = cmul(pp_conj, pk);`;

function makeDirectShader(n: number, thick = false, native = n, cacheProbe = false, fitStride = 0, phaseField = false, gateSidebands = false): string {
  const workgroupSize = Math.min(n, 256);
  const half = native / 2;
  const halfW = half + 1;
  const storedPlane = storedPlaneFor(native);
  const gqkDecl = "@group(0) @binding(1) var<storage, read> gqk: array<vec2<f32>>;";
  const fetchBody = `  var r = row % ${native}u;
  var c = x % ${native}u;
  var conj_sign = 1.0;
  if (c > ${half}u) {
    r = select(${native}u - r, 0u, r == 0u);
    c = ${native}u - c;
    conj_sign = -1.0;
  }
  let idx = local_bf * ${storedPlane}u + r * ${halfW}u + c;
  let v = gqk[idx];
  return vec2<f32>(v.x, v.y * conj_sign);`;
  return `
${makeFftConstants(n)}
struct Params {
  num_bf: u32,
  n: u32,
  plane: u32,
  bf_offset: u32,
  wavelength: f32,
  semiangle_rad: f32,
  ang_y_rad: f32,
  ang_x_rad: f32,
  C10: f32,
  C12: f32,
  cos2phi12: f32,
  sin2phi12: f32,
  factor: f32,
  dc_re: f32,
  dc_im: f32,
  inv_n2: f32,
  chunk_bf: u32,
  output_cols: u32,
  partial_groups: u32,
  compute_loss: u32,
  active_bf: u32,
  full_aberration: u32,${thick ? THICK_PARAMS_FIELDS : ""}
};
@group(0) @binding(0) var<uniform> params: Params;
${gqkDecl}
@group(0) @binding(2) var<storage, read_write> stage: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> bfGeom: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> bfTrig: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read> qx1d: array<f32>;
@group(0) @binding(6) var<storage, read> qy1d: array<f32>;
@group(0) @binding(7) var<storage, read_write> phaseStack: array<f32>;
@group(0) @binding(8) var<storage, read_write> partialSum: array<f32>;
@group(0) @binding(9) var<storage, read_write> partialSumSq: array<f32>;
@group(0) @binding(10) var<storage, read_write> phaseOut: array<f32>;
@group(0) @binding(11) var<storage, read_write> varianceOut: array<f32>;
@group(0) @binding(12) var<storage, read> abr: array<vec4<f32>, 14>;
${cacheProbe ? "@group(0) @binding(13) var<storage, read_write> cachedProbe: array<vec2<f32>>;" : ""}

var<workgroup> s: array<vec2<f32>, ${n}>;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> { return vec2<f32>(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x); }
fn fetch_g(local_bf: u32, bf_global: u32, row: u32, x: u32) -> vec2<f32> {
${fetchBody}
}
fn compute_geometry(dx: f32, dy: f32) -> vec4<f32> {
  let dx2 = dx * dx;
  let dy2 = dy * dy;
  let r2 = dx2 + dy2;
  let r = sqrt(r2);
  let alpha = r * params.wavelength;
  let alpha2 = alpha * alpha;
  let inv_r2 = select(0.0, 1.0 / r2, r2 > 1e-30);
  let cos2phi = (dx2 - dy2) * inv_r2;
  let sin2phi = 2.0 * dx * dy * inv_r2;
  let denom_num2 = (dx * params.ang_y_rad) * (dx * params.ang_y_rad) + (dy * params.ang_x_rad) * (dy * params.ang_x_rad);
  let inv_r = select(0.0, 1.0 / r, r > 1e-15);
  let denom = sqrt(denom_num2) * inv_r;
  let edge = select(1.0, (params.semiangle_rad - alpha) / denom + 0.5, denom > 1e-15);
  return vec4<f32>(alpha2, cos2phi, sin2phi, clamp(edge, 0.0, 1.0));
}
${thick ? "fn thick_sinc(x: f32) -> f32 { return select(sin(x) / x, 1.0, abs(x) < 1e-6); }\n" + THICK_GAMMA_SIGNATURE : THIN_GAMMA_SIGNATURE}
  let m = compute_geometry(qx - kx, qy - ky);
  let p = compute_geometry(qx + kx, qy + ky);
  ${cacheProbe ? `// Closed apertures contribute zero; retain the soft edge and BF sum order.
  if (m.w == 0.0 && p.w == 0.0) { return vec2<f32>(0.0); }` : ""}
  var chi_m: f32;
  var chi_p: f32;
  if (params.full_aberration != 0u) {
    // chi(k - q), not chi(q - k): the odd orders (coma, trefoil) change sign between them
    ${gateSidebands ? "if (m.w != 0.0) { chi_m = chi_full(kx - qx, ky - qy); }" : "chi_m = chi_full(kx - qx, ky - qy);"}
    ${gateSidebands ? "if (p.w != 0.0) { chi_p = chi_full(qx + kx, qy + ky); }" : "chi_p = chi_full(qx + kx, qy + ky);"}
  } else {
    let cos_term_m = m.y * params.cos2phi12 + m.z * params.sin2phi12;
    let cos_term_p = p.y * params.cos2phi12 + p.z * params.sin2phi12;
    chi_m = params.factor * m.x * (params.C12 * cos_term_m + params.C10);
    chi_p = params.factor * p.x * (params.C12 * cos_term_p + params.C10);
  }
  let pm = vec2<f32>(m.w * cos(chi_m), -m.w * sin(chi_m));
  let pp = vec2<f32>(p.w * cos(chi_p), -p.w * sin(chi_p));
  let pk_conj = vec2<f32>(pk.x, -pk.y);
  let pp_conj = vec2<f32>(pp.x, -pp.y);
${thick ? THICK_TERMS : THIN_TERMS}
  return t1 - t2;
}
fn gamma_mul(qx: f32, qy: f32, kx: f32, ky: f32, ${thick ? "alpha_k2: f32, " : ""}pk: vec2<f32>, G: vec2<f32>) -> vec2<f32> {
  var gg = transfer(qx, qy, kx, ky, ${thick ? "alpha_k2, " : ""}pk);
  let mag_sq = gg.x * gg.x + gg.y * gg.y;
  let inv_mag = select(1e8, inverseSqrt(mag_sq), mag_sq > 1e-16);
  gg *= inv_mag;
  return vec2<f32>(G.x * gg.x + G.y * gg.y, G.y * gg.x - G.x * gg.y);
}
fn chi_full(dx: f32, dy: f32) -> f32 {
  let r2 = dx * dx + dy * dy;
  let inv_r = select(0.0, inverseSqrt(r2), r2 > 1e-30);
  let r = r2 * inv_r;
  let alpha = r * params.wavelength;
  let c1 = dx * inv_r;
  let s1 = dy * inv_r;
  let c2 = 2.0 * c1 * c1 - 1.0;
  let s2 = 2.0 * s1 * c1;
  let c3 = c1 * c2 - s1 * s2;
  let s3 = s1 * c2 + c1 * s2;
  let c4 = c2 * c2 - s2 * s2;
  let s4 = 2.0 * s2 * c2;
  let c5 = c1 * c4 - s1 * s4;
  let s5 = s1 * c4 + c1 * s4;
  let c6 = c2 * c4 - s2 * s4;
  let s6 = s2 * c4 + c2 * s4;
  let a2 = alpha * alpha;
  let a3 = a2 * alpha;
  let a4 = a2 * a2;
  let a5 = a4 * alpha;
  let a6 = a3 * a3;
  var chi = 0.0;
  chi = chi + a2 * abr[0].x;
  chi = chi + a2 * abr[1].x * (c2 * abr[1].y + s2 * abr[1].z);
  chi = chi + a3 * abr[2].x * (c1 * abr[2].y + s1 * abr[2].z);
  chi = chi + a3 * abr[3].x * (c3 * abr[3].y + s3 * abr[3].z);
  chi = chi + a4 * abr[4].x;
  chi = chi + a4 * abr[5].x * (c2 * abr[5].y + s2 * abr[5].z);
  chi = chi + a4 * abr[6].x * (c4 * abr[6].y + s4 * abr[6].z);
  chi = chi + a5 * abr[7].x * (c1 * abr[7].y + s1 * abr[7].z);
  chi = chi + a5 * abr[8].x * (c3 * abr[8].y + s3 * abr[8].z);
  chi = chi + a5 * abr[9].x * (c5 * abr[9].y + s5 * abr[9].z);
  chi = chi + a6 * abr[10].x;
  chi = chi + a6 * abr[11].x * (c2 * abr[11].y + s2 * abr[11].z);
  chi = chi + a6 * abr[12].x * (c4 * abr[12].y + s4 * abr[12].z);
  chi = chi + a6 * abr[13].x * (c6 * abr[13].y + s6 * abr[13].z);
  return (6.283185307179586 / params.wavelength) * chi;
}
fn fft_inplace(tid: u32, sign: f32) {
  let _unused_sign = sign;
${makeFftStages(n)}
}

@compute @workgroup_size(${workgroupSize})
fn directRows(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
  let tid = lid.x;
  let row = wid.y;
  let local_bf = wid.z;
  if (local_bf >= params.chunk_bf) { return; }
  let bf = local_bf + params.bf_offset;
  if (bf >= params.active_bf) { return; }
  let bg = bfGeom[bf];
  if (bg.w == 0.0) { return; }
  let storage_bf = local_bf;
  let base = (storage_bf * params.plane) + row * ${n}u;
  let bt = bfTrig[bf];
  var chi_k: f32;
  if (params.full_aberration != 0u) {
    chi_k = chi_full(bg.x, bg.y);
  } else {
    let cos_term_k = bt.x * params.cos2phi12 + bt.y * params.sin2phi12;
    chi_k = params.factor * bg.z * (params.C12 * cos_term_k + params.C10);
  }
  let pk = vec2<f32>(bg.w * cos(chi_k), -bg.w * sin(chi_k));
  for (var off = 0u; off < ${n}u; off = off + ${workgroupSize}u) {
    let x = off + tid;
    if (x < ${n}u) {
      var v = gamma_mul(qx1d[row], qy1d[x], bg.x, bg.y, ${thick ? "bg.z, " : ""}pk, fetch_g(local_bf, bf, row, x));
      if (row == 0u && x == 0u) { v = vec2<f32>(params.dc_re, params.dc_im); }
      s[FFT_BITREV[x]] = v;
    }
  }
  fft_inplace(tid, 1.0);
  for (var off = 0u; off < ${n}u; off = off + ${workgroupSize}u) {
    let x = off + tid;
    if (x < ${n}u) { stage[base + x] = s[x]; }
  }
}

@compute @workgroup_size(${workgroupSize})
fn directCols(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
  let tid = lid.x;
  let col = wid.y;
  let local_bf = wid.z;
  if (local_bf >= params.chunk_bf) { return; }
  let bf = local_bf + params.bf_offset;
  if (bf >= params.active_bf) { return; }
  let bg = bfGeom[bf];
  if (bg.w == 0.0) { return; }
  let storage_bf = local_bf;
  for (var off = 0u; off < ${n}u; off = off + ${workgroupSize}u) {
    let y = off + tid;
    if (y < ${n}u) {
      s[FFT_BITREV[y]] = stage[storage_bf * params.plane + y * ${n}u + col];
    }
  }
  fft_inplace(tid, 1.0);
  for (var off = 0u; off < ${n}u; off = off + ${workgroupSize}u) {
    let y = off + tid;
    if (y < ${n}u) {
      let idx = storage_bf * params.plane + y * ${n}u + col;
      let v = s[y] * params.inv_n2;
      phaseStack[idx] = atan2(v.y, v.x);
    }
  }
}

@compute @workgroup_size(256)
fn reducePartialGroups(@builtin(global_invocation_id) gid: vec3<u32>) {
  let linear = gid.x;
  let groups_in_chunk = (params.chunk_bf + ${REDUCE_BF_GROUP - 1}u) / ${REDUCE_BF_GROUP}u;
  let total = groups_in_chunk * params.plane;
  if (linear >= total) { return; }
  let local_group = linear / params.plane;
  let idx = linear - local_group * params.plane;
  let start_bf = params.bf_offset + local_group * ${REDUCE_BF_GROUP}u;
  let end_bf = min(params.active_bf, min(params.bf_offset + params.chunk_bf, start_bf + ${REDUCE_BF_GROUP}u));
  let global_group = start_bf / ${REDUCE_BF_GROUP}u;
  var sum = 0.0;
  var sumsq = 0.0;
  if (params.compute_loss != 0u) {
    for (var bf = start_bf; bf < end_bf; bf = bf + 1u) {
      if (bfGeom[bf].w != 0.0) {
        let storage_bf = bf - params.bf_offset;
        let a = phaseStack[storage_bf * params.plane + idx];
        sum += a;
        sumsq += a * a;
      }
    }
  } else {
    for (var bf = start_bf; bf < end_bf; bf = bf + 1u) {
      if (bfGeom[bf].w != 0.0) {
        let storage_bf = bf - params.bf_offset;
        sum += phaseStack[storage_bf * params.plane + idx];
      }
    }
  }
  let out = global_group * params.plane + idx;
  partialSum[out] = sum;
  if (params.compute_loss != 0u) {
    partialSumSq[out] = sumsq;
  }
}

@compute @workgroup_size(256)
fn finalizePartialGroups(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.plane) { return; }
  var sum = 0.0;
  var sumsq = 0.0;
  if (params.compute_loss != 0u) {
    for (var group = 0u; group < params.partial_groups; group = group + 1u) {
      let off = group * params.plane + idx;
      sum += partialSum[off];
      sumsq += partialSumSq[off];
    }
  } else {
    for (var group = 0u; group < params.partial_groups; group = group + 1u) {
      sum += partialSum[group * params.plane + idx];
    }
  }
  let denom = f32(max(params.num_bf, 1u));
  let mean = sum / denom;
  phaseOut[idx] = mean;
  varianceOut[idx] = select(0.0, sumsq / denom - mean * mean, params.compute_loss != 0u);
}

// ---------------------------------------------------------------------------
// Object-redraw fast path (ported from the CUDA direct ptychography engine, 2026-07-17):
// mean_bf(ifft2(corrected_bf)) == ifft2(mean_bf(corrected_bf)), so the drag
// path sums gamma-corrected G(q,k) over BF pixels in Fourier space and runs
// ONE 2D inverse FFT, instead of two FFT passes + atan2 per BF pixel. This
// displays angle(mean(object)) - the same estimator as the Python
// DirectPtycho.result() reference - and skips the per-BF phase stack entirely.
// The loss/variance commit path still uses the exact per-BF pipeline above,
// because mean(angle(object)) != angle(mean(object)).
// ---------------------------------------------------------------------------
${cacheProbe ? `
// The incident probe depends on detector position and optics, not output q.
// Evaluate it once per BF pixel, retaining the original float32 expressions.
// Model agreement uses the unnormalized transfer function. Preserve the complex
// numerator across BF chunks before forming the power at each scan frequency.
@group(0) @binding(15) var<storage, read> fitFrequencies: array<vec2<u32>>;
@group(0) @binding(14) var<storage, read_write> agreementTerms: array<vec4<f32>>;
@compute @workgroup_size(64)
fn modelAgreement(@builtin(global_invocation_id) id: vec3<u32>) {
  let width = params.n / 2u + 1u;
  let q = id.x;
  let bandSize = arrayLength(&fitFrequencies);
  if (q >= bandSize) { return; }
  let row = fitFrequencies[q].x / width;
  let col = fitFrequencies[q].x % width;
  let qr = qx1d[row];
  let qc = qy1d[col];
  var sums = vec3<f32>(0.0);
  for (var k = 0u; k < params.chunk_bf; k++) {
    let bf = k + params.bf_offset;
    let bg = bfGeom[bf];
    if (bg.w == 0.0) { continue; }
    let pk = cachedProbe[bf];
    let gamma = transfer(qr, qc, bg.x, bg.y, ${thick ? "bg.z, " : ""}pk);
    let g = gqk[k * ${fitStride}u + fitFrequencies[q].y];
    sums += vec3<f32>(dot(g, gamma), g.y * gamma.x - g.x * gamma.y, dot(gamma, gamma));
  }
  agreementTerms[q] += vec4<f32>(sums, 0.0);
}
${thick ? agreementGradient(fitStride) : ""}
@compute @workgroup_size(256)
fn prepareProbe(@builtin(global_invocation_id) gid: vec3<u32>) {
  let bf = gid.x;
  if (bf >= params.active_bf) { return; }
  let bg = bfGeom[bf];
  let bt = bfTrig[bf];
  var chi_k: f32;
  if (params.full_aberration != 0u) {
    chi_k = chi_full(bg.x, bg.y);
  } else {
    let cos_term_k = bt.x * params.cos2phi12 + bt.y * params.sin2phi12;
    chi_k = params.factor * bg.z * (params.C12 * cos_term_k + params.C10);
  }
  cachedProbe[bf] = vec2<f32>(bg.w * cos(chi_k), -bg.w * sin(chi_k));
}
` : ""}
@compute @workgroup_size(256)
fn directObjSum(@builtin(global_invocation_id) gid: vec3<u32>) {
  let linear = gid.x + gid.y * 65535u * 256u;
  if (linear >= params.n * params.output_cols) { return; }
  let row = linear / params.output_cols;
  let x = linear - row * params.output_cols;
  let idx = row * ${n}u + x;
  var acc = vec2<f32>(0.0, 0.0);
  ${n > native ? `// Match native GPU single-origin handling of the scan background.
  // The output buffer was cleared; its DC replicas must remain zero.
  if (row % ${native}u == 0u && x % ${native}u == 0u && idx != 0u) { return; }` : ""}
  // Both shifted apertures are exactly zero outside this support. Avoid
  // evaluating thousands of BF terms in the unused corners of a finer grid.
  let support = 2.0 * (params.semiangle_rad + 0.5 * max(params.ang_y_rad, params.ang_x_rad)) / params.wavelength;
  if (qx1d[row] * qx1d[row] + qy1d[x] * qy1d[x] > support * support) {
    stage[idx] = vec2<f32>(0.0, 0.0); return;
  }
  let end_bf = min(params.active_bf, params.bf_offset + params.chunk_bf);
  for (var bf = params.bf_offset; bf < end_bf; bf = bf + 1u) {
    let bg = bfGeom[bf];
    if (bg.w == 0.0) { continue; }
    if (idx == 0u) {
      acc += vec2<f32>(params.dc_re, params.dc_im);
      continue;
    }
    ${cacheProbe ? "let pk = cachedProbe[bf];" : `let bt = bfTrig[bf];
    var chi_k: f32;
    if (params.full_aberration != 0u) {
      chi_k = chi_full(bg.x, bg.y);
    } else {
      let cos_term_k = bt.x * params.cos2phi12 + bt.y * params.sin2phi12;
      chi_k = params.factor * bg.z * (params.C12 * cos_term_k + params.C10);
    }
    let pk = vec2<f32>(bg.w * cos(chi_k), -bg.w * sin(chi_k));`}
    acc += gamma_mul(qx1d[row], qy1d[x], bg.x, bg.y, ${thick ? "bg.z, " : ""}pk, fetch_g(bf - params.bf_offset, bf, row, x));
  }
  // stage[0..plane] is the accumulation plane for the fast path (cleared
  // before the first chunk; chunks add sequentially via separate dispatches).
  stage[idx] = stage[idx] + acc;
}

@compute @workgroup_size(256)
fn directMirrorColumns(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x + gid.y * 65535u * 256u;
  if (idx >= params.plane) { return; }
  let row = idx / ${n}u;
  let col = idx % ${n}u;
  if (col <= ${n / 2}u) { return; }
  let mirrorRow = select(${n}u - row, 0u, row == 0u);
  let value = stage[mirrorRow * ${n}u + ${n}u - col];
  stage[idx] = vec2<f32>(-value.x, value.y);
}

@compute @workgroup_size(${workgroupSize})
fn directObjFftRows(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
  let tid = lid.x;
  let row = wid.y;
  for (var off = 0u; off < ${n}u; off = off + ${workgroupSize}u) {
    let x = off + tid;
    if (x < ${n}u) { s[FFT_BITREV[x]] = stage[row * ${n}u + x]; }
  }
  fft_inplace(tid, 1.0);
  for (var off = 0u; off < ${n}u; off = off + ${workgroupSize}u) {
    let x = off + tid;
    if (x < ${n}u) { stage[row * ${n}u + x] = s[x]; }
  }
}

@compute @workgroup_size(${workgroupSize})
fn directObjFftCols(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
  let tid = lid.x;
  let col = wid.y;
  for (var off = 0u; off < ${n}u; off = off + ${workgroupSize}u) {
    let y = off + tid;
    if (y < ${n}u) { s[FFT_BITREV[y]] = stage[y * ${n}u + col]; }
  }
  fft_inplace(tid, 1.0);
  for (var off = 0u; off < ${n}u; off = off + ${workgroupSize}u) {
    let y = off + tid;
    if (y < ${n}u) {
      let v = s[y] * params.inv_n2;
      phaseOut[y * ${n}u + col] = ${phaseField ? "atan2(sin(v.x),cos(v.x))" : "atan2(v.y, v.x)"};
    }
  }
}

`;
}

function makeBufferFromBytes(
  device: GPUDevice,
  bytes: Uint8Array,
  usage: GPUBufferUsageFlags,
  label: string,
): GPUBuffer {
  const buffer = device.createBuffer({ size: bytes.byteLength, usage: usage | GPUBufferUsage.COPY_DST, label });
  const chunk = 32 * 1024 * 1024;
  for (let off = 0; off < bytes.byteLength; off += chunk) {
    const len = Math.min(chunk, bytes.byteLength - off);
    device.queue.writeBuffer(buffer, off, bytes as unknown as BufferSource, off, len);
  }
  return buffer;
}

function makeBufferFromArray(
  device: GPUDevice,
  array: Float32Array | Uint32Array,
  usage: GPUBufferUsageFlags,
  label: string,
): GPUBuffer {
  return makeBufferFromBytes(
    device,
    new Uint8Array(array.buffer, array.byteOffset, array.byteLength),
    usage,
    label,
  );
}

function makeSourceFftConstants(n: SupportedDirectSize): string {
  const twiddles: string[] = [];
  for (let len = 2; len <= n; len *= 2) {
    const half = len / 2;
    for (let j = 0; j < half; j++) {
      const angle = -2 * Math.PI * j / len;
      twiddles.push(`vec2<f32>(${formatF32(Math.cos(angle))}, ${formatF32(Math.sin(angle))})`);
    }
  }
  const bitrev: number[] = [];
  const bits = Math.log2(n);
  for (let x = 0; x < n; x++) {
    let y = 0;
    for (let bit = 0; bit < bits; bit++) y = (y << 1) | ((x >> bit) & 1);
    bitrev.push(y);
  }
  return `
const SOURCE_FFT_TWIDDLES: array<vec2<f32>, ${twiddles.length}> = array<vec2<f32>, ${twiddles.length}>(
  ${twiddles.join(",\n  ")}
);
const SOURCE_FFT_BITREV: array<u32, ${n}> = array<u32, ${n}>(
  ${bitrev.map(value => `${value}u`).join(", ")}
);`;
}

function makeSourceFftStages(n: SupportedDirectSize, workgroupSize = Math.min(n, 256), columns = 1): string {
  const at = (index: string) => columns === 1 ? `s[${index}]` : `columnTile[${index} * ${columns}u + lane]`;
  const lines: string[] = ["  workgroupBarrier();"];
  for (let len = 2; len <= n; len *= 2) {
    const half = len / 2;
    const twiddleOffset = half - 1;
    lines.push(
      `  for (var butterfly = tid; butterfly < ${n / 2}u; butterfly = butterfly + ${workgroupSize}u) {`,
      `    let group = butterfly / ${half}u;`,
      `    let j = butterfly - group * ${half}u;`,
      `    let i0 = group * ${len}u + j;`,
      `    let i1 = i0 + ${half}u;`,
      `    let w = SOURCE_FFT_TWIDDLES[${twiddleOffset}u + j];`,
      `    let t = cmul_source(w, ${at("i1")});`,
      `    let u = ${at("i0")};`,
      `    ${at("i0")} = u + t;`,
      `    ${at("i1")} = u - t;`,
      "  }",
      "  workgroupBarrier();",
    );
  }
  return lines.join("\n");
}

// BF-column unpack (detector counts -> complex G(q,k) columns) and the 2D source FFT of each column.
function fftColumnTile(n: SupportedDirectSize): number {
  // Four adjacent columns share coalesced reads/writes within WebGPU's 16 KiB minimum.
  return n <= 512 ? 4 : 1;
}

function makeBfColumnGqkShader(n: SupportedDirectSize, compactOutput: boolean): string {
  const workgroupSize = Math.min(n, 256);
  const columns = fftColumnTile(n);
  const columnThreads = workgroupSize / columns;
  const tile = (index: string) => columns === 1 ? `s[${index}]` : `columnTile[(${index}) * ${columns}u + lane]`;
  return `
${makeSourceFftConstants(n)}
fn sample(gp: u32, mode: u32) -> u32 {
  if (mode == 1u) { let w = data[gp >> 2u]; return (w >> ((gp & 3u) * 8u)) & 0xffu; }
  let w = data[gp >> 1u];
  return select(w >> 16u, w & 0xffffu, (gp & 1u) == 0u);
}
fn cmul_source(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}
struct GatherParams {
  start_scan: u32,
  n_scan: u32,
  det_size: u32,
  mode: u32,
  chunk_bf: u32,
  plane: u32,
  _pad0: u32,
  _pad1: u32,
};
@group(0) @binding(0) var<storage, read> data: array<u32>;
@group(0) @binding(2) var<storage, read_write> gqk: array<vec2<f32>>;
@group(0) @binding(3) var<uniform> gatherParams: GatherParams;
@compute @workgroup_size(${BF_COLUMN_UNPACK_WORKGROUP_X}, ${BF_COLUMN_UNPACK_WORKGROUP_Y})
fn bfColumnsToGqk(@builtin(global_invocation_id) gid: vec3<u32>) {
  let scan = gid.x;
  let bf = gid.y;
  if (scan >= gatherParams.plane || bf >= gatherParams.chunk_bf) { return; }
  let gp = bf * gatherParams.plane + scan;
  gqk[gp] = vec2<f32>(f32(sample(gp, gatherParams.mode)), 0.0);
}

struct FftParams {
  chunk_bf: u32,
  plane: u32,
  _pad0: u32,
  _pad1: u32,
};
@group(1) @binding(0) var<storage, read_write> fftData: array<vec2<f32>>;
@group(1) @binding(1) var<uniform> fftParams: FftParams;
${compactOutput ? "@group(1) @binding(2) var<storage, read_write> spectrum: array<vec2<f32>>;" : ""}
var<workgroup> s: array<vec2<f32>, ${n}>;
fn source_fft_inplace(tid: u32) {
${makeSourceFftStages(n)}
}
@compute @workgroup_size(${workgroupSize})
fn fftRows(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
  let tid = lid.x;
  let row = wid.y;
  let bf = wid.z;
  if (bf >= fftParams.chunk_bf) { return; }
  let base = bf * fftParams.plane + row * ${n}u;
  for (var off = 0u; off < ${n}u; off = off + ${workgroupSize}u) {
    let x = off + tid;
    if (x < ${n}u) { s[SOURCE_FFT_BITREV[x]] = fftData[base + x]; }
  }
  source_fft_inplace(tid);
  for (var off = 0u; off < ${n}u; off = off + ${workgroupSize}u) {
    let x = off + tid;
    if (x < ${n}u) { fftData[base + x] = s[x]; }
  }
}
${columns === 1 ? "" : `var<workgroup> columnTile: array<vec2<f32>, ${n * columns}>;`}
@compute @workgroup_size(${workgroupSize})
fn fftCols(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
  let lane = lid.x % ${columns}u;
  let tid = lid.x / ${columns}u;
  let col = wid.y * ${columns}u + lane;
  let bf = wid.z;
  let base = bf * fftParams.plane;
  for (var y = tid; y < ${n}u; y += ${columnThreads}u) {
    if (col <= ${n/2}u) { ${tile("SOURCE_FFT_BITREV[y]")} = fftData[base + y * ${n}u + col]; }
  }
${makeSourceFftStages(n,columnThreads,columns)}
  for (var y = tid; y < ${n}u; y += ${columnThreads}u) {
    if (col <= ${n/2}u) {
      ${compactOutput ? `spectrum[(bf * ${n}u + y) * ${n/2+1}u + col]` : `fftData[base + y * ${n}u + col]`} = ${tile("y")};
    }
  }
}
`;
}


function makeGqkTransformShader(n: SupportedDirectSize, bandSize?: number): string {
  const half = n / 2;
  const halfW = half + 1;
  const storedPlane = bandSize ?? n * halfW;
  const dstDecl = "@group(0) @binding(1) var<storage, read_write> dst: array<vec2<f32>>;";
  const writeBody = `dst[bf * ${storedPlane}u + i] = v;`;
  return `
struct TransformParams {
  chunk_bf: u32,
  bf_offset: u32,
  _pad0: u32,
  _pad1: u32,
};
@group(0) @binding(0) var<storage, read> src: array<vec2<f32>>;
${dstDecl}
${bandSize ? "@group(0) @binding(2) var<storage, read> band: array<u32>;" : ""}
@group(0) @binding(3) var<uniform> tp: TransformParams;

@compute @workgroup_size(256)
fn compact(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let bf = gid.z;
  if (i >= ${storedPlane}u || bf >= tp.chunk_bf) { return; }
  let flat = ${bandSize ? "band[i]" : "i"};
  let row = flat / ${halfW}u;
  let c = flat % ${halfW}u;
  let v = src[bf * ${n * n}u + row * ${n}u + c];
  ${writeBody}
}
`;
}

/** Progress callback that stamps every event with the active/total BF counts and elapsed time. */
function progressEmitter(onProgress: WebGPUProgressHandler | undefined, activeBf: number, totalBf: number, started: number) {
  return (progress: Omit<WebGPULoadProgress, "elapsedMs" | "activeBf" | "totalBf">) => {
    onProgress?.({
      activeBf,
      totalBf,
      elapsedMs: performance.now() - started,
      ...progress,
    });
  };
}

function uniformU32(device: GPUDevice, values: number[], label: string): GPUBuffer {
  const buffer = device.createBuffer({ size: Math.max(16, Math.ceil(values.length / 4) * 16), usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label });
  const words = new Uint32Array(Math.ceil(values.length / 4) * 4);
  words.set(values.map(value => Math.max(0, Math.round(value))));
  device.queue.writeBuffer(buffer, 0, words.buffer, words.byteOffset, words.byteLength);
  return buffer;
}

function sampleThickness(sample?: WebGPUSampleOptions): number {
  return sample && Number.isFinite(sample.depthSpreadA) && sample.depthSpreadA > 0 ? sample.depthSpreadA : 0;
}

// Params uniform: 22 original words + thickness, tilt_row_rad, tilt_col_rad + 3 pad words (multiple of 16 bytes); the last
// six are declared only in the thick shader variant.
const DIRECT_PARAMS_BYTES = 112;

export function makeParams(
  cal: DirectCal,
  n: number,
  c10: number,
  c12: number,
  phi12Deg: number,
  numBrightFieldPixels?: number,
  bfOffset = 0,
  chunkBf?: number,
  computeLoss = true,
  activeBfCount = numBrightFieldPixels,
  fullAberration = false,
  sample?: WebGPUSampleOptions,
  outputCols = n,
): ArrayBuffer {
  const plane = n * n;
  const b = new ArrayBuffer(DIRECT_PARAMS_BYTES);
  const u = new Uint32Array(b);
  const f = new Float32Array(b);
  const phi12Rad = phi12Deg * Math.PI / 180;
  u[0] = Math.max(1, Math.min(cal.num_bright_field_pixels, Math.round(numBrightFieldPixels || cal.num_bright_field_pixels)));
  u[1] = n; u[2] = plane; u[3] = Math.round(bfOffset);
  f[4] = cal.wavelength_A;
  f[5] = cal.semiangle_rad;
  f[6] = cal.angular_sampling_rad[0];
  f[7] = cal.angular_sampling_rad[1];
  f[8] = c10;
  f[9] = c12;
  f[10] = Math.cos(2 * phi12Rad);
  f[11] = Math.sin(2 * phi12Rad);
  f[12] = Math.PI / cal.wavelength_A;
  f[13] = cal.dc_value[0];
  f[14] = cal.dc_value[1];
  f[15] = 1 / plane;
  u[16] = Math.max(1, Math.round(chunkBf || u[0]));
  u[17] = outputCols;
  u[18] = Math.max(1, Math.ceil(Math.max(1, Math.round(activeBfCount || u[0])) / REDUCE_BF_GROUP));
  u[19] = computeLoss ? 1 : 0;
  u[20] = Math.max(1, Math.round(activeBfCount || u[0]));
  u[21] = fullAberration ? 1 : 0;
  // Thick-sample depth average (Angstrom, radians). Absent or non-positive thickness leaves the standard direct ptychography gamma.
  const thickness = sampleThickness(sample);
  // A depth spread given without a tilt is an untilted crystal: indexing the omitted pair threw a TypeError.
  const tiltMrad = sample?.sampleTiltMrad ?? [0, 0];
  f[22] = thickness;
  f[23] = thickness > 0 ? (tiltMrad[0] || 0) * 1e-3 : 0;
  f[24] = thickness > 0 ? (tiltMrad[1] || 0) * 1e-3 : 0;
  return b;
}

const ABR_INV_N_PLUS_ONE = [
  1 / 2, 1 / 2,
  1 / 3, 1 / 3,
  1 / 4, 1 / 4, 1 / 4,
  1 / 5, 1 / 5, 1 / 5,
  1 / 6, 1 / 6, 1 / 6, 1 / 6,
];
const ABR_M_VALUES = [0, 2, 1, 3, 0, 2, 4, 1, 3, 5, 0, 2, 4, 6];
const HO_LAYOUT: Array<[string, number, boolean]> = [
  ["C21", 2, true], ["C23", 3, true],
  ["C30", 4, false], ["C32", 5, true], ["C34", 6, true],
  ["C41", 7, true], ["C43", 8, true], ["C45", 9, true],
  ["C50", 10, false], ["C52", 11, true],
  ["C54", 12, true], ["C56", 13, true],
];
// The keys `higherOrder` takes: a magnitude per term in Angstrom, and an azimuth in degrees where the term has one.
const HO_KEYS = HO_LAYOUT.flatMap(([name, , hasAngle]) => hasAngle ? [`${name}_A`, `phi${name.slice(1)}_deg`] : [`${name}_A`]);

export function packAberrations(
  c10: number,
  c12: number,
  phi12Deg: number,
  higherOrder?: Record<string, number>,
): { data: Float32Array; active: boolean } {
  const mags = new Float32Array(14);
  const angles = new Float32Array(14);
  mags[0] = c10;
  mags[1] = c12;
  angles[1] = phi12Deg * Math.PI / 180;
  let active = false;
  const ho = higherOrder || {};
  // A key the engine does not know read as 0, so an earlier spelling (C30 for C30_A, C21_mag for C21_A) gave the
  // image of no aberration without a word. Refused with the accepted keys, as the Python boundary does.
  const unknown = Object.keys(ho).filter(key => !HO_KEYS.includes(key));
  if (unknown.length > 0) {
    throw new Error(`Unknown higherOrder keys ${unknown.join(", ")}; the engine takes ${HO_KEYS.join(", ")}.`);
  }
  for (const [name, index, hasAngle] of HO_LAYOUT) {
    // magnitudes in Angstrom, azimuths in degrees: C21_A with phi21_deg, C30_A alone
    const mag = Number(ho[`${name}_A`] || 0);
    const angleDeg = hasAngle ? Number(ho[`phi${name.slice(1)}_deg`] || 0) : 0;
    mags[index] = mag;
    angles[index] = angleDeg * Math.PI / 180;
    if (mag !== 0) active = true;
  }
  const out = new Float32Array(14 * 4);
  for (let i = 0; i < 14; i++) {
    const theta = ABR_M_VALUES[i] * angles[i];
    out[i * 4 + 0] = mags[i] * ABR_INV_N_PLUS_ONE[i];
    out[i * 4 + 1] = Math.cos(theta);
    out[i * 4 + 2] = Math.sin(theta);
    out[i * 4 + 3] = 0;
  }
  return { data: out, active };
}

function baseRotationDeg(cal: DirectCal): number {
  return Number.isFinite(cal.scan_detector_rotation_deg) ? Number(cal.scan_detector_rotation_deg) : 0;
}

function rotateBfCoord(cal: DirectCal, kx: number, ky: number, scanDetectorRotationDeg?: number): [number, number] {
  if (scanDetectorRotationDeg == null || !Number.isFinite(scanDetectorRotationDeg)) return [kx, ky];
  const delta = (Number(scanDetectorRotationDeg) - baseRotationDeg(cal)) * Math.PI / 180;
  if (Math.abs(delta) < 1e-12) return [kx, ky];
  const cosA = Math.cos(-delta);
  const sinA = Math.sin(-delta);
  return [
    kx * cosA + ky * sinA,
    -kx * sinA + ky * cosA,
  ];
}

function bfGeometryFromCoord(cal: DirectCal, kx: number, ky: number): [number, number, number, number] {
  const dx2 = kx * kx;
  const dy2 = ky * ky;
  const r2 = dx2 + dy2;
  const r = Math.sqrt(r2);
  const alpha = r * cal.wavelength_A;
  const alpha2 = alpha * alpha;
  const invR2 = r2 > 1e-30 ? 1 / r2 : 0;
  const cos2phi = (dx2 - dy2) * invR2;
  const sin2phi = 2 * kx * ky * invR2;
  const denomNum2 = (kx * cal.angular_sampling_rad[0]) ** 2 + (ky * cal.angular_sampling_rad[1]) ** 2;
  const denom = r > 1e-15 ? Math.sqrt(denomNum2) / r : 0;
  const edge = denom > 1e-15 ? (cal.semiangle_rad - alpha) / denom + 0.5 : 1;
  return [alpha2, cos2phi, sin2phi, Math.max(0, Math.min(1, edge))];
}

function collectActiveBfIndices(cal: DirectCal, numBf: number, scanDetectorRotationDeg?: number): Uint32Array {
  const active: number[] = [];
  for (let i = 0; i < cal.num_bright_field_pixels; i++) {
    const [kx, ky] = rotateBfCoord(cal, cal.bright_field_k_row[i], cal.bright_field_k_col[i], scanDetectorRotationDeg);
    const [, , , aperture] = bfGeometryFromCoord(cal, kx, ky);
    if (aperture !== 0) active.push(i);
  }
  if (active.length === 0) return new Uint32Array();
  const count = Math.max(1, Math.min(active.length, Math.round(numBf)));
  if (count >= active.length) return new Uint32Array(active);
  const selected = new Uint32Array(count);
  const stride = active.length / count;
  for (let i = 0; i < count; i++) {
    selected[i] = active[Math.min(active.length - 1, Math.floor((i + 0.5) * stride))];
  }
  return selected;
}

function packGeometry(cal: DirectCal, indices: Uint32Array, scanDetectorRotationDeg?: number): { geom: Float32Array; trig: Float32Array } {
  const count = Math.max(1, indices.length);
  const geom = new Float32Array(count * 4);
  const trig = new Float32Array(count * 2);
  if (indices.length === 0) {
    return { geom, trig };
  }
  for (let i = 0; i < indices.length; i++) {
    const src = indices[i];
    const [kx, ky] = rotateBfCoord(cal, cal.bright_field_k_row[src], cal.bright_field_k_col[src], scanDetectorRotationDeg);
    const [alpha2, cos2phi, sin2phi, aperture] = bfGeometryFromCoord(cal, kx, ky);
    geom[i * 4 + 0] = kx;
    geom[i * 4 + 1] = ky;
    geom[i * 4 + 2] = alpha2;
    geom[i * 4 + 3] = aperture;
    trig[i * 2 + 0] = cos2phi;
    trig[i * 2 + 1] = sin2phi;
  }
  return { geom, trig };
}

async function readF32(device: GPUDevice, buffer: GPUBuffer, length: number): Promise<Float32Array> {
  const read = device.createBuffer({
    size: length * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(buffer, 0, read, 0, length * 4);
  device.queue.submit([enc.finish()]);
  await read.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(read.getMappedRange().slice(0));
  read.unmap();
  read.destroy();
  return out;
}

async function fetchRangeBytes(url: string, byteOffset: number, byteLength: number): Promise<Uint8Array> {
  const bytes = await readSourceBytes(url, byteOffset, byteLength);
  if (bytes.byteLength < byteLength) {
    throw new Error(
      `${url}: received ${(bytes.byteLength / 1e6).toFixed(1)} MB; `
      + `need ${(byteLength / 1e6).toFixed(1)} MB.`,
    );
  }
  return bytes.byteLength === byteLength ? bytes : bytes.slice(0, byteLength);
}

async function fetchPackedBfColumnBytes(
  url: string,
  activeIndices: Uint32Array,
  activeOffset: number,
  activeCount: number,
  bytesPerBf: number,
): Promise<Uint8Array> {
  const out = new Uint8Array(activeCount * bytesPerBf);
  if (activeIndices.length === 0) return out;
  let dstBf = 0;
  while (dstBf < activeCount) {
    const runActiveStart = activeOffset + dstBf;
    const srcStart = activeIndices[runActiveStart];
    let runBf = 1;
    while (
      dstBf + runBf < activeCount
      && activeIndices[runActiveStart + runBf] === srcStart + runBf
    ) {
      runBf += 1;
    }
    const srcByteOffset = srcStart * bytesPerBf;
    const runBytes = runBf * bytesPerBf;
    out.set(await fetchRangeBytes(url, srcByteOffset, runBytes), dstBf * bytesPerBf);
    dstBf += runBf;
  }
  return out;
}


// Build the resident half-plane G(q,k) chunk by chunk: each chunk's full n x n columns are
// unpacked (or ANS-decoded), FFT'd and compacted in one submission, and its full-plane buffer is
// freed before the next one is allocated, so the build peak is one full-plane chunk above the
// resident budget, not twice the budget. The host reads the next chunk's columns while the GPU
// works on the previous one.
async function buildBfColumnGqkChunks(
  device: GPUDevice,
  cal: DirectCal,
  source: DetectorDirectSource,
  n: SupportedDirectSize,
  plane: number,
  activeSourceIndices: Uint32Array,
  storageChunkCapacity: number,
  onProgress?: WebGPUProgressHandler,
  fitBand?: {indices: GPUBuffer; size: number},
  consume?: (encoder: GPUCommandEncoder, chunk: GPUBuffer, offset: number, count: number) => void,
): Promise<{ gqkChunks: GPUBuffer[]; chunkBfCounts: number[]; residentBytes: number }> {
  const t0 = performance.now();
  const emit = progressEmitter(onProgress, activeSourceIndices.length, cal.num_bright_field_pixels, t0);
  if (!source.url) throw new Error("BF-column source is missing a URL.");
  if (Number(source.plane) !== (source.kind === "full_detector" ? source.scan_shape[0] * source.scan_shape[1] : plane)) {
    throw new Error(`BF-column source plane ${source.plane} does not match ${n}x${n}=${plane}.`);
  }
  const maxActiveSource = activeSourceIndices.reduce((max, value) => Math.max(max, value), 0);
  if (Number(source.num_bright_field_pixels) <= maxActiveSource) {
    throw new Error(`BF-column source has ${source.num_bright_field_pixels} BF columns; need source BF index ${maxActiveSource}.`);
  }
  const detectorColumns = source.kind === "full_detector" ? fullDetectorColumns(cal, source) : null;
  const selectedColumns = detectorColumns ? activeSourceIndices.map(index => detectorColumns[index]) : activeSourceIndices;
  emit({
    stage: "pipeline",
    message: "Preparing WebGPU kernels",
    detail: `Building BF-column unpack and FFT kernels for ${n}x${n}`,
    current: 0,
    total: activeSourceIndices.length,
    percent: 10,
    sourceFrames: plane,
  });
  const module = device.createShaderModule({ code: makeBfColumnGqkShader(n, !fitBand), label: `Direct ptychography BF-column unpack+FFT ${n}` });
  const compactModule = device.createShaderModule({ code: makeGqkTransformShader(n, fitBand?.size), label: `Direct ptychography gqk compact ${n}` });
  const [unpackPipe, rowsPipe, colsPipe, compactPipe] = await Promise.all([
    device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "bfColumnsToGqk" } }),
    device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "fftRows" } }),
    device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "fftCols" } }),
    device.createComputePipelineAsync({ layout: "auto", compute: { module: compactModule, entryPoint: "compact" } }),
  ]);
  // uint8 columns pack 4 counts per word (mode 1), uint16 2 (mode 0); .qem columns arrive decoded
  const mode = source.encoding === "uint8" ? 1 : 0;
  const bytesPerBf = source.bytes_per_bright_field_pixel;
  const storedPlane = fitBand?.size ?? storedPlaneFor(n);
  const gqkChunks: GPUBuffer[] = [];
  const chunkBfCounts: number[] = [];
  let residentBytes = 0;
  let fetchBytes = 0;
  let fetchMs = 0;
  let admissionMs = 0;
  let columnGatherMs = 0;
  let gpuWaitMs = 0;
  let ans: RansResidentSet | null = null;
  let previous: Promise<void> = Promise.resolve();
  let pending: GPUBuffer[] = [];
  try {
  if (source.encoding === "qem") {
    const started = performance.now();
    // Admission reads authenticated chunks directly from the selected file or
    // HTTP stream, without holding a second complete encoded copy on the host.
    const admissionStarted = performance.now();
    if (source.kind === "full_detector" && source.resident) {
      ans = source.resident;
    } else {
    const file = await openQemFile(source.url);
    const input = source.kind === "full_detector" ? {
      name: file.name, size: file.size, mode: file.mode,
      slice: file.slice.bind(file), chunks: file.chunks?.bind(file),
      recycleChunk: file.recycleChunk?.bind(file),
      expectedHeaderSha256: source.header_sha256,
    } : file;
    fetchBytes = file.size; fetchMs = performance.now() - started;
    ans = await RansResidentSet.loadQemFile(device,
      input,
      detail => emit({stage: "gather", message: "Preparing lossless ANS counts", detail, percent: 35}));
    }
    const expectedShape = source.kind === "full_detector"
      ? [...source.scan_shape, ...source.detector_shape] : null;
    if (ans.scanCount !== source.plane || (expectedShape
      ? expectedShape.some((value, axis) => ans!.shape?.[axis] !== value) || ans.nativeDtype !== source.dtype
      : ans.detSize !== source.num_bright_field_pixels)) {
      throw new Error("QEM geometry or native count dtype does not match the direct ptychography source; export the review again.");
    }
    admissionMs = performance.now() - admissionStarted;
  }
  for (let bfOffset = 0; bfOffset < activeSourceIndices.length; bfOffset += storageChunkCapacity) {
    const chunkBf = Math.min(storageChunkCapacity, activeSourceIndices.length - bfOffset);
    emit({
      stage: "fetch",
      message: "Reading detector BF columns",
      detail: `${bfOffset + chunkBf}/${activeSourceIndices.length} active BF columns, ${(chunkBf * bytesPerBf / 1e6).toFixed(1)} MB`,
      current: bfOffset + chunkBf,
      total: activeSourceIndices.length,
      percent: Math.min(85, ((bfOffset + chunkBf) / activeSourceIndices.length) * 75 + 10),
      sourceFrames: plane,
    });
    let packed: Uint8Array | null = null;
    if (!ans) {
      const fetchT = performance.now();
      packed = await fetchPackedBfColumnBytes(source.url, activeSourceIndices, bfOffset, chunkBf, bytesPerBf);
      fetchMs += performance.now() - fetchT;
      fetchBytes += packed.byteLength;
    }
    const waitT = performance.now();
    await previous;
    gpuWaitMs += performance.now() - waitT;
    const full = device.createBuffer({ size: chunkBf * plane * 8, usage: GPUBufferUsage.STORAGE, label: `direct transient bf-column G(q,k) ${bfOffset}` });
    const temporaries: GPUBuffer[] = [full];
    pending = temporaries;
    const enc = device.createCommandEncoder();
    if (ans) {
      const decodeT = performance.now();
      ans.encodeColumnsComplex(enc, temporaries, 0, selectedColumns.slice(bfOffset, bfOffset + chunkBf), full, source.kind === "full_detector" ? source.region : undefined);
      columnGatherMs += performance.now() - decodeT;
    } else {
      const dataBuffer = makeBufferFromBytes(device, packed!, GPUBufferUsage.STORAGE, `direct bf-column packed ${bfOffset}`);
      const params = uniformU32(device, [0, 0, 0, mode, chunkBf, plane, 0, 0], `direct bf-column unpack params ${bfOffset}`);
      temporaries.push(dataBuffer, params);
      const pass = enc.beginComputePass({ label: "direct bf-column unpack" });
      pass.setPipeline(unpackPipe);
      pass.setBindGroup(0, device.createBindGroup({ layout: unpackPipe.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: dataBuffer } },
        { binding: 2, resource: { buffer: full } },
        { binding: 3, resource: { buffer: params } },
      ] }));
      pass.dispatchWorkgroups(Math.ceil(plane / BF_COLUMN_UNPACK_WORKGROUP_X), Math.ceil(chunkBf / BF_COLUMN_UNPACK_WORKGROUP_Y));
      pass.end();
    }
    const resident = device.createBuffer({ size: chunkBf * storedPlane * GQK_BYTES_PER_VALUE, usage: GPUBufferUsage.STORAGE, label: `direct gqk ${GQK_MODE} chunk ${gqkChunks.length}` });
    if (consume) temporaries.push(resident);
    else gqkChunks.push(resident);
    // Row then column FFT of every column of the chunk, in place.
    const fftParams = uniformU32(device, [chunkBf, plane, 0, 0], `direct bf-column fft params ${bfOffset}`);
    temporaries.push(fftParams);
    for (const pipeline of [rowsPipe, colsPipe]) {
      const pass = enc.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(1, device.createBindGroup({ layout: pipeline.getBindGroupLayout(1), entries: [
        { binding: 0, resource: { buffer: full } },
        { binding: 1, resource: { buffer: fftParams } },
        ...(!fitBand && pipeline === colsPipe ? [{ binding: 2, resource: { buffer: resident } }] : []),
      ] }));
      pass.dispatchWorkgroups(1, pipeline === colsPipe ? Math.ceil((n / 2 + 1) / fftColumnTile(n)) : n, chunkBf);
      pass.end();
    }
    if (fitBand) {
      const compactParams = uniformU32(device, [chunkBf, bfOffset, 0, 0], `direct gqk compact params ${bfOffset}`);
      temporaries.push(compactParams);
      const pass = enc.beginComputePass({ label: "direct gqk compact" });
      pass.setPipeline(compactPipe);
      pass.setBindGroup(0, device.createBindGroup({ layout: compactPipe.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: full } },
        { binding: 1, resource: { buffer: resident } },
        ...(fitBand ? [{ binding: 2, resource: { buffer: fitBand.indices } }] : []),
        { binding: 3, resource: { buffer: compactParams } },
      ] }));
      pass.dispatchWorkgroups(Math.ceil(storedPlane / 256), 1, chunkBf);
      pass.end();
    }
    consume?.(enc, resident, bfOffset, chunkBf);
    device.queue.submit([enc.finish()]);
    previous = device.queue.onSubmittedWorkDone().then(() => temporaries.forEach((buffer) => buffer.destroy()));
    chunkBfCounts.push(chunkBf);
    residentBytes += resident.size;
  }
  const waitT = performance.now();
  await previous;
  gpuWaitMs += performance.now() - waitT;
  const totalMs = performance.now() - t0;
  const readTiming = source.encoding === "qem"
    ? `open ${fetchMs.toFixed(1)} ms authenticated admission ${admissionMs.toFixed(1)} ms, column dispatch setup ${columnGatherMs.toFixed(1)} ms`
    : `fetch ${fetchMs.toFixed(1)} ms`;
  console.log(
    `[direct] BF-column source prepared ${activeSourceIndices.length} BF from ${plane} scan positions `
    + `read ${(fetchBytes / 1e9).toFixed(2)} GB ${source.encoding || source.dtype || "uint16"}, `
    + `${readTiming} GPU wait ${gpuWaitMs.toFixed(1)} ms total ${totalMs.toFixed(1)} ms`,
  );
  emit({
    stage: "ready",
    message: "WebGPU reducer ready",
    detail: `Loaded ${(fetchBytes / 1e9).toFixed(2)} GB detector BF columns`,
    current: activeSourceIndices.length,
    total: activeSourceIndices.length,
    percent: 100,
    sourceFrames: plane,
  });
  return { gqkChunks, chunkBfCounts, residentBytes };
  } catch (error) {
    await previous;
    pending.forEach(buffer => buffer.destroy());
    gqkChunks.forEach(buffer => buffer.destroy());
    throw error;
  } finally {
    if (source.kind !== "full_detector" || !source.resident) ans?.dispose();
  }
}

function chooseChunkCapacity(nbf: number, plane: number, device: GPUDevice): number {
  const maxStorage = device.limits.maxStorageBufferBindingSize || Number.MAX_SAFE_INTEGER;
  const maxBuffer = device.limits.maxBufferSize || Number.MAX_SAFE_INTEGER;
  const safeStorage = maxStorage === Number.MAX_SAFE_INTEGER ? maxStorage : Math.floor(maxStorage * 0.75);
  const safeBuffer = maxBuffer === Number.MAX_SAFE_INTEGER ? maxBuffer : Math.floor(maxBuffer * 0.75);
  // A mobile Adreno can advertise a 2 GiB binding yet lose the device when a
  // single decode and FFT submission spans hundreds of full scan planes.
  const batchBytes = device.adapterInfo?.vendor?.toLowerCase() === "qualcomm"
    ? Math.max(16 * 1024 * 1024, REDUCE_BF_GROUP * plane * 8) : MAX_GQK_CHUNK_BYTES;
  const maxBytes = Math.max(1, Math.min(safeStorage, safeBuffer, batchBytes));
  const maxByComplexBuffer = Math.max(1, Math.floor(maxBytes / (plane * 8)));
  const raw = Math.max(1, Math.min(nbf, MAX_BF_WORKGROUPS_PER_SUBMIT, maxByComplexBuffer));
  if (raw <= REDUCE_BF_GROUP) return raw;
  return Math.max(REDUCE_BF_GROUP, Math.floor(raw / REDUCE_BF_GROUP) * REDUCE_BF_GROUP);
}

type AgreementOptions = Pick<WebGPUReconstructionOptions, "higherOrder" | "sample" | "scanDetectorRotationDeg"> & {
  /** Alternating 0.05 inverse Å rings for the fifth-order consistency check. */
  ring?: 0 | 1;
};

export class WebGPUDirectBackend implements DirectProtocol<WebGPUDirectResult> {
  readonly backend = "webgpu" as const;
  readonly precision = {
    real_dtype: "float32",
    complex_dtype: "complex64",
  } as const;
  private outputPlan: {
    n: number; thick: boolean; owned: GPUBuffer[];
    phase: GPUBuffer; stage: GPUBuffer; params: GPUBuffer; chunks: GPUBuffer[];
    sum: GPUComputePipeline; rows: GPUComputePipeline; cols: GPUComputePipeline;
    sumGroups: GPUBindGroup[]; rowGroup: GPUBindGroup; colGroup: GPUBindGroup;
    probePipeline: GPUComputePipeline; probeGroup: GPUBindGroup;
    mirrorPipeline: GPUComputePipeline; mirrorGroup: GPUBindGroup;
  } | null = null;
  private agreementPlan: {
    rotation: number; retainSpectrum: boolean; rings: Uint8Array; bands: {frequencies: Uint32Array; indices: GPUBuffer}[]; owned: GPUBuffer[];
    paramsChunks: GPUBuffer[]; gqkChunks: GPUBuffer[]; chunkBfCounts: number[]; activeBfCount: number;
    bfGeom: GPUBuffer; bfTrig: GPUBuffer; qx: GPUBuffer; qy: GPUBuffer; aberrations: GPUBuffer;
    agreement: GPUComputePipeline; gradient: GPUComputePipeline; probePipeline: GPUComputePipeline;
  } | null = null;
  private cal: DirectCal;
  private n: SupportedDirectSize;
  private plane: number;
  private source: DetectorDirectSource;
  private device: GPUDevice | null = null;
  private buffers: DirectBuffers | null = null;
  private pipelines: DirectPipelines | null = null;
  private bindGroups: DirectBindGroups | null = null;
  private setupPromise: Promise<void> | null = null;
  private operationQueue: Promise<void> = Promise.resolve();
  private initialized = false;
  private loadedBfCount = 0;
  private geometryRotationDeg: number | null = null;
  private geometryBfCount: number | null = null;
  private onProgress: WebGPUProgressHandler | null = null;
  private closed = false;

  // `calJson` is the folder's `snapshots/cal.json`; the source holds exact BF columns or a full QEM.
  constructor(
    calJson: string,
    detectorSource: { json: string; resident?: RansResidentSet },
  ) {
    if (!globalThis.isSecureContext || !navigator.gpu) {
      throw new Error(
        "Direct ptychography WebGPU folder needs browser WebGPU. Open it from http://127.0.0.1, localhost, or HTTPS; LAN-IP HTTP pages cannot use WebGPU.",
      );
    }
    const calibration = JSON.parse(calJson) as BrowserCalibration;
    // Earlier calibrations name other quantities (num_bf, rotation_angle_deg below 180); none is reinterpreted.
    if (calibration.schema !== BROWSER_CALIBRATION_SCHEMA) {
      throw new Error(`Direct ptychography calibration schema ${JSON.stringify(calibration.schema)} is not read; this engine reads ${BROWSER_CALIBRATION_SCHEMA}. Export the ShowPtycho folder again.`);
    }
    this.cal = {
      ...calibration,
      semiangle_rad: calibration.semiangle_mrad * 1e-3,
      angular_sampling_rad: [calibration.detector_sampling_mrad[0] * 1e-3, calibration.detector_sampling_mrad[1] * 1e-3],
    };
    if (
      this.cal.precision?.real_dtype !== "float32"
      || this.cal.precision?.complex_dtype !== "complex64"
    ) {
      throw new Error(
        "Direct ptychography WebGPU requires the quantem.gpu float32/complex64 direct ptychography precision contract.",
      );
    }
    const ny = Number(this.cal.g_shape?.[1] || 0);
    const nx = Number(this.cal.g_shape?.[2] || 0);
    const n = supportedDirectSize(ny === nx ? ny : 0);
    if (!n) {
      throw new Error(
        `WebGPU direct ptychography supports square ${SUPPORTED_DIRECT_SIZES.join("/")} scan grids, got ${this.cal.g_shape?.join("x")}`,
      );
    }
    this.n = n;
    this.plane = n * n;
    this.source = JSON.parse(detectorSource.json) as DetectorDirectSource;
    if (this.source.kind === "full_detector") this.source.resident = detectorSource.resident;
    if (this.source.kind === "full_detector") {
      fullDetectorColumns(this.cal, this.source);
    } else if (this.source.kind !== "bf_columns") {
      throw new Error("Direct ptychography WebGPU reads BF columns or full-detector QEM; export this unsupported source again.");
    }
  }

  get readyLabel(): string {
    return `WebGPU direct ptychography ready: ${this.n}x${this.n}, ${this.cal.num_bright_field_pixels} BF pixels, ${this.source.kind === "full_detector" ? "full-detector QEM" : "detector BF columns"}`;
  }

  async prepare(): Promise<void> {
    await this.prepareBfCount(this.cal.num_bright_field_pixels);
  }

  private async prepareAgreement(rotation = baseRotationDeg(this.cal)) {
    if (this.agreementPlan?.rotation === rotation) return this.agreementPlan;
    this.releaseAgreement();
    this.resetGpuBuffers();
    const device = (this.source.kind === "full_detector" ? this.source.resident?.device : null) ?? await getGPUDevice(), n = this.n, cal = this.cal;
    if (!device) throw new Error("Direct ptychography fitting requires a WebGPU adapter.");
    const band: number[] = [];
    const width = n / 2 + 1;
    for (let row = 0; row < n; row++) {
      for (let col = 0; col < width; col++) {
        const q = Math.fround(Math.hypot(Math.fround(cal.scan_q_row[row]), Math.fround(cal.scan_q_col[col])));
        if (q > Math.fround(0.2) && q < Math.fround(0.9)) band.push(row * width + col);
      }
    }
    if (!band.length) throw new Error("The scan has no measured frequencies in the 0.2–0.9 inverse Å fit band.");
    const active = collectActiveBfIndices(cal, cal.num_bright_field_pixels, rotation);
    if (!active.length) throw new Error("The calibrated aperture contains no bright-field pixels.");
    if (active.length * band.length * 8 > gqkBudgetBytes(cal)) throw new Error("The full fit band exceeds the calibrated GPU budget.");
    const owned: GPUBuffer[] = [];
    const keep = (buffer: GPUBuffer) => { owned.push(buffer); return buffer; };
    const storage = (a: Float32Array, label: string) => keep(makeBufferFromArray(device, a, GPUBufferUsage.STORAGE, label));
    try {
      const frequencies = Uint32Array.from(band);
      const indices = keep(makeBufferFromBytes(device, new Uint8Array(frequencies.buffer), GPUBufferUsage.STORAGE, "fit band"));
      // Retain the Hermitian spectrum when it fits a small working set. Larger
      // acquisitions stream the final phase so their full spectrum never needs to fit.
      const retainSpectrum = active.length * storedPlaneFor(n) * GQK_BYTES_PER_VALUE <=
        Math.min(320 * 1024 * 1024, gqkBudgetBytes(cal));
      const module = device.createShaderModule({code: makeDirectShader(n, true, n, true, retainSpectrum ? storedPlaneFor(n) : band.length)});
      const pipeline = (entryPoint: string) => device.createComputePipelineAsync({layout: "auto", compute: {module, entryPoint}});
      const [probePipeline, agreement, gradient] = await Promise.all([pipeline("prepareProbe"), pipeline("modelAgreement"), pipeline("modelGradient")]);
      const {gqkChunks, chunkBfCounts} = await buildBfColumnGqkChunks(device, cal, this.source, n, n*n, active,
        chooseChunkCapacity(active.length, n*n, device), this.onProgress ?? undefined, retainSpectrum ? undefined : {indices, size: band.length});
      owned.push(...gqkChunks);
      const {geom, trig} = packGeometry(cal, active, rotation);
      this.device = device;
      const rings = Uint8Array.from(frequencies, index => {
        const q = Math.hypot(Math.fround(cal.scan_q_row[Math.floor(index / width)]), Math.fround(cal.scan_q_col[index % width]));
        return Math.floor(q / 0.05) % 2;
      });
      // Pack each ring in original frequency order. Its evaluations only
      // visit that ring while reading the shared resident spectrum.
      const bands = [undefined, 0, 1].map(ring => {
        const selected = [];
        for (let i = 0; i < band.length; i++) {
          if (ring === undefined || rings[i] === ring) selected.push(i);
        }
        const mapping = Uint32Array.from(selected.flatMap(i => [band[i], retainSpectrum ? band[i] : i]));
        return {
          frequencies: Uint32Array.from(selected, i => band[i]),
          indices: keep(makeBufferFromBytes(device, new Uint8Array(mapping.byteLength ? mapping.buffer : new ArrayBuffer(8)),
            GPUBufferUsage.STORAGE, "fit frequency map")),
        };
      });
      this.agreementPlan = {rotation, retainSpectrum, rings, bands, owned, probePipeline, agreement, gradient, activeBfCount: active.length,
        gqkChunks, chunkBfCounts, bfGeom: storage(geom, "fit BF coordinates"), bfTrig: storage(trig, "fit BF angles"),
        qx: storage(Float32Array.from(cal.scan_q_row), "fit scan row frequencies"),
        qy: storage(Float32Array.from(cal.scan_q_col), "fit scan column frequencies"),
        aberrations: storage(packAberrations(0, 0, 0).data, "fit aberrations"),
        paramsChunks: chunkBfCounts.map(() => keep(device.createBuffer({size: DIRECT_PARAMS_BYTES,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST})))};
      return this.agreementPlan;
    } catch (error) {
      owned.forEach(buffer => buffer.destroy());
      throw error;
    }
  }

  // A successful trust-region step asks for a gradient at the point just scored.
  private scoredPoint: {key: string; terms: GPUBuffer; score: number} | null = null;
  private releaseAgreement() {
    this.scoredPoint?.terms.destroy();
    this.scoredPoint = null;
    this.agreementPlan?.owned.forEach(buffer => buffer.destroy());
    this.agreementPlan = null;
  }

  /** Native-grid fifth-order model agreement over the measured 0.2–0.9 inverse Å band. */
  async modelAgreement(c10A: number, c12A: number, phi12Rad: number,
    options: AgreementOptions = {}): Promise<number> {
    return (await this.evaluateAgreement(c10A, c12A, phi12Rad, options, false)).score;
  }

  async modelAgreementGradient(c10A: number, c12A: number, phi12Rad: number,
    options: AgreementOptions = {}) {
    return this.evaluateAgreement(c10A, c12A, phi12Rad, options, true);
  }

  /** Independent points share the resident fit spectrum and queue their readbacks together. */
  async evaluateFitBatch(points: [number, number, number, AgreementOptions][], seedGradients: boolean) {
    if (!points.length || points.length > 8) throw Error("Select between one and eight independent fit points.");
    return this.runExclusive(async () => {
      const rotation = points[0][3].scanDetectorRotationDeg;
      if (points.some(p => p[3].scanDetectorRotationDeg !== rotation || p[3].ring !== undefined))
        throw Error("Batched fit points require the same geometry and full frequency band.");
      const buffers = await this.prepareAgreement(rotation);
      const jobs = points.map(([c10,c12,phi,options]) =>
        this.evaluatePrepared(buffers,c10,c12,phi,options,seedGradients,seedGradients,false));
      const results = await Promise.allSettled(jobs);
      return results.map(entry => {
        if (entry.status === "rejected") throw entry.reason;
        const r = entry.value;
        if (!seedGradients) return r.score;
        if (!r.ringResults) throw Error("No measured frequencies in the fitting band.");
        return [r,...r.ringResults].map(v => [v.score,Array.from(v.gradient)]);
      });
    });
  }

  private async evaluateAgreement(c10A: number, c12A: number, phi12Rad: number,
    options: AgreementOptions, derivatives: boolean) {
    return this.runExclusive(async () => this.evaluatePrepared(
      await this.prepareAgreement(options.scanDetectorRotationDeg), c10A, c12A, phi12Rad, options, derivatives));
  }

  private async evaluatePrepared(buffers: NonNullable<WebGPUDirectBackend["agreementPlan"]>,
    c10A: number, c12A: number, phi12Rad: number, options: AgreementOptions,
    derivatives: boolean, splitRings = false, cachePoint = true) {
    const device = this.device!, n = this.n;
    const {agreement, probePipeline} = buffers;
    const {frequencies: band, indices} = buffers.bands[options.ring === undefined ? 0 : options.ring + 1];
    if (!band.length) return {score: 0, gradient: new Float64Array(derivatives ? 28 : 0)};
    const owned: GPUBuffer[] = [];
    const allocate = (size: number, usage: GPUBufferUsageFlags) => {
      const b = device.createBuffer({size, usage});
      owned.push(b);
      return b;
    };
    const group = (p: GPUComputePipeline, entries: [number, GPUBuffer][]) => device.createBindGroup({
      layout: p.getBindGroupLayout(0), entries: entries.map(([binding, buffer]) => ({binding, resource: {buffer}})),
    });
    const width = n / 2 + 1;
    try {
      const key = JSON.stringify([c10A, c12A, phi12Rad, options.sample, options.higherOrder, options.ring]);
      const cached = cachePoint && this.scoredPoint?.key === key ? this.scoredPoint : null;
      if (cached && !derivatives) return {score: cached.score, gradient: new Float64Array(0)};
      const terms = cached?.terms ?? allocate(band.length * 16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
      const probe = allocate(buffers.activeBfCount * 8, GPUBufferUsage.STORAGE);
      const phi = phi12Rad * 180 / Math.PI;
      const abr = packAberrations(c10A, c12A, phi, options.higherOrder);
      device.queue.writeBuffer(buffers.aberrations, 0, abr.data as Float32Array<ArrayBuffer>);
      const encoder = device.createCommandEncoder();
      if (!cached) encoder.clearBuffer(terms);
      let offset = 0;
      for (const [i, count] of buffers.chunkBfCounts.entries()) {
        const params = buffers.paramsChunks[i];
        device.queue.writeBuffer(params, 0, makeParams(this.cal, n, c10A, c12A, phi,
          this.cal.num_bright_field_pixels, offset, count, false, buffers.activeBfCount, true, options.sample));
        if (cached) {
          offset += count;
          continue;
        }
        if (i === 0) {
          const pass = encoder.beginComputePass();
          pass.setPipeline(probePipeline);
          pass.setBindGroup(0, group(probePipeline, [[0, params], [3, buffers.bfGeom], [4, buffers.bfTrig],
            [12, buffers.aberrations], [13, probe]]));
          pass.dispatchWorkgroups(Math.ceil(buffers.activeBfCount / 256));
          pass.end();
        }
        const pass = encoder.beginComputePass();
        pass.setPipeline(agreement);
        pass.setBindGroup(0, group(agreement, [[0, params], [1, buffers.gqkChunks[i]], [3, buffers.bfGeom],
          [5, buffers.qx], [6, buffers.qy], [12, buffers.aberrations], [13, probe], [14, terms], [15, indices]]));
        pass.dispatchWorkgroups(Math.ceil(band.length / 64));
        pass.end();
        offset += count;
      }
      let gradientTerms: GPUBuffer | undefined;
      if (derivatives) {
        gradientTerms = allocate(band.length * 28 * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
        encoder.clearBuffer(gradientTerms);
        for (const [i, chunk] of buffers.gqkChunks.entries()) {
          const pass = encoder.beginComputePass();
          pass.setPipeline(buffers.gradient);
          pass.setBindGroup(0, group(buffers.gradient, [[0,buffers.paramsChunks[i]], [1,chunk], [3,buffers.bfGeom],
            [5,buffers.qx], [6,buffers.qy], [12,buffers.aberrations], [14,terms], [15,indices], [16,gradientTerms]]));
          pass.dispatchWorkgroups(Math.ceil(band.length / 64));
          pass.end();
        }
      }
      const scoreBytes = cached ? 0 : band.length * 16;
      const gradientBytes = gradientTerms ? band.length * 28 * 4 : 0;
      const read = allocate(scoreBytes + gradientBytes, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
      if (scoreBytes) encoder.copyBufferToBuffer(terms, 0, read, 0, scoreBytes);
      if (gradientTerms) encoder.copyBufferToBuffer(gradientTerms, 0, read, scoreBytes, gradientBytes);
      device.queue.submit([encoder.finish()]);
      await read.mapAsync(GPUMapMode.READ);
      const data = new Float32Array(read.getMappedRange());
      const ringResults = splitRings ? [0, 1].map(() => ({score:0, gradient:new Float64Array(28)})) : undefined;
      let score = cached?.score ?? 0;
      if (!cached) {
        for (let q = 0; q < band.length; q++) {
          const i = q * 4, col = band[q] % width;
          const weight = col === 0 || col === n / 2 ? 1 : 2;
          if (data[i+2] > 0) {
            const value = weight * (data[i] ** 2 + data[i+1] ** 2) / data[i+2];
            score += value;
            if (ringResults) ringResults[buffers.rings[q]].score += value;
          }
        }
        if (cachePoint) {
          this.scoredPoint?.terms.destroy();
          this.scoredPoint = {key, terms, score};
          owned.splice(owned.indexOf(terms), 1);
        }
      }
      const gradient = new Float64Array(derivatives ? 28 : 0);
      if (gradientTerms) {
        const g = data.subarray(scoreBytes / 4);
        for (let j = 0; j < 28; j++) {
          for (let q = 0; q < band.length; q++) {
            const col = band[q] % width;
            const value = g[j*band.length + q] * (col === 0 || col === n/2 ? 1 : 2);
            gradient[j] += value;
            if (ringResults) ringResults[buffers.rings[q]].gradient[j] += value;
          }
        }
      }
      read.unmap();
      return {score, gradient, ringResults};
    } finally {
      owned.forEach(b => b.destroy());
    }
  }

  async fit(
    options: WebGPUOptimizationOptions,
  ): Promise<DirectOptimizationResult> {
    return fit(options);
  }

  close(): void {
    this.closed = true;
    this.onProgress = null;
    // Submitted work may still reference these buffers when a view closes.
    void this.operationQueue.then(() => {
      this.releaseAgreement();
      this.resetGpuBuffers();
      this.device = null;
    });
  }

  setProgressHandler(handler: WebGPUProgressHandler | null): void {
    this.onProgress = handler;
  }

  private emitProgress(progress: Omit<WebGPULoadProgress, "totalBf">): void {
    this.onProgress?.({
      totalBf: this.cal.num_bright_field_pixels,
      elapsedMs: undefined,
      ...progress,
    });
  }

  private async setup(requiredBfCount = this.cal.num_bright_field_pixels, scanDetectorRotationDeg?: number): Promise<void> {
    this.releaseAgreement();
    const capacity = this.clampBfCount(requiredBfCount);
    if (this.initialized && this.residentCovers(capacity, scanDetectorRotationDeg)) {
      this.updateGeometry(capacity, scanDetectorRotationDeg);
      return;
    }
    if (this.setupPromise) {
      await this.setupPromise;
      if (this.initialized && this.residentCovers(capacity, scanDetectorRotationDeg)) {
        this.updateGeometry(capacity, scanDetectorRotationDeg);
        return;
      }
    }
    this.setupPromise = this.setupOnce(capacity, scanDetectorRotationDeg);
    try {
      await this.setupPromise;
    } finally {
      this.setupPromise = null;
    }
    this.updateGeometry(capacity, scanDetectorRotationDeg);
  }

  // The resident G(q,k) answer `count` BF pixels when they hold every pixel of its strided subset
  // (collectActiveBfIndices): a full load serves every smaller count without rebuilding.
  private residentCovers(count: number, scanDetectorRotationDeg?: number): boolean {
    if (this.loadedBfCount < count) return false;
    if (this.loadedBfCount === count) return true;
    const loaded = new Set(this.buffers!.activeSourceIndices);
    return collectActiveBfIndices(this.cal, count, scanDetectorRotationDeg).every((index) => loaded.has(index));
  }

  private async runExclusive<T>(operation: () => Promise<T>): Promise<T> {
    const previous = this.operationQueue;
    let release = () => {};
    this.operationQueue = new Promise<void>((resolve) => {
      release = resolve;
    });
    await previous.catch(() => undefined);
    try {
      if (this.closed) throw new Error("Direct ptychography backend is closed; create a new backend to reconstruct again.");
      return await operation();
    } finally {
      release();
    }
  }

  private resetGpuBuffers(): void {
    this.outputPlan?.owned.forEach(buffer => buffer.destroy());
    this.outputPlan = null;
    if (this.buffers) {
      this.buffers.gqkChunks.forEach(buffer => buffer.destroy());
      this.buffers.paramsChunks.forEach(buffer => buffer.destroy());
      this.buffers.params.destroy();
      this.buffers.aberrations.destroy();
      this.buffers.stage.destroy();
      this.buffers.phaseStack.destroy();
      this.buffers.partialSum.destroy();
      this.buffers.partialSumSq.destroy();
      this.buffers.phase.destroy();
      this.buffers.variance.destroy();
      this.buffers.bfGeom.destroy();
      this.buffers.bfTrig.destroy();
      this.buffers.qx.destroy();
      this.buffers.qy.destroy();
    }
    this.buffers = null;
    this.pipelines = null;
    this.bindGroups = null;
    this.initialized = false;
    this.loadedBfCount = 0;
    this.geometryRotationDeg = null;
    this.geometryBfCount = null;
  }

  private async setupOnce(capacity: number, scanDetectorRotationDeg?: number): Promise<void> {
    if (this.initialized && this.residentCovers(capacity, scanDetectorRotationDeg)) return;
    if (this.initialized || this.buffers) this.resetGpuBuffers();
    const setupT0 = performance.now();
    const n = this.n;
    const plane = this.plane;
    this.emitProgress({
      stage: "device",
      message: "Starting WebGPU for ptychography review",
      detail: `Preparing ${capacity}/${this.cal.num_bright_field_pixels} BF pixels at ${n}x${n}`,
      current: 0,
      total: capacity,
      percent: 0,
      activeBf: capacity,
    });
    const device = this.device ?? (this.source.kind === "full_detector" ? this.source.resident?.device : null) ?? await getGPUDevice();
    if (!device) throw new Error("WebGPU device unavailable");
    const fftConfig = getWebGPUFFTConfig(n);
    if (!fftConfig) throw new Error(`Missing WebGPU direct ptychography FFT configuration for ${n}.`);
    if (
      device.limits.maxComputeInvocationsPerWorkgroup < fftConfig.workgroupSize
      || device.limits.maxComputeWorkgroupSizeX < fftConfig.workgroupSize
    ) {
      throw new Error(
        `WebGPU adapter supports workgroup x=${device.limits.maxComputeWorkgroupSizeX}, `
        + `invocations=${device.limits.maxComputeInvocationsPerWorkgroup}; `
        + `${n}x${n} direct ptychography needs ${fftConfig.workgroupSize} threads plus ${(n * 8 / 1024).toFixed(1)} KB workgroup storage.`,
      );
    }
    this.device = device;
    const nbf = Math.max(1, Math.min(this.cal.num_bright_field_pixels, Math.round(capacity)));
    let activeSourceIndices = collectActiveBfIndices(this.cal, nbf, scanDetectorRotationDeg);
    // Full-BF scientific runs must use the identical detector evidence on
    // every backend. A memory budget may reject a run, but it must never
    // silently replace the exact objective with a uniformly strided subset.
    const perBfBytes = storedPlaneFor(n) * GQK_BYTES_PER_VALUE;
    const budgetMaxBf = Math.max(1, Math.floor(gqkBudgetBytes(this.cal) / perBfBytes));
    if (activeSourceIndices.length > budgetMaxBf) {
      throw new Error(
        `Exact full-BF WebGPU direct ptychography needs ${activeSourceIndices.length} active BF pixels, `
        + `but the ${(gqkBudgetBytes(this.cal) / 1e9).toFixed(1)} GB GPU budget fits ${budgetMaxBf} `
        + `(${(perBfBytes / 1e6).toFixed(1)} MB/BF in ${GQK_MODE} mode). `
        + "Export the review again with the current package and use a GPU with enough memory; "
        + "scientific fitting never subsamples automatically.",
      );
    }
    const nonzeroBfCount = activeSourceIndices.length;
    const activeIndices = nonzeroBfCount > 0 ? activeSourceIndices : new Uint32Array([0]);
    const activeBfCount = activeIndices.length;
    const bytesPerBf = plane * Float32Array.BYTES_PER_ELEMENT * 2;
    const expectedBytes = nbf * bytesPerBf;
    const activeBytes = activeBfCount * bytesPerBf;
    const storageChunkCapacity = chooseChunkCapacity(activeBfCount, plane, device);
    const dispatchCount = Math.ceil(activeBfCount / storageChunkCapacity);
    const partialGroups = Math.ceil(activeBfCount / REDUCE_BF_GROUP);
    console.log(
      `[direct] WebGPU direct ptychography setup start ${nbf}/${this.cal.num_bright_field_pixels} BF `
      + `${(expectedBytes / 1e6).toFixed(1)} MB source; `
      + `${nonzeroBfCount}/${nbf} active aperture BF `
      + `${(activeBytes / 1e6).toFixed(1)} MB upload `
      + `in chunks of ${storageChunkCapacity}`,
    );
    console.log(
      `[direct] WebGPU limits buffer=${device.limits.maxBufferSize} `
      + `storageBinding=${device.limits.maxStorageBufferBindingSize} `
      + `storageBuffers=${device.limits.maxStorageBuffersPerShaderStage}`,
    );
    this.emitProgress({
      stage: "pipeline",
      message: "Allocating WebGPU BF reducer buffers",
      detail: `${activeBfCount}/${this.cal.num_bright_field_pixels} active BF pixels, ${(activeBytes / 1e6).toFixed(1)} MB working data`,
      current: 1,
      total: 5,
      percent: 8,
      activeBf: activeBfCount,
    });
    const { geom, trig } = packGeometry(this.cal, activeIndices, scanDetectorRotationDeg);
    const maxStorage = device.limits.maxStorageBufferBindingSize || 0;
    const maxBuffer = device.limits.maxBufferSize || 0;
    const largestStorage = Math.max(storageChunkCapacity * plane * 8, storageChunkCapacity * plane * 4);
    if ((maxStorage > 0 && largestStorage > maxStorage) || (maxBuffer > 0 && largestStorage > maxBuffer)) {
      throw new Error(
        `WebGPU buffer limit is too small for chunked ${nbf}x${n}x${n}: `
        + `need ${(largestStorage / 1e6).toFixed(1)} MB, `
        + `maxStorage ${(maxStorage / 1e6).toFixed(1)} MB, maxBuffer ${(maxBuffer / 1e6).toFixed(1)} MB.`,
      );
    }
    const module = device.createShaderModule({ code: makeDirectShader(n), label: `Direct ptychography WGSL ${n} ${GQK_MODE}` });
    const thickModule = device.createShaderModule({ code: makeDirectShader(n, true), label: `Direct ptychography thick WGSL ${n} ${GQK_MODE}` });
    const [
      rows, cols, reducePartial, finalizeGroups, objSum, objFftRows, objFftCols, rowsThick, objSumThick,
    ] = await Promise.all([
      device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "directRows" } }),
      device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "directCols" } }),
      device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "reducePartialGroups" } }),
      device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "finalizePartialGroups" } }),
      device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "directObjSum" } }),
      device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "directObjFftRows" } }),
      device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "directObjFftCols" } }),
      device.createComputePipelineAsync({ layout: "auto", compute: { module: thickModule, entryPoint: "directRows" } }),
      device.createComputePipelineAsync({ layout: "auto", compute: { module: thickModule, entryPoint: "directObjSum" } }),
    ]);
    const pipelines: DirectPipelines = {
      rows,
      cols,
      reducePartial,
      finalizeGroups,
      objSum,
      objFftRows,
      objFftCols,
      rowsThick,
      objSumThick,
    };
    const built = await buildBfColumnGqkChunks(
      device,
      this.cal,
      this.source,
      n,
      plane,
      activeIndices,
      storageChunkCapacity,
      (progress) => this.onProgress?.({
        ...progress,
        elapsedMs: progress.elapsedMs ?? performance.now() - setupT0,
      }),
    );
    const { gqkChunks, chunkBfCounts, residentBytes } = built;
    console.log(
      `[direct] gqk mode ${GQK_MODE}: resident ${(residentBytes / 1e9).toFixed(2)} GB `
      + `for ${chunkBfCounts.reduce((acc, bf) => acc + bf, 0)} active BF pixels`,
    );
    const buffers: DirectBuffers = {
      params: device.createBuffer({ size: DIRECT_PARAMS_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label: "direct params" }),
      paramsChunks: Array.from({ length: dispatchCount }, (_, index) => device.createBuffer({
        size: DIRECT_PARAMS_BYTES,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        label: `direct params chunk ${index}`,
      })),
      aberrations: makeBufferFromArray(
        device,
        packAberrations(0, 0, 0).data,
        GPUBufferUsage.STORAGE,
        "direct aberrations",
      ),
      gqkChunks,
      gqkMode: GQK_MODE,
      gqkResidentBytes: residentBytes,
      chunkBfCounts,
      chunkCapacity: storageChunkCapacity,
      stage: device.createBuffer({ size: storageChunkCapacity * plane * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, label: "direct stage" }),
      phaseStack: device.createBuffer({ size: storageChunkCapacity * plane * 4, usage: GPUBufferUsage.STORAGE, label: "direct phase stack" }),
      partialSum: device.createBuffer({ size: partialGroups * plane * 4, usage: GPUBufferUsage.STORAGE, label: "direct phase partial sum" }),
      partialSumSq: device.createBuffer({ size: partialGroups * plane * 4, usage: GPUBufferUsage.STORAGE, label: "direct phase partial sumsq" }),
      partialGroups,
      activeBfCount,
      nonzeroBfCount,
      activeSourceIndices: activeIndices,
      phase: device.createBuffer({ size: plane * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, label: "direct phase" }),
      variance: device.createBuffer({ size: plane * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, label: "direct variance" }),
      bfGeom: makeBufferFromArray(device, geom, GPUBufferUsage.STORAGE, "direct bf geometry"),
      bfTrig: makeBufferFromArray(device, trig, GPUBufferUsage.STORAGE, "direct bf trig"),
      qx: makeBufferFromArray(device, new Float32Array(this.cal.scan_q_row), GPUBufferUsage.STORAGE, "direct qx"),
      qy: makeBufferFromArray(device, new Float32Array(this.cal.scan_q_col), GPUBufferUsage.STORAGE, "direct qy"),
    };
    device.pushErrorScope("validation");
    // directRows and directObjSum (thin and thick variants) read the same resources
    const correctionEntries = (params: GPUBuffer, index: number): GPUBindGroupEntry[] => [
      { binding: 0, resource: { buffer: params } },
      { binding: 1, resource: { buffer: buffers.gqkChunks[index] } },
      { binding: 2, resource: { buffer: buffers.stage } },
      { binding: 3, resource: { buffer: buffers.bfGeom } },
      { binding: 4, resource: { buffer: buffers.bfTrig } },
      { binding: 5, resource: { buffer: buffers.qx } },
      { binding: 6, resource: { buffer: buffers.qy } },
      { binding: 12, resource: { buffer: buffers.aberrations } },
    ];
    const bindGroups: DirectBindGroups = {
      rows: buffers.paramsChunks.map((params, index) => device.createBindGroup({
        layout: pipelines.rows.getBindGroupLayout(0),
        entries: correctionEntries(params, index),
      })),
      rowsThick: buffers.paramsChunks.map((params, index) => device.createBindGroup({
        layout: pipelines.rowsThick.getBindGroupLayout(0),
        entries: correctionEntries(params, index),
      })),
      cols: buffers.paramsChunks.map((params) => device.createBindGroup({
        layout: pipelines.cols.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: params } },
          { binding: 2, resource: { buffer: buffers.stage } },
          { binding: 3, resource: { buffer: buffers.bfGeom } },
          { binding: 7, resource: { buffer: buffers.phaseStack } },
        ],
      })),
      reducePartial: buffers.paramsChunks.map((params) => device.createBindGroup({
        layout: pipelines.reducePartial.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: params } },
          { binding: 3, resource: { buffer: buffers.bfGeom } },
          { binding: 7, resource: { buffer: buffers.phaseStack } },
          { binding: 8, resource: { buffer: buffers.partialSum } },
          { binding: 9, resource: { buffer: buffers.partialSumSq } },
        ],
      })),
      finalizeGroups: device.createBindGroup({
        layout: pipelines.finalizeGroups.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: buffers.params } },
          { binding: 8, resource: { buffer: buffers.partialSum } },
          { binding: 9, resource: { buffer: buffers.partialSumSq } },
          { binding: 10, resource: { buffer: buffers.phase } },
          { binding: 11, resource: { buffer: buffers.variance } },
        ],
      }),
      objSum: buffers.paramsChunks.map((params, index) => device.createBindGroup({
        layout: pipelines.objSum.getBindGroupLayout(0),
        entries: correctionEntries(params, index),
      })),
      objSumThick: buffers.paramsChunks.map((params, index) => device.createBindGroup({
        layout: pipelines.objSumThick.getBindGroupLayout(0),
        entries: correctionEntries(params, index),
      })),
      objFftRows: device.createBindGroup({
        layout: pipelines.objFftRows.getBindGroupLayout(0),
        // directObjFftRows uses only the stage plane (sizes are compile-time
        // constants), so the auto layout exposes just binding 2.
        entries: [
          { binding: 2, resource: { buffer: buffers.stage } },
        ],
      }),
      objFftCols: device.createBindGroup({
        layout: pipelines.objFftCols.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: buffers.params } },
          { binding: 2, resource: { buffer: buffers.stage } },
          { binding: 10, resource: { buffer: buffers.phase } },
        ],
      }),
    };
    const bindError = await device.popErrorScope();
    if (bindError) {
      throw new Error(`WebGPU direct ptychography bind group validation failed: ${bindError.message}`);
    }
    this.buffers = buffers;
    this.pipelines = pipelines;
    this.bindGroups = bindGroups;
    this.initialized = true;
    this.loadedBfCount = nbf;
    this.geometryRotationDeg = scanDetectorRotationDeg == null || !Number.isFinite(scanDetectorRotationDeg) ? baseRotationDeg(this.cal) : Number(scanDetectorRotationDeg);
    this.geometryBfCount = null;
    console.log(
      `[direct] WebGPU direct ptychography setup ready ${nbf}/${this.cal.num_bright_field_pixels} BF `
      + `(${nonzeroBfCount} active) in ${(performance.now() - setupT0).toFixed(1)} ms`,
    );
    this.emitProgress({
      stage: "ready",
      message: "WebGPU ptychography reducer ready",
      detail: `${nbf}/${this.cal.num_bright_field_pixels} BF pixels prepared in ${(performance.now() - setupT0).toFixed(0)} ms`,
      current: nbf,
      total: this.cal.num_bright_field_pixels,
      percent: 100,
      activeBf: activeBfCount,
      elapsedMs: performance.now() - setupT0,
    });
  }

  private clampBfCount(requested: number | undefined): number {
    if (requested == null || !Number.isFinite(requested)) return this.cal.num_bright_field_pixels;
    return Math.max(1, Math.min(this.cal.num_bright_field_pixels, Math.round(requested)));
  }

  async prepareBfCount(count: number): Promise<number> {
    const requested = this.clampBfCount(count);
    await this.runExclusive(async () => {
      await this.setup(requested);
    });
    return this.loadedBfCount;
  }

  // Write the resident BF geometry for this rotation. A count below the resident one keeps its
  // strided subset: every other resident BF pixel gets aperture 0, which each kernel skips, and
  // the mean divides by `count`.
  private updateGeometry(count: number, scanDetectorRotationDeg?: number): void {
    const device = this.device;
    const buffers = this.buffers;
    if (!device || !buffers) return;
    const requested = scanDetectorRotationDeg == null || !Number.isFinite(scanDetectorRotationDeg) ? baseRotationDeg(this.cal) : Number(scanDetectorRotationDeg);
    const subset = count < this.loadedBfCount ? count : null;
    if (this.geometryRotationDeg != null && Math.abs(requested - this.geometryRotationDeg) < 1e-9 && this.geometryBfCount === subset) return;
    const { geom, trig } = packGeometry(this.cal, buffers.activeSourceIndices, requested);
    if (subset !== null) {
      const keep = new Set(collectActiveBfIndices(this.cal, subset, scanDetectorRotationDeg));
      buffers.activeSourceIndices.forEach((index, i) => { if (!keep.has(index)) geom[i * 4 + 3] = 0; });
    }
    device.queue.writeBuffer(buffers.bfGeom, 0, geom as unknown as BufferSource);
    device.queue.writeBuffer(buffers.bfTrig, 0, trig as unknown as BufferSource);
    this.geometryRotationDeg = requested;
    this.geometryBfCount = subset;
  }

  // Reconstruct from the complete frequency grid. The fit only retains its
  // measured band, so stream the original columns again for the final phase.
  private async reconstructFitPhase(c10: number, c12: number, phi: number, options: WebGPUReconstructionOptions) {
    const plan = await this.prepareAgreement(options.scanDetectorRotationDeg);
    const device = this.device!, n = this.n, started = performance.now();
    const module = device.createShaderModule({code: makeDirectShader(n, true, n, true)});
    const pipelines = await Promise.all(["prepareProbe", "directObjSum", "directObjFftRows", "directObjFftCols"]
      .map(entryPoint => device.createComputePipelineAsync({layout: "auto", compute: {module, entryPoint}})));
    const [prepare, sum, rows, cols] = pipelines;
    const owned: GPUBuffer[] = [];
    const alloc = (size: number, usage: GPUBufferUsageFlags) => {
      const buffer = device.createBuffer({size, usage}); owned.push(buffer); return buffer;
    };
    const dispatch = (enc: GPUCommandEncoder, p: GPUComputePipeline, entries: [number, GPUBuffer][], x: number, y = 1) => {
      const pass = enc.beginComputePass(); pass.setPipeline(p);
      pass.setBindGroup(0, device.createBindGroup({layout: p.getBindGroupLayout(0),
        entries: entries.map(([binding, buffer]) => ({binding, resource: {buffer}}))}));
      pass.dispatchWorkgroups(x, y); pass.end();
    };
    const degrees = phi * 180 / Math.PI;
    const abr = packAberrations(c10, c12, degrees, options.higherOrder);
    const active = collectActiveBfIndices(this.cal, this.cal.num_bright_field_pixels, options.scanDetectorRotationDeg);
    const uniform = (offset: number, count: number) => makeParams(this.cal, n, c10, c12, degrees,
      this.cal.num_bright_field_pixels, offset, count, false, active.length, abr.active, options.sample);
    try {
      const stage = alloc(n*n*8, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
      const phase = alloc(n*n*4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const probe = alloc(active.length*8, GPUBufferUsage.STORAGE);
      const params = alloc(DIRECT_PARAMS_BYTES, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
      device.queue.writeBuffer(plan.aberrations, 0, abr.data as Float32Array<ArrayBuffer>);
      device.queue.writeBuffer(params, 0, uniform(0, 1));
      const init = device.createCommandEncoder(); init.clearBuffer(stage);
      dispatch(init, prepare, [[0, params], [3, plan.bfGeom], [4, plan.bfTrig], [12, plan.aberrations], [13, probe]],
        Math.ceil(active.length/256));
      device.queue.submit([init.finish()]);
      const sumChunk = (encoder: GPUCommandEncoder, buffer: GPUBuffer, offset: number, count: number) => {
        options.signal?.throwIfAborted();
        device.queue.writeBuffer(params, 0, uniform(offset, count));
        dispatch(encoder, sum, [[0, params], [1, buffer], [2, stage], [3, plan.bfGeom],
          [5, plan.qx], [6, plan.qy], [12, plan.aberrations], [13, probe]], Math.ceil(n*n/256));
      };
      if (plan.retainSpectrum) {
        let offset = 0;
        for (const [i, count] of plan.chunkBfCounts.entries()) {
          const encoder = device.createCommandEncoder();
          sumChunk(encoder, plan.gqkChunks[i], offset, count);
          device.queue.submit([encoder.finish()]);
          offset += count;
        }
      } else {
        await buildBfColumnGqkChunks(device, this.cal, this.source, n, n*n, active,
          chooseChunkCapacity(active.length, n*n, device), this.onProgress ?? undefined, undefined, sumChunk);
      }
      const end = device.createCommandEncoder();
      dispatch(end, rows, [[2, stage]], 1, n);
      dispatch(end, cols, [[0, params], [2, stage], [10, phase]], 1, n);
      device.queue.submit([end.finish()]);
      const pixels = await readF32(device, phase, n*n);
      return {phase: pixels, width: n, height: n, gpuMs: performance.now() - started,
        numBrightFieldPixels: this.cal.num_bright_field_pixels, scanDetectorRotationDeg: plan.rotation,
        loss: null, adapterInfo: getGPUInfo(), softwareAdapter: isSoftwareGPUAdapter()};
    } finally {
      owned.forEach(buffer => buffer.destroy());
    }
  }

  /** Full-field direct phase inversion. Damping acts on the coupled alias system. */
  async reconstructPhase(c10A: number, c12A: number, phi12Rad: number,
    options: Omit<AgreementOptions, "ring"> & {phaseDamp?: number; upsample?: "auto" | 1 | 2 | 3 | 4 | 8} = {}) {
    return this.runExclusive(async () => {
      const started = performance.now(), n = this.n, cal = this.cal;
      const rowStep = Math.fround(cal.scan_q_row[1]), colStep = Math.fround(cal.scan_q_col[1]);
      const required = 4 * cal.semiangle_rad / (cal.wavelength_A * n * Math.min(rowStep, colStep));
      const auto = [1,2,3,4].find(value => value >= required - 1e-12) ?? 4;
      const factor = options.upsample === undefined || options.upsample === "auto" ? auto : options.upsample;
      const size = n * factor, damping = options.phaseDamp ?? 0.3;
      if (!supportedDirectSize(size)) throw new Error(`No WebGPU FFT for the requested ${size} × ${size} phase grid.`);
      if (!(Number.isFinite(damping) && damping > 0)) throw new Error("Phase damping must be finite and positive.");
      const device = (this.source.kind === "full_detector" ? this.source.resident?.device : null) ?? await getGPUDevice();
      if (!device) throw new Error("Direct phase reconstruction needs WebGPU.");
      const rotation = options.scanDetectorRotationDeg ?? baseRotationDeg(cal);
      const cutoff = (2 * cal.semiangle_rad + Math.max(...cal.angular_sampling_rad)) / cal.wavelength_A;
      const layout = phaseLayout(n, factor, rowStep, colStep, cutoff);
      const {count,aliases} = layout, entries = aliases*(aliases+1)/2+aliases;
      const bytes = count * entries * 8;
      if (bytes > device.limits.maxStorageBufferBindingSize) throw new Error("The coupled phase system exceeds this adapter's buffer limit.");
      const active = Uint32Array.from({length:cal.num_bright_field_pixels}, (_,i)=>i);
      const {geom, trig: angleData} = packGeometry(cal, active, rotation);
      let aperturePower=0, scale=0;
      for(let i=3;i<geom.length;i+=4){const p=Math.fround(geom[i]*geom[i]);aperturePower+=p;scale+=Math.fround(p*p);}
      const owned:GPUBuffer[]=[];
      const keep=(b:GPUBuffer)=>{owned.push(b);return b;};
      const alloc=(size:number,usage:GPUBufferUsageFlags)=>keep(device.createBuffer({size,usage}));
      const storage=(v:Float32Array | Uint32Array,label:string)=>keep(makeBufferFromBytes(device,
        new Uint8Array(v.buffer,v.byteOffset,v.byteLength),GPUBufferUsage.STORAGE,label));
      const compile=(module:GPUShaderModule,entryPoint:string)=>device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint}});
      const dispatch=(enc:GPUCommandEncoder,p:GPUComputePipeline,entries:[number,GPUBuffer][],x:number,y=1)=>{
        const pass=enc.beginComputePass();pass.setPipeline(p);
        pass.setBindGroup(0,device.createBindGroup({layout:p.getBindGroupLayout(0),entries:entries.map(([binding,buffer])=>({binding,resource:{buffer}}))}));
        pass.dispatchWorkgroups(x,y);pass.end();
      };
      try {
        const source=makeDirectShader(n,true,n,true,0,false,true)+phaseNormalShader(n,size,aliases,rowStep,colStep);
        const normal=device.createShaderModule({code:source});
        const fft=device.createShaderModule({code:makeDirectShader(size,true,n,true,0,true)});
        const [prepare,accumulate,measureDose,rows,cols]=await Promise.all([compile(normal,"prepareProbe"),compile(normal,"accumulatePhase"),
          compile(normal,"phaseDetectorCounts"),compile(fft,"directObjFftRows"),compile(fft,"directObjFftCols")]);
        const mapped=storage(layout.map,"phase aliases"),geometry=storage(geom,"phase detector coordinates");
        const trig=storage(angleData,"phase detector angles");
        const degrees=phi12Rad*180/Math.PI;
        const coefficients=storage(packAberrations(c10A,c12A,degrees,options.higherOrder).data,"phase optics");
        const params=alloc(DIRECT_PARAMS_BYTES,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST);
        const probe=alloc(active.length*8,GPUBufferUsage.STORAGE);
        const counts=alloc(active.length*4,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC);
        const gram=alloc(bytes,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST);
        const uniform=(grid:number,offset:number,count:number)=>makeParams(cal,grid,c10A,c12A,degrees,active.length,offset,count,false,active.length,true,options.sample);
        device.queue.writeBuffer(params,0,uniform(n,0,1));
        let encoder=device.createCommandEncoder(); encoder.clearBuffer(gram);
        dispatch(encoder,prepare,[[0,params],[3,geometry],[4,trig],[12,coefficients],[13,probe]],Math.ceil(active.length/256));device.queue.submit([encoder.finish()]);
        await buildBfColumnGqkChunks(device,cal,this.source,n,n*n,active,chooseChunkCapacity(active.length,n*n,device),this.onProgress??undefined,undefined,
          (enc,buffer,offset,countBf)=>{
            device.queue.writeBuffer(params,0,uniform(n,offset,countBf));
            dispatch(enc,measureDose,[[0,params],[1,buffer],[7,counts]],Math.ceil(countBf/64));
            dispatch(enc,accumulate,[[0,params],[1,buffer],[2,gram],[3,geometry],[12,coefficients],[13,probe],[17,mapped]],Math.ceil(count/64));
          });
        const scanSums=await readF32(device,counts,active.length);
        const dose=scanSums.reduce((a,b)=>a+b,0)/(n*n*aperturePower);
        if (!(dose>0) || !Number.isFinite(dose)) throw new Error("The measured incident dose must be positive.");
        const field=solvePhase(layout,await readF32(device,gram,bytes/4),size,factor,rowStep,colStep,dose,damping*scale);
        const stage=storage(field,"damped phase spectrum"),phase=alloc(size*size*4,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC);
        device.queue.writeBuffer(params,0,uniform(size,0,1));
        encoder=device.createCommandEncoder();
        dispatch(encoder,rows,[[2,stage]],1,size);
        dispatch(encoder,cols,[[0,params],[2,stage],[10,phase]],1,size);
        device.queue.submit([encoder.finish()]);
        return {phase:await readF32(device,phase,size*size),width:size,height:size,upsample:factor,phaseDamp:damping,elapsedMs:performance.now()-started};
      } finally {owned.forEach(b=>b.destroy());}
    });
  }

  async reconstruct(c10A: number, c12A: number, phi12Rad: number,
    options: WebGPUReconstructionOptions = {}): Promise<WebGPUDirectResult> {
    const factor = options.upsample ?? 1;
    if (![1, 2, 4, 8].includes(factor)) throw new Error("WebGPU direct ptychography upsample must be 1, 2, 4, or 8.");
    return this.runExclusive(async () => {
      options.signal?.throwIfAborted();
      // Loss is a native-grid diagnostic. It must not change the displayed estimator on release.
      const wantsLoss = options.computeLoss !== false;
      if (this.agreementPlan && !wantsLoss && factor === 1 && this.clampBfCount(options.numBrightFieldPixels) === this.cal.num_bright_field_pixels) {
        return this.reconstructFitPhase(c10A, c12A, phi12Rad, options);
      }
      const native = wantsLoss || factor === 1
        ? await this.reconstructNative(c10A, c12A, phi12Rad, {...options, computeLoss: wantsLoss}) : null;
      let result: WebGPUDirectResult;
      if (factor === 1) {
        result = wantsLoss
          ? await this.reconstructNative(c10A, c12A, phi12Rad, {...options, computeLoss: false}) : native!;
      } else {
        result = await this.reconstructOutput(c10A, c12A, phi12Rad, factor, options);
      }
      return {...result, loss: native?.loss ?? null,
        gpuMs: result.gpuMs + (wantsLoss ? native?.gpuMs ?? 0 : 0)};
    });
  }

  private async reconstructOutput(c10: number, c12: number, phi12: number,
    factor: number, options: WebGPUReconstructionOptions): Promise<WebGPUDirectResult> {
    const numBrightFieldPixels = this.clampBfCount(options.numBrightFieldPixels);
    await this.setup(numBrightFieldPixels, options.scanDetectorRotationDeg);
    const device = this.device!, buffers = this.buffers!, n = this.n * factor;
    const thick = sampleThickness(options.sample) > 0;
    if (n * 8 > device.limits.maxComputeWorkgroupStorageSize) {
      throw new Error(`${n}×${n} direct ptychography needs ${n * 8} bytes of workgroup storage; select a smaller output factor on this GPU.`);
    }
    if (!this.outputPlan || this.outputPlan.n !== n || this.outputPlan.thick !== thick) {
      this.outputPlan?.owned.forEach(buffer => buffer.destroy()); this.outputPlan = null;
      const owned: GPUBuffer[] = [];
      const make = (size: number, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST) => {
        const out = device.createBuffer({size, usage}); owned.push(out); return out;
      };
      try {
        const module = device.createShaderModule({code: makeDirectShader(n, thick, this.n, true)});
        const [sum, rows, cols, probePipeline, mirrorPipeline] = await Promise.all(["directObjSum", "directObjFftRows", "directObjFftCols", "prepareProbe", "directMirrorColumns"].map(entryPoint =>
          device.createComputePipelineAsync({layout: "auto", compute: {module, entryPoint}})));
        const stage = make(n * n * 8), phase = make(n * n * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
        const params = make(DIRECT_PARAMS_BYTES, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
        const probe = make(buffers.activeBfCount * 8);
        const chunks = buffers.gqkChunks.map(() => make(DIRECT_PARAMS_BYTES, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST));
        const frequencies = [this.cal.scan_q_row, this.cal.scan_q_col].map(native => {
          const q = Float32Array.from({length: n}, (_, i) => (i < n / 2 ? i : i - n) * native[1]);
          const buffer = make(q.byteLength); device.queue.writeBuffer(buffer, 0, q); return buffer;
        });
        const group = (pipeline: GPUComputePipeline, entries: [number, GPUBuffer][]) => device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0), entries: entries.map(([binding, buffer]) => ({binding, resource: {buffer}})),
        });
        const sumGroups = chunks.map((params, i) => group(sum, [
          [0, params], [1, buffers.gqkChunks[i]], [2, stage], [3, buffers.bfGeom],
          [5, frequencies[0]], [6, frequencies[1]], [12, buffers.aberrations], [13, probe],
        ]));
        this.outputPlan = {n, thick, owned, stage, phase, params, chunks, sum, rows, cols, sumGroups,
          probePipeline, probeGroup: group(probePipeline, [[0, params], [3, buffers.bfGeom],
            [4, buffers.bfTrig], [12, buffers.aberrations], [13, probe]]),
          mirrorPipeline, mirrorGroup: group(mirrorPipeline, [[0, params], [2, stage]]),
          rowGroup: group(rows, [[2, stage]]), colGroup: group(cols, [[0, params], [2, stage], [10, phase]])};
      } catch (error) { owned.forEach(buffer => buffer.destroy()); throw error; }
    }
    const plan = this.outputPlan;
    const aberrations = packAberrations(c10, c12, phi12 * 180 / Math.PI, options.higherOrder);
    // Even C10/C12 optics pair corrected frequencies. Require both Nyquist
    // edges outside the complete soft-aperture support before using symmetry.
    const support = 2 * (this.cal.semiangle_rad + .5 * Math.max(...this.cal.angular_sampling_rad))
      / this.cal.wavelength_A * (1 + 8 * 2 ** -23);
    const paired = !aberrations.active && n / 2 * Math.min(Math.abs(this.cal.scan_q_row[1]), Math.abs(this.cal.scan_q_col[1])) > support;
    const outputCols = paired ? n / 2 + 1 : n;
    device.queue.writeBuffer(buffers.aberrations, 0, aberrations.data as Float32Array<ArrayBuffer>);
    const started = performance.now(), enc = device.createCommandEncoder();
    enc.clearBuffer(plan.stage);
    device.queue.writeBuffer(plan.params, 0, makeParams(this.cal, n, c10, c12, phi12 * 180 / Math.PI,
      numBrightFieldPixels, 0, 1, false, buffers.activeBfCount, aberrations.active, options.sample));
    const probePass = enc.beginComputePass();
    probePass.setPipeline(plan.probePipeline); probePass.setBindGroup(0, plan.probeGroup);
    probePass.dispatchWorkgroups(Math.ceil(buffers.activeBfCount / 256)); probePass.end();
    let offset = 0;
    for (let i = 0; i < buffers.gqkChunks.length; i++) {
      const count = buffers.chunkBfCounts[i];
      device.queue.writeBuffer(plan.chunks[i], 0, makeParams(this.cal, n, c10, c12, phi12 * 180 / Math.PI,
        numBrightFieldPixels, offset, count, false, buffers.activeBfCount, aberrations.active, options.sample, outputCols));
      const pass = enc.beginComputePass(); pass.setPipeline(plan.sum); pass.setBindGroup(0, plan.sumGroups[i]);
      pass.dispatchWorkgroups(Math.min(65535, Math.ceil(n * outputCols / 256)), Math.ceil(n * outputCols / (256 * 65535))); pass.end(); offset += count;
    }
    if (paired) {
      const pass = enc.beginComputePass();
      pass.setPipeline(plan.mirrorPipeline); pass.setBindGroup(0, plan.mirrorGroup);
      pass.dispatchWorkgroups(Math.min(65535, Math.ceil(n * n / 256)), Math.ceil(n * n / (256 * 65535))); pass.end();
    }
    for (const [pipeline, bind] of [[plan.rows, plan.rowGroup], [plan.cols, plan.colGroup]] as const) {
      const pass = enc.beginComputePass(); pass.setPipeline(pipeline); pass.setBindGroup(0, bind);
      pass.dispatchWorkgroups(1, n); pass.end();
    }
    device.queue.submit([enc.finish()]); await device.queue.onSubmittedWorkDone();
    return {phase: await readF32(device, plan.phase, n * n), width: n, height: n,
      gpuMs: performance.now() - started, numBrightFieldPixels, scanDetectorRotationDeg: options.scanDetectorRotationDeg ?? baseRotationDeg(this.cal),
      loss: null, adapterInfo: getGPUInfo(), softwareAdapter: isSoftwareGPUAdapter()};
  }

  private async reconstructNative(
    c10: number,
    c12: number,
    phi12: number,
    options: WebGPUReconstructionOptions = {},
  ): Promise<WebGPUDirectResult> {
    // Engine units: Angstrom and radians. The scientist-facing widget converts nm at its boundary.
    const phi12Deg = phi12 * 180 / Math.PI;
    const numBrightFieldPixels = this.clampBfCount(options.numBrightFieldPixels);
    const computeLoss = options.computeLoss ?? numBrightFieldPixels === this.cal.num_bright_field_pixels;
    const scanDetectorRotationDeg = options.scanDetectorRotationDeg == null || !Number.isFinite(options.scanDetectorRotationDeg)
      ? baseRotationDeg(this.cal)
      : Number(options.scanDetectorRotationDeg);
    {
      await this.setup(numBrightFieldPixels, scanDetectorRotationDeg);
      const device = this.device;
      const buffers = this.buffers;
      const pipelines = this.pipelines;
      const bindGroups = this.bindGroups;
      if (!device || !buffers || !pipelines || !bindGroups) {
        throw new Error("WebGPU direct ptychography buffers are not ready after setup");
      }
      const aberrations = packAberrations(c10, c12, phi12Deg, options.higherOrder);
      // thickness > 0 selects the thick-sample shader variant; otherwise the standard pipelines run unchanged
      const thick = sampleThickness(options.sample) > 0;
      device.queue.writeBuffer(buffers.aberrations, 0, aberrations.data as unknown as BufferSource);
      const t0 = performance.now();
      let enc = device.createCommandEncoder();
      // Object-redraw fast path (ported from the CUDA direct ptychography engine): sum
      // corrected G over BF pixels in Fourier space, one iFFT, angle once -
      // skips two FFT passes + atan2 per BF pixel. Displays angle(mean(obj)),
      // the same estimator as the Python DirectPtycho.result() reference (verified
      // corr 0.997 vs backend, 0.9965 vs the exact browser path at identical
      // state). Loss commits always use the exact per-BF path below.
      if (!computeLoss) {
        enc.clearBuffer(buffers.stage, 0, this.plane * 8);
        for (let bfOffset = 0; bfOffset < buffers.activeBfCount; bfOffset += buffers.chunkCapacity) {
          const chunkBf = Math.min(buffers.chunkCapacity, buffers.activeBfCount - bfOffset);
          const chunkIndex = Math.floor(bfOffset / buffers.chunkCapacity);
          device.queue.writeBuffer(
            buffers.paramsChunks[chunkIndex],
            0,
            makeParams(this.cal, this.n, c10, c12, phi12Deg, numBrightFieldPixels, bfOffset, chunkBf, false, buffers.activeBfCount, aberrations.active, options.sample),
          );
          const pass = enc.beginComputePass({ label: "direct obj sum" });
          pass.setPipeline(thick ? pipelines.objSumThick : pipelines.objSum);
          pass.setBindGroup(0, (thick ? bindGroups.objSumThick : bindGroups.objSum)[chunkIndex]);
          pass.dispatchWorkgroups(Math.ceil(this.plane / 256));
          pass.end();
        }
        device.queue.writeBuffer(
          buffers.params,
          0,
          makeParams(this.cal, this.n, c10, c12, phi12Deg, numBrightFieldPixels, 0, 1, false, buffers.activeBfCount, aberrations.active, options.sample),
        );
        let pass = enc.beginComputePass({ label: "direct obj fft rows" });
        pass.setPipeline(pipelines.objFftRows);
        pass.setBindGroup(0, bindGroups.objFftRows);
        pass.dispatchWorkgroups(1, this.n);
        pass.end();
        pass = enc.beginComputePass({ label: "direct obj fft cols" });
        pass.setPipeline(pipelines.objFftCols);
        pass.setBindGroup(0, bindGroups.objFftCols);
        pass.dispatchWorkgroups(1, this.n);
        pass.end();
        device.queue.submit([enc.finish()]);
        await device.queue.onSubmittedWorkDone();
        const gpuMsFast = performance.now() - t0;
        const phaseFast = await readF32(device, buffers.phase, this.plane);
        const resultFast = {
          phase: phaseFast,
          width: this.n,
          height: this.n,
          gpuMs: gpuMsFast,
          numBrightFieldPixels,
          scanDetectorRotationDeg,
          loss: null,
          adapterInfo: getGPUInfo(),
          softwareAdapter: isSoftwareGPUAdapter(),
        };
        return resultFast;
      }
      for (let bfOffset = 0; bfOffset < buffers.activeBfCount; bfOffset += buffers.chunkCapacity) {
        const chunkBf = Math.min(buffers.chunkCapacity, buffers.activeBfCount - bfOffset);
        const chunkIndex = Math.floor(bfOffset / buffers.chunkCapacity);
        device.queue.writeBuffer(
          buffers.paramsChunks[chunkIndex],
          0,
          makeParams(
            this.cal,
            this.n,
            c10,
            c12,
            phi12Deg,
            numBrightFieldPixels,
            bfOffset,
            chunkBf,
            computeLoss,
            buffers.activeBfCount,
            aberrations.active,
            options.sample,
          ),
        );
        let pass = enc.beginComputePass({ label: "direct rows" });
        pass.setPipeline(thick ? pipelines.rowsThick : pipelines.rows);
        pass.setBindGroup(0, (thick ? bindGroups.rowsThick : bindGroups.rows)[chunkIndex]);
        pass.dispatchWorkgroups(1, this.n, chunkBf);
        pass.end();
        pass = enc.beginComputePass({ label: "direct cols" });
        pass.setPipeline(pipelines.cols);
        pass.setBindGroup(0, bindGroups.cols[chunkIndex]);
        pass.dispatchWorkgroups(1, this.n, chunkBf);
        pass.end();
        pass = enc.beginComputePass({ label: "direct reduce chunk" });
        pass.setPipeline(pipelines.reducePartial);
        pass.setBindGroup(0, bindGroups.reducePartial[chunkIndex]);
        pass.dispatchWorkgroups(Math.ceil(Math.ceil(chunkBf / REDUCE_BF_GROUP) * this.plane / 256));
        pass.end();
        if (options.signal) {
          // Diagnostics yield between complete BF chunks. A renewed drag can
          // stop the remaining work without returning a partial phase or loss.
          device.queue.submit([enc.finish()]);
          await device.queue.onSubmittedWorkDone();
          options.signal.throwIfAborted();
          enc = device.createCommandEncoder();
        }
      }
      device.queue.writeBuffer(
        buffers.params,
        0,
        makeParams(
          this.cal,
          this.n,
          c10,
          c12,
          phi12Deg,
          numBrightFieldPixels,
          0,
          1,
          computeLoss,
          buffers.activeBfCount,
          aberrations.active,
          options.sample,
        ),
      );
      const pass = enc.beginComputePass({ label: "direct reduce" });
      pass.setPipeline(pipelines.finalizeGroups);
      pass.setBindGroup(0, bindGroups.finalizeGroups);
      pass.dispatchWorkgroups(Math.ceil(this.plane / 256));
      pass.end();
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
      const gpuMs = performance.now() - t0;
      const phase = await readF32(device, buffers.phase, this.plane);
      let loss: number | null = null;
      if (computeLoss) {
        const variance = await readF32(device, buffers.variance, this.plane);
        let sum = 0;
        for (let i = 0; i < variance.length; i++) sum += variance[i];
        loss = sum / variance.length;
      }
      const result = {
        phase,
        width: this.n,
        height: this.n,
        gpuMs,
        numBrightFieldPixels,
        scanDetectorRotationDeg,
        loss,
        adapterInfo: getGPUInfo(),
        softwareAdapter: isSoftwareGPUAdapter(),
      };
      return result;
    }
  }
}
