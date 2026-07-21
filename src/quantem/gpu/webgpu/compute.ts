/// <reference types="@webgpu/types" />
// Offline 4D-STEM compute in the browser via WebGPU - no Python kernel.
// Primitives (browser siblings of the Python backends):
//   maskedSum(detectorMask) -> virtual image  (one thread per scan position)
//   maskedDpc(detectorMask) -> DPC row/col image (CoM + mean subtraction on GPU)
//   reduceFrames(scanMask)  -> diffraction DP  (one thread per detector pixel)
//
// CHUNKED: the stack is split into scan-row ranges, each in its own GPU buffer
// (<= the 1 GB per-buffer cap). This lets a stack far larger than one buffer
// (e.g. 512x512x192x192 = 9.7 GB) live across N buffers and be reduced by
// dispatching per chunk and accumulating. A single-buffer dataset is just the
// N=1 case. Verified bit-exact vs numpy (chunked masked_sum maxErr 0).
//
// Stack ships as uint8 (clip(0,255): real detector counts are 0-~200, so the
// value IS the count, near-lossless) or uint16; dtype inferred from byte length.
import { getGPUDevice } from "./device";
import { decodeBslz4ToStack, decodeBslz4Batch, type Bslz4Spec } from "./bslz4";
import { FFT_2D_SHADER } from "./fft-shader";

// `mode`: 0 = uint16 (2 samples/u32), 1 = uint8 (4/u32). `sample(gp)` reads a
// detector value at a chunk-local global pixel index.
const SAMPLE = `
fn sample(gp: u32, mode: u32) -> u32 {
  if (mode == 1u) { let w = data[gp >> 2u]; return (w >> ((gp & 3u) * 8u)) & 0xffu; }
  let w = data[gp >> 1u];
  return select(w >> 16u, w & 0xffffu, (gp & 1u) == 0u);
}
// Float value of a pixel. mode 2 = float32 (1 u32/pixel = IEEE-754 bit pattern -> bitcast,
// full precision); modes 0/1 return the integer count as f32.
fn sampleF(gp: u32, mode: u32) -> f32 {
  if (mode == 2u) { return bitcast<f32>(data[gp]); }
  return f32(sample(gp, mode));
}`;

// One WORKGROUP per scan position; its 64 threads COOPERATIVELY sum the aperture pixels, then
// a shared-memory tree reduction writes one VI value. The old "one thread per scan position"
// kernel was uncoalesced: thread sl read data[sl*detSize + idx[j]], so a warp's 64 threads
// touched 64 addresses detSize apart = 64 cache lines per access = ~1/64 of memory bandwidth
// (180-300 ms for a BF drag). Here consecutive threads read consecutive idx entries (idx is
// row-major sorted) -> consecutive detector pixels -> COALESCED, so the drag hits the memory
// floor. 2D dispatch (gridX in u2.x) because a single-buffer stack can exceed 65535 scans.
// `sg` = use subgroup (warp) reduction: one subgroup (32 lanes) per scan, summed with a single
// `subgroupAdd` - NO shared memory, NO barriers. The shared-memory tree reduction (fallback for
// GPUs without the subgroups feature) pays 6 workgroupBarriers per scan position x 262144 scans,
// which dominated the kernel (~45-58ms); subgroupAdd is a register-level warp shuffle. Lanes read
// consecutive idx entries -> consecutive detector pixels (idx is row-major sorted) -> coalesced.
// WGSZ threads per scan position so many loads are in flight at once (the kernel is memory-bound
// on a gather; only 32 lanes left bandwidth at ~4% of peak). Each warp reduces with subgroupAdd,
// the per-warp partials combine through a tiny shared array (one barrier). Fallback (no subgroups)
// is the full shared tree reduction.
const WGSZ = 128;
const SGSZ = 32;   // subgroup (warp) size on the target GPUs
const maskedSumSrc = (sg: boolean) => `
${sg ? "enable subgroups;" : ""}
@group(0) @binding(0) var<storage,read> data: array<u32>;
@group(0) @binding(1) var<storage,read> idx: array<u32>;   // ACTIVE detector pixel indices only
@group(0) @binding(2) var<storage,read_write> vi: array<f32>;
@group(0) @binding(3) var<uniform> u: vec4<u32>;   // startScan, nScanInChunk, detSize, mode
@group(0) @binding(4) var<uniform> u2: vec4<u32>;  // gridX, 0, 0, 0
${SAMPLE}
var<workgroup> part: array<f32, ${WGSZ}>;
@compute @workgroup_size(${WGSZ})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let sl = wid.y * u2.x + wid.x; let tid = lid.x;
  // f32 accumulate via sampleF: exact for uint8/uint16 counts (sum << 2^24 for a screening
  // aperture), and the only correct path for float32 (mode 2) values.
  let n = arrayLength(&idx); let base = sl * u.z; var sum: f32 = 0.0;
  if (sl < u.y) { for (var j = tid; j < n; j = j + ${WGSZ}u) { sum = sum + sampleF(base + idx[j], u.w); } }
${sg
  ? `  sum = subgroupAdd(sum);                       // per-warp partial
  if (subgroupElect()) { part[tid / ${SGSZ}u] = sum; }   // one slot per warp
  workgroupBarrier();
  if (tid == 0u && sl < u.y) {
    var total = 0.0; for (var w = 0u; w < ${WGSZ / SGSZ}u; w = w + 1u) { total = total + part[w]; }
    vi[u.x + sl] = total;
  }`
  : `  part[tid] = sum; workgroupBarrier();
  for (var s: u32 = ${WGSZ / 2}u; s > 0u; s = s >> 1u) { if (tid < s) { part[tid] = part[tid] + part[tid + s]; } workgroupBarrier(); }
  if (tid == 0u && sl < u.y) { vi[u.x + sl] = part[0]; }`}
}`;

// One thread per detector pixel; ACCUMULATES this chunk's in-ROI scan positions
// into the DP (chunks dispatched serially, so += across chunks is safe). dims:
// startScan, nScanInChunk, detSize, mode; plus extra: total scanMask is global,
// indexed by startScan+sl.
// One thread per (detector pixel, FRAME-BLOCK): the 2D grid parallelizes the reduction over
// FRAMES too, not just pixels. The old "one thread per pixel, serial loop over all frames"
// launched only detSize (~37K) threads -> ~12% occupancy on a big GPU (88% idle), so the
// memory-bound sum ran ~70x over its bandwidth floor. Here gid.y splits the frames into
// FRAME_BLOCKS strided slices; each thread sums its slice locally (bit-exact integer) then does
// ONE atomicAdd into dp[k]. ~FRAME_BLOCKS x more threads saturate the GPU; atomic contention is
// only FRAME_BLOCKS-way per pixel (one add per thread), trivially cheap vs the memory traffic.
const FRAME_BLOCKS = 64;
const REDUCE_FRAMES_WGSL = `
@group(0) @binding(0) var<storage,read> data: array<u32>;
@group(0) @binding(1) var<storage,read> scanMask: array<u32>;  // GLOBAL scanCount
@group(0) @binding(2) var<storage,read_write> dp: array<atomic<u32>>;  // detSize, INTEGER (exact)
@group(0) @binding(3) var<uniform> u: vec4<u32>;   // startScan, nScanInChunk, detSize, mode
${SAMPLE}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let k = gid.x; let detSize = u.z; if (k >= detSize) { return; }
  var sum: u32 = 0u;   // integer accumulate: bit-exact, no f32 rounding on large/dead-pixel sums
  for (var sl: u32 = gid.y; sl < u.y; sl = sl + ${FRAME_BLOCKS}u) {   // strided frame slice
    if (scanMask[u.x + sl] != 0u) { sum = sum + sample(sl * detSize + k, u.w); }
  }
  if (sum != 0u) { atomicAdd(&dp[k], sum); }   // one add per thread; skip empty slices
}`;

// Extract ONE frame's diffraction pattern (detSize values) from a chunk buffer -
// the offline replacement for the kernel's per-probe frame_bytes. One thread/pixel.
const FRAME_WGSL = `
@group(0) @binding(0) var<storage,read> data: array<u32>;
@group(0) @binding(1) var<storage,read_write> frame: array<f32>;
@group(0) @binding(2) var<uniform> u: vec4<u32>;   // localBase (pixels), detSize, mode
${SAMPLE}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let k = gid.x; if (k >= u.y) { return; }
  frame[k] = sampleF(u.x + k, u.z);
}`;

// One thread per scan position: intensity-weighted centroid (center of mass) of the
// detector over the active mask pixels. Output is the per-position CoM in detector px:
// comY at [gi], comX at [scanCount+gi]. Drives CoMx/CoMy/CoMmag/iCoM (DPC).
// One WORKGROUP per scan position (same coalescing fix as MASKED_SUM): 64 threads cooperatively
// accumulate the intensity-weighted centroid over the aperture, three shared-memory reductions
// (weight, y*weight, x*weight). gridX in u2.z for the 2D dispatch.
const maskedComSrc = (sg: boolean) => `
${sg ? "enable subgroups;" : ""}
@group(0) @binding(0) var<storage,read> data: array<u32>;
@group(0) @binding(1) var<storage,read> idx: array<u32>;   // ACTIVE detector pixel indices
@group(0) @binding(2) var<storage,read_write> com: array<f32>;  // 2*scanCount: [gi]=comY, [scanCount+gi]=comX
@group(0) @binding(3) var<uniform> u: vec4<u32>;   // startScan, nScanInChunk, detSize, mode
@group(0) @binding(4) var<uniform> u2: vec4<u32>;  // detCols, scanCount, gridX, 0
${SAMPLE}
fn ratioU32(num: u32, den: u32) -> f32 {
  if (den == 0u) { return 0.0; }
  let q = num / den;
  let rem = num - q * den;
  return f32(q) + f32(rem) / f32(den);
}
var<workgroup> pw: array<f32, ${WGSZ}>;
var<workgroup> py: array<f32, ${WGSZ}>;
var<workgroup> px: array<f32, ${WGSZ}>;
var<workgroup> pwu: array<u32, ${WGSZ}>;
var<workgroup> pyu: array<u32, ${WGSZ}>;
var<workgroup> pxu: array<u32, ${WGSZ}>;
@compute @workgroup_size(${WGSZ})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let sl = wid.y * u2.z + wid.x; let tid = lid.x;
  let base = sl * u.z; let n = arrayLength(&idx); let detCols = u2.x;
  var wsum: f32 = 0.0; var ysum: f32 = 0.0; var xsum: f32 = 0.0;
  var wsumU: u32 = 0u; var ysumU: u32 = 0u; var xsumU: u32 = 0u;
  if (u.w == 1u) {
    if (sl < u.y) { for (var j: u32 = tid; j < n; j = j + ${WGSZ}u) {
      let p = idx[j]; let v = sample(base + p, u.w);
      wsumU = wsumU + v; ysumU = ysumU + (p / detCols) * v; xsumU = xsumU + (p % detCols) * v;
    } }
  } else {
    if (sl < u.y) { for (var j: u32 = tid; j < n; j = j + ${WGSZ}u) {
      let p = idx[j]; let v = sampleF(base + p, u.w);
      wsum = wsum + v; ysum = ysum + f32(p / detCols) * v; xsum = xsum + f32(p % detCols) * v;
    } }
  }
${sg
  ? `  if (u.w == 1u) {
    wsumU = subgroupAdd(wsumU); ysumU = subgroupAdd(ysumU); xsumU = subgroupAdd(xsumU);
    if (subgroupElect()) { let w = tid / ${SGSZ}u; pwu[w] = wsumU; pyu[w] = ysumU; pxu[w] = xsumU; }
    workgroupBarrier();
    if (tid == 0u && sl < u.y) {
      var w = 0u; var y = 0u; var x = 0u;
      for (var k = 0u; k < ${WGSZ / SGSZ}u; k = k + 1u) { w = w + pwu[k]; y = y + pyu[k]; x = x + pxu[k]; }
      let gi = u.x + sl;
      if (w > 0u) { com[gi] = ratioU32(y, w); com[u2.y + gi] = ratioU32(x, w); } else { com[gi] = 0.0; com[u2.y + gi] = 0.0; }
    }
  } else {
    wsum = subgroupAdd(wsum); ysum = subgroupAdd(ysum); xsum = subgroupAdd(xsum);
  if (subgroupElect()) { let w = tid / ${SGSZ}u; pw[w] = wsum; py[w] = ysum; px[w] = xsum; }
  workgroupBarrier();
  if (tid == 0u && sl < u.y) {
    var w = 0.0; var y = 0.0; var x = 0.0;
    for (var k = 0u; k < ${WGSZ / SGSZ}u; k = k + 1u) { w = w + pw[k]; y = y + py[k]; x = x + px[k]; }
    let gi = u.x + sl;
    if (w > 0.0) { com[gi] = y / w; com[u2.y + gi] = x / w; } else { com[gi] = 0.0; com[u2.y + gi] = 0.0; }
  }
  }`
  : `  if (u.w == 1u) {
    pwu[tid] = wsumU; pyu[tid] = ysumU; pxu[tid] = xsumU; workgroupBarrier();
    for (var s: u32 = ${WGSZ / 2}u; s > 0u; s = s >> 1u) {
      if (tid < s) { pwu[tid] = pwu[tid] + pwu[tid + s]; pyu[tid] = pyu[tid] + pyu[tid + s]; pxu[tid] = pxu[tid] + pxu[tid + s]; }
      workgroupBarrier();
    }
    if (tid == 0u && sl < u.y) {
      let gi = u.x + sl;
      if (pwu[0] > 0u) { com[gi] = ratioU32(pyu[0], pwu[0]); com[u2.y + gi] = ratioU32(pxu[0], pwu[0]); } else { com[gi] = 0.0; com[u2.y + gi] = 0.0; }
    }
  } else {
    pw[tid] = wsum; py[tid] = ysum; px[tid] = xsum; workgroupBarrier();
  for (var s: u32 = ${WGSZ / 2}u; s > 0u; s = s >> 1u) {
    if (tid < s) { pw[tid] = pw[tid] + pw[tid + s]; py[tid] = py[tid] + py[tid + s]; px[tid] = px[tid] + px[tid + s]; }
    workgroupBarrier();
  }
  if (tid == 0u && sl < u.y) {
    let gi = u.x + sl;
    if (pw[0] > 0.0) { com[gi] = py[0] / pw[0]; com[u2.y + gi] = px[0] / pw[0]; } else { com[gi] = 0.0; com[u2.y + gi] = 0.0; }
  }
  }`}
}`;

// CoM -> DPC mean reduction. One workgroup reduces the scan-position CoM arrays
// to their global row/col means. DPC display is then centered without pulling the
// two full CoM maps back to JavaScript.
const DPC_MEAN_WGSL = `
@group(0) @binding(0) var<storage,read> com: array<f32>;       // [row...][col...]
@group(0) @binding(1) var<storage,read_write> mean: array<f32>; // mean[0]=row, mean[1]=col
@group(0) @binding(2) var<uniform> u: vec4<u32>;                // scanCount, 0, 0, 0
var<workgroup> partRow: array<f32, 256>;
var<workgroup> partCol: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x; let n = u.x;
  var rowSum = 0.0; var colSum = 0.0;
  var rowComp = 0.0; var colComp = 0.0;
  for (var i = tid; i < n; i = i + 256u) {
    let rowY = com[i] - rowComp;
    let rowT = rowSum + rowY;
    rowComp = (rowT - rowSum) - rowY;
    rowSum = rowT;
    let colY = com[n + i] - colComp;
    let colT = colSum + colY;
    colComp = (colT - colSum) - colY;
    colSum = colT;
  }
  partRow[tid] = rowSum; partCol[tid] = colSum; workgroupBarrier();
  for (var s = 128u; s > 0u; s = s >> 1u) {
    if (tid < s) {
      partRow[tid] = partRow[tid] + partRow[tid + s];
      partCol[tid] = partCol[tid] + partCol[tid + s];
    }
    workgroupBarrier();
  }
  if (tid == 0u) {
    let denom = max(f32(n), 1.0);
    mean[0] = partRow[0] / denom;
    mean[1] = partCol[0] / denom;
  }
}`;

// Select one centered DPC component for display. component=0 -> row/Y, 1 -> col/X.
const DPC_COMPONENT_WGSL = `
@group(0) @binding(0) var<storage,read> com: array<f32>;
@group(0) @binding(1) var<storage,read> mean: array<f32>;
@group(0) @binding(2) var<storage,read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> u: vec4<u32>; // scanCount, component, 0, 0
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; let n = u.x;
  if (i >= n) { return; }
  if (u.y == 0u) {
    out[i] = com[i] - mean[0];
  } else {
    out[i] = com[n + i] - mean[1];
  }
}`;

const DPC_COMPONENT_PAIR_WGSL = `
@group(0) @binding(0) var<storage,read> com: array<f32>;
@group(0) @binding(1) var<storage,read> mean: array<f32>;
@group(0) @binding(2) var<storage,read_write> rowOut: array<f32>;
@group(0) @binding(3) var<storage,read_write> colOut: array<f32>;
@group(0) @binding(4) var<uniform> u: vec4<u32>; // scanCount, 0, 0, 0
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let n = u.x;
  if (i >= n) { return; }
  rowOut[i] = com[i] - mean[0];
  colOut[i] = com[n + i] - mean[1];
}`;

// Mean reduction for one already-centered DPC component. The first DPC centering
// matches the backend semantics; this pass lets the next shader choose the same
// one-ulp rounding side as the NumPy/CUDA reference without a CPU readback.
const DPC_OUTPUT_MEAN_WGSL = `
@group(0) @binding(0) var<storage,read> values: array<f32>;
@group(0) @binding(1) var<storage,read_write> mean: array<f32>;
@group(0) @binding(2) var<uniform> u: vec4<u32>; // scanCount, 0, 0, 0
var<workgroup> part: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  let n = u.x;
  var sum = 0.0;
  var comp = 0.0;
  for (var i = tid; i < n; i = i + 256u) {
    let y = values[i] - comp;
    let t = sum + y;
    comp = (t - sum) - y;
    sum = t;
  }
  part[tid] = sum;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s = s >> 1u) {
    if (tid < s) {
      part[tid] = part[tid] + part[tid + s];
    }
    workgroupBarrier();
  }
  if (tid == 0u) {
    mean[0] = part[0] / max(f32(n), 1.0);
  }
}`;

const DPC_OUTPUT_ULP_CORRECT_WGSL = `
@group(0) @binding(0) var<storage,read_write> values: array<f32>;
@group(0) @binding(1) var<storage,read> residualMean: array<f32>;
@group(0) @binding(2) var<storage,read> dpcMean: array<f32>;
@group(0) @binding(3) var<uniform> u: vec4<u32>; // scanCount, component, 0, 0
fn oneUlpBelowDelta(value: f32) -> f32 {
  if (value <= 0.0) { return 0.0; }
  let bits = bitcast<u32>(value);
  if (bits == 0u) { return 0.0; }
  return value - bitcast<f32>(bits - 1u);
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= u.x) { return; }
  let delta = select(0.0, oneUlpBelowDelta(dpcMean[u.y]), residualMean[0] < 0.0);
  values[i] = values[i] + delta;
}`;

// Pack centered DPC row/col maps into one complex dual-real FFT input for iDPC.
// flags bit 0 selects the transpose convention used by the Python DPC solver.
const IDPC_PACK_WGSL = `
struct PackParams {
  n: u32,
  flags: u32,
  _pad0: u32,
  _pad1: u32,
  rot: vec4<f32>, // cos(theta), sin(theta), 0, 0
}
@group(0) @binding(0) var<storage,read> rowDpc: array<f32>;
@group(0) @binding(1) var<storage,read> colDpc: array<f32>;
@group(0) @binding(2) var<storage,read_write> gradFft: array<vec2<f32>>;
@group(0) @binding(3) var<uniform> u: PackParams;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= u.n) { return; }
  let row = rowDpc[i];
  let col = colDpc[i];
  var gradRow: f32;
  var gradCol: f32;
  if ((u.flags & 1u) != 0u) {
    gradRow = u.rot.y * col + u.rot.x * row;
    gradCol = u.rot.x * col - u.rot.y * row;
  } else {
    gradRow = u.rot.x * row - u.rot.y * col;
    gradCol = u.rot.y * row + u.rot.x * col;
  }
  gradFft[i] = vec2<f32>(gradRow, gradCol);
}`;

const IDPC_POISSON_WGSL = `
@group(0) @binding(0) var<storage,read> gradFft: array<vec2<f32>>;
@group(0) @binding(1) var<storage,read_write> phaseFft: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> u: vec4<u32>; // width, height, n, 0
fn freq(i: u32, n: u32) -> f32 {
  if (i < (n + 1u) / 2u) {
    return f32(i) / f32(n);
  }
  return -f32(n - i) / f32(n);
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= u.z) { return; }
  if (i == 0u) {
    phaseFft[i] = vec2<f32>(0.0, 0.0);
    return;
  }
  let row = i / u.x;
  let col = i - row * u.x;
  let mirrorRow = (u.y - row) % u.y;
  let mirrorCol = (u.x - col) % u.x;
  let mirror = mirrorRow * u.x + mirrorCol;
  let z = gradFft[i];
  let zMirrorConj = vec2<f32>(gradFft[mirror].x, -gradFft[mirror].y);
  let rowF = 0.5 * (z + zMirrorConj);
  let diff = z - zMirrorConj;
  let colF = vec2<f32>(0.5 * diff.y, -0.5 * diff.x);
  let k0 = freq(row, u.y);
  let k1 = freq(col, u.x);
  let k2 = k0 * k0 + k1 * k1;
  let g = rowF * k0 + colF * k1;
  let scale = 0.25 / k2;
  phaseFft[i] = vec2<f32>(g.y * scale, -g.x * scale);
}`;

const IDPC_EXTRACT_WGSL = `
@group(0) @binding(0) var<storage,read> phaseComplex: array<vec2<f32>>;
@group(0) @binding(1) var<storage,read_write> phase: array<f32>;
@group(0) @binding(2) var<uniform> u: vec4<u32>; // n, 0, 0, 0
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= u.x) { return; }
  phase[i] = -phaseComplex[i].x;
}`;

// Dense DF/ADF helper: out = full-detector total - complement sum. This mirrors
// CUDA/MPS dense-mask behavior so dragging a large annulus reads the smaller
// complement after the per-scan total image is cached.
const SUBTRACT_FROM_TOTAL_WGSL = `
@group(0) @binding(0) var<storage,read> total: array<f32>;
@group(0) @binding(1) var<storage,read> subtract: array<f32>;
@group(0) @binding(2) var<storage,read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> u: vec4<u32>; // scanCount, 0, 0, 0
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= u.x) { return; }
  out[i] = total[i] - subtract[i];
}`;

const MAX_WG = 65535;   // max workgroups per dispatch dimension; >this needs a 2D grid

interface Chunk { buffer: GPUBuffer; startScan: number; nScan: number; }

interface TraitReader { get(name: string): any; }

// Detector mask for the offline WebGPU virtual-image sum. Mirrors the Python
// mask geometry exactly (show4dstem.py _create_*_mask): cx pairs with column,
// cy with row, so the browser virtual image matches the native backend result
// pixel-for-pixel.
export function buildDetectorMask(model: TraitReader, detRows: number, detCols: number): Uint32Array {
  const mask = new Uint32Array(detRows * detCols);
  const cx = model.get("roi_center_col");
  const cy = model.get("roi_center_row");
  const mode = model.get("roi_mode") || "circle";
  const radius = model.get("roi_radius") || 0;
  const inner = model.get("roi_radius_inner") || 0;
  const halfW = (model.get("roi_width") || 0) / 2;
  const halfH = (model.get("roi_height") || 0) / 2;
  for (let row = 0; row < detRows; row++) {
    for (let col = 0; col < detCols; col++) {
      const dx = col - cx, dy = row - cy, d2 = dx * dx + dy * dy;
      let inside = false;
      if (mode === "circle") inside = d2 <= radius * radius;
      else if (mode === "annular") inside = d2 > inner * inner && d2 <= radius * radius;
      else if (mode === "square") inside = Math.abs(dx) <= radius && Math.abs(dy) <= radius;
      else if (mode === "rect") inside = Math.abs(dx) <= halfW && Math.abs(dy) <= halfH;
      else if (mode === "point") inside = Math.round(cx) === col && Math.round(cy) === row;
      mask[row * detCols + col] = inside ? 1 : 0;
    }
  }
  return mask;
}

export function buildFullDetectorMask(detRows: number, detCols: number): Uint32Array {
  const mask = new Uint32Array(detRows * detCols);
  mask.fill(1);
  return mask;
}

// Scan-ROI mask for the offline DP-from-region reduce (mirrors the vi_roi_mode
// geometry in show4dstem.py _compute_vi_roi_dp).
export function buildScanMask(model: TraitReader, scanRows: number, scanCols: number): Uint32Array {
  const mask = new Uint32Array(scanRows * scanCols);
  const cx = model.get("vi_roi_center_col");
  const cy = model.get("vi_roi_center_row");
  const mode = model.get("vi_roi_mode") || "circle";
  const radius = model.get("vi_roi_radius") || 0;
  const halfW = (model.get("vi_roi_width") || 0) / 2;
  const halfH = (model.get("vi_roi_height") || 0) / 2;
  for (let row = 0; row < scanRows; row++) {
    for (let col = 0; col < scanCols; col++) {
      const dx = col - cx, dy = row - cy;
      let inside = false;
      if (mode === "circle") inside = dx * dx + dy * dy <= radius * radius;
      else if (mode === "square") inside = Math.abs(dx) <= radius && Math.abs(dy) <= radius;
      else if (mode === "rect") inside = Math.abs(dx) <= halfW && Math.abs(dy) <= halfH;
      mask[row * scanCols + col] = inside ? 1 : 0;
    }
  }
  return mask;
}

export class Show4DSTEMCompute {
  private device: GPUDevice;
  private maskedSumPipe: GPUComputePipeline;
  private maskedComPipe: GPUComputePipeline;
  private dpcMeanPipe: GPUComputePipeline;
  private dpcComponentPipe: GPUComputePipeline;
  private dpcComponentPairPipe: GPUComputePipeline;
  private dpcOutputMeanPipe: GPUComputePipeline;
  private dpcOutputUlpCorrectPipe: GPUComputePipeline;
  private idpcPackPipe: GPUComputePipeline;
  private idpcPoissonPipe: GPUComputePipeline;
  private idpcExtractPipe: GPUComputePipeline;
  private fftPipes: {
    bitReverseRows: GPUComputePipeline;
    bitReverseCols: GPUComputePipeline;
    butterflyRows: GPUComputePipeline;
    butterflyCols: GPUComputePipeline;
    normalize: GPUComputePipeline;
  };
  private subtractPipe: GPUComputePipeline;
  private reduceFramesPipe: GPUComputePipeline;
  private frameAtPipe: GPUComputePipeline;
  private chunks: Chunk[];
  private dpcBufferCache = new Map<string, { buffer: GPUBuffer; n: number }>();
  readonly scanCount: number;
  readonly detSize: number;
  readonly mode: number;
  // Bad/hot detector pixel indices (from the HDF5 pixel_mask). Auto-excluded from
  // every reduction so the offline result matches CUDA's apply_mask path - the
  // browser data is filtered automatically, no per-call masking needed.
  badPx: Uint32Array = new Uint32Array(0);

  private constructor(device: GPUDevice, chunks: Chunk[], scanCount: number, detSize: number, mode: number) {
    this.device = device; this.chunks = chunks; this.scanCount = scanCount; this.detSize = detSize; this.mode = mode;
    const sg = device.features.has("subgroups");   // warp reduction in maskedSum/CoM when available
    const ms = device.createShaderModule({ code: maskedSumSrc(sg) });
    const rf = device.createShaderModule({ code: REDUCE_FRAMES_WGSL });
    this.maskedSumPipe = device.createComputePipeline({ layout: "auto", compute: { module: ms, entryPoint: "main" } });
    this.maskedComPipe = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code: maskedComSrc(sg) }), entryPoint: "main" } });
    this.dpcMeanPipe = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code: DPC_MEAN_WGSL }), entryPoint: "main" } });
    this.dpcComponentPipe = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code: DPC_COMPONENT_WGSL }), entryPoint: "main" } });
    this.dpcComponentPairPipe = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code: DPC_COMPONENT_PAIR_WGSL }), entryPoint: "main" } });
    this.dpcOutputMeanPipe = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code: DPC_OUTPUT_MEAN_WGSL }), entryPoint: "main" } });
    this.dpcOutputUlpCorrectPipe = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code: DPC_OUTPUT_ULP_CORRECT_WGSL }), entryPoint: "main" } });
    this.idpcPackPipe = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code: IDPC_PACK_WGSL }), entryPoint: "main" } });
    this.idpcPoissonPipe = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code: IDPC_POISSON_WGSL }), entryPoint: "main" } });
    this.idpcExtractPipe = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code: IDPC_EXTRACT_WGSL }), entryPoint: "main" } });
    const fftModule = device.createShaderModule({ code: FFT_2D_SHADER });
    this.fftPipes = {
      bitReverseRows: device.createComputePipeline({ layout: "auto", compute: { module: fftModule, entryPoint: "bitReverseRows" } }),
      bitReverseCols: device.createComputePipeline({ layout: "auto", compute: { module: fftModule, entryPoint: "bitReverseCols" } }),
      butterflyRows: device.createComputePipeline({ layout: "auto", compute: { module: fftModule, entryPoint: "butterflyRows" } }),
      butterflyCols: device.createComputePipeline({ layout: "auto", compute: { module: fftModule, entryPoint: "butterflyCols" } }),
      normalize: device.createComputePipeline({ layout: "auto", compute: { module: fftModule, entryPoint: "normalize2D" } }),
    };
    this.subtractPipe = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code: SUBTRACT_FROM_TOTAL_WGSL }), entryPoint: "main" } });
    this.reduceFramesPipe = device.createComputePipeline({ layout: "auto", compute: { module: rf, entryPoint: "main" } });
    this.frameAtPipe = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code: FRAME_WGSL }), entryPoint: "main" } });
  }

  getDevice(): GPUDevice {
    return this.device;
  }

  async readFloatBuffer(buf: GPUBuffer, n: number): Promise<Float32Array> {
    return await this.readF32(buf, n);
  }

  async checksumFrames(scanIndices: number[]): Promise<Array<{ scanIndex: number; sum: number; min: number; max: number; n: number }>> {
    const bytesPerPixel = this.mode === 0 ? 2 : this.mode === 2 ? 4 : 1;
    const bad = this.badPx.length ? new Set(this.badPx) : null;
    const out: Array<{ scanIndex: number; sum: number; min: number; max: number; n: number }> = [];
    for (const rawIndex of scanIndices) {
      const scanIndex = Math.max(0, Math.min(this.scanCount - 1, Math.round(Number(rawIndex) || 0)));
      const ch = this.chunks.find((c) => scanIndex >= c.startScan && scanIndex < c.startScan + c.nScan) ?? this.chunks[0];
      const localBase = (scanIndex - ch.startScan) * this.detSize;
      const byteOffset = localBase * bytesPerPixel;
      const byteLength = this.detSize * bytesPerPixel;
      const rb = this.device.createBuffer({
        size: Math.max(4, Math.ceil(byteLength / 4) * 4),
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      const enc = this.device.createCommandEncoder();
      enc.copyBufferToBuffer(ch.buffer, byteOffset, rb, 0, byteLength);
      this.device.queue.submit([enc.finish()]);
      await rb.mapAsync(GPUMapMode.READ);
      const mapped = rb.getMappedRange();
      let sum = 0, min = Number.POSITIVE_INFINITY, max = Number.NEGATIVE_INFINITY;
      if (this.mode === 0) {
        const values = new Uint16Array(mapped, 0, this.detSize);
        for (let i = 0; i < values.length; i++) { const v = bad?.has(i) ? 0 : values[i]; sum += v; if (v < min) min = v; if (v > max) max = v; }
      } else if (this.mode === 2) {
        const values = new Float32Array(mapped, 0, this.detSize);
        for (let i = 0; i < values.length; i++) { const v = bad?.has(i) ? 0 : values[i]; sum += v; if (v < min) min = v; if (v > max) max = v; }
      } else {
        const values = new Uint8Array(mapped, 0, this.detSize);
        for (let i = 0; i < values.length; i++) { const v = bad?.has(i) ? 0 : values[i]; sum += v; if (v < min) min = v; if (v > max) max = v; }
      }
      rb.unmap();
      rb.destroy();
      out.push({ scanIndex, sum, min, max, n: this.detSize });
    }
    return out;
  }

  // One frame's diffraction pattern (f32[detSize]) for scan position scanIdx -
  // a GPU extract from whichever chunk holds it. Drives the offline DP panel.
  async frameAt(scanIdx: number): Promise<Float32Array> {
    const ch = this.chunks.find((c) => scanIdx >= c.startScan && scanIdx < c.startScan + c.nScan) ?? this.chunks[0];
    const localBase = (scanIdx - ch.startScan) * this.detSize;
    const out = this.device.createBuffer({ size: this.detSize * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const dims = this.uniform([localBase, this.detSize, this.mode, 0]);
    const bind = this.device.createBindGroup({ layout: this.frameAtPipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: ch.buffer } }, { binding: 1, resource: { buffer: out } }, { binding: 2, resource: { buffer: dims } } ] });
    this.dispatch(this.frameAtPipe, bind, Math.ceil(this.detSize / 64));
    const frame = await this.readF32(out, this.detSize);
    for (const bp of this.badPx) frame[bp] = 0;   // auto-filter hot px in the diffraction pattern
    out.destroy(); dims.destroy(); return frame;
  }

  // Per-scan-position center of mass (intensity-weighted detector centroid) over the
  // active mask. Returns {comY, comX} in detector px (length scanCount each). Drives DPC
  // (CoMx/CoMy/CoMmag/iCoM). detCols unravels the flat pixel index to (row,col); badPx
  // are excluded from the mask, matching every other reduction.
  async maskedCoM(mask: Uint32Array, detCols: number): Promise<{ comY: Float32Array; comX: Float32Array }> {
    const { buffer: com, n } = this.maskedCoMBuffer(mask, detCols);
    const flat = await this.readF32(com, this.scanCount * 2);
    com.destroy();
    const comY = flat.slice(0, this.scanCount), comX = flat.slice(this.scanCount, this.scanCount * 2);
    if (n === 0) { comY.fill(0); comX.fill(0); }
    return { comY, comX };
  }

  // GPU-resident CoM maps. Caller owns the returned buffer.
  maskedCoMBuffer(mask: Uint32Array, detCols: number): { buffer: GPUBuffer; n: number } {
    const device = this.device;
    const { idx, n } = this.detectorIndices(mask);
    const idxBuf = this.upload(idx, GPUBufferUsage.STORAGE);
    const com = device.createBuffer({ size: this.scanCount * 2 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    if (n === 0) {
      idxBuf.destroy();
      return { buffer: com, n };
    }
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    this.encodeMaskedCoM(pass, idxBuf, com, detCols);
    pass.end();
    device.queue.submit([enc.finish()]);
    this.retireBuffers([idxBuf]);
    return { buffer: com, n };
  }

  // GPU-resident centered DPC component. This keeps the common CoMx/CoMy display path on GPU:
  // CoM reduction -> global mean -> component subtraction, with no JavaScript pass over scan pixels.
  maskedDpcBuffer(mask: Uint32Array, detCols: number, component: "row" | "col" | 0 | 1): { buffer: GPUBuffer; n: number; cleanup?: () => void } {
    const device = this.device;
    const { idx, n } = this.detectorIndices(mask);
    const comp = component === "row" || component === 0 ? 0 : 1;
    const key = this.dpcCacheKey(mask, detCols, comp, n);
    const cached = this.dpcBufferCache.get(key);
    if (cached) return { buffer: this.copyScanBuffer(cached.buffer), n: cached.n };
    const out = device.createBuffer({ size: this.scanCount * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    if (n === 0) return { buffer: out, n };
    const idxBuf = this.upload(idx, GPUBufferUsage.STORAGE);
    const com = device.createBuffer({ size: this.scanCount * 2 * 4, usage: GPUBufferUsage.STORAGE });
    const mean = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE });
    const residualMean = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE });
    const cache = device.createBuffer({ size: this.scanCount * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const meanDims = this.uniform([this.scanCount, 0, 0, 0]);
    const compDims = this.uniform([this.scanCount, comp, 0, 0]);
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    this.encodeMaskedCoM(pass, idxBuf, com, detCols);
    pass.setPipeline(this.dpcMeanPipe);
    pass.setBindGroup(0, device.createBindGroup({ layout: this.dpcMeanPipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: com } }, { binding: 1, resource: { buffer: mean } }, { binding: 2, resource: { buffer: meanDims } } ] }));
    pass.dispatchWorkgroups(1);
    pass.setPipeline(this.dpcComponentPipe);
    pass.setBindGroup(0, device.createBindGroup({ layout: this.dpcComponentPipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: com } }, { binding: 1, resource: { buffer: mean } },
      { binding: 2, resource: { buffer: cache } }, { binding: 3, resource: { buffer: compDims } } ] }));
    pass.dispatchWorkgroups(Math.ceil(this.scanCount / 256));
    pass.setPipeline(this.dpcOutputMeanPipe);
    pass.setBindGroup(0, device.createBindGroup({ layout: this.dpcOutputMeanPipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: cache } }, { binding: 1, resource: { buffer: residualMean } }, { binding: 2, resource: { buffer: meanDims } } ] }));
    pass.dispatchWorkgroups(1);
    pass.setPipeline(this.dpcOutputUlpCorrectPipe);
    pass.setBindGroup(0, device.createBindGroup({ layout: this.dpcOutputUlpCorrectPipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: cache } }, { binding: 1, resource: { buffer: residualMean } },
      { binding: 2, resource: { buffer: mean } }, { binding: 3, resource: { buffer: compDims } } ] }));
    pass.dispatchWorkgroups(Math.ceil(this.scanCount / 256));
    pass.end();
    enc.copyBufferToBuffer(cache, 0, out, 0, this.scanCount * 4);
    device.queue.submit([enc.finish()]);
    this.storeDpcCache(key, cache, n);
    return {
      buffer: out,
      n,
      cleanup: () => { idxBuf.destroy(); com.destroy(); mean.destroy(); residualMean.destroy(); meanDims.destroy(); compDims.destroy(); },
    };
  }

  async maskedDpc(mask: Uint32Array, detCols: number, component: "row" | "col" | 0 | 1): Promise<Float32Array> {
    const { buffer, n, cleanup } = this.maskedDpcBuffer(mask, detCols, component);
    const out = await this.readF32(buffer, this.scanCount);
    cleanup?.();
    buffer.destroy();
    if (n === 0) out.fill(0);
    return out;
  }

  private maskedDpcPairBuffers(mask: Uint32Array, detCols: number): { row: GPUBuffer; col: GPUBuffer; n: number; cleanup?: () => void } {
    const device = this.device;
    const { idx, n } = this.detectorIndices(mask);
    const rowKey = this.dpcCacheKey(mask, detCols, 0, n);
    const colKey = this.dpcCacheKey(mask, detCols, 1, n);
    const rowCached = this.dpcBufferCache.get(rowKey);
    const colCached = this.dpcBufferCache.get(colKey);
    if (rowCached && colCached) {
      return { row: rowCached.buffer, col: colCached.buffer, n };
    }
    const rowCache = device.createBuffer({ size: this.scanCount * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const colCache = device.createBuffer({ size: this.scanCount * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    if (n === 0) return { row: rowCache, col: colCache, n, cleanup: () => { rowCache.destroy(); colCache.destroy(); } };
    const idxBuf = this.upload(idx, GPUBufferUsage.STORAGE);
    const com = device.createBuffer({ size: this.scanCount * 2 * 4, usage: GPUBufferUsage.STORAGE });
    const mean = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE });
    const rowResidualMean = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE });
    const colResidualMean = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE });
    const meanDims = this.uniform([this.scanCount, 0, 0, 0]);
    const rowDims = this.uniform([this.scanCount, 0, 0, 0]);
    const colDims = this.uniform([this.scanCount, 1, 0, 0]);
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    this.encodeMaskedCoM(pass, idxBuf, com, detCols);
    pass.setPipeline(this.dpcMeanPipe);
    pass.setBindGroup(0, device.createBindGroup({ layout: this.dpcMeanPipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: com } }, { binding: 1, resource: { buffer: mean } }, { binding: 2, resource: { buffer: meanDims } } ] }));
    pass.dispatchWorkgroups(1);
    pass.setPipeline(this.dpcComponentPairPipe);
    pass.setBindGroup(0, device.createBindGroup({ layout: this.dpcComponentPairPipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: com } }, { binding: 1, resource: { buffer: mean } },
      { binding: 2, resource: { buffer: rowCache } }, { binding: 3, resource: { buffer: colCache } },
      { binding: 4, resource: { buffer: meanDims } } ] }));
    pass.dispatchWorkgroups(Math.ceil(this.scanCount / 256));
    pass.setPipeline(this.dpcOutputMeanPipe);
    pass.setBindGroup(0, device.createBindGroup({ layout: this.dpcOutputMeanPipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: rowCache } }, { binding: 1, resource: { buffer: rowResidualMean } }, { binding: 2, resource: { buffer: rowDims } } ] }));
    pass.dispatchWorkgroups(1);
    pass.setPipeline(this.dpcOutputUlpCorrectPipe);
    pass.setBindGroup(0, device.createBindGroup({ layout: this.dpcOutputUlpCorrectPipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: rowCache } }, { binding: 1, resource: { buffer: rowResidualMean } },
      { binding: 2, resource: { buffer: mean } }, { binding: 3, resource: { buffer: rowDims } } ] }));
    pass.dispatchWorkgroups(Math.ceil(this.scanCount / 256));
    pass.setPipeline(this.dpcOutputMeanPipe);
    pass.setBindGroup(0, device.createBindGroup({ layout: this.dpcOutputMeanPipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: colCache } }, { binding: 1, resource: { buffer: colResidualMean } }, { binding: 2, resource: { buffer: colDims } } ] }));
    pass.dispatchWorkgroups(1);
    pass.setPipeline(this.dpcOutputUlpCorrectPipe);
    pass.setBindGroup(0, device.createBindGroup({ layout: this.dpcOutputUlpCorrectPipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: colCache } }, { binding: 1, resource: { buffer: colResidualMean } },
      { binding: 2, resource: { buffer: mean } }, { binding: 3, resource: { buffer: colDims } } ] }));
    pass.dispatchWorkgroups(Math.ceil(this.scanCount / 256));
    pass.end();
    device.queue.submit([enc.finish()]);
    this.storeDpcCache(rowKey, rowCache, n);
    this.storeDpcCache(colKey, colCache, n);
    return {
      row: rowCache,
      col: colCache,
      n,
      cleanup: () => this.retireBuffers([idxBuf, com, mean, rowResidualMean, colResidualMean, meanDims, rowDims, colDims]),
    };
  }

  // GPU-resident fixed-rotation iDPC phase. Auto-rotation is intentionally not
  // hidden here; callers must pass the explicit rotation/transpose policy they
  // want to compare against.
  async maskedIDpcBuffer(
    mask: Uint32Array,
    detCols: number,
    scanRows: number,
    scanCols: number,
    rotationDeg = 0,
    useTranspose = false,
  ): Promise<{ buffer: GPUBuffer; n: number }> {
    if (scanRows * scanCols !== this.scanCount) {
      throw new Error(`WebGPU iDPC scan shape ${scanRows}x${scanCols} does not match scanCount=${this.scanCount}.`);
    }
    if (!this.isPowerOfTwo(scanRows) || !this.isPowerOfTwo(scanCols)) {
      throw new Error(`WebGPU iDPC requires power-of-two scan dimensions, got ${scanRows}x${scanCols}.`);
    }
    const dpc = this.maskedDpcPairBuffers(mask, detCols);
    await this.device.queue.onSubmittedWorkDone().catch(() => {});
    dpc.cleanup?.();
    const n = dpc.n;
    const out = this.device.createBuffer({ size: this.scanCount * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    if (n === 0) {
      return { buffer: out, n };
    }
    const complexBytes = this.scanCount * 8;
    const gradFft = this.device.createBuffer({ size: complexBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const phaseFft = this.device.createBuffer({ size: complexBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const theta = rotationDeg * Math.PI / 180.0;
    const packParams = this.idpcPackParams(Math.cos(theta), Math.sin(theta), useTranspose);
    const packBind = this.device.createBindGroup({ layout: this.idpcPackPipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: dpc.row } },
      { binding: 1, resource: { buffer: dpc.col } },
      { binding: 2, resource: { buffer: gradFft } },
      { binding: 3, resource: { buffer: packParams } },
    ] });
    this.dispatch(this.idpcPackPipe, packBind, Math.ceil(this.scanCount / 256));
    this.runFFT2DInPlace(gradFft, scanCols, scanRows, false);
    const poissonDims = this.uniform([scanCols, scanRows, this.scanCount, 0]);
    const poissonBind = this.device.createBindGroup({ layout: this.idpcPoissonPipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: gradFft } },
      { binding: 1, resource: { buffer: phaseFft } },
      { binding: 2, resource: { buffer: poissonDims } },
    ] });
    this.dispatch(this.idpcPoissonPipe, poissonBind, Math.ceil(this.scanCount / 256));
    this.runFFT2DInPlace(phaseFft, scanCols, scanRows, true);
    const extractDims = this.uniform([this.scanCount, 0, 0, 0]);
    const extractBind = this.device.createBindGroup({ layout: this.idpcExtractPipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: phaseFft } },
      { binding: 1, resource: { buffer: out } },
      { binding: 2, resource: { buffer: extractDims } },
    ] });
    this.dispatch(this.idpcExtractPipe, extractBind, Math.ceil(this.scanCount / 256));
    this.retireBuffers([gradFft, phaseFft, packParams, poissonDims, extractDims]);
    return { buffer: out, n };
  }

  async maskedIDpc(
    mask: Uint32Array,
    detCols: number,
    scanRows: number,
    scanCols: number,
    rotationDeg = 0,
    useTranspose = false,
  ): Promise<Float32Array> {
    const { buffer, n } = await this.maskedIDpcBuffer(mask, detCols, scanRows, scanCols, rotationDeg, useTranspose);
    const out = await this.readF32(buffer, this.scanCount);
    buffer.destroy();
    if (n === 0) out.fill(0);
    return out;
  }

  // Single decompressed stack -> one chunk (the common, fits-in-one-buffer case).
  static async create(stack: Uint8Array, scanCount: number, detSize: number): Promise<Show4DSTEMCompute | null> {
    return Show4DSTEMCompute.createChunked([{ bytes: stack, startScan: 0, nScan: scanCount }], scanCount, detSize);
  }

  // Decompress a native HDF5 bitshuffle+LZ4 (bslz4) stack on the GPU and wrap the
  // decoded buffer as the (single) compute chunk - the offline "ship compressed,
  // decompress in browser, no Python" path. dtype "uint8" (offline default, clip
  // 0-255, half memory) or "uint16" (lossless). The decoded buffer is packed in
  // [scanPos][detPixel] order matching sample() for that mode, so masked_sum /
  // reduceFrames run on it unchanged.
  static async createFromBslz4(spec: Bslz4Spec, dtype: "uint8" | "uint16" | "float32" = "uint8", srcDtype: "uint8" | "uint16" | "uint32" | "float32" = "uint16"): Promise<Show4DSTEMCompute | null> {
    const decoded = await decodeBslz4ToStack(spec, dtype, srcDtype);
    if (!decoded) return null;
    const chunks: Chunk[] = [{ buffer: decoded.buffer, startScan: 0, nScan: spec.nFrames }];
    return new Show4DSTEMCompute(decoded.device, chunks, spec.nFrames, spec.detSize, decoded.mode);
  }

  // Wrap already-decoded GPU stack buffers as a compute (no decode). Lets a caller pipeline
  // parse + decode itself (decode group N while parsing group N+1) and hand the finished
  // chunk buffers here. mode: 1 = uint8, 0 = uint16.
  static fromGpuChunks(device: GPUDevice, chunks: { buffer: GPUBuffer; startScan: number; nScan: number }[], scanCount: number, detSize: number, mode: number): Show4DSTEMCompute {
    return new Show4DSTEMCompute(device, chunks, scanCount, detSize, mode);
  }

  // Chunked bslz4: decode each scan-row chunk's compressed bytes into its OWN GPU
  // buffer and hold them all, so a stack far bigger than one 1 GB buffer (full
  // 512x512x192x192 = 9.6 GB uint8) lives across N buffers and masked_sum /
  // reduceFrames reduce across them. Each decode reuses a ~1 GB scratch internally.
  static async createFromBslz4Chunked(
    chunkSpecs: (Bslz4Spec & { startScan: number; nScan: number })[],
    scanCount: number, detSize: number, dtype: "uint8" | "uint16" | "float32" = "uint8",
    srcDtype: "uint8" | "uint16" | "uint32" | "float32" = "uint16",
  ): Promise<Show4DSTEMCompute | null> {
    // Batch the per-chunk decodes (one submit + await per group) so the GPU overlaps
    // upload and compute instead of draining after every chunk.
    const decoded = await decodeBslz4Batch(chunkSpecs, dtype, srcDtype);
    if (!decoded) return null;
    const chunks: Chunk[] = decoded.buffers.map((buffer, i) => ({ buffer, startScan: chunkSpecs[i].startScan, nScan: chunkSpecs[i].nScan }));
    return new Show4DSTEMCompute(decoded.device, chunks, scanCount, detSize, decoded.mode);
  }

  // N chunks, each {bytes, startScan, nScan}. Each chunk's bytes hold its scan
  // range's frames contiguously. dtype inferred from total bytes vs total pixels.
  static async createChunked(chunkSpecs: { bytes: Uint8Array; startScan: number; nScan: number }[], scanCount: number, detSize: number): Promise<Show4DSTEMCompute | null> {
    const device = await getGPUDevice();
    if (!device) return null;
    const totalBytes = chunkSpecs.reduce((a, c) => a + c.bytes.byteLength, 0);
    const mode = totalBytes <= scanCount * detSize ? 1 : 0;  // <= 1 byte/pixel => uint8
    const chunks: Chunk[] = chunkSpecs.map((c) => {
      const padLen = Math.ceil(c.bytes.byteLength / 4) * 4;
      const buffer = device.createBuffer({ size: Math.max(4, padLen), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(buffer, 0, c.bytes.buffer as ArrayBuffer, c.bytes.byteOffset, c.bytes.byteLength);
      return { buffer, startScan: c.startScan, nScan: c.nScan };
    });
    return new Show4DSTEMCompute(device, chunks, scanCount, detSize, mode);
  }

  // Virtual image: f32[scanCount]. Each chunk writes its disjoint VI slice.
  // Loops only the ACTIVE (in-aperture) detector pixels, not all detSize - a BF
  // disk (~9k of 36864 px) or ADF annulus is then 4-10x fewer reads per scan pos.
  async maskedSum(mask: Uint32Array): Promise<Float32Array> {
    const device = this.device;
    const bad = this.badPx.length ? new Set(this.badPx) : null;
    const idxArr = new Uint32Array(this.detSize); let n = 0;
    for (let k = 0; k < this.detSize; k++) if (mask[k] !== 0 && !(bad && bad.has(k))) idxArr[n++] = k;  // skip hot px
    const idx = idxArr.subarray(0, n || 1);   // active pixel indices (>=1 to keep a valid binding)
    const idxBuf = this.upload(idx, GPUBufferUsage.STORAGE);
    const vi = device.createBuffer({ size: this.scanCount * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const temps: GPUBuffer[] = [];
    // One workgroup per scan position (2D grid for >65535), ALL chunks in ONE encoder + submit,
    // and the readback folded in - so a BF/ADF drag is a single GPU pass + single sync.
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass(); pass.setPipeline(this.maskedSumPipe);
    for (const ch of this.chunks) {
      const gx = Math.min(ch.nScan, MAX_WG), gy = Math.ceil(ch.nScan / MAX_WG);
      const dims = this.uniform([ch.startScan, ch.nScan, this.detSize, this.mode]); temps.push(dims);
      const dims2 = this.uniform([gx, 0, 0, 0]); temps.push(dims2);
      const bind = device.createBindGroup({ layout: this.maskedSumPipe.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: ch.buffer } }, { binding: 1, resource: { buffer: idxBuf } },
        { binding: 2, resource: { buffer: vi } }, { binding: 3, resource: { buffer: dims } }, { binding: 4, resource: { buffer: dims2 } } ] });
      pass.setBindGroup(0, bind); pass.dispatchWorkgroups(gx, gy);
    }
    pass.end();
    const rb = device.createBuffer({ size: this.scanCount * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    enc.copyBufferToBuffer(vi, 0, rb, 0, this.scanCount * 4);
    device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(rb.getMappedRange().slice(0)); rb.unmap(); rb.destroy();
    if (n === 0) out.fill(0);   // empty mask -> all zero (idx had a dummy entry)
    idxBuf.destroy(); vi.destroy(); temps.forEach((b) => b.destroy()); return out;
  }

  // GPU-RESIDENT virtual image: identical to maskedSum but returns the vi GPU buffer WITHOUT
  // the readback. The 60fps drag path feeds this straight into the colormap engine + canvas, so
  // there is NO GPU->CPU->GPU bounce and NO mapAsync fence (the two fences - maskedSum readback +
  // colormap rgba readback - were the ~100ms/repaint that capped the drag at ~9fps). Caller owns
  // the returned buffer (the colormap slot adopts it; freed on the next adopt / dispose).
  maskedSumBuffer(mask: Uint32Array): { buffer: GPUBuffer; n: number } {
    const selected = this.detectorIndices(mask);
    if (selected.n > this.detSize * 0.5) {
      const complement = this.detectorComplementIndices(mask);
      if (complement.n < selected.n) {
        const total = this.totalSumBuffer();
        if (complement.n === 0) {
          return { buffer: this.copyScanBuffer(total), n: selected.n };
        }
        const complementSum = this.maskedSumBufferFromIndices(complement.idx, complement.n);
        const out = this.subtractFromTotalBuffer(total, complementSum.buffer);
        this.retireBuffers([complementSum.buffer]);
        return { buffer: out, n: selected.n };
      }
    }
    return this.maskedSumBufferFromIndices(selected.idx, selected.n);
  }

  private maskedSumBufferFromIndices(idx: Uint32Array, n: number): { buffer: GPUBuffer; n: number } {
    const device = this.device;
    const vi = device.createBuffer({ size: this.scanCount * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    if (n === 0) return { buffer: vi, n };
    const idxBuf = this.upload(idx, GPUBufferUsage.STORAGE);
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass(); pass.setPipeline(this.maskedSumPipe);
    for (const cd of this.sumDims()) {   // per-chunk dims are constant -> built once, reused every drag frame
      const bind = device.createBindGroup({ layout: this.maskedSumPipe.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: cd.chunk } }, { binding: 1, resource: { buffer: idxBuf } },
        { binding: 2, resource: { buffer: vi } }, { binding: 3, resource: { buffer: cd.dims } }, { binding: 4, resource: { buffer: cd.dims2 } } ] });
      pass.setBindGroup(0, bind); pass.dispatchWorkgroups(cd.gx, cd.gy);
    }
    pass.end();
    device.queue.submit([enc.finish()]);
    this.retireBuffers([idxBuf]);   // vi handed to caller; dims are cached + reused, NOT destroyed
    return { buffer: vi, n };
  }

  // Per-chunk maskedSum dispatch params (chunk buffer, dims uniforms, 2D grid). These never change
  // for a dataset, so build them ONCE and reuse across every drag frame instead of allocating +
  // destroying ~2 uniform buffers per chunk per frame (~3300 buffer creates/s during a drag).
  private sumDimsCache: { chunk: GPUBuffer; dims: GPUBuffer; dims2: GPUBuffer; gx: number; gy: number }[] | null = null;
  private sumDims() {
    if (!this.sumDimsCache) {
      this.sumDimsCache = this.chunks.map((ch) => {
        const gx = Math.min(ch.nScan, MAX_WG), gy = Math.ceil(ch.nScan / MAX_WG);
        return { chunk: ch.buffer, dims: this.uniform([ch.startScan, ch.nScan, this.detSize, this.mode]), dims2: this.uniform([gx, 0, 0, 0]), gx, gy };
      });
    }
    return this.sumDimsCache;
  }

  private comDimsCache: { detCols: number; rows: { chunk: GPUBuffer; dims: GPUBuffer; dims2: GPUBuffer; gx: number; gy: number }[] } | null = null;
  private comDims(detCols: number) {
    if (!this.comDimsCache || this.comDimsCache.detCols !== detCols) {
      if (this.comDimsCache) {
        for (const cd of this.comDimsCache.rows) { cd.dims.destroy(); cd.dims2.destroy(); }
      }
      this.comDimsCache = {
        detCols,
        rows: this.chunks.map((ch) => {
          const gx = Math.min(ch.nScan, MAX_WG), gy = Math.ceil(ch.nScan / MAX_WG);
          return { chunk: ch.buffer, dims: this.uniform([ch.startScan, ch.nScan, this.detSize, this.mode]), dims2: this.uniform([detCols, this.scanCount, gx, 0]), gx, gy };
        }),
      };
    }
    return this.comDimsCache.rows;
  }

  private encodeMaskedCoM(pass: GPUComputePassEncoder, idxBuf: GPUBuffer, com: GPUBuffer, detCols: number) {
    pass.setPipeline(this.maskedComPipe);
    for (const cd of this.comDims(detCols)) {
      const bind = this.device.createBindGroup({ layout: this.maskedComPipe.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: cd.chunk } }, { binding: 1, resource: { buffer: idxBuf } },
        { binding: 2, resource: { buffer: com } }, { binding: 3, resource: { buffer: cd.dims } }, { binding: 4, resource: { buffer: cd.dims2 } } ] });
      pass.setBindGroup(0, bind); pass.dispatchWorkgroups(cd.gx, cd.gy);
    }
  }

  // DP over a real-space ROI: f32[detSize]. scanMask is GLOBAL; chunks accumulate
  // in INTEGER (u32, bit-exact) - the mean divide happens once in f64 at readback,
  // so the result matches the torch/CUDA integer-sum-then-divide exactly (even on
  // saturated 65535 dead pixels, where f32 accumulation would drift ~1 count).
  async reduceFrames(scanMask: Uint32Array, mean = true): Promise<Float32Array> {
    const device = this.device;
    const maskBuf = this.upload(scanMask, GPUBufferUsage.STORAGE);
    const dp = device.createBuffer({ size: this.detSize * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(dp, 0, new Uint32Array(this.detSize));  // zero-init integer accumulator
    const temps: GPUBuffer[] = [];
    // Record EVERY chunk's reduce pass into ONE command encoder + ONE submit. Per-chunk submits
    // (27 buffers for a 27-file dataset) each pay kernel-launch + queue round-trip overhead and
    // do not overlap; batching lets the GPU run them back-to-back into the shared dp accumulator.
    const grid = Math.ceil(this.detSize / 64);
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.reduceFramesPipe);
    for (const ch of this.chunks) {
      const dims = this.uniform([ch.startScan, ch.nScan, this.detSize, this.mode]); temps.push(dims);
      const bind = device.createBindGroup({ layout: this.reduceFramesPipe.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: ch.buffer } }, { binding: 1, resource: { buffer: maskBuf } },
        { binding: 2, resource: { buffer: dp } }, { binding: 3, resource: { buffer: dims } } ] });
      pass.setBindGroup(0, bind); pass.dispatchWorkgroups(grid, FRAME_BLOCKS);
    }
    pass.end();
    // Fold the dp -> readback copy into the SAME encoder: one submit, one GPU->CPU sync. A
    // separate readU32 would submit + fence a second time (~30-50ms of mapAsync round-trip for
    // a 147 KB buffer - all latency, no bandwidth), doubling the sync cost of a tiny readback.
    const rb = device.createBuffer({ size: this.detSize * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    enc.copyBufferToBuffer(dp, 0, rb, 0, this.detSize * 4);
    device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const sums = new Uint32Array(rb.getMappedRange().slice(0)); rb.unmap(); rb.destroy();
    temps.forEach((b) => b.destroy());
    const n = mean ? (scanMask.reduce((a, v) => a + (v ? 1 : 0), 0) || 1) : 1;
    const out = new Float32Array(this.detSize);
    for (let i = 0; i < this.detSize; i++) out[i] = sums[i] / n;  // f64 divide -> f32 store
    for (const bp of this.badPx) out[bp] = 0;   // auto-filter hot px (matches CUDA apply_mask)
    maskBuf.destroy(); dp.destroy(); return out;
  }

  private upload(arr: Uint32Array, usage: number): GPUBuffer {
    const b = this.device.createBuffer({ size: Math.max(16, arr.byteLength), usage: usage | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(b, 0, arr.buffer as ArrayBuffer, arr.byteOffset, arr.byteLength); return b;
  }
  private detectorIndices(mask: Uint32Array): { idx: Uint32Array; n: number } {
    const bad = this.badPx.length ? new Set(this.badPx) : null;
    const idxArr = new Uint32Array(this.detSize); let n = 0;
    for (let k = 0; k < this.detSize; k++) if (mask[k] !== 0 && !(bad && bad.has(k))) idxArr[n++] = k;
    return { idx: idxArr.subarray(0, n || 1), n };
  }
  private detectorComplementIndices(mask: Uint32Array): { idx: Uint32Array; n: number } {
    const bad = this.badPx.length ? new Set(this.badPx) : null;
    const idxArr = new Uint32Array(this.detSize); let n = 0;
    for (let k = 0; k < this.detSize; k++) if (mask[k] === 0 && !(bad && bad.has(k))) idxArr[n++] = k;
    return { idx: idxArr.subarray(0, n || 1), n };
  }
  private fullDetectorIndices(): { idx: Uint32Array; n: number } {
    const bad = this.badPx.length ? new Set(this.badPx) : null;
    const idxArr = new Uint32Array(this.detSize); let n = 0;
    for (let k = 0; k < this.detSize; k++) if (!(bad && bad.has(k))) idxArr[n++] = k;
    return { idx: idxArr.subarray(0, n || 1), n };
  }

  private totalSumCache: GPUBuffer | null = null;
  private totalSumCacheKey: string | null = null;
  private totalSumBuffer(): GPUBuffer {
    const key = this.badPixelKey();
    if (!this.totalSumCache || this.totalSumCacheKey !== key) {
      if (this.totalSumCache) this.totalSumCache.destroy();
      const full = this.fullDetectorIndices();
      this.totalSumCache = this.maskedSumBufferFromIndices(full.idx, full.n).buffer;
      this.totalSumCacheKey = key;
    }
    return this.totalSumCache;
  }
  private badPixelKey(): string {
    let hash = 2166136261;
    for (const value of this.badPx) {
      hash = Math.imul(hash ^ value, 16777619);
    }
    return `${this.badPx.length}:${hash >>> 0}`;
  }
  private detectorMaskHash(mask: Uint32Array): string {
    let hash = 2166136261;
    for (let i = 0; i < mask.length; i++) {
      hash = Math.imul(hash ^ (mask[i] ? 1 : 0), 16777619);
    }
    return `${mask.length}:${hash >>> 0}`;
  }
  private dpcCacheKey(mask: Uint32Array, detCols: number, component: number, activePixels: number): string {
    return [
      this.badPixelKey(),
      this.detectorMaskHash(mask),
      `detCols:${detCols}`,
      `component:${component}`,
      `active:${activePixels}`,
      `scan:${this.scanCount}`,
      `det:${this.detSize}`,
      `mode:${this.mode}`,
    ].join("|");
  }
  private storeDpcCache(key: string, buffer: GPUBuffer, n: number): void {
    const previous = this.dpcBufferCache.get(key);
    if (previous && previous.buffer !== buffer) previous.buffer.destroy();
    this.dpcBufferCache.set(key, { buffer, n });
    while (this.dpcBufferCache.size > 8) {
      const oldest = this.dpcBufferCache.keys().next().value;
      if (oldest === undefined) break;
      const entry = this.dpcBufferCache.get(oldest);
      entry?.buffer.destroy();
      this.dpcBufferCache.delete(oldest);
    }
  }
  private copyScanBuffer(src: GPUBuffer): GPUBuffer {
    const out = this.device.createBuffer({ size: this.scanCount * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(src, 0, out, 0, this.scanCount * 4);
    this.device.queue.submit([enc.finish()]);
    return out;
  }
  private subtractFromTotalBuffer(total: GPUBuffer, subtract: GPUBuffer): GPUBuffer {
    const out = this.device.createBuffer({ size: this.scanCount * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const dims = this.uniform([this.scanCount, 0, 0, 0]);
    const bind = this.device.createBindGroup({ layout: this.subtractPipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: total } }, { binding: 1, resource: { buffer: subtract } },
      { binding: 2, resource: { buffer: out } }, { binding: 3, resource: { buffer: dims } } ] });
    this.dispatch(this.subtractPipe, bind, Math.ceil(this.scanCount / 256));
    this.retireBuffers([dims]);
    return out;
  }
  private uniform(vals: number[]): GPUBuffer {
    const b = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const a = new Uint32Array(vals); this.device.queue.writeBuffer(b, 0, a.buffer as ArrayBuffer, a.byteOffset, a.byteLength); return b;
  }
  private idpcPackParams(cosTheta: number, sinTheta: number, useTranspose: boolean): GPUBuffer {
    const buf = new ArrayBuffer(32);
    const u32 = new Uint32Array(buf);
    const f32 = new Float32Array(buf);
    u32[0] = this.scanCount;
    u32[1] = useTranspose ? 1 : 0;
    f32[4] = cosTheta;
    f32[5] = sinTheta;
    const b = this.device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(b, 0, buf);
    return b;
  }
  private isPowerOfTwo(value: number): boolean {
    const n = Math.max(0, Math.round(value));
    return n > 0 && (n & (n - 1)) === 0;
  }
  private runFFT2DInPlace(dataBuffer: GPUBuffer, width: number, height: number, inverse: boolean): void {
    if (!this.isPowerOfTwo(width) || !this.isPowerOfTwo(height)) {
      throw new Error(`WebGPU FFT requires power-of-two dimensions, got ${height}x${width}.`);
    }
    const workgroupsX = Math.ceil(width / 16);
    const workgroupsY = Math.ceil(height / 16);
    const passes: { pipeline: GPUComputePipeline; params: ArrayBuffer }[] = [];
    const pushPass = (
      pipeline: GPUComputePipeline,
      stageCount: number,
      stageIndex: number,
      rowAxis: boolean,
    ) => {
      const params = new ArrayBuffer(24);
      const paramsU32 = new Uint32Array(params);
      const paramsF32 = new Float32Array(params);
      paramsU32[0] = width;
      paramsU32[1] = height;
      paramsU32[2] = stageCount;
      paramsU32[3] = stageIndex;
      paramsF32[4] = inverse ? 1.0 : -1.0;
      paramsU32[5] = rowAxis ? 1 : 0;
      passes.push({ pipeline, params });
    };
    const widthStages = Math.log2(width);
    const heightStages = Math.log2(height);
    pushPass(this.fftPipes.bitReverseRows, widthStages, 0, true);
    for (let stage = 0; stage < widthStages; stage++) {
      pushPass(this.fftPipes.butterflyRows, widthStages, stage, true);
    }
    pushPass(this.fftPipes.bitReverseCols, heightStages, 0, false);
    for (let stage = 0; stage < heightStages; stage++) {
      pushPass(this.fftPipes.butterflyCols, heightStages, stage, false);
    }
    if (inverse) pushPass(this.fftPipes.normalize, heightStages, 0, false);

    const paramsBuffers = passes.map((item) => {
      const buffer = this.device.createBuffer({ size: 24, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(buffer, 0, item.params);
      return buffer;
    });
    const enc = this.device.createCommandEncoder();
    passes.forEach((item, index) => {
      const bind = this.device.createBindGroup({ layout: item.pipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: paramsBuffers[index] } },
        { binding: 1, resource: { buffer: dataBuffer } },
      ] });
      const pass = enc.beginComputePass();
      pass.setPipeline(item.pipeline);
      pass.setBindGroup(0, bind);
      pass.dispatchWorkgroups(workgroupsX, workgroupsY);
      pass.end();
    });
    this.device.queue.submit([enc.finish()]);
    this.retireBuffers(paramsBuffers);
  }
  private retireBuffers(buffers: GPUBuffer[]): void {
    if (!buffers.length) return;
    void this.device.queue.onSubmittedWorkDone()
      .catch(() => {})
      .finally(() => {
        for (const b of buffers) b.destroy();
      });
  }
  private dispatch(pipe: GPUComputePipeline, bind: GPUBindGroup, groups: number, gy = 1) {
    const enc = this.device.createCommandEncoder(); const pass = enc.beginComputePass();
    pass.setPipeline(pipe); pass.setBindGroup(0, bind); pass.dispatchWorkgroups(groups, gy); pass.end();
    this.device.queue.submit([enc.finish()]);
  }
  private async readF32(buf: GPUBuffer, n: number): Promise<Float32Array> {
    const rb = this.device.createBuffer({ size: n * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.device.createCommandEncoder(); enc.copyBufferToBuffer(buf, 0, rb, 0, n * 4); this.device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ); const out = new Float32Array(rb.getMappedRange().slice(0)); rb.unmap(); rb.destroy(); return out;
  }

  dispose() {
    for (const c of this.chunks) c.buffer.destroy();
    if (this.totalSumCache) { this.totalSumCache.destroy(); this.totalSumCache = null; this.totalSumCacheKey = null; }
    for (const entry of this.dpcBufferCache.values()) entry.buffer.destroy();
    this.dpcBufferCache.clear();
    if (this.sumDimsCache) { for (const cd of this.sumDimsCache) { cd.dims.destroy(); cd.dims2.destroy(); } this.sumDimsCache = null; }
    if (this.comDimsCache) { for (const cd of this.comDimsCache.rows) { cd.dims.destroy(); cd.dims2.destroy(); } this.comDimsCache = null; }
  }
}

export class Show4DSTEMCpuCompute {
  readonly scanCount: number;
  readonly detSize: number;
  readonly mode: number;
  badPx: Uint32Array = new Uint32Array(0);
  private stack: Uint8Array;

  private constructor(stack: Uint8Array, scanCount: number, detSize: number, mode: number) {
    this.stack = stack;
    this.scanCount = scanCount;
    this.detSize = detSize;
    this.mode = mode;
  }

  static create(stack: Uint8Array, scanCount: number, detSize: number): Show4DSTEMCpuCompute {
    const expectedU8 = scanCount * detSize;
    const mode = stack.byteLength <= expectedU8 ? 1 : 0;
    return new Show4DSTEMCpuCompute(stack, scanCount, detSize, mode);
  }

  async frameAt(scanIdx: number): Promise<Float32Array> {
    const out = new Float32Array(this.detSize);
    const base = Math.max(0, Math.min(this.scanCount - 1, scanIdx | 0)) * this.detSize;
    for (let k = 0; k < this.detSize; k++) out[k] = this.sample(base + k);
    for (const bp of this.badPx) out[bp] = 0;
    return out;
  }

  async maskedSum(mask: Uint32Array): Promise<Float32Array> {
    const bad = this.badPx.length ? new Set(this.badPx) : null;
    const idx: number[] = [];
    for (let k = 0; k < this.detSize; k++) {
      if (mask[k] !== 0 && !(bad && bad.has(k))) idx.push(k);
    }
    const out = new Float32Array(this.scanCount);
    if (idx.length === 0) return out;
    for (let scan = 0; scan < this.scanCount; scan++) {
      const base = scan * this.detSize;
      let sum = 0;
      for (let j = 0; j < idx.length; j++) sum += this.sample(base + idx[j]);
      out[scan] = sum;
    }
    return out;
  }

  async maskedCoM(mask: Uint32Array, detCols: number): Promise<{ comY: Float32Array; comX: Float32Array }> {
    const bad = this.badPx.length ? new Set(this.badPx) : null;
    const idx: number[] = [];
    for (let k = 0; k < this.detSize; k++) {
      if (mask[k] !== 0 && !(bad && bad.has(k))) idx.push(k);
    }
    const comY = new Float32Array(this.scanCount);
    const comX = new Float32Array(this.scanCount);
    if (idx.length === 0) return { comY, comX };
    for (let scan = 0; scan < this.scanCount; scan++) {
      const base = scan * this.detSize;
      let wsum = 0;
      let ysum = 0;
      let xsum = 0;
      for (let j = 0; j < idx.length; j++) {
        const p = idx[j];
        const v = this.sample(base + p);
        wsum += v;
        ysum += Math.floor(p / detCols) * v;
        xsum += (p % detCols) * v;
      }
      if (wsum > 0) {
        comY[scan] = ysum / wsum;
        comX[scan] = xsum / wsum;
      }
    }
    return { comY, comX };
  }

  async maskedDpc(mask: Uint32Array, detCols: number, component: "row" | "col" | 0 | 1): Promise<Float32Array> {
    const { comY, comX } = await this.maskedCoM(mask, detCols);
    const values = component === "row" || component === 0 ? comY : comX;
    let mean = 0;
    for (let i = 0; i < values.length; i++) mean += values[i];
    mean /= Math.max(1, values.length);
    const out = new Float32Array(values.length);
    for (let i = 0; i < values.length; i++) out[i] = values[i] - mean;
    return out;
  }

  async reduceFrames(scanMask: Uint32Array, mean = true): Promise<Float32Array> {
    const out = new Float32Array(this.detSize);
    let n = 0;
    for (let scan = 0; scan < this.scanCount; scan++) {
      if (!scanMask[scan]) continue;
      n++;
      const base = scan * this.detSize;
      for (let k = 0; k < this.detSize; k++) out[k] += this.sample(base + k);
    }
    if (mean && n > 0) {
      for (let k = 0; k < this.detSize; k++) out[k] /= n;
    }
    for (const bp of this.badPx) out[bp] = 0;
    return out;
  }

  dispose() {
    // CPU fallback owns no browser resources.
  }

  private sample(globalPixel: number): number {
    if (this.mode === 1) return this.stack[globalPixel] ?? 0;
    const i = globalPixel * 2;
    return (this.stack[i] ?? 0) | ((this.stack[i + 1] ?? 0) << 8);
  }
}
