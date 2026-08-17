/** WebGPU shaders for DPC centering and integrated-DPC reconstruction. */
export const DPC_MEAN_WGSL = `
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
export const DPC_COMPONENT_WGSL = `
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

export const DPC_COMPONENT_PAIR_WGSL = `
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

/** Magnitude of an already-centered row/column DPC pair. */
export const DPC_MAGNITUDE_WGSL = `
@group(0) @binding(0) var<storage,read> rowDpc: array<f32>;
@group(0) @binding(1) var<storage,read> colDpc: array<f32>;
@group(0) @binding(2) var<storage,read_write> magnitude: array<f32>;
@group(0) @binding(3) var<uniform> u: vec4<u32>; // scanCount, 0, 0, 0
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= u.x) { return; }
  magnitude[i] = length(vec2<f32>(rowDpc[i], colDpc[i]));
}`;

// Mean reduction for one already-centered DPC component. The first DPC centering
// matches the backend semantics; this pass lets the next shader choose the same
// one-ulp rounding side as the NumPy/CUDA reference without a CPU readback.
export const DPC_OUTPUT_MEAN_WGSL = `
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

export const DPC_OUTPUT_ULP_CORRECT_WGSL = `
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
export const IDPC_PACK_WGSL = `
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

export const IDPC_POISSON_WGSL = `
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

export const IDPC_EXTRACT_WGSL = `
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
