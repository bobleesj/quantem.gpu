/// <reference types="@webgpu/types" />
/**
 * Browser-resident lossless 4D-STEM source: detector-conditioned byte rANS
 * streams (ryg_rans byte renormalisation, LOWER = 2^23) decoded on the GPU
 * where the pixels are painted. Nothing is uploaded per interaction: a
 * detector change decodes only the columns that entered or left the mask,
 * for every 256-frame window of every block of every acquisition, and adds
 * or subtracts their exact counts into GPU-resident uint32 images.
 *
 * Exactness: every count is decoded as stored; window checkpoints are
 * verified to terminate each column stream exactly at build time.
 */

import { copyRansPayload, payloadGroupLimit, type RansByteSource, type RansPayloadProfile, type RansPreparedGroup } from "./rans-source";
import { qemFilesSource, type QemByteFile, type QemCheckpointGroup } from "./qem-source";
import { DetectorCompute } from "./backend";
import { EXACT_INTEGER_COM_WGSL } from "./exact-com";
import type { Uint32ImageView } from "./borrowed-image";

const WINDOW = 256;
const LOWER = 8388608;

const COMMON_WGSL = /* wgsl */ `
struct Params { frames: u32, K: u32, scale: u32, windows: u32, n_cols: u32, unit0: u32, n_units: u32, N: u32 }
struct Unit { payload_word: u32, offsets_base: u32, colmeta_word: u32, entries_word: u32, lut_word: u32, chk_base: u32, out_base: u32, pad: u32 }
@group(0) @binding(0) var<storage, read> payload: array<u32>;
@group(0) @binding(1) var<storage, read> offsets: array<u32>;
@group(0) @binding(2) var<storage, read> tables: array<u32>;    // column metadata, packed symbol entries, slot lookup bytes
@group(0) @binding(3) var<storage, read_write> chk: array<u32>;
@group(0) @binding(4) var<storage, read> cols: array<u32>;
@group(0) @binding(5) var<storage, read_write> out: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read> units: array<Unit>;
@group(0) @binding(7) var<uniform> p: Params;
const LOWER: u32 = ${LOWER}u;
override BINARY_LOOKUP: bool = false;
override DIRECT_LOOKUP: bool = false;
struct Col { cursor: u32, end: u32, state: u32, left0: u32, right0: u32, raw: bool, bad: bool, lutbyte: u32, pay: u32, ebase: u32, mode: u32 }
fn byte_at(c: ptr<function, Col>, at: u32) -> u32 { let w = (*c).pay + (at >> 2u); return (payload[w] >> ((at & 3u) * 8u)) & 255u; }
fn open_col(u: Unit, k: u32) -> Col {
  var c: Col;
  if (offsets[u.offsets_base + p.K] == 0xffffffffu) {
    let code = offsets[u.offsets_base + k];
    c.mode = 6u; c.pay = code >> 5u; c.right0 = code & 31u;
    return c;
  }
  c.pay = u.payload_word; c.ebase = u.entries_word;
  // A block staged inside a resident payload group may start mid-word: bits
  // 28-29 of pad carry its byte offset within payload_word.
  let byte_bias = (u.pad >> 28u) & 3u;
  c.cursor = offsets[u.offsets_base + k] + byte_bias; c.end = offsets[u.offsets_base + k + 1u] + byte_bias;
  c.right0 = tables[u.colmeta_word + k * 3u + 1u];
  c.left0 = tables[u.colmeta_word + k * 3u]; c.mode = tables[u.colmeta_word + k * 3u + 2u]; c.raw = c.mode == 1u;
  c.bad = false; c.state = LOWER; c.lutbyte = u.lut_word * 4u + k * 256u;
  if (DIRECT_LOOKUP) { c.lutbyte = u.lut_word + (c.left0 / 33u) * 1024u; }
  return c;
}
fn unit_frames(u: Unit) -> u32 { return select(p.frames, u.pad & 0x0fffffffu, (u.pad & 0x0fffffffu) != 0u); }
fn start_col(c: ptr<function, Col>) {
  if ((*c).mode >= 3u) { if ((*c).mode == 5u) { (*c).state = 0u; } return; }
  if ((*c).raw) { return; }
  if ((*c).end - (*c).cursor < 4u) { (*c).bad = true; return; }
  var s: u32 = 0u;
  for (var b = 0u; b < 4u; b++) { s |= byte_at(c, (*c).cursor) << (8u * b); (*c).cursor += 1u; }
  (*c).state = s;
  if (s < LOWER || s >= 2147483648u) { (*c).bad = true; }
}
fn packed_value(base: u32, width: u32, row: u32) -> u32 {
  if (width == 0u) { return 0u; }
  let bit = row * width; let word = base + (bit >> 5u); let shift = bit & 31u;
  var value = payload[word] >> shift;
  if (shift + width > 32u) { value |= payload[word + 1u] << (32u - shift); }
  return value & ((1u << width) - 1u);
}
fn seek(c: ptr<function, Col>, u: Unit, k: u32, w: u32) {
  if ((*c).mode == 6u) { (*c).cursor = w * 256u; return; }
  let i = u.chk_base + (k * p.windows + w) * 2u;
  (*c).state = chk[i]; (*c).cursor = chk[i + 1u];
}
fn decode_one(c: ptr<function, Col>) -> u32 {
  if ((*c).mode == 6u) {
    let row = (*c).cursor; (*c).cursor++; return packed_value((*c).pay, (*c).right0, row);
  }
  if ((*c).mode == 3u) { return 0u; }
  if ((*c).mode == 4u) {
    let value = byte_at(c, (*c).end - 2u) | (byte_at(c, (*c).end - 1u) << 8u);
    (*c).cursor = (*c).end; return value;
  }
  if ((*c).mode == 5u) {
    let row = (*c).state; (*c).state += 1u;
    if ((*c).cursor == (*c).end) { return 0u; }
    let event = byte_at(c, (*c).cursor) | (byte_at(c, (*c).cursor + 1u) << 8u);
    let position = event >> 7u;
    if (position < row) { (*c).bad = true; return 0u; }
    if (position != row) { return 0u; }
    (*c).cursor += 2u; return (event & 127u) + 1u;
  }
  if ((*c).raw) {
    if ((*c).cursor + 2u > (*c).end) { (*c).bad = true; return 0u; }
    let lo = byte_at(c, (*c).cursor); let hi = byte_at(c, (*c).cursor + 1u); (*c).cursor += 2u;
    return lo | (hi << 8u);
  }
  let slot = (*c).state & ((1u << p.scale) - 1u);
  var e: u32; var e0: u32; var freq: u32;
  if (DIRECT_LOOKUP) {
    let entry = tables[(*c).lutbyte + slot];
    e0 = ((entry & 1023u) << 16u) | (entry >> 21u);
    freq = (entry >> 10u) & 2047u;
  } else if (BINARY_LOOKUP) {
    var low = (*c).left0; var high = (*c).right0;
    loop {
      if (high - low <= 1u) { break; }
      let middle = low + (high - low) / 2u;
      if ((tables[(*c).ebase + middle * 2u] >> 16u) <= slot) { low = middle; }
      else { high = middle; }
    }
    e = (*c).ebase + low * 2u;
    e0 = tables[e]; freq = tables[e + 1u];
  } else {
    let li = (*c).lutbyte + (slot >> 7u);
    var idx = (tables[li >> 2u] >> ((li & 3u) * 8u)) & 255u;
    e = (*c).ebase + ((*c).left0 + idx) * 2u;
    loop {
      e0 = tables[e]; freq = tables[e + 1u];
      if (slot < (e0 >> 16u) + freq) { break; }
      idx += 1u; e = (*c).ebase + ((*c).left0 + idx) * 2u;
    }
  }
  let cum = e0 >> 16u;
  let v = e0 & 65535u;
  (*c).state = freq * ((*c).state >> p.scale) + slot - cum;
  loop {
    if ((*c).state >= LOWER) { break; }
    if ((*c).cursor >= (*c).end) { (*c).bad = true; return 0u; }
    (*c).state = ((*c).state << 8u) | byte_at(c, (*c).cursor); (*c).cursor += 1u;
  }
  if ((*c).mode == 2u && v == 32u) {
    if ((*c).cursor + 2u > (*c).end) { (*c).bad = true; return 0u; }
    let escaped = byte_at(c, (*c).cursor) | (byte_at(c, (*c).cursor + 1u) << 8u);
    (*c).cursor += 2u; return escaped;
  }
  return v;
}
`;
const BUILD_WGSL = COMMON_WGSL + /* wgsl */ `
@compute @workgroup_size(64) fn build_checkpoints(@builtin(global_invocation_id) g: vec3u) {
  let k = g.x; if (k >= p.K) { return; }
  let u = units[p.unit0 + g.z];
  var c = open_col(u, k); start_col(&c);
  var w = 0u;
  for (var r = 0u; r < unit_frames(u); r++) {
    if (c.bad) { break; }
    if ((r & 255u) == 0u) { let ci = u.chk_base + (k * p.windows + w) * 2u; chk[ci] = c.state; chk[ci + 1u] = c.cursor; w += 1u; }
    let v = decode_one(&c);
    if ((u.pad & 0x40000000u) != 0u && v > 255u) { c.bad = true; }
  }
  if (c.bad || c.cursor != c.end || (!c.raw && c.mode != 5u && c.state != LOWER)) { atomicAdd(&out[p.N * 2u], 1u); }
}`;
// Exact BF columns for direct ptychography: reuse the validated recurrence and checkpoints.
// Each selected column and scan position has a single writer.
const COLUMNS_WGSL = COMMON_WGSL.replace("out: array<atomic<u32>>", "out: array<vec2<f32>>") + /* wgsl */ `
@compute @workgroup_size(64) fn decode_columns(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; if (i >= p.n_cols) { return; }
  let u = units[p.unit0 + g.z]; let k = cols[i]; let w = g.y;
  let first = w * 256u;
  if (first >= unit_frames(u)) { return; }
  let scan0 = (u.out_base % p.N) + first;
  let stride = cols[p.n_cols]; let row0 = cols[p.n_cols + 1u];
  let col0 = cols[p.n_cols + 2u]; let width = cols[p.n_cols + 3u];
  let height = cols[p.n_cols + 4u];
  if (scan0 >= (row0 + height) * stride || scan0 + 256u <= row0 * stride) { return; }
  let count = min(256u, unit_frames(u) - first);
  let whole = width == stride && width * height == p.N;
  if (!whole && scan0 / stride == (scan0 + count - 1u) / stride &&
      (scan0 % stride >= col0 + width || scan0 % stride + count <= col0)) { return; }
  var c = open_col(u, k);
  seek(&c, u, k, w);
  if (whole) {
    let dst = i * p.N + scan0;
    for (var r = 0u; r < count; r++) { out[dst + r] = vec2<f32>(f32(decode_one(&c)), 0.0); }
    return;
  }
  for (var r = 0u; r < count; r++) {
    let value = decode_one(&c);
    let scan = scan0 + r;
    let row = scan / stride; let col = scan % stride;
    if (row < row0 || row >= row0 + height || col < col0 || col >= col0 + width) { continue; }
    let dst = i * width * height + (row - row0) * width + col - col0;
    out[dst] = vec2<f32>(f32(value), 0.0);
  }
}`;
// Two passes convert one authenticated group. The widest value selects the
// bit width of each column, including literal and escaped uint16 counts.
const PACK_WGSL = COMMON_WGSL + /* wgsl */ `
@compute @workgroup_size(64) fn measure(@builtin(global_invocation_id) id: vec3u) {
  let k = id.x; if (k >= p.K) { return; }
  let u = units[p.unit0 + id.z]; let count = unit_frames(u);
  var c = open_col(u,k); start_col(&c); var bits = 0u;
  for (var r = 0u; r < count; r++) { bits |= decode_one(&c); }
  let width = 32u - countLeadingZeros(bits);
  let base = atomicAdd(&out[p.N], (width * count + 31u) >> 5u);
  atomicStore(&out[u.offsets_base+k], (base << 5u) | width);
  if (k == 0u) { atomicStore(&out[u.offsets_base+p.K], 0xffffffffu); }
}
@compute @workgroup_size(64) fn pack(@builtin(global_invocation_id) id: vec3u) {
  let k = id.x; if (k >= p.K) { return; }
  let u = units[p.unit0 + id.z]; let code = cols[u.offsets_base+k]; let width = code & 31u;
  if (width == 0u) { return; }
  var word = code >> 5u; var shift = 0u; var value = 0u;
  var c = open_col(u,k); start_col(&c);
  for (var r = 0u; r < unit_frames(u); r++) {
    let v = decode_one(&c); value |= v << shift;
    if (shift + width >= 32u) {
      atomicStore(&out[word], value); word++;
      value = select(0u, v >> (32u-shift), shift + width > 32u);
    }
    shift = (shift + width) & 31u;
  }
  if (shift != 0u) { atomicStore(&out[word], value); }
}`;
const PACKED_INTEGRATE = COMMON_WGSL + /* wgsl */ `
@compute @workgroup_size(64) fn integrate(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) lane: u32) {
  let u = units[p.unit0 + wg.z]; let row = wg.y * 64u + lane;
  if (row >= unit_frames(u)) { return; }
  var delta = 0u;
  for (var i=wg.x*64u; i<min(p.n_cols,(wg.x+1u)*64u); i++) {
    let item=cols[i]; let code=offsets[u.offsets_base+(item & 16777215u)];
    let v=packed_value(code >> 5u, code & 31u, row);
    delta += select(0u,v,(item & 0x1000000u)!=0u) - select(0u,v,(item & 0x2000000u)!=0u);
  }
  atomicAdd(&out[u.out_base + row],delta);
}`;
const INTEGRATE_WGSL = COMMON_WGSL + /* wgsl */ `
var<workgroup> s_add: array<atomic<u32>, 256>;
var<workgroup> s_sub: array<atomic<u32>, 256>;
@compute @workgroup_size(64) fn integrate_windows(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  for (var r = t; r < 256u; r += 64u) { atomicStore(&s_add[r], 0u); atomicStore(&s_sub[r], 0u); }
  workgroupBarrier();
  let u = units[p.unit0 + wg.z];
  let idx = wg.x * 64u + t; let w = wg.y;
  let valid = idx < p.n_cols;
  let count = min(256u, unit_frames(u) - min(unit_frames(u), w * 256u));
  var c: Col; var flag = 0u;
  if (valid) {
    let entry = cols[idx]; let k = entry & 16777215u; flag = entry >> 24u;
    c = open_col(u, k); seek(&c, u, k, w);
  }
  for (var r = 0u; r < count; r++) {
    var v = 0u;
    if (valid && !c.bad) { v = decode_one(&c); }
    if (v != 0u) {
      if ((flag & 1u) != 0u) { atomicAdd(&s_add[r], v); }
      if ((flag & 2u) != 0u) { atomicAdd(&s_sub[r], v); }
    }
  }
  if (valid && c.bad) { atomicAdd(&out[p.N * 2u], 1u); }
  workgroupBarrier();
  for (var r = t; r < count; r += 64u) {
    let a = atomicLoad(&s_add[r]); let s = atomicLoad(&s_sub[r]);
    if (a != 0u) { atomicAdd(&out[u.out_base + w * 256u + r], a); }
    if (s != 0u) { atomicAdd(&out[p.N + u.out_base + w * 256u + r], s); }
  }
}`;
// A window header points to sorted local rows, including repeated selections.
const GATHER_WGSL = COMMON_WGSL + /* wgsl */ `
@compute @workgroup_size(64) fn gather_accumulate(@builtin(global_invocation_id) g: vec3u) {
  let k = g.x; if (k >= p.K) { return; }
  let base = g.y * 4u;
  let u = units[cols[base]]; let w = cols[base + 1u];
  var index = cols[base + 2u]; let end = cols[base + 3u];
  var c = open_col(u, k); seek(&c, u, k, w);
  for (var r = 0u; index < end; r++) {
    let v = decode_one(&c);
    if (c.bad) { atomicAdd(&out[p.N * 2u], 1u); break; }
    while (index < end && cols[index] == r) {
      if (v != 0u) {
        let previous = atomicAdd(&out[k], v);
        if (previous > 0xffffffffu - v) { atomicAdd(&out[p.K + k], 1u); }
      }
      index++;
    }
  }
}`;
// Gather each selected detector column once per checkpoint window. Integer
// moments use paired words so saturated pixels cannot overflow weighted sums.
const MOMENTS_WGSL = COMMON_WGSL + /* wgsl */ `
// Each workgroup combines 64 detector columns before writing to global memory.
var<workgroup> sums: array<atomic<u32>, 1536>;
fn add_local(index: u32, value: u32) {
  let previous = atomicAdd(&sums[index], value);
  if (previous > 0xffffffffu - value) { atomicAdd(&sums[index + 1u], 1u); }
}
@compute @workgroup_size(64) fn gather_moments(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) lane: u32) {
  for (var i = lane; i < 1536u; i += 64u) { atomicStore(&sums[i], 0u); }
  workgroupBarrier();
  let col = wg.x * 64u + lane;
  let u = units[p.unit0 + wg.z]; let window = wg.y;
  let count = min(256u, unit_frames(u) - min(window * 256u, unit_frames(u)));
  if (col < p.n_cols) {
    let k = cols[col]; var c = open_col(u, k);
    seek(&c, u, k, window);
    let row = k / p.n_units; let column = k % p.n_units;
    for (var r = 0u; r < count; r++) {
      let value = decode_one(&c);
      if (value != 0u) {
        add_local(r * 2u, value);
        add_local(512u + r * 2u, value * row);
        add_local(1024u + r * 2u, value * column);
      }
    }
  }
  workgroupBarrier();
  for (var r = lane; r < count; r += 64u) {
    let scan = u.out_base % p.N + window * 256u + r;
    for (var m = 0u; m < 3u; m++) {
      let at = m * 512u + 2u * r;
      let low = atomicLoad(&sums[at]); var high = atomicLoad(&sums[at + 1u]);
      let dest = 2u * (m * p.N + scan);
      if (low != 0u) {
        let previous = atomicAdd(&out[dest], low);
        high += select(0u, 1u, previous > 0xffffffffu - low);
      }
      if (high != 0u) { atomicAdd(&out[dest + 1u], high); }
    }
  }
}`;
// Center of mass from the exact paired-word moments: each moment divided by the weight and
// rounded once (ratioU64), the same quotient the dense backend and CUDA/MPS produce. One call
// site in a two-component loop keeps the division inlined once (Windows FXC).
const MOMENT_RATIO_WGSL = /* wgsl */ `
${EXACT_INTEGER_COM_WGSL}
@group(0) @binding(0) var<storage, read> moments: array<u32>;
@group(0) @binding(1) var<storage, read_write> com: array<f32>;
@group(0) @binding(2) var<uniform> p: vec4u;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3u) {
  let scan = g.x; if (scan >= p.x) { return; }
  let weight = U64Words(moments[scan * 2u], moments[scan * 2u + 1u]);
  for (var component = 1u; component <= 2u; component++) {
    let index = p.x * component + scan;
    com[p.x * (component - 1u) + scan] = ratioU64(U64Words(moments[index * 2u], moments[index * 2u + 1u]), weight);
  }
}`;
// images += add - sub exactly (mod 2^32; every true sum fits), then clear the accumulators.
const APPLY_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@group(0) @binding(1) var<storage, read_write> images: array<u32>;
@group(0) @binding(2) var<storage, read_write> imagesF32: array<f32>;
@group(0) @binding(3) var<uniform> n: vec4u;
@compute @workgroup_size(256) fn apply(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; if (i >= n.x) { return; }
  let v = images[i] + out[i] - out[n.x + i];
  images[i] = v; imagesF32[i] = f32(v);
  out[i] = 0u; out[n.x + i] = 0u;
}`;

const NORMALIZE_DISPLAY_WGSL = /* wgsl */ `
struct Params { n: u32, area: f32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<storage, read_write> values: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;
@compute @workgroup_size(256)
fn normalize_display(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x < params.n) { values[gid.x] = values[gid.x] / params.area; }
}`;

export interface RansLoadProfile extends RansPayloadProfile {
  metadataReadMs: number;
  payloadBuffers: number;
  /** These groups were validated during admission, not by the final timed pass. */
  checkpointPrebuiltGroups?: number;
}

export interface RansManifest { tilts: RansTiltMeta[]; bad_pixels?: number[]; scan_shape?: number[]; detector_shape?: number[]; source_metadata?: Record<string, unknown>; native_dtype?: "uint8" | "uint16" }
export interface RansTiltMeta { tilt: number; K: number; frames: number; blocks: number; scale: number; model_frames: number; payload_url?: string; blocks_meta: { index: number; first?: number; bytes: number; model: number; byte_start?: number; byte_end?: number; frames?: number }[]; models: { index: number; symbols: number; colmeta_url?: string; entries_url?: string; lut_url?: string }[]; binary_lookup?: boolean; direct_lookup?: boolean }

interface Span { unit0: number; n: number; params: GPUBuffer; group: GPUBindGroup }
interface Group { packed?: boolean; replacePayload: (data: GPUBuffer, index: GPUBuffer) => void; payload: GPUBuffer; offsets: GPUBuffer; chk: GPUBuffer; params: GPUBuffer; group: GPUBindGroup; unit0: number; units: UnitRec[]; spans: Map<number, Span>; bind: (paramsBuf: GPUBuffer, outBuf?: GPUBuffer, colsB?: GPUBuffer, unitsB?: GPUBuffer) => GPUBindGroup }
interface UnitRec { payload_word: number; offsets_base: number; colmeta_word: number; entries_word: number; lut_word: number; chk_base: number; out_base: number; tilt: number; block: number; frameFlags: number }

const pad4 = (n: number) => Math.ceil(n / 4) * 4;
function concatU32(parts: Uint32Array[]): Uint32Array { const total = parts.reduce((a, b) => a + b.length, 0); const o = new Uint32Array(total); let off = 0; for (const p of parts) { o.set(p, off); off += p.length; } return o; }

/** Submit the same decoder on verified groups while later bytes are in flight.
 * Submission parameters are temporary; the source owns bookmarks and reusable tables.
 * A single fault readback gates admission of every submitted group.
 */
function checkpointAdmission(device: GPUDevice) {
  let pipeline: GPUComputePipeline | undefined, faults: GPUBuffer | undefined;
  return {
    submit({ payload, K, nativeDtype, entries, directLookup, blocks }: QemCheckpointGroup): RansPreparedGroup {
      pipeline ??= device.createComputePipeline({ layout: "auto", compute: {
        module: device.createShaderModule({ code: BUILD_WGSL }), entryPoint: "build_checkpoints",
        constants: { BINARY_LOOKUP: 1 },
      } });
      faults ??= device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      const temporary: GPUBuffer[] = [];
      const upload = (parts: Uint32Array[], usage = GPUBufferUsage.STORAGE) => {
        const size = parts.reduce((total, words) => total + words.byteLength, 0);
        const buffer = device.createBuffer({ size: Math.max(4, size), usage, mappedAtCreation: true });
        temporary.push(buffer);
        const target = new Uint32Array(buffer.getMappedRange());
        let offset = 0;
        for (const words of parts) { target.set(words, offset); offset += words.length; }
        buffer.unmap(); return buffer;
      };
      const chk = device.createBuffer({ size: blocks.length * K * 2 * 2 * 4, usage: GPUBufferUsage.STORAGE });
      try {
        const columns = blocks.length * K * 3;
        const table = upload([...blocks.map(block => block.columns), entries, directLookup]);
        const offsets = upload(blocks.map(block => block.offsets));
        const units = upload([Uint32Array.from(blocks.flatMap((block, i) => [
          Math.floor(block.offset / 4), i * (K + 1), i * K * 3, columns, 0, i * K * 4, 0,
          (block.frames | ((block.offset % 4) << 28) | 0x80000000 | (nativeDtype === "uint8" ? 0x40000000 : 0)) >>> 0,
        ]))]);
        const params = upload([new Uint32Array([512, K, 10, 2, 0, 0, blocks.length, 0])], GPUBufferUsage.UNIFORM);
        const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
          { binding: 0, resource: { buffer: payload } }, { binding: 1, resource: { buffer: offsets } },
          { binding: 2, resource: { buffer: table } }, { binding: 3, resource: { buffer: chk } },
          { binding: 5, resource: { buffer: faults } }, { binding: 6, resource: { buffer: units } },
          { binding: 7, resource: { buffer: params } },
        ] });
        const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
        pass.setPipeline(pipeline); pass.setBindGroup(0, group);
        pass.dispatchWorkgroups(Math.ceil(K / 64), 1, blocks.length); pass.end();
        device.queue.submit([encoder.finish()]);
        // Keep the group's tables for resident operations too: no second host
        // concatenation or GPU upload when the final set takes ownership.
        temporary.splice(temporary.indexOf(table), 1);
        temporary.splice(temporary.indexOf(offsets), 1);
        return { checkpoints: chk, tables: table, offsets, entriesWord: columns, lookupWord: columns + entries.length };
      } catch (error) { chk.destroy(); throw error; }
      finally { for (const buffer of temporary) buffer.destroy(); }
    },
    async finish() {
      if (!faults) return;
      const readback = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      try {
        const encoder = device.createCommandEncoder(); encoder.copyBufferToBuffer(faults, 0, readback, 0, 16);
        device.queue.submit([encoder.finish()]); await readback.mapAsync(GPUMapMode.READ);
        const count = new Uint32Array(readback.getMappedRange())[0];
        if (count) throw new Error(`rANS streams did not terminate exactly (${count} columns); the export is corrupt`);
      } finally { readback.destroy(); }
    },
    dispose() { faults?.destroy(); },
  };
}

/** All acquisitions of one exported rANS series, resident in browser GPU memory. */
export class RansResidentSet {
  readonly computes: RansDetectorCompute[] = [];
  readonly scanCount: number;
  readonly detSize: number;
  readonly frames: number;
  readonly windows: number;
  readonly scale: number;
  readonly K: number;
  readonly T: number;
  readonly badPx: Uint32Array;
  readonly shape: readonly [number, number, number, number] | null;
  readonly sourceMetadata: Record<string, unknown>;
  readonly nativeDtype: "uint8" | "uint16";
  readonly payloadBytes!: number;
  readonly loadMs!: number;
  readonly checkpointMs!: number;
  readonly readyMs!: number;
  readonly loadProfile!: RansLoadProfile;
  readonly acquisitionMode!: RansByteSource["mode"];
  private groups: Group[] = [];
  private out!: GPUBuffer;
  private images!: GPUBuffer;
  private imagesF32!: GPUBuffer;
  private colsBuf!: GPUBuffer;
  private colsList: Uint32Array;
  private unitsBuf!: GPUBuffer;
  private intPipe!: GPUComputePipeline;
  private gatherPipe!: GPUComputePipeline;
  private columnsPipe?: GPUComputePipeline;
  private momentsPipe?: GPUComputePipeline;
  private momentRatioPipe?: GPUComputePipeline;
  private applyPipe!: GPUComputePipeline;
  private normalizeDisplayPipe?: GPUComputePipeline;
  private normalizeDisplayParams: GPUBuffer[] = [];
  private displayCopies = new WeakSet<GPUBuffer>();
  private applyGroup!: GPUBindGroup;
  private imageCountParams!: GPUBuffer;
  private disposed = false;
  private readonly binaryLookup: boolean;
  private readonly directLookup: boolean;

  private constructor(readonly device: GPUDevice, manifest: RansManifest, built: {
    groups: Group[]; out: GPUBuffer; images: GPUBuffer; imagesF32: GPUBuffer; colsBuf: GPUBuffer; unitsBuf: GPUBuffer;
    intPipe: GPUComputePipeline; gatherPipe: GPUComputePipeline; applyPipe: GPUComputePipeline; applyGroup: GPUBindGroup; imageCountParams: GPUBuffer;
    payloadBytes: number; loadMs: number; checkpointMs: number; readyMs: number; loadProfile: RansLoadProfile; acquisitionMode: RansByteSource["mode"];
  }) {
    const first = manifest.tilts[0];
    this.binaryLookup = Boolean(first.binary_lookup);
    this.directLookup = Boolean(first.direct_lookup);
    this.K = first.K; this.frames = first.frames; this.scale = first.scale; this.windows = Math.ceil(first.frames / WINDOW);
    this.scanCount = manifest.scan_shape ? manifest.scan_shape[0] * manifest.scan_shape[1] : first.frames * first.blocks; this.detSize = first.K; this.T = manifest.tilts.length;
    this.badPx = new Uint32Array(manifest.bad_pixels ?? []);
    this.shape = manifest.scan_shape && manifest.detector_shape
      ? [manifest.scan_shape[0], manifest.scan_shape[1], manifest.detector_shape[0], manifest.detector_shape[1]] : null;
    this.sourceMetadata = manifest.source_metadata ?? {}; this.nativeDtype = manifest.native_dtype ?? "uint16";
    Object.assign(this, built);
    this.colsList = new Uint32Array(this.K);
    for (let t = 0; t < this.T; t++) this.computes.push(new RansDetectorCompute(this, t));
  }

  /** Load one .qem count acquisition through the same GPU decoder; `signal` cancels it, downloads included. */
  static async loadQemFile(device: GPUDevice, file: QemByteFile, onStatus: (text: string) => void = () => {}, badPixels: number[] = [], signal?: AbortSignal): Promise<RansResidentSet> {
    return this.loadQemFiles(device, [file], onStatus, badPixels, signal);
  }

  /** Load an ordered, compatible series of .qem count acquisitions into one batched resident set. */
  static async loadQemFiles(device: GPUDevice, files: ArrayLike<QemByteFile>, onStatus: (text: string) => void = () => {}, badPixels: number[] = [], signal?: AbortSignal): Promise<RansResidentSet> {
    const started = performance.now();
    const admission = checkpointAdmission(device);
    let source: RansByteSource | undefined;
    // The source owns its staged payload until the resident set takes it over.
    try {
      source = await qemFilesSource(files, onStatus, badPixels, device, signal, admission.submit);
      await admission.finish();
      signal?.throwIfAborted();
      const set = await this.loadSource(device, source, onStatus, started);
      // Cancelled while the decoder was being built: nothing of the load survives.
      if (signal?.aborted) { set.dispose(); signal.throwIfAborted(); }
      return set;
    }
    catch (error) { source?.dispose?.(); throw error; }
    finally { admission.dispose(); }
  }

  private static async loadSource(device: GPUDevice, source: RansByteSource, onStatus: (text: string) => void, started = performance.now()): Promise<RansResidentSet> {
    const loadProfile: RansLoadProfile = { metadataReadMs: 0, payloadReadMs: 0, payloadReadWaitMs: 0, payloadStageMs: 0, payloadChunks: 0, payloadBuffers: 0 };
    const readFile = async (name: string) => {
      const begin = performance.now();
      try { return await source.read(name); }
      finally { loadProfile.metadataReadMs += performance.now() - begin; }
    };
    const manifest = JSON.parse(new TextDecoder().decode(await readFile("manifest.json"))) as RansManifest;
    const tilts = manifest.tilts; const T = tilts.length;
    const binaryLookup = Boolean(tilts[0].binary_lookup);
    const directLookup = Boolean(tilts[0].direct_lookup);
    if (tilts.some(tilt => Boolean(tilt.binary_lookup) !== binaryLookup || Boolean(tilt.direct_lookup) !== directLookup)) throw new Error("Use one rANS table profile per resident series");
    const { K, frames, blocks, scale } = tilts[0]; const windows = Math.ceil(frames / WINDOW); const N = manifest.scan_shape ? manifest.scan_shape[0] * manifest.scan_shape[1] : frames * blocks;
    const preparedOnly = tilts.every(tilt => tilt.blocks_meta.every(block => {
      const segment = tilt.payload_url && block.byte_start !== undefined && block.byte_end !== undefined
        ? source.residentPayload?.(tilt.payload_url, block.byte_start, block.byte_end) : undefined;
      return segment && source.residentCheckpoints?.(segment.buffer);
    }));
    const upload = (data: ArrayBufferView, usage: GPUBufferUsageFlags) => {
      const size = pad4(data.byteLength);
      const buffer = device.createBuffer({ size, usage: usage | GPUBufferUsage.COPY_DST, mappedAtCreation: true });
      new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength)); buffer.unmap(); return buffer;
    };
    // Decode tables for every (acquisition, model), concatenated into ONE u32 buffer:
    // [column metadata words | packed symbol entries | slot lookup bytes]. One binding
    // keeps the layout inside the default 8 storage buffers per stage.
    const modelBases: { colmeta_word: number; entries_word: number; lut_word: number }[][] = [];
    const colmetaParts: Uint32Array[] = [], entriesParts: Uint32Array[] = [], lutParts: Uint8Array[] = [];
    let colmetaLen = 0, entriesLen = 0, lutBytes = 0;
    const sharedEntries = new Map<string, number>();
    const sharedLookup = new Map<string, number>();
    for (let tiltIndex = 0; tiltIndex < T; tiltIndex++) {
      const prefix = `t${tilts[tiltIndex].tilt}-`; modelBases.push([]);
      if (preparedOnly) continue;
      for (const model of tilts[tiltIndex].models) {
        onStatus(`Loading rANS tables ${tiltIndex + 1}/${T} (model ${model.index + 1}/${tilts[tiltIndex].models.length})`);
        let columnMeta: Uint32Array;
        if (model.colmeta_url) columnMeta = new Uint32Array(await readFile(model.colmeta_url));
        else {
          const contexts = new Uint32Array(await readFile(`${prefix}ctx-${model.index}.u32`)); const literals = new Uint8Array(await readFile(`${prefix}literal-${model.index}.u8`));
          columnMeta = new Uint32Array(K * 3);
          for (let k = 0; k < K; k++) { columnMeta[k * 3] = contexts[k]; columnMeta[k * 3 + 1] = contexts[k + 1]; columnMeta[k * 3 + 2] = literals[k]; }
        }
        const entriesName = model.entries_url ?? `${prefix}entries-${model.index}.u32`;
        const lookupName = model.lut_url ?? `${prefix}lut-${model.index}.u8`;
        if (!sharedEntries.has(entriesName)) {
          const entries = new Uint32Array(await readFile(entriesName));
          sharedEntries.set(entriesName, entriesLen); entriesParts.push(entries); entriesLen += entries.length;
        }
        if (!sharedLookup.has(lookupName)) {
          const lookup = new Uint8Array(await readFile(lookupName));
          sharedLookup.set(lookupName, lutBytes / 4); lutParts.push(lookup); lutBytes += pad4(lookup.length);
        }
        modelBases[tiltIndex].push({ colmeta_word: colmetaLen, entries_word: sharedEntries.get(entriesName)!, lut_word: sharedLookup.get(lookupName)! });
        colmetaParts.push(columnMeta); colmetaLen += columnMeta.length;
      }
    }
    // Each part is copied straight into the final mapping, so the column
    // metadata, the largest table, is never concatenated on the host.
    const tablesBuf = device.createBuffer({ size: Math.max(4, (colmetaLen + entriesLen) * 4 + lutBytes), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, mappedAtCreation: true });
    const tableBytes = new Uint8Array(tablesBuf.getMappedRange());
    let tableAt = 0;
    for (const part of [...colmetaParts, ...entriesParts]) {
      tableBytes.set(new Uint8Array(part.buffer, part.byteOffset, part.byteLength), tableAt);
      tableAt += part.byteLength;
    }
    for (const part of lutParts) { tableBytes.set(part, tableAt); tableAt += pad4(part.byteLength); }
    tablesBuf.unmap();
    for (const bases of modelBases) for (const base of bases) { base.entries_word += colmetaLen; base.lut_word += colmetaLen + entriesLen; }
    // Payload blocks packed into as few buffers as the group limit allows; each buffer is one dispatch.
    const limit = payloadGroupLimit(device);
    type Pending = { payload: GPUBuffer; payBytes: number; offParts: Uint32Array[]; offLen: number; units: UnitRec[]; resident: boolean };
    const groups: Group[] = []; let current: Pending | null = null;
    const pending: Pending[] = []; let payloadBytes = 0;
    // Plan from authenticated export lengths before reading. Upload each block
    // directly into its final packed GPU buffer instead of retaining the series
    // and making a second multi-gigabyte concatenation in JavaScript memory.
    const plans: { tiltIndex: number; block: RansTiltMeta["blocks_meta"][number]; group: Pending; offset: number }[] = [];
    // A payload the source already staged on the GPU is bound where it lies.
    const residentGroups = new Map<GPUBuffer, Pending>();
    for (let tiltIndex = 0; tiltIndex < T; tiltIndex++) for (const block of tilts[tiltIndex].blocks_meta) {
      const size = Math.max(4, pad4(block.bytes));
      if (!Number.isSafeInteger(block.bytes) || block.bytes < 0 || size > limit) throw new Error(`rANS block ${block.index} exceeds device buffer limits or has invalid length`);
      const tilt = tilts[tiltIndex];
      const segment = tilt.payload_url && block.byte_start !== undefined && block.byte_end !== undefined
        ? source.residentPayload?.(tilt.payload_url, block.byte_start, block.byte_end) : undefined;
      if (segment) {
        let group = residentGroups.get(segment.buffer);
        if (!group) {
          group = { payload: segment.buffer, payBytes: segment.buffer.size, offParts: [], offLen: 0, units: [], resident: true };
          residentGroups.set(segment.buffer, group);
          pending.push(group);
        }
        plans.push({ tiltIndex, block, group, offset: segment.offset });
        continue;
      }
      if (!current || current.payBytes + size > limit) {
        current = { payload: null as unknown as GPUBuffer, payBytes: 0, offParts: [], offLen: 0, units: [], resident: false };
        pending.push(current);
      }
      plans.push({ tiltIndex, block, group: current, offset: current.payBytes }); current.payBytes += size;
    }
    loadProfile.payloadBuffers = pending.length;
    let mapped: Uint8Array | null = null;
    let active: Pending | null = null;
    try {
      for (const { tiltIndex, block, group, offset } of plans) {
        if (!group.resident && active !== group) {
          const stageBegin = performance.now();
          if (active) active.payload.unmap();
          mapped = null;
          group.payload = device.createBuffer({ size: group.payBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, mappedAtCreation: true });
          mapped = new Uint8Array(group.payload.getMappedRange());
          active = group;
          loadProfile.payloadStageMs += performance.now() - stageBegin;
        }
        const tilt = tilts[tiltIndex];
        const prefix = `t${tilt.tilt}-`;
        onStatus(`Loading rANS payload ${tiltIndex + 1}/${T} block ${block.index + 1}/${tilt.blocks_meta.length}`);
        const linked = tilt.payload_url && block.byte_start !== undefined && block.byte_end !== undefined;
        const name = linked ? tilt.payload_url! : `${prefix}payload-${String(block.index).padStart(2, "0")}.bin`;
        const start = linked ? block.byte_start! : 0;
        if (linked && block.byte_end! - start !== block.bytes) throw new Error(`${name}: manifest payload range length mismatch`);
        // Four bounded reads feed one mapped payload group. Read service times
        // overlap; payloadReadWaitMs measures only waits exposed to this loop.
        if (!group.resident) await copyRansPayload(source, name, start, block.bytes, mapped!, offset, loadProfile);
        const prepared = source.residentCheckpoints?.(group.payload);
        const blockOffsets = prepared ? undefined : new Uint32Array(await readFile(`${prefix}offsets-${String(block.index).padStart(2, "0")}.u32`));
        payloadBytes += block.bytes;
        const modelBase = prepared
          ? { colmeta_word: group.units.length * K * 3, entries_word: prepared.entriesWord, lut_word: prepared.lookupWord ?? 0 }
          : modelBases[tiltIndex][block.model];
        const unit: UnitRec = { payload_word: Math.floor(offset / 4), offsets_base: group.offLen, ...modelBase, chk_base: group.units.length * K * windows * 2, out_base: tiltIndex * N + (block.first ?? block.index * frames), tilt: tiltIndex, block: block.index, frameFlags: ((block.frames ?? 0) | ((offset % 4) << 28) | (tilt.binary_lookup ? 0x80000000 : 0) | (manifest.native_dtype === "uint8" ? 0x40000000 : 0)) >>> 0 };
        if (blockOffsets) group.offParts.push(blockOffsets);
        group.offLen += blockOffsets?.length ?? K + 1; group.units.push(unit);
      }
    } catch (error) {
      // The read helper drains its bounded requests before this storage is freed.
      // Staged groups stay with the source, which releases them itself.
      mapped = null;
      for (const group of pending) if (!group.resident) group.payload?.destroy();
      tablesBuf.destroy();
      throw error;
    }
    const unmapBegin = performance.now();
    if (active) active.payload.unmap();
    mapped = null;
    loadProfile.payloadStageMs += performance.now() - unmapBegin;
    // Rows follow the groups, as every group's unit0 does: per-block uploads and
    // staged segments interleave in plan order, so plan order is not group order.
    const units = pending.flatMap(group => group.units);
    const unitTable = new Uint32Array(units.length * 8); units.forEach((unit, i) => unitTable.set([unit.payload_word, unit.offsets_base, unit.colmeta_word, unit.entries_word, unit.lut_word, unit.chk_base, unit.out_base, unit.frameFlags], i * 8));
    const unitsBuf = upload(unitTable, GPUBufferUsage.STORAGE);
    const out = device.createBuffer({ size: (T * N * 2 + 4) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const images = device.createBuffer({ size: T * N * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const imagesF32 = device.createBuffer({ size: T * N * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const colsBuf = device.createBuffer({ size: K * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const layoutEntries: GPUBindGroupLayoutEntry[] = [
      ...[0, 1, 2, 4, 6].map((binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" as GPUBufferBindingType } })),
      ...[3, 5].map((binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" as GPUBufferBindingType } })),
      { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" as GPUBufferBindingType } },
    ];
    const layout = device.createBindGroupLayout({ entries: layoutEntries });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const pipeline = (code: string, entryPoint: string) => device.createComputePipeline({ layout: pipelineLayout, compute: { module: device.createShaderModule({ code }), entryPoint, constants: { BINARY_LOOKUP: binaryLookup ? 1 : 0, DIRECT_LOOKUP: directLookup ? 1 : 0 } } });
    const buildPipe = pipeline(BUILD_WGSL, "build_checkpoints"), intPipe = pipeline(INTEGRATE_WGSL, "integrate_windows"), gatherPipe = pipeline(GATHER_WGSL, "gather_accumulate");
    const applyPipe = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code: APPLY_WGSL }), entryPoint: "apply" } });
    const imageCountParams = upload(new Uint32Array([T * N, 0, 0, 0]), GPUBufferUsage.UNIFORM);
    const applyGroup = device.createBindGroup({ layout: applyPipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: out } }, { binding: 1, resource: { buffer: images } }, { binding: 2, resource: { buffer: imagesF32 } }, { binding: 3, resource: { buffer: imageCountParams } }] });
    let unitCursor = 0;
    for (const planned of pending) {
      let payload = planned.payload;
      const prepared = source.residentCheckpoints?.(payload);
      let offsets = prepared?.offsets ?? upload(concatU32(planned.offParts), GPUBufferUsage.STORAGE);
      let chk = prepared?.checkpoints ?? device.createBuffer({ size: planned.units.length * K * windows * 2 * 4, usage: GPUBufferUsage.STORAGE });
      if (prepared) loadProfile.checkpointPrebuiltGroups = (loadProfile.checkpointPrebuiltGroups ?? 0) + 1;
      const params = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      let tables = prepared?.tables ?? tablesBuf;
      const bind = (paramsBuf: GPUBuffer, outBuf: GPUBuffer = out, colsB: GPUBuffer = colsBuf, unitsB: GPUBuffer = unitsBuf) => device.createBindGroup({ layout, entries: [
        { binding: 0, resource: { buffer: payload } }, { binding: 1, resource: { buffer: offsets } }, { binding: 2, resource: { buffer: tables } },
        { binding: 3, resource: { buffer: chk } }, { binding: 4, resource: { buffer: colsB } }, { binding: 5, resource: { buffer: outBuf } },
        { binding: 6, resource: { buffer: unitsB } }, { binding: 7, resource: { buffer: paramsBuf } }] });
      const group = bind(params);
      // Per-acquisition spans inside this buffer, each with its own uniform and
      // bind group: several spans are recorded into one command encoder, and a
      // shared uniform would be overwritten before the GPU executes any of them.
      const spans = new Map<number, Span>();
      planned.units.forEach((unit, i) => {
        const existing = spans.get(unit.tilt);
        if (existing) { existing.n += 1; return; }
        const spanParams = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        spans.set(unit.tilt, { unit0: unitCursor + i, n: 1, params: spanParams, group: bind(spanParams) });
      });
      const record: Group = { payload, offsets, chk, params, group, unit0: unitCursor, units: planned.units, spans, bind,
        replacePayload(data, index) {
          const priorData = payload, priorIndex = offsets;
          // Packed readers need neither ANS probabilities nor decoder bookmarks.
          // Keep a minimal writable binding to satisfy the shared pipeline layout.
          const oldBookmarks = chk;
          record.chk = chk = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE });
          if (tables !== tablesBuf) tables.destroy();
          tables = tablesBuf;
          record.payload = payload = data;
          record.offsets = offsets = index;
          record.group = bind(params);
          for (const span of spans.values()) span.group = bind(span.params);
          record.packed = true;
          priorData.destroy(); priorIndex.destroy(); oldBookmarks.destroy();
        },
      };
      groups.push(record);
      unitCursor += planned.units.length;
    }
    const loadMs = performance.now() - started;
    // One full decode of every column: checkpoints every 256 frames, exact termination verified.
    onStatus("Building rANS decode checkpoints");
    device.queue.writeBuffer(out, 0, new Uint32Array(T * N * 2 + 4));
    const checkpointBegin = performance.now();
    const encoder = device.createCommandEncoder(); const pass = encoder.beginComputePass(); pass.setPipeline(buildPipe);
    for (const group of groups) {
      if (source.residentCheckpoints?.(group.payload)) continue;
      device.queue.writeBuffer(group.params, 0, new Uint32Array([frames, K, scale, windows, 0, group.unit0, group.units.length, T * N])); pass.setBindGroup(0, group.group); pass.dispatchWorkgroups(Math.ceil(K / 64), 1, group.units.length);
    }
    pass.end(); device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone();
    const checkpointMs = performance.now() - checkpointBegin;
    const faultReadback = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }); const faultEncoder = device.createCommandEncoder(); faultEncoder.copyBufferToBuffer(out, T * N * 2 * 4, faultReadback, 0, 16); device.queue.submit([faultEncoder.finish()]);
    await faultReadback.mapAsync(GPUMapMode.READ); const faults = new Uint32Array(faultReadback.getMappedRange().slice(0))[0]; faultReadback.unmap(); faultReadback.destroy();
    const set = new RansResidentSet(device, manifest, { groups, out, images, imagesF32, colsBuf, unitsBuf, intPipe, gatherPipe, applyPipe, applyGroup, imageCountParams, payloadBytes, loadMs, checkpointMs, readyMs: performance.now() - started, loadProfile, acquisitionMode: source.mode });
    set._setTables([tablesBuf, ...pending.flatMap(group => {
      const prepared = source.residentCheckpoints?.(group.payload);
      return prepared ? [prepared.tables] : [];
    })]);
    // A rejected export keeps nothing on the device.
    if (faults) { set.dispose(); throw new Error(`rANS streams did not terminate exactly (${faults} columns); the export is corrupt`); }
    onStatus("");
    return set;
  }

  private packedPipe?: GPUComputePipeline;
  private packing?: Promise<void>;
  private foreground = new Set<Promise<void>>();

  /** Keep optional drag preparation off the queue while a requested operation runs. */
  async withForegroundCompute<T>(compute: () => Promise<T>): Promise<T> {
    let release!: () => void;
    const finished = new Promise<void>(resolve => { release = resolve; });
    this.foreground.add(finished);
    try { return await compute(); }
    finally { this.foreground.delete(finished); release(); }
  }

  /** Prepare random-access integer counts one group at a time; ANS remains usable until each swap. */
  prepareDetectorDrags(): Promise<void> {
    return this.packing ??= this.packDetectorGroups();
  }

  private async packDetectorGroups(): Promise<void> {
    const device = this.device;
    const layout = device.createPipelineLayout({bindGroupLayouts: [this.intPipe.getBindGroupLayout(0)]});
    const pipeline = (code: string, entryPoint: string) => device.createComputePipelineAsync({layout, compute: {
      module: device.createShaderModule({code}), entryPoint,
      constants: {BINARY_LOOKUP: Number(this.binaryLookup), DIRECT_LOOKUP: Number(this.directLookup)},
    }});
    const [measure, pack, integrate] = await Promise.all([
      pipeline(PACK_WGSL, "measure"), pipeline(PACK_WGSL, "pack"), pipeline(PACKED_INTEGRATE, "integrate"),
    ]);
    this.packedPipe = integrate;
    for (const group of this.groups) {
      if (this.disposed) return;
      const n = group.units.length * (this.K + 1);
      // The packed index reserves five bits per column for the width.
      if (group.units.length * this.K * Math.ceil(this.frames / 2) >= 2**32) continue;
      const index = device.createBuffer({size:(n+1)*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
      const params = device.createBuffer({size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
      const size = device.createBuffer({size:4,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
      let data: GPUBuffer | undefined;
      try {
        const run = async (pipe: GPUComputePipeline, binding: GPUBindGroup) => {
          for (let i = 0; i < group.units.length && !this.disposed; i++) {
            while (this.foreground.size && !this.disposed) await Promise.all(this.foreground);
            if (this.disposed) return;
            device.queue.writeBuffer(params, 0, new Uint32Array([this.frames, this.K, this.scale, this.windows, 0, group.unit0 + i, 1, n]));
            const enc = device.createCommandEncoder();
            const pass = enc.beginComputePass();
            pass.setPipeline(pipe);
            pass.setBindGroup(0, binding);
            pass.dispatchWorkgroups(Math.ceil(this.K / 64));
            pass.end();
            device.queue.submit([enc.finish()]);
            await device.queue.onSubmittedWorkDone();
          }
        };
        await run(measure,group.bind(params,index));
        if (this.disposed) return;
        const enc = device.createCommandEncoder();
        enc.copyBufferToBuffer(index, n * 4, size, 0, 4);
        device.queue.submit([enc.finish()]);
        await size.mapAsync(GPUMapMode.READ);
        const words = new Uint32Array(size.getMappedRange())[0];
        size.unmap();
        if (this.disposed) return;
        if (words >= 2**27 || words*4 > Math.min(device.limits.maxStorageBufferBindingSize,device.limits.maxBufferSize)) continue;
        // Highly sparse or outlier-heavy columns can be smaller as ANS. Limit expansion and
        // preserve the verified decoder when the optional interactive index cannot fit.
        if (words * 4 > group.payload.size * 1.75 + 1024 * 1024) continue;
        let allocationError: GPUError | null;
        device.pushErrorScope("out-of-memory");
        try {
          data = device.createBuffer({size: Math.max(4, words * 4), usage: GPUBufferUsage.STORAGE});
        } finally {
          allocationError = await device.popErrorScope();
        }
        if (allocationError || this.disposed) return;
        await run(pack,group.bind(params,data,index));
        if (this.disposed) return;
        group.replacePayload(data,index);
      } finally {
        params.destroy(); size.destroy();
        if (!group.packed) { index.destroy(); data?.destroy(); }
      }
    }
  }

  /** Decode only the listed columns (bit0 add, bit1 subtract) for the given acquisitions; images stay on the GPU.
   * With `encoder`, the work is recorded there and the caller submits it.
   */
  integrate(tilts: Set<number> | null, added: Uint8Array | Uint32Array | null, removed: Uint8Array | Uint32Array | null, encoder?: GPUCommandEncoder): number {
    if (this.disposed) throw new Error("rANS resident set disposed");
    let n = 0;
    for (let k = 0; k < this.K; k++) { const flags = (added && added[k] ? 1 : 0) | (removed && removed[k] ? 2 : 0); if (flags) this.colsList[n++] = k | (flags << 24); }
    if (!n) return 0;
    this.device.queue.writeBuffer(this.colsBuf, 0, this.colsList.buffer as ArrayBuffer, 0, n * 4);
    const chunked = !encoder && this.device.adapterInfo?.vendor?.toLowerCase() === "qualcomm";
    const unitsPerSubmit = chunked ? Math.max(1, Math.floor(16_777_216 / (n * this.frames))) : Infinity;
    let enc = encoder ?? this.device.createCommandEncoder();
    let pass = enc.beginComputePass();
    pass.setPipeline(this.intPipe);
    for (const group of this.groups) {
      const spans = tilts
        ? [...group.spans].filter(([tilt]) => tilts.has(tilt)).map(([, span]) => span)
        : [{ unit0: group.unit0, n: group.units.length, params: group.params, group: group.group }];
      for (const span of spans) {
        for (let first = 0; first < span.n; first += unitsPerSubmit) {
          const count = Math.min(span.n - first, unitsPerSubmit);
          this.device.queue.writeBuffer(span.params, 0, new Uint32Array([this.frames, this.K, this.scale, this.windows, n, span.unit0 + first, count, this.T * this.scanCount]));
          pass.setPipeline(group.packed ? this.packedPipe! : this.intPipe);
          pass.setBindGroup(0, span.group);
          pass.dispatchWorkgroups(Math.ceil(n / 64), group.packed ? Math.ceil(this.frames / 64) : this.windows, count);
          if (chunked && n * this.frames * span.n > 16_777_216) {
            pass.end();
            this.device.queue.submit([enc.finish()]);
            enc = this.device.createCommandEncoder();
            pass = enc.beginComputePass();
            pass.setPipeline(this.intPipe);
          }
        }
      }
    }
    pass.setPipeline(this.applyPipe); pass.setBindGroup(0, this.applyGroup); pass.dispatchWorkgroups(Math.ceil(this.T * this.scanCount / 256));
    pass.end(); if (!encoder) this.device.queue.submit([enc.finish()]);
    return n;
  }

  /** Decode selected detector columns directly into caller-owned complex64 GPU storage.
   * Layout is [column, scan, real/imaginary]. Every native integer is exact.
   */
  async columnsComplex(tilt: number, indices: Uint32Array, output: GPUBuffer, region?: readonly [number, number, number, number]): Promise<void> {
    const encoder = this.device.createCommandEncoder();
    const temps: GPUBuffer[] = [];
    try {
      this.encodeColumnsComplex(encoder, temps, tilt, indices, output, region);
      this.device.queue.submit([encoder.finish()]);
      await this.device.queue.onSubmittedWorkDone();
    } finally { temps.forEach(buffer => buffer.destroy()); }
  }

  /** Record the decode before its consumer in the same GPU submission. */
  encodeColumnsComplex(encoder: GPUCommandEncoder, temps: GPUBuffer[], tilt: number,
    indices: Uint32Array, output: GPUBuffer, region?: readonly [number, number, number, number]): void {
    if (this.disposed) throw new Error("rANS resident set disposed");
    if (!Number.isInteger(tilt) || tilt < 0 || tilt >= this.T || !indices.length
      || indices.some(index => index >= this.K)) {
      throw new Error("Select existing detector columns from one acquisition.");
    }
    const stride = this.shape?.[1] ?? this.scanCount;
    const [row, col, height, width] = region ?? [0, 0, this.scanCount / stride, stride];
    if (![row, col, height, width].every(Number.isSafeInteger) || row < 0 || col < 0 || height < 1 || width < 1
      || row + height > this.scanCount / stride || col + width > stride) throw new Error("Scan crop exceeds the QEM field.");
    if (output.size < indices.length * height * width * 8) {
      throw new Error("BF output buffer must hold all selected columns and scan positions as complex64.");
    }
    const device = this.device;
    this.columnsPipe ??= device.createComputePipeline({
      layout: device.createPipelineLayout({bindGroupLayouts: [this.intPipe.getBindGroupLayout(0)]}),
      compute: {module: device.createShaderModule({code: COLUMNS_WGSL}), entryPoint: "decode_columns",
        constants: {BINARY_LOOKUP: this.binaryLookup ? 1 : 0, DIRECT_LOOKUP: this.directLookup ? 1 : 0}},
    });
    const geometry = new Uint32Array(indices.length + 5);
    geometry.set(indices); geometry.set([stride, row, col, width, height], indices.length);
    const selected = device.createBuffer({size: geometry.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST});
    device.queue.writeBuffer(selected, 0, geometry);
    temps.push(selected);
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.columnsPipe);
    for (const group of this.groups) {
      const span = group.spans.get(tilt); if (!span) continue;
      const params = device.createBuffer({size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST});
      temps.push(params);
      device.queue.writeBuffer(params, 0, new Uint32Array([
        this.frames, this.K, this.scale, this.windows, indices.length, span.unit0, span.n, this.scanCount,
      ]));
      pass.setBindGroup(0, group.bind(params, output, selected, this.unitsBuf));
      pass.dispatchWorkgroups(Math.ceil(indices.length / 64), this.windows, span.n);
    }
    pass.end();
  }

  /** Sum or mean of selected patterns; accumulate paired-word integers before f64 division. */
  async reduceMany(tilt: number, scanIdxs: number[], mean: boolean): Promise<Float32Array> {
    const K = this.K;
    for (const index of scanIdxs) {
      if (!Number.isInteger(index) || index < 0 || index >= this.scanCount) {
        throw new Error(`Scan index ${index} is outside 0..${this.scanCount - 1}; select positions within the acquisition.`);
      }
    }
    const scratch = this.device.createBuffer({ size: (K * 2 + 4) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const temps: GPUBuffer[] = [scratch];
    try {
      // Decode each requested checkpoint window once. Bound each submission so a
      // full-pattern mean cannot monopolize an integrated GPU's command queue.
      const windowsPerBatch = Math.max(1, Math.min(this.device.limits.maxComputeWorkgroupsPerDimension,
        Math.floor(16_777_216 / (K * WINDOW))));
      const positionsInOrder = [...scanIdxs].sort((a, b) => a - b);
      for (const group of this.groups) {
        const windows = new Map<number, { unit: number; window: number; rows: number[] }>();
        let selected = 0;
        for (const [i, record] of group.units.entries()) {
          if (record.tilt !== tilt) continue;
          const unit = group.unit0 + i;
          const first = record.out_base - tilt * this.scanCount;
          const count = (record.frameFlags & 0x0fffffff) || this.frames;
          while (selected < positionsInOrder.length && positionsInOrder[selected] < first) selected++;
          while (selected < positionsInOrder.length && positionsInOrder[selected] < first + count) {
            const local = positionsInOrder[selected++] - first;
            const window = Math.floor(local / WINDOW);
            const key = unit * this.windows + window;
            if (!windows.has(key)) windows.set(key, { unit, window, rows: [] });
            windows.get(key)!.rows.push(local % WINDOW);
          }
        }
        const all = [...windows.values()];
        for (let offset = 0; offset < all.length; offset += windowsPerBatch) {
          const batch = all.slice(offset, offset + windowsPerBatch);
          const positions = new Uint32Array(batch.length * 4 + batch.reduce((n, entry) => n + entry.rows.length, 0));
          let cursor = batch.length * 4;
          batch.forEach(({ unit, window, rows }, i) => {
            positions.set([unit, window, cursor, cursor + rows.length], i * 4);
            positions.set(rows.sort((a, b) => a - b), cursor);
            cursor += rows.length;
          });
          const cols = this.device.createBuffer({ size: positions.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
          const params = this.device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
          try {
            this.device.queue.writeBuffer(cols, 0, positions);
            this.device.queue.writeBuffer(params, 0, new Uint32Array([this.frames, K, this.scale, this.windows, batch.length, 0, 0, K]));
            const enc = this.device.createCommandEncoder();
            const pass = enc.beginComputePass();
            pass.setPipeline(this.gatherPipe);
            pass.setBindGroup(0, group.bind(params, scratch, cols, this.unitsBuf));
            pass.dispatchWorkgroups(Math.ceil(K / 64), batch.length);
            pass.end();
            this.device.queue.submit([enc.finish()]);
            if (all.length > windowsPerBatch) await this.device.queue.onSubmittedWorkDone();
          } finally { cols.destroy(); params.destroy(); }
        }
      }
    } catch (error) { scratch.destroy(); throw error; }
    const enc = this.device.createCommandEncoder();
    const readback = this.device.createBuffer({ size: (K * 2 + 1) * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    enc.copyBufferToBuffer(scratch, 0, readback, 0, (K * 2 + 1) * 4);
    this.device.queue.submit([enc.finish()]);
    try {
      await readback.mapAsync(GPUMapMode.READ);
      const counts = new Uint32Array(readback.getMappedRange());
      if (counts[K * 2]) throw new Error("rANS pattern decoding failed; reload and verify the exported source.");
      const result = new Float32Array(K); const divisor = mean && scanIdxs.length ? scanIdxs.length : 1;
      for (let k = 0; k < K; k++) result[k] = (counts[k] + counts[K + k] * 4294967296) / divisor;
      return result;
    } finally { readback.destroy(); temps.forEach((buffer) => buffer.destroy()); }
  }

  /** @internal Encode exact masked moments, then retain the CoM maps on the GPU. */
  encodeCoM(pass: GPUComputePassEncoder, tilt: number, indices: GPUBuffer, output: GPUBuffer, detCols: number, count: number): GPUBuffer[] {
    if (!Number.isInteger(detCols) || detCols < 1 || this.K % detCols !== 0 || Math.max(detCols, this.K / detCols) > 65536) {
      throw new Error(`Detector width ${detCols} does not describe ${this.K} pixels with dimensions <=65536; supply the source detector width.`);
    }
    const device = this.device;
    if (!this.momentsPipe) {
      this.momentsPipe = device.createComputePipeline({ layout: device.createPipelineLayout({ bindGroupLayouts: [this.intPipe.getBindGroupLayout(0)] }), compute: { module: device.createShaderModule({ code: MOMENTS_WGSL }), entryPoint: "gather_moments", constants: { BINARY_LOOKUP: this.binaryLookup ? 1 : 0, DIRECT_LOOKUP: this.directLookup ? 1 : 0 } } });
      this.momentRatioPipe = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code: MOMENT_RATIO_WGSL }), entryPoint: "main" } });
    }
    const moments = device.createBuffer({ size: this.scanCount * 6 * 4, usage: GPUBufferUsage.STORAGE });
    const temps = [moments];
    pass.setPipeline(this.momentsPipe);
    for (const group of this.groups) {
      const span = group.spans.get(tilt); if (!span) continue;
      const mobile = device.adapterInfo?.vendor?.toLowerCase() === "qualcomm";
      const unitsPerSubmit = mobile ? Math.max(1, Math.floor(16_777_216 / (count * this.frames))) : span.n;
      for (let offset = 0; offset < span.n; offset += unitsPerSubmit) {
        const params = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        temps.push(params);
        device.queue.writeBuffer(params, 0, new Uint32Array([this.frames, this.K, this.scale, this.windows, count, span.unit0 + offset, detCols, this.scanCount]));
        // Queue bounded work before the caller submits the final ratio and display pass.
        const encoder = mobile ? device.createCommandEncoder() : null;
        const work = encoder?.beginComputePass() ?? pass;
        work.setPipeline(this.momentsPipe);
        work.setBindGroup(0, group.bind(params, moments, indices, this.unitsBuf));
        work.dispatchWorkgroups(Math.ceil(count / 64), this.windows, Math.min(unitsPerSubmit, span.n - offset));
        if (encoder) { work.end(); device.queue.submit([encoder.finish()]); }
      }
    }
    const ratioParams = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }); temps.push(ratioParams);
    device.queue.writeBuffer(ratioParams, 0, new Uint32Array([this.scanCount, 0, 0, 0]));
    const ratioPipe = this.momentRatioPipe!;
    pass.setPipeline(ratioPipe);
    pass.setBindGroup(0, device.createBindGroup({ layout: ratioPipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: moments } }, { binding: 1, resource: { buffer: output } }, { binding: 2, resource: { buffer: ratioParams } },
    ] }));
    pass.dispatchWorkgroups(Math.ceil(this.scanCount / 256));
    return temps;
  }
  pattern(tilt: number, scanIdx: number): Promise<Float32Array> { return this.reduceMany(tilt, [scanIdx], false); }
  private tablesBuffers!: GPUBuffer[];

  /** Copy current images into caller-owned display buffers in one submission.
   * Existing destinations remain stable across drag steps; queue order ensures
   * the preceding render consumes its image before the next copy overwrites it.
   * With `encoder`, the copies are recorded there and the caller submits them.
   */
  imageBuffersF32(tilts: number[], previous?: GPUBuffer[], encoder?: GPUCommandEncoder): GPUBuffer[] {
    const bytes = this.scanCount * 4;
    const buffers = previous ?? tilts.map(() => this.device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    }));
    buffers.forEach(buffer => this.displayCopies.add(buffer));
    const copies = encoder ?? this.device.createCommandEncoder();
    tilts.forEach((tilt, index) => copies.copyBufferToBuffer(this.imagesF32, tilt * bytes, buffers[index], 0, bytes));
    if (!encoder) this.device.queue.submit([copies.finish()]);
    return buffers;
  }

  /**
   * Convert fresh caller-owned display copies from sums to mean detector intensity.
   *
   * Call once after imageBuffersF32 (including a batch delta's copy refresh),
   * before adopting the buffers into compare-display slots. Canonical integer
   * counts and single-view sums are untouched. Preview float division may differ
   * from CPU division by one ULP; it is not an exact-count representation.
   * Empty masks use area 1. No
   * image readback or upload occurs; one encoder covers all supplied panels.
   * With `encoder`, the work is recorded there and the caller submits it.
   */
  normalizeDisplayBuffers(buffers: GPUBuffer[], maskArea: number, encoder?: GPUCommandEncoder): void {
    if (this.disposed) throw new Error("rANS resident set disposed");
    if (!Number.isInteger(maskArea) || maskArea < 0 || maskArea > this.detSize) {
      throw new Error(`Detector mask area must be an integer from 0 to ${this.detSize}; count the selected mask pixels`);
    }
    if (new Set(buffers).size !== buffers.length || buffers.some(buffer => !this.displayCopies.has(buffer))) {
      throw new Error("Normalize distinct display copies returned by imageBuffersF32; never pass source or unrelated buffers");
    }
    const area = Math.max(1, maskArea);
    if (area === 1 || !buffers.length) return;
    const device = this.device;
    if (!this.normalizeDisplayPipe) this.normalizeDisplayPipe = device.createComputePipeline({
      layout: "auto", compute: { module: device.createShaderModule({ code: NORMALIZE_DISPLAY_WGSL }), entryPoint: "normalize_display" },
    });
    const pipeline = this.normalizeDisplayPipe;
    const commands = encoder ?? device.createCommandEncoder();
    const pass = commands.beginComputePass(); pass.setPipeline(pipeline);
    buffers.forEach((buffer, index) => {
      let params = this.normalizeDisplayParams[index];
      if (!params) {
        params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.normalizeDisplayParams[index] = params;
      }
      const values = new ArrayBuffer(16);
      new Uint32Array(values)[0] = this.scanCount;
      new Float32Array(values)[1] = area;
      // One uniform per dispatch. Queue writes precede this encoder's submit;
      // reuse on a subsequent call is ordered after the preceding submission.
      device.queue.writeBuffer(params, 0, values);
      const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer } }, { binding: 1, resource: { buffer: params } },
      ] });
      pass.setBindGroup(0, group); pass.dispatchWorkgroups(Math.ceil(this.scanCount / 256));
    });
    pass.end(); if (!encoder) device.queue.submit([commands.finish()]);
  }

  /** Return a caller-owned copy for a single-view display. */
  imageBufferF32(tilt: number): GPUBuffer {
    return this.imageBuffersF32([tilt])[0];
  }

  /** Read the current quantitative detector sums without float32 rounding.
   * The copy snapshots this acquisition in GPU queue order. Display scaling
   * never changes these counts; callers own the returned uint32 array.
   */
  async readImageU32(tilt: number): Promise<Uint32Array> {
    if (this.disposed) throw new Error("rANS resident set disposed");
    if (!Number.isInteger(tilt) || tilt < 0 || tilt >= this.T) {
      throw new Error(`Acquisition index ${tilt} is outside 0..${this.T - 1}; select an acquisition in this resident set`);
    }
    const bytes = this.scanCount * 4;
    const readback = this.device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.images, tilt * bytes, readback, 0, bytes);
    this.device.queue.submit([encoder.finish()]);
    try {
      await readback.mapAsync(GPUMapMode.READ);
      return new Uint32Array(readback.getMappedRange().slice(0));
    } finally { readback.destroy(); }
  }

  /** Lend the exact uint32 detector sums of the given acquisitions to a display.
   * The views stay valid until this set is disposed; the borrower must not write
   * or destroy the buffer, and shows f32(count) / divisor.
   */
  imageViewsU32(tilts: number[], divisor: number): Uint32ImageView[] {
    if (this.disposed) throw new Error("rANS resident set disposed");
    if (tilts.some(tilt => !Number.isInteger(tilt) || tilt < 0 || tilt >= this.T)) {
      throw new Error(`Acquisition indices must be within 0..${this.T - 1}; select acquisitions in this resident set`);
    }
    // A storage binding must start on the device's offset alignment (256 bytes);
    // later acquisitions start at acquisition * scanCount * 4 bytes.
    const alignment = this.device.limits.minStorageBufferOffsetAlignment;
    const unaligned = tilts.find(tilt => tilt * this.scanCount * 4 % alignment !== 0);
    if (unaligned !== undefined) {
      throw new Error(`Acquisition ${unaligned} cannot be lent as a count view: its image starts at byte ${unaligned * this.scanCount * 4}, not a multiple of ${alignment}. Load one acquisition per resident set (RansResidentSeries) or display copies from imageBuffersF32.`);
    }
    return tilts.map(tilt => ({ device: this.device, buffer: this.images, byteOffset: tilt * this.scanCount * 4, count: this.scanCount, divisor }));
  }

  /** Read float32 display sums; use readImageU32 for exact quantitative counts. */
  async readImage(tilt: number): Promise<Float32Array> {
    const bytes = this.scanCount * 4;
    const readback = this.device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = this.device.createCommandEncoder(); encoder.copyBufferToBuffer(this.imagesF32, tilt * bytes, readback, 0, bytes); this.device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ); const image = new Float32Array(readback.getMappedRange().slice(0)); readback.unmap(); readback.destroy(); return image;
  }

  resetImages(tilts: Set<number> | null): void {
    if (tilts === null) { this.device.queue.writeBuffer(this.images, 0, new Uint32Array(this.T * this.scanCount)); this.device.queue.writeBuffer(this.imagesF32, 0, new Float32Array(this.T * this.scanCount)); return; }
    for (const tilt of tilts) { this.device.queue.writeBuffer(this.images, tilt * this.scanCount * 4, new Uint32Array(this.scanCount)); this.device.queue.writeBuffer(this.imagesF32, tilt * this.scanCount * 4, new Float32Array(this.scanCount)); }
  }

  dispose(): void {
    if (this.disposed) return; this.disposed = true;
    for (const compute of this.computes) compute.dispose();
    for (const params of this.normalizeDisplayParams) params.destroy();
    this.normalizeDisplayParams = [];
    for (const group of this.groups) { group.payload.destroy(); group.offsets.destroy(); group.chk.destroy(); group.params.destroy(); for (const span of group.spans.values()) span.params.destroy(); }
    for (const buffer of this.tablesBuffers) buffer.destroy();
    this.out.destroy(); this.images.destroy(); this.imagesF32.destroy(); this.colsBuf.destroy(); this.unitsBuf.destroy(); this.imageCountParams.destroy();
  }
  /** @internal set by load() */
  _setTables(tables: GPUBuffer[]): void { this.tablesBuffers = tables; }
}

/** One acquisition of a RansResidentSet, presenting the subset of the DetectorCompute surface the widget uses. */
export class RansDetectorCompute {
  readonly scanCount: number;
  readonly detSize: number;
  readonly mode = 2;          // exact uint32 sums
  badPx: Uint32Array;
  readonly isRansResident = true;
  currentMask: Uint8Array | null = null;
  constructor(readonly set: RansResidentSet, readonly tilt: number) {
    this.scanCount = set.scanCount; this.detSize = set.detSize; this.badPx = set.badPx;
  }
  effective(mask: Uint32Array | Uint8Array): Uint8Array {
    const selected = new Uint8Array(this.detSize);
    for (let k = 0; k < this.detSize; k++) selected[k] = mask[k] ? 1 : 0;
    for (const bp of this.badPx) selected[bp] = 0;
    return selected;
  }
  /** Bring this acquisition's resident image to `mask`, decoding only the changed columns. */
  update(mask: Uint32Array | Uint8Array): void {
    const next = this.effective(mask);
    const tilts = new Set([this.tilt]);
    if (!this.currentMask) { this.set.resetImages(tilts); this.set.integrate(tilts, next, null); }
    else {
      const { add, sub, changed } = maskDelta(next, this.currentMask);
      if (changed) this.set.integrate(tilts, add, sub);
    }
    this.currentMask = next;
  }
  /** Apply a caller-computed delta (added/removed detector pixels) without a full mask. */
  applyDelta(added: Uint32Array | Uint8Array, removed: Uint32Array | Uint8Array): void {
    if (!this.currentMask) throw new Error("applyDelta before the first full mask");
    const add = this.effective(added), sub = this.effective(removed);
    for (let k = 0; k < this.detSize; k++) { if (add[k]) this.currentMask[k] = 1; if (sub[k]) this.currentMask[k] = 0; }
    this.set.integrate(new Set([this.tilt]), add, sub);
  }
  async maskedSum(mask: Uint32Array): Promise<Float32Array> { this.update(mask); return this.set.readImage(this.tilt); }
  maskedSumBuffer(mask: Uint32Array): { buffer: GPUBuffer; n: number } { this.update(mask); return { buffer: this.set.imageBufferF32(this.tilt), n: this.scanCount }; }
  async frameAt(scanIdx: number): Promise<Float32Array> { const pattern = await this.set.pattern(this.tilt, scanIdx); for (const bp of this.badPx) pattern[bp] = 0; return pattern; }
  async reduceFrames(scanMask: Uint32Array, mean = true): Promise<Float32Array> {
    const idx: number[] = []; for (let i = 0; i < scanMask.length; i++) if (scanMask[i]) idx.push(i);
    const out = idx.length ? await this.set.reduceMany(this.tilt, idx, mean) : new Float32Array(this.detSize);
    for (const bp of this.badPx) out[bp] = 0;
    return out;
  }
  getDevice(): GPUDevice { return this.set.device; }
  private products?: DetectorCompute;
  private numericalProducts(): DetectorCompute {
    if (!this.products) {
      this.products = DetectorCompute.fromResidentCoM(this.set.device, this.scanCount, this.detSize,
        (pass, indices, output, detCols, count) => this.set.encodeCoM(pass, this.tilt, indices, output, detCols, count));
    }
    this.products.badPx = this.badPx;
    return this.products;
  }
  maskedCoM(...args: Parameters<DetectorCompute["maskedCoM"]>): ReturnType<DetectorCompute["maskedCoM"]> { return this.numericalProducts().maskedCoM(...args); }
  maskedCoMBuffer(...args: Parameters<DetectorCompute["maskedCoMBuffer"]>): ReturnType<DetectorCompute["maskedCoMBuffer"]> { return this.numericalProducts().maskedCoMBuffer(...args); }
  maskedDpc(...args: Parameters<DetectorCompute["maskedDpc"]>): ReturnType<DetectorCompute["maskedDpc"]> { return this.numericalProducts().maskedDpc(...args); }
  maskedDpcBuffer(...args: Parameters<DetectorCompute["maskedDpcBuffer"]>): ReturnType<DetectorCompute["maskedDpcBuffer"]> { return this.numericalProducts().maskedDpcBuffer(...args); }
  maskedIDpc(...args: Parameters<DetectorCompute["maskedIDpc"]>): ReturnType<DetectorCompute["maskedIDpc"]> { return this.numericalProducts().maskedIDpc(...args); }
  maskedIDpcBuffer(...args: Parameters<DetectorCompute["maskedIDpcBuffer"]>): ReturnType<DetectorCompute["maskedIDpcBuffer"]> { return this.numericalProducts().maskedIDpcBuffer(...args); }
  dispose(): void { this.products?.dispose(); this.products = undefined; }

}

/** Batch helpers matching DetectorCompute's static contract for the compare grid. */
export function isRansBatch(computes: unknown[]): computes is RansDetectorCompute[] {
  return computes.length > 0 && computes.every((compute) => Boolean((compute as { isRansResident?: boolean }).isRansResident));
}
export function ransMaskedSumBuffersBatch(computes: RansDetectorCompute[], mask: Uint32Array): { buffers: GPUBuffer[]; n: number; path: "batched-submit" } {
  const buffers = eachResidentSet(computes, (members, _indices, encoder) => {
    const set = members[0].set;
    const next = members[0].effective(mask);
    const shared = members.every((compute) => compute.currentMask !== null && sameMask(compute.currentMask, members[0].currentMask!));
    if (members.every((compute) => compute.currentMask === null)) {
      const tilts = new Set(members.map((compute) => compute.tilt));
      set.resetImages(tilts);
      set.integrate(tilts, next, null, encoder);
      for (const compute of members) compute.currentMask = next.slice();
    } else if (shared) {
      // Every panel carries the same mask history: one column diff, one integrate over all of them.
      const { add, sub, changed } = maskDelta(next, members[0].currentMask!);
      if (changed) set.integrate(new Set(members.map((compute) => compute.tilt)), add, sub, encoder);
      for (const compute of members) compute.currentMask = next.slice();
    } else {
      for (const compute of members) compute.update(mask);
    }
    return set.imageBuffersF32(members.map((compute) => compute.tilt), undefined, encoder);
  });
  return { buffers, n: computes[0].scanCount, path: "batched-submit" };
}

export function ransMaskedSumDeltaBuffersBatch(computes: RansDetectorCompute[], addedMask: Uint32Array, removedMask: Uint32Array, previous?: GPUBuffer[]): { buffers: GPUBuffer[]; path: "delta"; addedPixels: number; removedPixels: number } {
  let addedPixels = 0, removedPixels = 0;
  for (let k = 0; k < addedMask.length; k++) { if (addedMask[k]) addedPixels++; if (removedMask[k]) removedPixels++; }
  // Checked before any set records work: a later failure would leave earlier
  // sets' masks advanced while their images were never updated.
  if (computes.some((compute) => !compute.currentMask)) throw new Error("rANS delta update before the first full mask");
  const buffers = eachResidentSet(computes, (members, indices, encoder) => {
    const set = members[0].set;
    const add = members[0].effective(addedMask), sub = members[0].effective(removedMask);
    for (const compute of members) {
      for (let k = 0; k < add.length; k++) { if (add[k]) compute.currentMask![k] = 1; if (sub[k]) compute.currentMask![k] = 0; }
    }
    set.integrate(new Set(members.map((compute) => compute.tilt)), add, sub, encoder);
    return set.imageBuffersF32(members.map((compute) => compute.tilt), previous && indices.map(index => previous[index]), encoder);
  });
  return { buffers, path: "delta", addedPixels, removedPixels };
}

/** Run `batch` once per resident set and submit all of their work together.
 * A .qem series loads every acquisition as its own set; batching only the first
 * set would integrate and copy its images into every panel. Buffers come back
 * in the order of `computes`.
 */
function eachResidentSet(computes: RansDetectorCompute[], batch: (members: RansDetectorCompute[], indices: number[], encoder: GPUCommandEncoder) => GPUBuffer[]): GPUBuffer[] {
  const device = computes[0].set.device;
  const encoder = device.createCommandEncoder();
  const buffers: GPUBuffer[] = new Array(computes.length);
  for (const set of new Set(computes.map((compute) => compute.set))) {
    const indices = computes.flatMap((compute, index) => compute.set === set ? [index] : []);
    batch(indices.map(index => computes[index]), indices, encoder).forEach((buffer, member) => { buffers[indices[member]] = buffer; });
  }
  device.queue.submit([encoder.finish()]);
  return buffers;
}

/** Detector columns that enter (`add`) or leave (`sub`) the mask between two effective masks. */
function maskDelta(next: Uint8Array, previous: Uint8Array): { add: Uint8Array; sub: Uint8Array; changed: number } {
  const add = new Uint8Array(next.length), sub = new Uint8Array(next.length); let changed = 0;
  for (let k = 0; k < next.length; k++) { if (next[k] && !previous[k]) { add[k] = 1; changed++; } else if (!next[k] && previous[k]) { sub[k] = 1; changed++; } }
  return { add, sub, changed };
}

function sameMask(a: Uint8Array, b: Uint8Array): boolean {
  if (a.length !== b.length) return false;
  for (let k = 0; k < a.length; k++) if (a[k] !== b[k]) return false;
  return true;
}
