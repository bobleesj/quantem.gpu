/** Exact preserved source112 dense-pair tANS / sparse-event kernels.
 *
 * Bindings 0..7 are storage buffers; binding 8 is the sole 64-byte uniform:
 * 0 packed record payloads + compact offsets (u32 words)
 * 1 records (12 u32 per descriptor, see Record below)
 * 2 columns: stream rank -> native detector index (u32)
 * 3 model IDs (packed original u8, little-endian)
 * 4 original 1024-state decoding tables (u32)
 * 5 selected: rank in low24 bits, bit24 means subtraction, otherwise addition
 * 6 output (atomic u32): canonical sums or caller-cleared pattern/gather output
 * 7 errors (atomic u32): OR-ed error bits, caller checks before accepting output
 *
 * Descriptor bases are WORD offsets except modelIdsBase (BYTE offset). outBase
 * addresses the first of this record's 16384 native scan sums in output.
 * Uniform fields are in declaration order, all u32. Dense/sparse calls bind
 * their respective columns and selections. lengthWordBase is the compact
 * offsets' byte-length section: 17466 dense, 19399 sparse in source112.
 *
 * Entry points decode_dense / decode_sparse use workgroup_size(64).
 * Sum mode0: dispatch(ceil(selectedCount/64), 32, recordCount); output remains
 * resident, clear once for a full mask, retain for subsequent signed deltas.
 * Pattern mode1: dispatch(ceil(selectedCount/64), 1, recordCount), patternFrame
 * is 0..16383 within each record. Clear output first; every native column can
 * be selected, including uint16 hardware literals and sparse pixels.
 * Gather mode2: same dispatch, repeated for chosen patternFrame values, adds
 * paired-word uint64 counts; low at patternOutBase, high at wideOutBase, both
 * with patternOutStride per relative dispatch record. Clear both before use.
 * Pattern/gather output index is base + workgroup_id.z * stride + detector.
 *
 * Caller admits disjoint dense/sparse column maps covering the full geometry,
 * valid descriptor ranges, table lengths and unique selected ranks. No source
 * stream or model is decoded or approximated on the host. Uint32 detector sums
 * are exact because 36864 * 65535 < 2^32; output deltas use modular arithmetic.
 * Error bits mirror the bounded CUDA source112 decoder: dense bounds1,
 * model/state2, truncated bits4, sparse header8, sparse bounds16, events32.
 */
export const SOURCE112_WGSL = /* wgsl */ `
struct Record {
  denseBase: u32, denseOffsetsBase: u32, sparseBase: u32, sparseOffsetsBase: u32,
  modelIdsBase: u32, outBase: u32, denseWords: u32, sparseWords: u32,
  decodingBase: u32, modelCount: u32, pad0: u32, pad1: u32,
}
struct Params {
  recordFirst: u32, recordCount: u32, columnCount: u32, lengthWordBase: u32,
  selectedCount: u32, mode: u32, patternFrame: u32, detectorPixels: u32,
  patternOutBase: u32, patternOutStride: u32, wideOutBase: u32, pad0: u32,
  pad1: u32, pad2: u32, pad3: u32, pad4: u32,
}
@group(0) @binding(0) var<storage, read> packed: array<u32>;
@group(0) @binding(1) var<storage, read> records: array<Record>;
@group(0) @binding(2) var<storage, read> columns: array<u32>;
@group(0) @binding(3) var<storage, read> ids: array<u32>;
@group(0) @binding(4) var<storage, read> decoding: array<u32>;
@group(0) @binding(5) var<storage, read> selected: array<u32>;
@group(0) @binding(6) var<storage, read_write> output: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> errors: array<atomic<u32>>;
@group(0) @binding(8) var<uniform> p: Params;
var<workgroup> sums: array<atomic<u32>, 512>;
fn fault(bits: u32) { atomicOr(&errors[0], bits); }
fn length_byte(offsetBase: u32, stream: u32) -> u32 {
  return (packed[offsetBase + p.lengthWordBase + (stream >> 2u)] >> ((stream & 3u) * 8u)) & 255u;
}
fn bounds(offsetBase: u32, stream: u32, extra: u32) -> vec2u {
  var first = packed[offsetBase + (stream >> 5u)];
  for (var j = stream & ~31u; j < stream; j++) { first += length_byte(offsetBase, j) + extra; }
  return vec2u(first, first + length_byte(offsetBase, stream) + extra);
}
fn emit(q: u32, frame: u32, value: u32, subtract: bool, relativeRecord: u32) {
  if (value == 0u) { return; }
  if (p.mode == 0u) {
    atomicAdd(&sums[frame], select(value, 0u - value, subtract));
  } else if (frame == (p.patternFrame & 511u)) {
    let index = p.patternOutBase + relativeRecord * p.patternOutStride + q;
    if (p.mode == 2u) {
      // Gather is an unsigned sum of native patterns, never a signed mask delta.
      let previous = atomicAdd(&output[index], value);
      if (previous > 0xffffffffu - value) { atomicAdd(&output[p.wideOutBase + relativeRecord * p.patternOutStride + q], 1u); }
    } else {
      atomicAdd(&output[index], value);
    }
  }
}
struct Bits { cursor: u32, end: u32, reservoir: u32, available: u32, bad: bool }
fn take_bits(b: ptr<function, Bits>, count: u32) -> u32 {
  if (count == 0u) { return 0u; }
  let mask = (1u << count) - 1u;
  if ((*b).available >= count) {
    let value = (*b).reservoir & mask;
    (*b).reservoir >>= count; (*b).available -= count;
    return value;
  }
  if ((*b).cursor >= (*b).end) { (*b).bad = true; return 0u; }
  // Consume only the required low bits from the next word. This is identical
  // to the CUDA uint64 reservoir, using u32-only operations available in WGSL.
  let word = packed[(*b).cursor]; (*b).cursor += 1u;
  let value = ((*b).reservoir | (word << (*b).available)) & mask;
  let consumed = count - (*b).available;
  (*b).reservoir = word >> consumed; (*b).available = 32u - consumed;
  return value;
}
fn dense_column(rec: Record, packet: u32, rank: u32, subtract: bool, relativeRecord: u32) {
  let q = columns[rank];
  let range = bounds(rec.denseOffsetsBase, packet * p.columnCount + rank, 1u);
  if (q >= p.detectorPixels || range.x >= range.y || range.y > rec.denseWords) { fault(1u); return; }
  let first = rec.denseBase + range.x; let end = rec.denseBase + range.y;
  let idIndex = rec.modelIdsBase + q;
  let model = (ids[idIndex >> 2u] >> ((idIndex & 3u) * 8u)) & 255u;
  if (model == 255u) {
    if (range.y - range.x != 256u) { fault(1u); return; }
    if (p.mode != 0u) {
      let frame = p.patternFrame & 511u;
      let word = packed[first + (frame >> 1u)];
      emit(q, frame, (word >> ((frame & 1u) * 16u)) & 65535u, subtract, relativeRecord);
    } else {
      for (var pair = 0u; pair < 256u; pair++) {
        let word = packed[first + pair];
        emit(q, pair * 2u, word & 65535u, subtract, relativeRecord);
        emit(q, pair * 2u + 1u, word >> 16u, subtract, relativeRecord);
      }
    }
    return;
  }
  if (model >= rec.modelCount || model == 254u) { fault(2u); return; }
  let header = packed[first]; var state = header & 1023u;
  var bits = Bits(first + 1u, end, header >> 10u, 22u, false);
  let pairs = select(256u, ((p.patternFrame & 511u) >> 1u) + 1u, p.mode != 0u);
  for (var pairIndex = 0u; pairIndex < pairs; pairIndex++) {
    if (state >= 1024u) { fault(2u); return; }
    let code = decoding[rec.decodingBase + model * 1024u + state];
    let count = (code >> 12u) & 15u;
    let low = take_bits(&bits, count);
    state = (code >> 16u) + low;
    var pair = code & 4095u;
    if (pair == 4095u) { pair = take_bits(&bits, 12u); }
    if (bits.bad) { fault(4u); return; }
    emit(q, pairIndex * 2u, pair & 63u, subtract, relativeRecord);
    emit(q, pairIndex * 2u + 1u, pair >> 6u, subtract, relativeRecord);
  }
  if (state >= 1024u) { fault(2u); }
}
fn sparse_column(rec: Record, packet: u32, rank: u32, subtract: bool, relativeRecord: u32) {
  if (rec.sparseWords < 4u) { fault(8u); return; }
  let base = rec.sparseBase;
  let n = packed[base]; let pwords = packed[base + 1u];
  let fwords = packed[base + 2u]; let rwords = packed[base + 3u];
  let valueWord = 4u + pwords + fwords + rwords;
  // Quotient/remainder forms avoid multiplying a potentially large event count.
  if (pwords != (n / 32u) * 9u + ((n % 32u) * 9u + 31u) / 32u ||
      fwords != n / 32u + select(0u, 1u, n % 32u != 0u) ||
      rwords != n / 256u + select(0u, 1u, n % 256u != 0u) || valueWord > rec.sparseWords) { fault(8u); return; }
  let q = columns[rank];
  let range = bounds(rec.sparseOffsetsBase, packet * p.columnCount + rank, 0u);
  if (q >= p.detectorPixels || range.x > range.y || range.y > n || packed[rec.sparseOffsetsBase + p.columnCount] != n) { fault(16u); return; }
  let positions = base + 4u; let flags = positions + pwords; let ranks = flags + fwords;
  var previous = 0u; var havePrevious = false;
  for (var at = range.x; at < range.y; at++) {
    let bit = (at % 32u) * 9u;
    let word = (at / 32u) * 9u + (bit >> 5u); let shift = bit & 31u;
    var position = packed[positions + word] >> shift;
    if (shift > 23u && word + 1u < pwords) { position |= packed[positions + word + 1u] << (32u - shift); }
    position &= 511u;
    let flag = packed[flags + (at >> 5u)]; var value = 1u;
    if (((flag >> (at & 31u)) & 1u) != 0u) {
      var valueIndex = packed[ranks + (at >> 8u)];
      for (var j = (at >> 8u) * 8u; j < (at >> 5u); j++) { valueIndex += countOneBits(packed[flags + j]); }
      valueIndex += countOneBits(flag & ((1u << (at & 31u)) - 1u));
      if ((valueIndex >> 2u) >= rec.sparseWords - valueWord) { fault(32u); return; }
      value = (packed[base + valueWord + (valueIndex >> 2u)] >> ((valueIndex & 3u) * 8u)) & 255u;
    }
    if ((havePrevious && position <= previous) || value == 0u || value > 127u) { fault(32u); return; }
    previous = position; havePrevious = true;
    emit(q, position, value, subtract, relativeRecord);
  }
}
fn finish_packet(rec: Record, packet: u32, t: u32) {
  if (p.mode == 0u) {
    for (var frame = t; frame < 512u; frame += 64u) {
      let value = atomicLoad(&sums[frame]);
      if (value != 0u) { atomicAdd(&output[rec.outBase + packet * 512u + frame], value); }
    }
  }
}
@compute @workgroup_size(64)
fn decode_dense(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  for (var frame = t; frame < 512u; frame += 64u) { atomicStore(&sums[frame], 0u); }
  workgroupBarrier();
  let rec = records[p.recordFirst + wg.z];
  let packet = select(wg.y, p.patternFrame / 512u, p.mode != 0u);
  let index = wg.x * 64u + t;
  if (wg.z < p.recordCount && packet < 32u && index < p.selectedCount) {
    let entry = select(selected[index], index, p.mode == 1u); let rank = entry & 0xffffffu;
    if (rank < p.columnCount) { dense_column(rec, packet, rank, (entry & 0x1000000u) != 0u, wg.z); }
    else { fault(1u); }
  }
  workgroupBarrier();
  finish_packet(rec, packet, t);
}
@compute @workgroup_size(64)
fn decode_sparse(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  for (var frame = t; frame < 512u; frame += 64u) { atomicStore(&sums[frame], 0u); }
  workgroupBarrier();
  let rec = records[p.recordFirst + wg.z];
  let packet = select(wg.y, p.patternFrame / 512u, p.mode != 0u);
  let index = wg.x * 64u + t;
  if (wg.z < p.recordCount && packet < 32u && index < p.selectedCount) {
    let entry = select(selected[index], index, p.mode == 1u); let rank = entry & 0xffffffu;
    if (rank < p.columnCount) { sparse_column(rec, packet, rank, (entry & 0x1000000u) != 0u, wg.z); }
    else { fault(16u); }
  }
  workgroupBarrier();
  finish_packet(rec, packet, t);
}
`;

/** Compile-time detector-sum specialization of the same exact decoder.
 * Use only for mode0 integration/deltas; pattern and gather keep SOURCE112_WGSL.
 * This removes output-mode branches without duplicating the scientific kernel.
 */
export const SOURCE112_SUM_WGSL = SOURCE112_WGSL.replace(/\bp\.mode\b/g, '0u');
/** Experimental GPU restart cache; the original encoded payload is unchanged. */
const RESTART_RECORDS = /* wgsl */ `
fn load_record(index: u32) -> Record {
  let b = index * 12u;
  return Record(recordWords[b], recordWords[b+1u], recordWords[b+2u], recordWords[b+3u],
    recordWords[b+4u], recordWords[b+5u], recordWords[b+6u], recordWords[b+7u],
    recordWords[b+8u], recordWords[b+9u], recordWords[b+10u], recordWords[b+11u]);
}
fn restart_word(rec: Record, packet: u32, rank: u32, segment: u32) -> u32 {
  return rec.pad0 + (packet * p.columnCount + rank) * 3u + segment - 1u;
}
`;
const RESTART_BASE = SOURCE112_WGSL
  .replace('var<storage, read> records: array<Record>;', 'var<storage, read> recordWords: array<u32>;')
  .split('records[p.recordFirst + wg.z]').join('load_record(p.recordFirst + wg.z)') + RESTART_RECORDS;

export const SOURCE112_RESTART_WGSL = RESTART_BASE + /* wgsl */ `
fn dense_segment(rec: Record, packet: u32, rank: u32, segment: u32, subtract: bool, relativeRecord: u32) {
  let q = columns[rank];
  let range = bounds(rec.denseOffsetsBase, packet * p.columnCount + rank, 1u);
  if (q >= p.detectorPixels || range.x >= range.y || range.y > rec.denseWords) { fault(1u); return; }
  let first = rec.denseBase + range.x; let end = rec.denseBase + range.y;
  let idIndex = rec.modelIdsBase + q;
  let model = (ids[idIndex >> 2u] >> ((idIndex & 3u) * 8u)) & 255u;
  let beginPair = segment * 64u;
  if (model == 255u) {
    if (range.y - range.x != 256u) { fault(1u); return; }
    for (var pair = beginPair; pair < beginPair + 64u; pair++) {
      let word = packed[first + pair];
      emit(q, pair * 2u, word & 65535u, subtract, relativeRecord);
      emit(q, pair * 2u + 1u, word >> 16u, subtract, relativeRecord);
    }
    return;
  }
  if (model >= rec.modelCount || model == 254u) { fault(2u); return; }
  let header = packed[first]; var state = header & 1023u;
  var bits = Bits(first + 1u, end, header >> 10u, 22u, false);
  if (segment > 0u) {
    let checkpoint = recordWords[restart_word(rec, packet, rank, segment)];
    let cursor = (checkpoint >> 10u) & 8191u;
    if ((checkpoint & 0x80000000u) == 0u || cursor < 10u || cursor > (end - first) * 32u) { fault(64u); return; }
    state = checkpoint & 1023u;
    let word = first + (cursor >> 5u); let shift = cursor & 31u;
    bits = Bits(word, end, 0u, 0u, false);
    if (word < end) { bits = Bits(word + 1u, end, packed[word] >> shift, 32u - shift, false); }
  }
  for (var pairIndex = beginPair; pairIndex < beginPair + 64u; pairIndex++) {
    if (state >= 1024u) { fault(2u); return; }
    let code = decoding[rec.decodingBase + model * 1024u + state];
    let low = take_bits(&bits, (code >> 12u) & 15u);
    state = (code >> 16u) + low;
    var pair = code & 4095u;
    if (pair == 4095u) { pair = take_bits(&bits, 12u); }
    if (bits.bad) { fault(4u); return; }
    emit(q, pairIndex * 2u, pair & 63u, subtract, relativeRecord);
    emit(q, pairIndex * 2u + 1u, pair >> 6u, subtract, relativeRecord);
  }
  if (state >= 1024u) { fault(2u); }
}
@compute @workgroup_size(64)
fn decode_dense_restart(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  for (var frame = t; frame < 512u; frame += 64u) { atomicStore(&sums[frame], 0u); }
  workgroupBarrier();
  let rec = load_record(p.recordFirst + wg.z);
  let index = wg.x * 16u + (t >> 2u);
  if (wg.z < p.recordCount && wg.y < 32u && index < p.selectedCount) {
    let entry = selected[index]; let rank = entry & 0xffffffu;
    if (rank < p.columnCount) { dense_segment(rec, wg.y, rank, t & 3u, (entry & 0x1000000u) != 0u, wg.z); }
    else { fault(1u); }
  }
  workgroupBarrier();
  finish_packet(rec, wg.y, t);
}
`;

export const SOURCE112_RESTART_BUILD_WGSL = RESTART_BASE
  .replace('var<storage, read> recordWords:', 'var<storage, read_write> recordWords:') + /* wgsl */ `
@compute @workgroup_size(64)
fn build_dense_restarts(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  let rank = wg.x * 64u + t;
  if (wg.z >= p.recordCount || wg.y >= 32u || rank >= p.columnCount) { return; }
  let rec = load_record(p.recordFirst + wg.z);
  let q = columns[rank];
  let range = bounds(rec.denseOffsetsBase, wg.y * p.columnCount + rank, 1u);
  if (q >= p.detectorPixels || range.x >= range.y || range.y > rec.denseWords) { fault(1u); return; }
  let first = rec.denseBase + range.x; let end = rec.denseBase + range.y;
  let idIndex = rec.modelIdsBase + q;
  let model = (ids[idIndex >> 2u] >> ((idIndex & 3u) * 8u)) & 255u;
  if (model == 255u) { if (range.y - range.x != 256u) { fault(1u); } return; }
  if (model >= rec.modelCount || model == 254u) { fault(2u); return; }
  let header = packed[first]; var state = header & 1023u;
  var bits = Bits(first + 1u, end, header >> 10u, 22u, false);
  for (var pairIndex = 0u; pairIndex < 256u; pairIndex++) {
    if (state >= 1024u) { fault(2u); return; }
    let code = decoding[rec.decodingBase + model * 1024u + state];
    let low = take_bits(&bits, (code >> 12u) & 15u);
    state = (code >> 16u) + low;
    if ((code & 4095u) == 4095u) { let ignored = take_bits(&bits, 12u); }
    if (bits.bad) { fault(4u); return; }
    if ((pairIndex & 63u) == 63u && pairIndex < 192u) {
      let cursor = (bits.cursor - first) * 32u - bits.available;
      if (state >= 1024u || cursor > 8191u) { fault(64u); return; }
      recordWords[restart_word(rec, wg.y, rank, (pairIndex + 1u) / 64u)] = 0x80000000u | (cursor << 10u) | state;
    }
  }
  if (state >= 1024u) { fault(2u); }
}
`;

/** GPU-authored absolute stream starts accompany the128-value restart cache.
 * The original compact length remains authoritative and is checked on every decode.
 * pad0 addresses restart words; pad1 addresses one absolute first word per stream.
 * Shared sums pad each32-word row to separate the four128-value segments across
 * banks; initialization, accumulation, and readback use the same frame mapping.
 */
export const SOURCE112_OFFSET_RESTART_WGSL = SOURCE112_RESTART_WGSL.replace(
  `fn dense_segment(rec: Record, packet: u32, rank: u32, segment: u32, subtract: bool, relativeRecord: u32) {
  let q = columns[rank];
  let range = bounds(rec.denseOffsetsBase, packet * p.columnCount + rank, 1u);`,
  `fn dense_segment(rec: Record, packet: u32, rank: u32, segment: u32, subtract: bool, relativeRecord: u32) {
  let q = columns[rank];
  let range = cached_dense_bounds(rec, packet * p.columnCount + rank);`,
).replace('var<workgroup> sums: array<atomic<u32>, 512>;',
  'var<workgroup> sums: array<atomic<u32>, 528>;')
  .replace(/sums\[frame\]/g, 'sums[frame + (frame >> 5u)]') + /* wgsl */ `
fn cached_dense_bounds(rec: Record, stream: u32) -> vec2u {
  let first = recordWords[rec.pad1 + stream];
  if (first < rec.denseBase) { return vec2u(0u); }
  let relative = first - rec.denseBase;
  let length = length_byte(rec.denseOffsetsBase, stream) + 1u;
  if (relative >= rec.denseWords || length > rec.denseWords - relative) { return vec2u(0u); }
  return vec2u(relative, relative + length);
}
`;

export const SOURCE112_OFFSET_RESTART_BUILD_WGSL = SOURCE112_RESTART_BUILD_WGSL.replace(
  `fn build_dense_restarts(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  let rank = wg.x * 64u + t;
  if (wg.z >= p.recordCount || wg.y >= 32u || rank >= p.columnCount) { return; }
  let rec = load_record(p.recordFirst + wg.z);
  let q = columns[rank];
  let range = bounds(rec.denseOffsetsBase, wg.y * p.columnCount + rank, 1u);
  if (q >= p.detectorPixels || range.x >= range.y || range.y > rec.denseWords) { fault(1u); return; }
  let first = rec.denseBase + range.x; let end = rec.denseBase + range.y;`,
  `fn build_dense_restarts(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  let rank = wg.x * 64u + t;
  if (wg.z >= p.recordCount || wg.y >= 32u || rank >= p.columnCount) { return; }
  let rec = load_record(p.recordFirst + wg.z);
  let q = columns[rank];
  let range = bounds(rec.denseOffsetsBase, wg.y * p.columnCount + rank, 1u);
  if (q >= p.detectorPixels || range.x >= range.y || range.y > rec.denseWords) { fault(1u); return; }
  let first = rec.denseBase + range.x; let end = rec.denseBase + range.y;
  recordWords[rec.pad1 + wg.y * p.columnCount + rank] = first;`,
);
