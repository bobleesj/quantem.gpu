#include <metal_stdlib>
using namespace metal;

// Image-only exact widening; never constructs a decoded 4D resident.
kernel void tans_display_counts(device const ushort* source [[buffer(0)]],
    device uint* output [[buffer(1)]], uint q [[thread_position_in_grid]]) {
  if (q < 36864) output[q] = uint(source[q]);
}

// source112-tans1024-pair-v1 and position9-flag1-count8-rank256-v1.
// Counts remain exact; sparse and literal exceptions are not discarded.
inline uint byte_sum(uint word) {
  return (word & 255u) + ((word >> 8) & 255u) + ((word >> 16) & 255u) + (word >> 24);
}
inline uint2 stream_bounds(device const uint* offsets, uint stream, uint columns, bool sparse) {
  uint lane = stream & 31u, whole = lane >> 2, tail = lane & 3u;
  device const uint* words = offsets + columns + (sparse ? 1u : 0u) + (stream >> 5) * 8;
  uint sum = sparse ? 0u : lane;
  for (uint i = 0; i < whole; ++i) sum += byte_sum(words[i]);
  if (tail) sum += byte_sum(words[whole] & ((1u << (tail * 8)) - 1u));
  uint begin = offsets[stream >> 5] + sum;
  return uint2(begin, begin + ((words[whole] >> (tail * 8)) & 255u) + (sparse ? 0u : 1u));
}
inline uint sparse_event(device const uint* data, uint i) {
  uint pwords = data[1], fwords = data[2], rwords = data[3];
  device const uint* positions = data + 4;
  device const uint* flags = positions + pwords;
  device const uint* ranks = flags + fwords;
  device const uchar* values = reinterpret_cast<device const uchar*>(ranks + rwords);
  uint bit = i * 9u, word = bit >> 5, shift = bit & 31u;
  ulong joined = ulong(positions[word]);
  if (word + 1 < pwords) joined |= ulong(positions[word + 1]) << 32;
  uint position = uint(joined >> shift) & 511u;
  uint flag = flags[i >> 5], count = 1;
  if ((flag >> (i & 31u)) & 1u) {
    uint rank = ranks[i >> 8];
    for (uint j = (i >> 8) * 8; j < (i >> 5); ++j) rank += popcount(flags[j]);
    rank += popcount(flag & ((1u << (i & 31u)) - 1u));
    count = values[rank];
  }
  return position | (count << 9);
}
inline uint read_bits(device const uint* payload, thread uint& cursor, uint end,
                      uint bits, thread ulong& reservoir, thread uint& available) {
  if (available < bits) {
    if (cursor < end) reservoir |= ulong(payload[cursor++]) << available;
    available += 32;
  }
  uint result = uint(reservoir) & ((1u << bits) - 1u);
  reservoir >>= bits;
  available -= bits;
  return result;
}
inline uint next_pair(device const uint* payload, thread uint& cursor, uint end,
                      device const uint* table, thread uint& state,
                      thread ulong& reservoir, thread uint& available) {
  uint code = table[state], bits = (code >> 12) & 15u;
  state = (code >> 16) + read_bits(payload, cursor, end, bits, reservoir, available);
  uint pair = code & 4095u;
  return pair == 4095u ? read_bits(payload, cursor, end, 12, reservoir, available) : pair;
}
struct TANSQuery {
  uint scan;
  uint retainedColumns;
  uint sparseColumns;
  uint modelOffset;
};
// Alternative exact reader. Keep at most one word in the reservoir; the
// crossing branch consumes only the required low bits of the next word.
inline uint read_bits32(device const uint* payload, thread uint& cursor, uint end,
                       uint bits, thread uint& reservoir, thread uint& available) {
  if (bits == 0) return 0;
  uint mask = (1u << bits) - 1u;
  if (available >= bits) {
    uint result = reservoir & mask;
    reservoir >>= bits; available -= bits;
    return result;
  }
  uint next = cursor < end ? payload[cursor++] : 0u;
  uint used = bits - available;
  uint result = (reservoir | (next << available)) & mask;
  reservoir = next >> used; available = 32u - used;
  return result;
}
inline uint next_pair32(device const uint* payload, thread uint& cursor, uint end,
                       device const uint* table, thread uint& state,
                       thread uint& reservoir, thread uint& available) {
  uint code = table[state], bits = (code >> 12) & 15u;
  state = (code >> 16) + read_bits32(payload, cursor, end, bits, reservoir, available);
  uint pair = code & 4095u;
  return pair == 4095u ? read_bits32(payload, cursor, end, 12, reservoir, available) : pair;
}
kernel void tans_diffraction_word32(
    device const uint* payload [[buffer(0)]], device const uint* offsets [[buffer(1)]],
    device const uint* events [[buffer(2)]], device const uint* sparseOffsets [[buffer(3)]],
    device const uint* decoding [[buffer(4)]], device const uchar* models [[buffer(5)]],
    device const int* cacheMap [[buffer(6)]], device const uint* retainedRank [[buffer(7)]],
    device ushort* output [[buffer(8)]], constant TANSQuery& query [[buffer(9)]],
    uint q [[thread_position_in_grid]]) {
  if (q >= 36864) return;
  uint packet = query.scan / 512, target = query.scan % 512, value = 0;
  int cached = cacheMap[q];
  if (cached >= 0) {
    uint2 bounds = stream_bounds(sparseOffsets, packet * query.sparseColumns + uint(cached), query.sparseColumns, true);
    uint begin = bounds.x, end = bounds.y;
    while (begin < end) {
      uint mid = begin + (end - begin) / 2, event = sparse_event(events, mid);
      if ((event & 511u) < target) begin = mid + 1;
      else if ((event & 511u) > target) end = mid;
      else { value = event >> 9; break; }
    }
  } else {
    uint2 bounds = stream_bounds(offsets, packet * query.retainedColumns + retainedRank[q], query.retainedColumns, false);
    uint cursor = bounds.x, end = bounds.y, model = models[query.modelOffset + q];
    if (model == 255u) value = (payload[cursor + target / 2] >> ((target & 1u) * 16)) & 65535u;
    else {
      uint header = payload[cursor++], state = header & 1023u, available = 22, pair = 0;
      uint reservoir = header >> 10;
      for (uint i = 0; i <= target / 2; ++i)
        pair = next_pair32(payload, cursor, end, decoding + model * 1024, state, reservoir, available);
      value = (pair >> ((target & 1u) * 6)) & 63u;
    }
  }
  output[q] = ushort(value);
}
kernel void tans_diffraction(
    device const uint* payload [[buffer(0)]],
    device const uint* offsets [[buffer(1)]],
    device const uint* events [[buffer(2)]],
    device const uint* sparseOffsets [[buffer(3)]],
    device const uint* decoding [[buffer(4)]],
    device const uchar* models [[buffer(5)]],
    device const int* cacheMap [[buffer(6)]],
    device const uint* retainedRank [[buffer(7)]],
    device ushort* output [[buffer(8)]],
    constant TANSQuery& query [[buffer(9)]],
    uint q [[thread_position_in_grid]]) {
  if (q >= 36864) return;
  uint packet = query.scan / 512, target = query.scan % 512, value = 0;
  int cached = cacheMap[q];
  if (cached >= 0) {
    uint2 bounds = stream_bounds(sparseOffsets, packet * query.sparseColumns + uint(cached), query.sparseColumns, true);
    uint begin = bounds.x, end = bounds.y;
    while (begin < end) {
      uint mid = begin + (end - begin) / 2, event = sparse_event(events, mid);
      if ((event & 511u) < target) begin = mid + 1;
      else if ((event & 511u) > target) end = mid;
      else { value = event >> 9; break; }
    }
  } else {
    uint2 bounds = stream_bounds(offsets, packet * query.retainedColumns + retainedRank[q], query.retainedColumns, false);
    uint cursor = bounds.x, end = bounds.y;
    uint model = models[query.modelOffset + q];
    if (model == 255u) value = (payload[cursor + target / 2] >> ((target & 1u) * 16)) & 65535u;
    else {
      uint header = payload[cursor++], state = header & 1023u, available = 22, pair = 0;
      ulong reservoir = header >> 10;
      for (uint i = 0; i <= target / 2; ++i)
        pair = next_pair(payload, cursor, end, decoding + model * 1024, state, reservoir, available);
      value = (pair >> ((target & 1u) * 6)) & 63u;
    }
  }
  output[q] = ushort(value);
}

// Bounded independent audit only: one512-scan packet (36MiB), never a full
// decoded4D resident. The consumer receives nothing until the command completes.
kernel void tans_audit_packet(
    device const uint* payload [[buffer(0)]],
    device const uint* offsets [[buffer(1)]],
    device const uint* events [[buffer(2)]],
    device const uint* sparseOffsets [[buffer(3)]],
    device const uint* decoding [[buffer(4)]],
    device const uchar* models [[buffer(5)]],
    device const int* cacheMap [[buffer(6)]],
    device const uint* retainedRank [[buffer(7)]],
    device ushort* output [[buffer(8)]],
    constant TANSQuery& query [[buffer(9)]],
    uint q [[thread_position_in_grid]]) {
  if (q >= 36864) return;
  uint packet = query.scan / 512;
  int cached = cacheMap[q];
  if (cached >= 0) {
    for (uint scan = 0; scan < 512; ++scan) output[scan * 36864 + q] = 0;
    uint2 bounds = stream_bounds(sparseOffsets, packet * query.sparseColumns + uint(cached), query.sparseColumns, true);
    for (uint at = bounds.x; at < bounds.y; ++at) {
      uint event = sparse_event(events, at);
      output[(event & 511u) * 36864 + q] = ushort(event >> 9);
    }
  } else {
    uint2 bounds = stream_bounds(offsets, packet * query.retainedColumns + retainedRank[q], query.retainedColumns, false);
    uint cursor = bounds.x, end = bounds.y, model = models[query.modelOffset + q];
    if (model == 255u) {
      for (uint scan = 0; scan < 512; scan += 2) {
        uint pair = payload[cursor + scan / 2];
        output[scan * 36864 + q] = ushort(pair & 65535u);
        output[(scan + 1) * 36864 + q] = ushort(pair >> 16);
      }
    } else {
      uint header = payload[cursor++], state = header & 1023u, available = 22;
      ulong reservoir = header >> 10;
      for (uint scan = 0; scan < 512; scan += 2) {
        uint pair = next_pair(payload, cursor, end, decoding + model * 1024, state, reservoir, available);
        output[scan * 36864 + q] = ushort(pair & 63u);
        output[(scan + 1) * 36864 + q] = ushort(pair >> 6);
      }
    }
  }
}

// One SIMD group decodes 32 selected detector columns across a 512-scan
// entropy packet. Reduce while decoding: no decoded 4D counts are stored.
// Each partial fits uint32: 32 * 65535 < 2^32. The complete detector sum
// also fits uint32: 36864 * 65535 < 2^32, with no count truncation.
struct TANSDetectorQuery {
  uint retainedColumns, sparseColumns, modelOffset, selectedCount, groups;
};
inline void tans_detector_reduce(
    device const uint* payload, device const uint* offsets,
    device const uint* events, device const uint* sparseOffsets,
    device const uint* decoding, device const uchar* models,
    device const int* cacheMap, device const uint* retainedRank,
    device uint* partials, TANSDetectorQuery query,
    device const uint* selected, uint lane, uint2 group,
    device const int* coefficients, bool atomicOutput, bool accumulate) {
  uint index = group.x * 32u + lane;
  bool active = index < query.selectedCount;
  uint q = active ? selected[index] : 0u;
  int cached = active ? cacheMap[q] : -1;
  uint cursor=0,end=0,state=0,available=0,model=0,event=0;
  ulong reservoir=0;
  if (active) {
    uint2 bounds = cached >= 0
      ? stream_bounds(sparseOffsets, group.y * query.sparseColumns + uint(cached), query.sparseColumns, true)
      : stream_bounds(offsets, group.y * query.retainedColumns + retainedRank[q], query.retainedColumns, false);
    cursor=bounds.x; end=bounds.y;
    if (cached >= 0) { if (cursor < end) event=sparse_event(events,cursor); }
    else {
      model=models[query.modelOffset+q];
      if (model != 255u) { uint header=payload[cursor++]; state=header&1023u; available=22; reservoir=header>>10; }
    }
  }
  // The exact detector path always consumes one complete 512-scan packet.
  // Keep the stateful tANS dependency inside the lane, but ask Metal to
  // unroll the fixed loop so address arithmetic and the mask reduction stay
  // in registers. This changes no decode order or integer arithmetic.
  #pragma unroll
  for (uint scan=0; scan<512; scan+=2) {
    uint a=0,b=0;
    if (active) {
      if (cached >= 0) {
        if (cursor<end && (event&511u)==scan) { a=event>>9; if (++cursor<end) event=sparse_event(events,cursor); }
        if (cursor<end && (event&511u)==scan+1) { b=event>>9; if (++cursor<end) event=sparse_event(events,cursor); }
      } else if (model==255u) {
        uint pair=payload[cursor++]; a=pair&65535u; b=pair>>16;
      } else {
        uint pair=next_pair(payload,cursor,end,decoding+model*1024,state,reservoir,available);
        a=pair&63u; b=pair>>6;
      }
    }
    // Unsigned modular arithmetic is exact for signed mask differences.
    // The final binary-mask sum is bounded by 36864 * 65535 < 2^32.
    uint sign=active ? uint(coefficients[index]) : 0u;
    uint sumA=simd_sum(a*sign),sumB=simd_sum(b*sign);
    if (lane==0) {
      if (atomicOutput) {
        device atomic_uint* output=reinterpret_cast<device atomic_uint*>(partials);
        atomic_fetch_add_explicit(output+group.y*512+scan,sumA,memory_order_relaxed);
        atomic_fetch_add_explicit(output+group.y*512+scan+1,sumB,memory_order_relaxed);
      } else {
        uint address=(group.y*query.groups+group.x)*512+scan;
        if (accumulate) {
          partials[address]+=sumA; partials[address+1]+=sumB;
        } else {
          partials[address]=sumA; partials[address+1]=sumB;
        }
      }
    }
  }
}
kernel void tans_detector_partials(
    device const uint* payload [[buffer(0)]], device const uint* offsets [[buffer(1)]],
    device const uint* events [[buffer(2)]], device const uint* sparseOffsets [[buffer(3)]],
    device const uint* decoding [[buffer(4)]], device const uchar* models [[buffer(5)]],
    device const int* cacheMap [[buffer(6)]], device const uint* retainedRank [[buffer(7)]],
    device uint* partials [[buffer(8)]], constant TANSDetectorQuery& query [[buffer(9)]],
    device const uint* selected [[buffer(10)]], uint lane [[thread_index_in_simdgroup]],
    uint2 group [[threadgroup_position_in_grid]], device const int* coefficients [[buffer(11)]]) {
  tans_detector_reduce(payload,offsets,events,sparseOffsets,decoding,models,cacheMap,
    retainedRank,partials,query,selected,lane,group,coefficients,false,false);
}
struct TANSDetectorRecord {
  device const uint* payload [[id(0)]];
  device const uint* offsets [[id(1)]];
  device const uint* events [[id(2)]];
  device const uint* sparseOffsets [[id(3)]];
  device uint* output [[id(4)]];
};

// Reduced-work diagnostic only. Never used to return detector images.
// Dense mode reads each selected compressed stream verbatim, without ANS,
// sparse-event decoding, or a scientific reduction.
kernel void tans_submission_probe(
    device const TANSDetectorRecord* records [[buffer(0)]],
    device const uint* retainedRank [[buffer(1)]],
    device const uint* selected [[buffer(2)]],
    device atomic_uint* output [[buffer(3)]],
    constant uint* parameters [[buffer(4)]],
    uint2 index [[thread_position_in_grid]]) {
  uint columns=parameters[0], count=parameters[1], recordCount=parameters[2];
  if (index.y>=recordCount) return;
  uint slot=(parameters[3]+index.y)*4u;
  if (parameters[4]==1u) {
    if(index.x==0u)
      atomic_store_explicit(output+slot+3u,0x51A70000u+parameters[3]+index.y,memory_order_relaxed);
    return;
  }
  if (index.x>=count*32u) return;
  TANSDetectorRecord record=records[index.y];
  uint packet=index.x/count,q=selected[index.x%count];
  uint2 bounds=stream_bounds(record.offsets,packet*columns+retainedRank[q],columns,false);
  uint checksum=0;
  for(uint word=bounds.x;word<bounds.y;++word) checksum+=record.payload[word];
  atomic_fetch_add_explicit(output+slot,checksum,memory_order_relaxed);
  atomic_fetch_add_explicit(output+slot+1u,bounds.y-bounds.x,memory_order_relaxed);
  atomic_fetch_add_explicit(output+slot+2u,1u,memory_order_relaxed);
}
kernel void tans_detector_batch(
    device const TANSDetectorRecord* records [[buffer(0)]],
    device const uint* decoding [[buffer(1)]], device const uchar* models [[buffer(2)]],
    device const int* cacheMap [[buffer(3)]], device const uint* retainedRank [[buffer(4)]],
    device const uint* selected [[buffer(5)]], device const int* coefficients [[buffer(6)]],
    constant TANSDetectorQuery& base [[buffer(7)]], device const uint* modelOffsets [[buffer(8)]],
    uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  TANSDetectorRecord record=records[group.z];
  TANSDetectorQuery query=base;
  query.modelOffset=modelOffsets[group.z];
  tans_detector_reduce(record.payload,record.offsets,record.events,record.sparseOffsets,
    decoding,models,cacheMap,retainedRank,record.output,query,selected,lane,group.xy,
    coefficients,true,false);
}

// Experimental exact ILP variant: multiple independent streams per lane.
// Source records and 512-scan packet boundaries are unchanged. Only register
// state is added; never constructs a decoded 4D resident or persistent index.
constant uint tans_streams_per_lane [[function_constant(0)]];
constant bool tans_coalesced_output [[function_constant(1)]];
kernel void tans_detector_interleaved_batch(
    device const TANSDetectorRecord* records [[buffer(0)]],
    device const uint* decoding [[buffer(1)]], device const uchar* models [[buffer(2)]],
    device const int* cacheMap [[buffer(3)]], device const uint* retainedRank [[buffer(4)]],
    device const uint* selected [[buffer(5)]], device const int* coefficients [[buffer(6)]],
    constant TANSDetectorQuery& base [[buffer(7)]], device const uint* modelOffsets [[buffer(8)]],
    uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  TANSDetectorRecord record = records[group.z];
  uint cursor[4], end[4], state[4], available[4], model[4], sign[4];
  ulong reservoir[4];
  bool active[4];
  #pragma unroll
  for (uint j=0; j<tans_streams_per_lane; ++j) {
    uint index = (group.x*tans_streams_per_lane+j)*32u+lane;
    active[j] = index < base.selectedCount;
    cursor[j]=0; end[j]=0; state[j]=0; available[j]=0; model[j]=255u;
    reservoir[j]=0; sign[j]=0;
    if (active[j]) {
      uint q = selected[index];
      uint2 bounds = stream_bounds(record.offsets,
        group.y*base.retainedColumns+retainedRank[q],base.retainedColumns,false);
      cursor[j]=bounds.x; end[j]=bounds.y;
      model[j]=models[modelOffsets[group.z]+q];
      sign[j]=uint(coefficients[index]);
      if (model[j]!=255u) {
        uint header=record.payload[cursor[j]++];
        state[j]=header&1023u; available[j]=22; reservoir[j]=header>>10;
      }
    }
  }
  // Keep the packet loop rolled so ILP does not multiply a 256-step code
  // expansion. The inner independent-stream loop is specialized and unrolled.
  #pragma clang loop unroll(disable)
  for (uint batch=0;batch<512;batch+=32) {
    uint saved=0;
    #pragma clang loop unroll(disable)
    for (uint step=0;step<32;step+=2) {
    uint scan=batch+step;
    uint sumA=0,sumB=0;
    #pragma unroll
    for (uint j=0;j<tans_streams_per_lane;++j) {
      if (active[j]) {
        uint a,b;
        if (model[j]==255u) {
          uint pair=record.payload[cursor[j]++];
          a=pair&65535u; b=pair>>16;
        } else {
          uint pair=next_pair(record.payload,cursor[j],end[j],decoding+model[j]*1024u,
            state[j],reservoir[j],available[j]);
          a=pair&63u; b=pair>>6;
        }
        sumA+=a*sign[j]; sumB+=b*sign[j];
      }
    }
    sumA=simd_sum(sumA); sumB=simd_sum(sumB);
    if (tans_coalesced_output) {
      if (lane==step) saved=sumA;
      if (lane==step+1) saved=sumB;
    } else if (lane==0) {
      device atomic_uint* output=reinterpret_cast<device atomic_uint*>(record.output);
      atomic_fetch_add_explicit(output+group.y*512u+scan,sumA,memory_order_relaxed);
      atomic_fetch_add_explicit(output+group.y*512u+scan+1,sumB,memory_order_relaxed);
    }
    }
    if (tans_coalesced_output) {
      device atomic_uint* output=reinterpret_cast<device atomic_uint*>(record.output);
      atomic_fetch_add_explicit(output+group.y*512u+batch+lane,saved,memory_order_relaxed);
    }
  }
}

// Experimental model-homogeneous groups share one exact 1024-state table.
// Four SIMD groups process distinct source packets; no scan count is omitted.
// Explicit specialization gates. Ordinary shared-model behavior stays frozen.
constant bool tans_shared_pair_requested [[function_constant(2)]];
constant bool tans_shared_word32_requested [[function_constant(3)]];
constant uint tans_shared_threads_requested [[function_constant(4)]];
constant bool tans_shared_pair = is_function_constant_defined(tans_shared_pair_requested)
    ? tans_shared_pair_requested : false;
constant bool tans_shared_word32 = is_function_constant_defined(tans_shared_word32_requested)
    ? tans_shared_word32_requested : false;
constant uint tans_shared_threads = is_function_constant_defined(tans_shared_threads_requested)
    ? tans_shared_threads_requested : 128u;
constant bool tans_shared_refill16_requested [[function_constant(7)]];
constant bool tans_shared_refill16 = is_function_constant_defined(tans_shared_refill16_requested)
    ? tans_shared_refill16_requested : false;
constant bool tans_direct_table_requested [[function_constant(8)]];
constant bool tans_direct_table = is_function_constant_defined(tans_direct_table_requested)
    ? tans_direct_table_requested : false;
constant bool tans_zero_runs_requested [[function_constant(9)]];
constant bool tans_zero_runs = is_function_constant_defined(tans_zero_runs_requested)
    ? tans_zero_runs_requested : false;
constant bool tans_zero_arithmetic_requested [[function_constant(10)]];
constant bool tans_zero_arithmetic = is_function_constant_defined(tans_zero_arithmetic_requested)
    ? tans_zero_arithmetic_requested : false;
constant bool tans_mixed_tails_requested [[function_constant(11)]];
constant bool tans_mixed_tails = is_function_constant_defined(tans_mixed_tails_requested)
    ? tans_mixed_tails_requested : false;
constant bool tans_mixed_only_requested [[function_constant(12)]];
constant bool tans_mixed_only = is_function_constant_defined(tans_mixed_only_requested)
    ? tans_mixed_only_requested : false;
constant bool tans_signed_pair_requested [[function_constant(13)]];
constant bool tans_signed_pair = is_function_constant_defined(tans_signed_pair_requested)
    ? tans_signed_pair_requested : false;
constant bool tans_prepared_zero_runs_requested [[function_constant(14)]];
constant bool tans_prepared_zero_runs = is_function_constant_defined(tans_prepared_zero_runs_requested)
    ? tans_prepared_zero_runs_requested : false;
constant uint tans_pair_unroll_requested [[function_constant(15)]];
constant uint tans_pair_unroll = is_function_constant_defined(tans_pair_unroll_requested)
    ? tans_pair_unroll_requested : 1u;
constant bool tans_staged_reduction_requested [[function_constant(16)]];
constant bool tans_staged_reduction = is_function_constant_defined(tans_staged_reduction_requested)
    ? tans_staged_reduction_requested : false;
constant bool tans_prefetch_code_requested [[function_constant(17)]];
constant bool tans_prefetch_code = is_function_constant_defined(tans_prefetch_code_requested)
    ? tans_prefetch_code_requested : false;
constant uint tans_deferred_pairs_requested [[function_constant(18)]];
constant uint tans_deferred_pairs = is_function_constant_defined(tans_deferred_pairs_requested)
    ? tans_deferred_pairs_requested : 1u;
constant bool tans_bit_extract_requested [[function_constant(19)]];
constant bool tans_bit_extract = is_function_constant_defined(tans_bit_extract_requested)
    ? tans_bit_extract_requested : false;
constant bool tans_packet_major_requested [[function_constant(20)]];
constant bool tans_packet_major = is_function_constant_defined(tans_packet_major_requested)
    ? tans_packet_major_requested : false;
constant uint tans_pair_lookup_bits_requested [[function_constant(21)]];
constant uint tans_pair_lookup_bits = is_function_constant_defined(tans_pair_lookup_bits_requested)
    ? tans_pair_lookup_bits_requested : 0u;
// Exact packed transpose reduction for the plain shared32 decoder (six-bit
// lanes, coefficients in {-1,0,1}); the host leaves it off only for the frozen
// decoder experiments that keep the per-pair reductions below.
constant bool tans_transpose_reduction_requested [[function_constant(23)]];
constant bool tans_transpose_reduction = is_function_constant_defined(tans_transpose_reduction_requested)
    ? tans_transpose_reduction_requested : false;
// Grouped refill checks, three pairs per check, in the transposed decoders. A
// state read consumes at most 10 bits (host-validated table contract). At the
// first pair of a group of G <= 3 pairs the lane refills one word if fewer than
// 10*G bits remain (result: >= 32 >= 10*G bits, at most 29+32 = 61 < 64), so
// the G state reads need no further check. An escape literal at group position
// p first ensures 12 + 10*(G-1-p) <= 32 bits (at most 31+32 = 63 < 64),
// restoring the same guarantee for the rest of the group. The same bits are
// consumed in the same order as the per-pair check; only when the (divergent)
// refill branch is evaluated changes. K = 1 and 2 measured slower.
constant uint tans_refill_every = 3u;
// Compact launch: group.x indexes a host-built list of (record << 16 | model
// group) work items, so no threadgroup is launched for a padded group slot.
// Every (record, group, packet) triple is visited exactly once, as before.
constant bool tans_work_list_requested [[function_constant(32)]];
constant bool tans_work_list = is_function_constant_defined(tans_work_list_requested)
    ? tans_work_list_requested : false;
// Narrow mixed groups (host-verified <=32/P active six-bit lanes, coefficients
// in {-1,0,1}): each SIMD group decodes P packets, 32/P lanes per packet. The
// exact per-packet reduction only spans that packet's lane segment.
constant uint tans_lane_packets_requested [[function_constant(33)]];
constant uint tans_lane_packets = is_function_constant_defined(tans_lane_packets_requested)
    ? tans_lane_packets_requested : 1u;

// One butterfly level of the exact transposed batch reduction. The lane whose
// selector bit is clear keeps the low-index word and receives its partner's
// copy of that same word; the other lane keeps the high-index word.
inline uint tans_butterfly(uint low, uint high, uint bit, ushort offset) {
  uint keep=bit!=0u?high:low, send=bit!=0u?low:high;
  return keep+simd_shuffle_xor(send,offset);
}

// Two exact non-escape transitions; invalid entries retain the original reader.
// Compact payload: second pair[11:0], next state[21:12], consumed bits[27:22],
// valid[28]. The first pair is already in the original state table.
// This contains codebook metadata only, never decoded source counts.
kernel void tans_prepare_pair_lookup(device const uint* original [[buffer(0)]],
    device uint* prepared [[buffer(1)]], constant uint2& info [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  if (index>=info.x) return;
  uint width=info.y,entry=index>>width,look=index&((1u<<width)-1u);
  uint first=original[entry],bits0=(first>>12)&15u;
  uint result=0;
  if (bits0<=width && (first&4095u)!=4095u) {
    uint mid=(first>>16)+(look&((1u<<bits0)-1u));
    if (mid<1024u) {
      uint second=original[(entry&~1023u)+mid],bits1=(second>>12)&15u;
      if (bits0+bits1<=width && (second&4095u)!=4095u) {
        uint next=(second>>16)+((look>>bits0)&((1u<<bits1)-1u));
        if (next<1024u)
          result=(second&4095u)|(next<<12)|((bits0+bits1)<<22)|(1u<<28);
      }
    }
  }
  prepared[index]=result;
}

// Exact query-local lookup metadata. No counts are decoded or approximated here.
kernel void tans_prepare_zero_runs(device const uint* original [[buffer(0)]],
    device uint* prepared [[buffer(1)]], constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  if (index>=count) return;
  uint code=original[index];
  if ((code&65535u)==0u) {
    uint modelBase=(index/1024u)*1024u,next=index%1024u,run=0;
    while (run<63u) {
      uint transition=original[modelBase+next];
      if ((transition&65535u)!=0u) break;
      next=transition>>16;++run;
    }
    code=(next<<16)|(run<<26);
  }
  prepared[index]=code;
}

kernel void tans_detector_shared_model_batch(
    device const TANSDetectorRecord* records [[buffer(0)]],
    device const uint* decoding [[buffer(1)]],
    device const uchar* models [[buffer(2)]],
    device const uint* retainedRank [[buffer(4)]],
    constant TANSDetectorQuery& base [[buffer(7)]],
    device const uint* modelOffsets [[buffer(8)]],
    device const uint2* groups [[buffer(9)]],
    device const uint* selected [[buffer(10)]],
    device const int* coefficients [[buffer(11)]],
    device const uint* groupOffsets [[buffer(12)]],
    device const uint* groupCounts [[buffer(13)]],
    device const uint* pairLookup [[buffer(14)]],
    device const uint* workItems [[buffer(15)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint threadID [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  uint modelGroup=tans_packet_major?group.y:group.x;
  uint packetGroup=tans_packet_major?group.x:group.y;
  uint recordIndex=group.z;
  if (tans_work_list) {
    uint item=workItems[group.x];
    recordIndex=item>>16;modelGroup=item&65535u;packetGroup=group.y;
  } else if (modelGroup>=groupCounts[group.z]) return;
  uint2 descriptor=groups[groupOffsets[recordIndex]+modelGroup];
  uint model=descriptor.x,index=descriptor.y+(lane&(32u/tans_lane_packets-1u));
  bool mixed=tans_mixed_only || (tans_mixed_tails && model==256u);
  uint q=selected[index],sign=uint(coefficients[index]);
  bool active=q!=0xffffffffu;
  if (mixed) model=active?uint(models[modelOffsets[recordIndex]+q]):255u;
  // Uniform SIMD decision: literal uint16 must never enter the six-bit bound.
  bool signedPair=tans_signed_pair && !simd_any(active && model==255u);
  threadgroup uint table[1024];
  // Four independent SIMD32 scratch tiles. Each word preserves two UInt16
  // counts; no persistent raw volume or scientific narrowing is introduced.
  threadgroup uint stagedPairs[2176];
  if (!tans_direct_table && !mixed && model!=255u) {
    for (uint i=threadID;i<1024;i+=tans_shared_threads) {
      uint code=decoding[model*1024u+i];
      if (tans_zero_runs && !tans_prepared_zero_runs && (code&65535u)==0u) {
        uint next=i,run=0;
        // Zero symbols consuming zero bits follow a deterministic state chain.
        // Encode at most63 pairs in otherwise-unused high bits of this entry.
        while (run<63u) {
          uint transition=decoding[model*1024u+next];
          if ((transition&65535u)!=0u) break;
          next=transition>>16;++run;
        }
        code=(next<<16)|(run<<26);
      }
      table[i]=code;
    }
  }
  if (!tans_direct_table && !mixed) threadgroup_barrier(mem_flags::mem_threadgroup);
  uint packet=(packetGroup*(tans_shared_threads/32u)+threadID/32u)*tans_lane_packets
    +lane/(32u/tans_lane_packets);
  TANSDetectorRecord record=records[recordIndex];
  uint cursor=0,end=0,state=0,available=0,lookahead=0;
  ulong reservoir=0;
  uint reservoir32=0,halfCursor=0,zeroPairs=0;
  uint pendingPair=0;
  bool hasPendingPair=false;
  if (active) {
    uint2 bounds=stream_bounds(record.offsets,
      packet*base.retainedColumns+retainedRank[q],base.retainedColumns,false);
    cursor=bounds.x;end=bounds.y;
    if (model!=255u) {
      uint header=record.payload[cursor++];
      state=header&1023u;available=22;reservoir=header>>10;reservoir32=header>>10;
      lookahead=cursor<end?record.payload[cursor]:0;
      halfCursor=cursor*2u;
    }
  }
  // Carry the next exact transition across the output-reduction work. No
  // extra source reads, state approximation, or persisted decoded counts.
  uint prefetchedCode=0;
  if (tans_prefetch_code && active && model!=255u)
    prefetchedCode=mixed?decoding[model*1024u+state]:table[state];
  // Exact reduction-free batch accumulate. Each lane packs its signed six-bit
  // pair products as A + 65536*B modulo 2^32, and the SIMD group combines the
  // 16 packed words of a 32-scan batch with a shuffle butterfly as they are
  // decoded (16 shuffles per batch instead of 32 full simd_sum reductions).
  // Per-lane decode order and consumed bits are unchanged. Bounds: with
  // coefficients in {-1,0,1}, |A|,|B| <= 32*63 = 2016 < 2^15 per scan, so both
  // halves are recovered exactly. Literal uint16 lanes or any other
  // coefficient keep the wide per-pair reduction below (uniform SIMD choice).
  if (tans_lane_packets>1u && simd_any(active && (model==255u || sign+1u>2u))) {
    // Exact per-lane fallback for a narrow group the host should never route
    // here (a literal lane or a wide coefficient): every scan is still added.
    device atomic_uint* output=reinterpret_cast<device atomic_uint*>(record.output);
    if (active) {
      for (uint scan=0;scan<512;scan+=2) {
        uint a,b;
        if (model==255u) { uint pair=record.payload[cursor++];a=pair&65535u;b=pair>>16; }
        else {
          uint pair=next_pair(record.payload,cursor,end,decoding+model*1024u,state,
            reservoir,available);
          a=pair&63u;b=pair>>6;
        }
        atomic_fetch_add_explicit(output+packet*512u+scan,a*sign,memory_order_relaxed);
        atomic_fetch_add_explicit(output+packet*512u+scan+1u,b*sign,memory_order_relaxed);
      }
    }
    return;
  }
  if (tans_transpose_reduction
      && (tans_lane_packets>1u || !simd_any(active && (model==255u || sign+1u>2u)))) {
    // Butterfly levels span only this packet's 32/P-lane segment: offsets
    // 16/P, 8/P, ... 2, then xor 1. P=1 is the original 16,8,4,2,1 chain.
    const ushort o0=ushort(16u/tans_lane_packets);
    uint bit4=(lane&uint(o0))!=0u?1u:0u,bit3=(lane&uint(o0>>1))!=0u?1u:0u;
    uint bit2=(lane&uint(o0>>2))!=0u?1u:0u,bit1=(lane&uint(o0>>3))!=0u?1u:0u;
    // After the butterfly this lane owns pair word j of its emission block.
    uint word=tans_lane_packets==1u?(bit4|(bit3<<1)|(bit2<<2)|(bit1<<3))
      :(tans_lane_packets==2u?(bit4|(bit3<<1)|(bit2<<2)):(bit4|(bit3<<1)));
    uint scanInBatch=2u*word+(lane&1u);
    device atomic_uint* output=reinterpret_cast<device atomic_uint*>(record.output);
    #pragma clang loop unroll(disable)
    for (uint batch=0;batch<512;batch+=32) {
      uint level[4]={0u,0u,0u,0u};
      #pragma clang loop unroll(full)
      for (uint step=0;step<16;++step) {
        uint a=0,b=0;
        // Grouped checks restart every batch; step is a compile-time constant.
        const uint groupStart=step-step%tans_refill_every;
        const uint groupSize=min(tans_refill_every,16u-groupStart);
        const uint position=step-groupStart;
        if (active) {
          if (position==0u && available<10u*groupSize) {
            reservoir|=ulong(lookahead)<<available;available+=32;
            if (cursor<end) ++cursor;
            lookahead=cursor<end?record.payload[cursor]:0;
          }
          uint code=tans_prefetch_code?prefetchedCode
            :(mixed?decoding[model*1024u+state]:table[state]);
          uint bits=(code>>12)&15u;
          uint low=uint(reservoir)&((1u<<bits)-1u);
          reservoir>>=bits;available-=bits;state=(code>>16)+low;
          uint pair=code&4095u;
          if (pair==4095u) {
            if (available<12u+10u*(groupSize-1u-position)) {
              reservoir|=ulong(lookahead)<<available;available+=32;
              if (cursor<end) ++cursor;
              lookahead=cursor<end?record.payload[cursor]:0;
            }
            pair=uint(reservoir)&4095u;reservoir>>=12;available-=12;
          }
          a=pair&63u;b=pair>>6;
          if (tans_prefetch_code)
            prefetchedCode=mixed?decoding[model*1024u+state]:table[state];
        }
        uint packed=(a|(b<<16))*sign;
        if ((step&1u)==0u) { level[0]=packed; continue; }
        packed=tans_butterfly(level[0],packed,bit4,o0);
        if ((step&2u)==0u) { level[1]=packed; continue; }
        packed=tans_butterfly(level[1],packed,bit3,o0>>1);
        if (tans_lane_packets<=2u) {
          if ((step&4u)==0u) { level[2]=packed; continue; }
          packed=tans_butterfly(level[2],packed,bit2,o0>>2);
          if (tans_lane_packets==1u) {
            if ((step&8u)==0u) { level[3]=packed; continue; }
            packed=tans_butterfly(level[3],packed,bit1,o0>>3);
          }
        }
        packed+=simd_shuffle_xor(packed,ushort(1));
        int sumA=int(short(packed&65535u));
        int sumB=int(packed-uint(sumA))>>16;
        uint saved=(lane&1u)!=0u?uint(sumB):uint(sumA);
        // Emission block of 16/P steps ending at this step.
        uint blockScan=2u*(step+1u-16u/tans_lane_packets);
        atomic_fetch_add_explicit(output+packet*512u+batch+blockScan+scanInBatch,saved,
          memory_order_relaxed);
      }
    }
    return;
  }
  #pragma clang loop unroll(disable)
  for (uint batch=0;batch<512;batch+=32) {
    uint saved=0;
    #pragma clang loop unroll(disable)
    for (uint stepBase=0;stepBase<32;stepBase+=2*tans_pair_unroll*tans_deferred_pairs) {
      #pragma clang loop unroll(full)
      for (uint pairIndex=0;pairIndex<tans_pair_unroll;++pairIndex) {
        uint2 decoded[8];
        // Decode exact pairs into registers before collective reductions.
        // This changes scheduling only, not packet or scientific coverage.
        #pragma clang loop unroll(full)
        for (uint deferred=0;deferred<tans_deferred_pairs;++deferred) {
          uint a=0,b=0;
          uint zeroCode=tans_zero_arithmetic && active && model!=255u ? table[state] : 1u;
          if (active) {
            if (model==255u) {
              uint pair=record.payload[cursor++];
              a=pair&65535u;b=pair>>16;
            } else if (tans_shared_refill16) {
              // Identical little-endian source bits, but refill16 keeps at most27
              // available bits. No64-bit shifts or persistent repacked source.
              device const ushort* halves=reinterpret_cast<device const ushort*>(record.payload);
              if (available<12) {
                uint next=halfCursor<end*2u?uint(halves[halfCursor++]):0u;
                reservoir32|=next<<available;available+=16;
              }
              uint code=(tans_direct_table||mixed)?decoding[model*1024u+state]:table[state],bits=(code>>12)&15u;
              uint low=reservoir32&((1u<<bits)-1u);
              reservoir32>>=bits;available-=bits;state=(code>>16)+low;
              uint pair=code&4095u;
              if (pair==4095u) {
                if (available<12) {
                  uint next=halfCursor<end*2u?uint(halves[halfCursor++]):0u;
                  reservoir32|=next<<available;available+=16;
                }
                pair=reservoir32&4095u;reservoir32>>=12;available-=12;
              }
              a=pair&63u;b=pair>>6;
            } else if (tans_shared_word32) {
              uint code=(tans_direct_table||mixed)?decoding[model*1024u+state]:table[state],bits=(code>>12)&15u;
              uint low=read_bits32(record.payload,cursor,end,bits,reservoir32,available);
              state=(code>>16)+low;
              uint pair=code&4095u;
              if (pair==4095u)
                pair=read_bits32(record.payload,cursor,end,12,reservoir32,available);
              a=pair&63u;b=pair>>6;
            } else if (tans_zero_arithmetic && (zeroCode&65535u)==0u) {
              // Exact zero pair with no consumed input bits: only advance state.
              state=zeroCode>>16;
            } else if (tans_zero_runs && zeroPairs>0u) {
              --zeroPairs;
            } else if (tans_pair_lookup_bits>0u && hasPendingPair) {
              a=pendingPair&63u;b=pendingPair>>6;hasPendingPair=false;
            } else {
              // Refill can begin independently of the next state-table lookup.
              if (available<12) {
                reservoir|=ulong(lookahead)<<available;available+=32;
                if (cursor<end) ++cursor;
                lookahead=cursor<end?record.payload[cursor]:0;
              }
              bool usedPairLookup=false;
              uint position=batch+stepBase+(pairIndex*tans_deferred_pairs+deferred)*2u;
              if (tans_pair_lookup_bits>0u && position<510u) {
                uint key=((model*1024u+state)<<tans_pair_lookup_bits)
                  |(uint(reservoir)&((1u<<tans_pair_lookup_bits)-1u));
                uint folded=pairLookup[key];
                if ((folded>>28)!=0u) {
                  uint first=((tans_direct_table||mixed)
                    ?decoding[model*1024u+state]:table[state])&4095u;
                  uint consumed=(folded>>22)&63u;
                  reservoir>>=consumed;available-=consumed;
                  state=(folded>>12)&1023u;
                  pendingPair=folded&4095u;hasPendingPair=true;
                  a=first&63u;b=first>>6;usedPairLookup=true;
                }
              }
              if (!usedPairLookup) {
              uint code=tans_prefetch_code?prefetchedCode:(tans_zero_arithmetic?zeroCode:((tans_direct_table||mixed)?decoding[model*1024u+state]:table[state]));
              uint bits=(code>>12)&15u;
              if (tans_zero_runs && (code>>26)!=0u) {
                zeroPairs=(code>>26)-1u;state=(code>>16)&1023u;
              } else {
                // Same low-bit field; probe the native integer bit-edit intrinsic.
                uint low=tans_bit_extract?extract_bits(uint(reservoir),0u,bits)
                    :(uint(reservoir)&((1u<<bits)-1u));
                reservoir>>=bits;available-=bits;state=(code>>16)+low;
                uint pair=code&4095u;
                if (pair==4095u) {
                  if (available<12) {
                    reservoir|=ulong(lookahead)<<available;available+=32;
                    if (cursor<end) ++cursor;
                    lookahead=cursor<end?record.payload[cursor]:0;
                  }
                  pair=uint(reservoir)&4095u;reservoir>>=12;available-=12;
                }
                a=pair&63u;b=pair>>6;
              }
              }
            }
          }
          if (tans_prefetch_code && active && model!=255u)
            prefetchedCode=mixed?decoding[model*1024u+state]:table[state];
          decoded[deferred]=uint2(a,b);
        }
        #pragma clang loop unroll(full)
        for (uint deferred=0;deferred<tans_deferred_pairs;++deferred) {
          uint step=stepBase+(pairIndex*tans_deferred_pairs+deferred)*2u;
          uint a=decoded[deferred].x,b=decoded[deferred].y;
          if (tans_staged_reduction) {
            stagedPairs[(threadID/32u)*544u+lane*17u+step/2u]=a|(b<<16);
          } else if (signedPair) {
            // For signed six-bit counts, A and B are each within [-2016,2016].
            // Sum A + 65536*B modulo2^32, then undo the low-half borrow.
            // No sign-based regrouping or narrowed scientific output is needed.
            uint packed=simd_sum((a|(b<<16))*sign);
            int sumA=int(short(packed&65535u));
            int sumB=int(short((packed>>16)+uint(sumA<0)));
            if (lane==step) saved=uint(sumA);
            if (lane==step+1) saved=uint(sumB);
          } else if (tans_shared_pair && model!=255u) {
            // Each half <= 32*63 = 2016, hence no cross-half carry.
            // Host groups only identical nonzero signs. Apply it after reduction.
            uint packed=simd_sum(a|(b<<16));
            uint uniformSign=simd_broadcast_first(sign);
            if ((lane>>1)==(step>>1))
              saved=((packed>>((lane&1u)*16u))&65535u)*uniformSign;
          } else {
            // Literal uint16 is not bounded to six bits; keep wide sums.
            uint sumA=simd_sum(a*sign),sumB=simd_sum(b*sign);
            if (lane==step) saved=sumA;
            if (lane==step+1) saved=sumB;
          }
        }
      }
    }
    if (tans_staged_reduction) {
      simdgroup_barrier(mem_flags::mem_threadgroup);
      uint start=(threadID/32u)*544u+lane/2u;
      saved=0;
      for (uint column=0;column<32u;++column) {
        uint pair=stagedPairs[start+column*17u];
        uint value=(pair>>((lane&1u)*16u))&65535u;
        saved+=value*uint(coefficients[descriptor.y+column]);
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    device atomic_uint* output=reinterpret_cast<device atomic_uint*>(record.output);
    atomic_fetch_add_explicit(output+packet*512u+batch+lane,saved,memory_order_relaxed);
  }
}

// Packet-owner exact detector kernel. One SIMD group owns one (record, 512-scan
// packet): it loops over every dense 32-lane model group of that record, adds
// the packet's sparse events, and writes each of its 512 output scans once (no
// device atomics: no other SIMD group writes these scans in this dispatch).
// Per-lane decode order, consumed bits and the modular integer arithmetic are
// those of tans_detector_shared_model_batch with transpose reduction; only the
// ownership of the sums moves. Measured on all 66 acquisitions and fixed here:
// four SIMD groups per threadgroup, device-resident tables, per-scan sums in
// threadgroup memory, the packed decode table (no per-step activity
// predicate), balanced sparse events with rank bytes, staged steps, host cache
// slots and a ballot column search, and the owner-written base.
// Must equal ownerSimds at the dispatch in MetalTANSResidentSeries.swift.
constant uint tans_owner_simds = 4u;
struct TANSOwnerBytes { device const uchar* values; };
// Rank bytes of one record: bytes[w] = popcount of flag words (w & ~7) ... w-1
// (at most 7 * 32 = 224), so the global count rank of event `at` is
// ranks[at >> 8] + bytes[at >> 5] + the flags below `at` in its own word.
// Built once per series from the immutable flags; words at or beyond the
// record's flag-word count are never written or read.
kernel void tans_owner_rank_bytes_build(device const uint* events [[buffer(0)]],
    device uchar* bytes [[buffer(1)]], uint w [[thread_position_in_grid]]) {
  uint pwords=events[1],fwords=events[2];
  if (w>=fwords) return;
  device const uint* flags=events+4+pwords;
  uint sum=0;
  for (uint j=w&~7u;j<w;++j) sum+=popcount(flags[j]);
  bytes[w]=uchar(sum);
}
// Flag-word count (sparse header word 2) of one record, for the host layout.
kernel void tans_owner_sparse_header(device const uint* events [[buffer(0)]],
    device uint* word [[buffer(1)]], uint id [[thread_position_in_grid]]) {
  if (id==0u) word[0]=events[2];
}
// Column of balanced event window + lane: the largest lane k with
// start_k <= window + lane (a zero-length lane shares its start with the next
// lane, so never wins). Each column publishes d = start - window (0 before the
// window) through five bit-plane ballots and every lane counts the columns
// inside the window with d <= lane by a bitwise <= comparator.
inline uint tans_owner_event_column(uint start, uint window, uint lane) {
  bool inside=start<window+32u;
  uint d=start>window?start-window:0u;
  uint eq=uint(ulong(simd_ballot(inside))),lt=0u;
  #pragma clang loop unroll(full)
  for (uint b=5u;b>0u;--b) {
    uint plane=uint(ulong(simd_ballot(inside && ((d>>(b-1u))&1u)!=0u)));
    uint laneBit=0u-((lane>>(b-1u))&1u);
    lt|=eq&~plane&laneBit;
    eq&=~(plane^laneBit);
  }
  return popcount(lt|eq)-1u;
}
struct TANSOwnerBase { device const uint* values; };
// baseMode: 0 add onto an initialized output (tile or atlas fields), 1 store
// zero + sums, 2 store the record's seed image slice from `bases` + sums.
struct TANSOwnerQuery { uint retainedColumns, sparseColumns, sparseCount, baseMode; };
struct TANSOwnerLane {
  uint cursor, end, state, available, lookahead, code;
  ulong reservoir;
};
inline void tans_owner_refill(device const uint* payload, thread TANSOwnerLane& s) {
  s.reservoir|=ulong(s.lookahead)<<s.available;s.available+=32;
  if (s.cursor<s.end) ++s.cursor;
  s.lookahead=s.cursor<s.end?payload[s.cursor]:0;
}
// One exact tANS pair from the unpacked table with the next entry prefetched:
// the shared-model decoder's reader with its per-pair 12-bit refill check.
inline uint tans_owner_pair(device const uint* payload, device const uint* model,
    thread TANSOwnerLane& s) {
  if (s.available<12) tans_owner_refill(payload,s);
  uint code=s.code;
  uint bits=(code>>12)&15u;
  uint low=uint(s.reservoir)&((1u<<bits)-1u);
  s.reservoir>>=bits;s.available-=bits;s.state=(code>>16)+low;
  uint pair=code&4095u;
  if (pair==4095u) {
    if (s.available<12) tans_owner_refill(payload,s);
    pair=uint(s.reservoir)&4095u;s.reservoir>>=12;s.available-=12;
  }
  s.code=model[s.state];
  return pair;
}
// Literal uint16 lanes or other coefficients: full-width modular sums. The
// lane keeps the scan it owns in the transpose layout, so ownership is fixed.
inline uint tans_owner_wide_batch(device const uint* payload, device const uint* model,
    thread TANSOwnerLane& s, bool active, bool literal, uint sign, uint scanInBatch) {
  uint result=0;
  #pragma clang loop unroll(full)
  for (uint step=0;step<16;++step) {
    uint a=0,b=0;
    if (active) {
      if (literal) {
        uint pair=payload[s.cursor++];a=pair&65535u;b=pair>>16;
      } else {
        uint pair=tans_owner_pair(payload,model,s);
        a=pair&63u;b=pair>>6;
      }
    }
    uint sumA=simd_sum(a*sign),sumB=simd_sum(b*sign);
    if (scanInBatch==2u*step) result=sumA;
    if (scanInBatch==2u*step+1u) result=sumB;
  }
  return result;
}
// One dense model group as seen by this lane: its model (255 for a literal or
// inactive mixed lane), packed table (inactive lanes read model 0, never
// added), coefficient (0 when inactive) and whether the whole SIMD group is
// six-bit with coefficients in {-1,0,1} (otherwise the wide path decodes it).
struct TANSOwnerGroup {
  device const uint* table;
  uint q, sign, model;
  bool active, fast;
};
inline TANSOwnerGroup tans_owner_group(uint2 d, uint lane,
    device const uint* selected, device const int* coefficients, device const uchar* models,
    uint modelBase, device const uint* packedTable) {
  bool mixed=d.x==256u;
  uint index=d.y+lane;
  uint q=selected[index],sign=uint(coefficients[index]);
  bool active=q!=0xffffffffu;
  uint model=mixed?(active?uint(models[modelBase+q]):255u):d.x;
  TANSOwnerGroup f;
  f.fast=!simd_any(active && (model==255u || sign+1u>2u));
  f.table=packedTable+(model<255u?model:0u)*1024u;
  f.q=q;f.sign=active?sign:0u;f.model=model;f.active=active;
  return f;
}
// Stream header and first packed entry of one lane of a fast group. An
// inactive lane decodes an empty stream (cursor = end = 0, zero bits) through
// its group's table with coefficient 0, so the step needs no predicate.
inline TANSOwnerLane tans_owner_fast_lane(TANSDetectorRecord rec,
    device const uint* retainedRank, uint retainedColumns, uint packet, TANSOwnerGroup f) {
  TANSOwnerLane s;
  s.cursor=0;s.end=0;s.state=0;s.available=0;s.lookahead=0;s.reservoir=0;
  if (f.active) {
    uint2 bounds=stream_bounds(rec.offsets,packet*retainedColumns+retainedRank[f.q],
      retainedColumns,false);
    s.cursor=bounds.x;s.end=bounds.y;
    uint header=rec.payload[s.cursor++];
    s.state=header&1023u;s.available=22;s.reservoir=header>>10;
    s.lookahead=s.cursor<s.end?rec.payload[s.cursor]:0;
  }
  s.code=f.table[s.state];
  return s;
}
// One exact pair of one lane as A + 65536*B from the packed table (host-built
// from the validated codebook, metadata only): a[5:0] | state bits[9:6] |
// escape[10] | b[21:16] | next base[31:22]; escape entries carry a = b = 0.
// Refill points and thresholds are the grouped checks of tans_refill_every;
// the same stream bits are consumed in the same order.
inline uint tans_owner_fast_step(device const uint* payload, thread const TANSOwnerGroup& f,
    thread TANSOwnerLane& s, uint groupSize, uint position) {
  if (position==0u && s.available<10u*groupSize) tans_owner_refill(payload,s);
  uint e=s.code;
  uint bits=extract_bits(e,6u,4u);
  uint low=uint(s.reservoir)&((1u<<bits)-1u);
  s.reservoir>>=bits;s.available-=bits;
  s.state=(e>>22)+low;
  uint packed=e&0x003F003Fu;
  if ((e&0x400u)!=0u) {
    if (s.available<12u+10u*(groupSize-1u-position)) tans_owner_refill(payload,s);
    uint lit=uint(s.reservoir)&4095u;s.reservoir>>=12;s.available-=12;
    packed=(lit&63u)|((lit>>6)<<16);
  }
  s.code=f.table[s.state];
  return packed;
}
// One 32-scan batch of a fast group: the packed A + 65536*B butterfly of the
// shared-model decoder (|A|,|B| <= 2016 < 2^15). Returns this lane's exact
// signed sum for scan 2*word+(lane&1) of the batch as a modular uint32.
inline uint tans_owner_fast_batch(device const uint* payload, thread TANSOwnerLane& s,
    thread const TANSOwnerGroup& f, uint bit4, uint bit3, uint bit2, uint bit1, bool odd) {
  uint level0=0,level1=0,level2=0,level3=0,result=0;
  #pragma clang loop unroll(full)
  for (uint step=0;step<16;++step) {
    const uint groupStart=step-step%tans_refill_every;
    const uint groupSize=min(tans_refill_every,16u-groupStart);
    const uint position=step-groupStart;
    uint packed=tans_owner_fast_step(payload,f,s,groupSize,position)*f.sign;
    if ((step&1u)==0u) { level0=packed; continue; }
    packed=tans_butterfly(level0,packed,bit4,16);
    if ((step&2u)==0u) { level1=packed; continue; }
    packed=tans_butterfly(level1,packed,bit3,8);
    if ((step&4u)==0u) { level2=packed; continue; }
    packed=tans_butterfly(level2,packed,bit2,4);
    if ((step&8u)==0u) { level3=packed; continue; }
    packed=tans_butterfly(level3,packed,bit1,2);
    packed+=simd_shuffle_xor(packed,ushort(1));
    int sumA=int(short(packed&65535u));
    int sumB=int(packed-uint(sumA))>>16;
    result=odd?uint(sumB):uint(sumA);
  }
  return result;
}
kernel void tans_detector_packet_owner_batch(
    device const TANSDetectorRecord* records [[buffer(0)]],
    device const uint* decoding [[buffer(1)]],
    device const uchar* models [[buffer(2)]],
    device const uint* retainedRank [[buffer(4)]],
    constant TANSOwnerQuery& query [[buffer(7)]],
    device const uint* modelOffsets [[buffer(8)]],
    device const uint2* groups [[buffer(9)]],
    device const uint* selected [[buffer(10)]],
    device const int* coefficients [[buffer(11)]],
    device const uint* groupOffsets [[buffer(12)]],
    device const uint* groupCounts [[buffer(13)]],
    device const uint* sparseSlots [[buffer(14)]],
    device const int* sparseCoefficients [[buffer(15)]],
    device const uint* packedTable [[buffer(16)]],
    device const TANSOwnerBase* bases [[buffer(17)]],
    device const TANSOwnerBytes* rankTables [[buffer(18)]],
    threadgroup uint* scratch [[threadgroup(0)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint threadID [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  uint record=tg.y,simd=threadID/32u,packet=tg.x*tans_owner_simds+simd;
  TANSDetectorRecord rec=records[record];
  uint first=groupOffsets[record],count=groupCounts[record],modelBase=modelOffsets[record];
  uint bit4=(lane>>4)&1u,bit3=(lane>>3)&1u,bit2=(lane>>2)&1u,bit1=(lane>>1)&1u;
  uint word=bit4|(bit3<<1)|(bit2<<2)|(bit1<<3);
  uint scanInBatch=2u*word+(lane&1u);
  bool odd=(lane&1u)!=0u;
  // This SIMD group's 512 per-scan sums; lane-owned words during the dense loop.
  threadgroup uint* tgSums=scratch+simd*512u;
  #pragma clang loop unroll(full)
  for (uint k=0;k<16;++k) tgSums[k*32u+lane]=0u;
  simdgroup_barrier(mem_flags::mem_threadgroup);
  #pragma clang loop unroll(disable)
  for (uint g=0;g<count;++g) {
    TANSOwnerGroup f=tans_owner_group(groups[first+g],lane,selected,coefficients,models,
      modelBase,packedTable);
    if (f.fast) {
      TANSOwnerLane s=tans_owner_fast_lane(rec,retainedRank,query.retainedColumns,packet,f);
      #pragma clang loop unroll(disable)
      for (uint batch=0;batch<16;++batch)
        tgSums[batch*32u+scanInBatch]+=tans_owner_fast_batch(rec.payload,s,f,bit4,bit3,bit2,bit1,odd);
      continue;
    }
    device const uint* modelTable=decoding+f.model*1024u;
    TANSOwnerLane s;
    s.cursor=0;s.end=0;s.state=0;s.available=0;s.lookahead=0;s.code=0;s.reservoir=0;
    if (f.active) {
      uint2 bounds=stream_bounds(rec.offsets,
        packet*query.retainedColumns+retainedRank[f.q],query.retainedColumns,false);
      s.cursor=bounds.x;s.end=bounds.y;
      if (f.model!=255u) {
        uint header=rec.payload[s.cursor++];
        s.state=header&1023u;s.available=22;s.reservoir=header>>10;
        s.lookahead=s.cursor<s.end?rec.payload[s.cursor]:0;
        s.code=modelTable[s.state];
      }
    }
    #pragma clang loop unroll(disable)
    for (uint batch=0;batch<16;++batch)
      tgSums[batch*32u+scanInBatch]+=
        tans_owner_wide_batch(rec.payload,modelTable,s,f.active,f.model==255u,f.sign,scanInBatch);
  }
  // Sparse events of this packet, balanced across the SIMD group: per round of
  // 32 selected sparse columns (host cache slots) the group takes an exclusive
  // prefix of the stream lengths and all 32 lanes walk the concatenated events
  // together, two 32-event steps per iteration (every load that depends only
  // on the event index first, then the count values of flagged events, then
  // the adds). Every event of every selected stream is added exactly once;
  // count and position decoding is sparse_event's with the rank bytes; only
  // the order of the exact threadgroup atomic adds differs.
  simdgroup_barrier(mem_flags::mem_threadgroup);
  threadgroup atomic_uint* sums=reinterpret_cast<threadgroup atomic_uint*>(tgSums);
  device const uint* data=rec.events;
  uint pwords=data[1],fwords=data[2],rwords=data[3];
  device const uint* positions=data+4;
  device const uint* flags=positions+pwords;
  device const uint* ranks=flags+fwords;
  device const uchar* values=reinterpret_cast<device const uchar*>(ranks+rwords);
  device const uchar* rankBytes=rankTables[record].values;
  // Uniform loops: every lane runs every shuffle and ballot.
  for (uint round=0;round<query.sparseCount;round+=32u) {
    uint c=round+lane,begin=0,length=0,sign=0;
    if (c<query.sparseCount) {
      sign=uint(sparseCoefficients[c]);
      uint2 bounds=stream_bounds(rec.sparseOffsets,packet*query.sparseColumns+sparseSlots[c],
        query.sparseColumns,true);
      begin=bounds.x;length=bounds.y-bounds.x;
    }
    uint start=simd_prefix_exclusive_sum(length);
    uint total=simd_shuffle(start+length,ushort(31));
    uint delta=begin-start;
    for (uint e0=0;e0<total;e0+=64u) {
      uint at[2],coefficient[2],low[2],high[2],flag[2],rank[2];
      bool valid[2];
      #pragma clang loop unroll(full)
      for (uint i=0;i<2u;++i) {
        at[i]=0u;coefficient[i]=0u;valid[i]=false;
        uint window=e0+i*32u;
        // Uniform: this step holds at least one event.
        if (window<total) {
          uint e=window+lane;
          ushort k=ushort(tans_owner_event_column(start,window,lane));
          at[i]=e+simd_shuffle(delta,k);coefficient[i]=simd_shuffle(sign,k);
          valid[i]=e<total;
        }
      }
      #pragma clang loop unroll(full)
      for (uint i=0;i<2u;++i) {
        low[i]=0u;high[i]=0u;flag[i]=0u;rank[i]=0u;
        if (valid[i]) {
          // A field crossing into word + 1 implies word + 1 < pwords.
          uint word=(at[i]*9u)>>5;
          low[i]=positions[word];
          high[i]=word+1u<pwords?positions[word+1u]:0u;
          flag[i]=flags[at[i]>>5];
          rank[i]=ranks[at[i]>>8]+uint(rankBytes[at[i]>>5]);
        }
      }
      #pragma clang loop unroll(full)
      for (uint i=0;i<2u;++i) {
        if (valid[i]) {
          // (joined >> shift) & 511 with joined = low | high << 32: the high
          // word enters at bit 32 - shift (not at all for shift 0).
          uint shift=(at[i]*9u)&31u,bit=at[i]&31u;
          uint position=((low[i]>>shift)|((high[i]<<1)<<(31u-shift)))&511u;
          uint countValue=1u;
          if ((flag[i]>>bit)&1u)
            countValue=values[rank[i]+popcount(flag[i]&((1u<<bit)-1u))];
          atomic_fetch_add_explicit(sums+position,countValue*coefficient[i],
            memory_order_relaxed);
        }
      }
    }
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  device uint* out=rec.output+packet*512u;
  if (query.baseMode!=0u) {
    // Sole writer of these scans: store base + sums (no initialization pass).
    bool seeded=query.baseMode==2u;
    device const uint* seed=seeded?bases[record].values+packet*512u:out;
    #pragma clang loop unroll(full)
    for (uint k=0;k<16;++k) {
      uint scan=k*32u+scanInBatch;
      out[scan]=(seeded?seed[scan]:0u)+atomic_load_explicit(sums+scan,memory_order_relaxed);
    }
    return;
  }
  // Tile or atlas fields: the initialization pass wrote base + fields.
  #pragma clang loop unroll(full)
  for (uint k=0;k<16;++k) {
    uint scan=k*32u+scanInBatch;
    out[scan]+=atomic_load_explicit(sums+scan,memory_order_relaxed);
  }
}

// Explicit packet ILP experiment. Each lane decodes independent scan packets
// from the same detector column/model. No persistent representation changes.
constant uint tans_shared_packets [[function_constant(5)]];
kernel void tans_detector_shared_packet_ilp_batch(
    device const TANSDetectorRecord* records [[buffer(0)]],
    device const uint* decoding [[buffer(1)]],
    device const uint* retainedRank [[buffer(4)]],
    constant TANSDetectorQuery& base [[buffer(7)]],
    device const uint2* groups [[buffer(9)]],
    device const uint* selected [[buffer(10)]],
    device const int* coefficients [[buffer(11)]],
    device const uint* groupOffsets [[buffer(12)]],
    device const uint* groupCounts [[buffer(13)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint threadID [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  if (group.x>=groupCounts[group.z]) return;
  uint2 descriptor=groups[groupOffsets[group.z]+group.x];
  uint model=descriptor.x,index=descriptor.y+lane;
  threadgroup uint table[1024];
  if (model!=255u)
    for (uint i=threadID;i<1024;i+=tans_shared_threads) table[i]=decoding[model*1024u+i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint q=selected[index],sign=uint(coefficients[index]);
  bool active=q!=0xffffffffu;
  uint packetBase=(group.y*(tans_shared_threads/32u)+threadID/32u)*tans_shared_packets;
  TANSDetectorRecord record=records[group.z];
  uint cursor[4],end[4],state[4],available[4],lookahead[4];
  ulong reservoir[4];
  #pragma unroll
  for (uint j=0;j<tans_shared_packets;++j) {
    cursor[j]=0;end[j]=0;state[j]=0;available[j]=0;lookahead[j]=0;reservoir[j]=0;
    if (active) {
      uint2 bounds=stream_bounds(record.offsets,
        (packetBase+j)*base.retainedColumns+retainedRank[q],base.retainedColumns,false);
      cursor[j]=bounds.x;end[j]=bounds.y;
      if (model!=255u) {
        uint header=record.payload[cursor[j]++];
        state[j]=header&1023u;available[j]=22;reservoir[j]=header>>10;
        lookahead[j]=cursor[j]<end[j]?record.payload[cursor[j]]:0;
      }
    }
  }
  #pragma clang loop unroll(disable)
  for (uint batch=0;batch<512;batch+=32) {
    uint saved[4]={0,0,0,0};
    #pragma clang loop unroll(disable)
    for (uint step=0;step<32;step+=2) {
      #pragma unroll
      for (uint j=0;j<tans_shared_packets;++j) {
        uint a=0,b=0;
        if (active) {
          if (model==255u) {
            uint pair=record.payload[cursor[j]++];a=pair&65535u;b=pair>>16;
          } else {
            if (available[j]<12) {
              reservoir[j]|=ulong(lookahead[j])<<available[j];available[j]+=32;
              if (cursor[j]<end[j]) ++cursor[j];
              lookahead[j]=cursor[j]<end[j]?record.payload[cursor[j]]:0;
            }
            uint code=table[state[j]],bits=(code>>12)&15u;
            uint low=uint(reservoir[j])&((1u<<bits)-1u);
            reservoir[j]>>=bits;available[j]-=bits;state[j]=(code>>16)+low;
            uint pair=code&4095u;
            if (pair==4095u) {
              if (available[j]<12) {
                reservoir[j]|=ulong(lookahead[j])<<available[j];available[j]+=32;
                if (cursor[j]<end[j]) ++cursor[j];
                lookahead[j]=cursor[j]<end[j]?record.payload[cursor[j]]:0;
              }
              pair=uint(reservoir[j])&4095u;reservoir[j]>>=12;available[j]-=12;
            }
            a=pair&63u;b=pair>>6;
          }
        }
        uint sumA=simd_sum(a*sign),sumB=simd_sum(b*sign);
        if (lane==step) saved[j]=sumA;
        if (lane==step+1) saved[j]=sumB;
      }
    }
    device atomic_uint* output=reinterpret_cast<device atomic_uint*>(record.output);
    #pragma unroll
    for (uint j=0;j<tans_shared_packets;++j)
      atomic_fetch_add_explicit(output+(packetBase+j)*512u+batch+lane,saved[j],memory_order_relaxed);
  }
}

// Word/funnel reader matching the newer CUDA query topology. No 64-bit
// reservoir; bit is always in 0...31 and each read crosses at most once.
inline uint tans_funnel_bits(device const uint* payload, thread uint& cursor,
    uint end, uint bits, thread uint& word, thread uint& bit, thread uint& next) {
  uint joined=(word>>bit)|(bit==0u?0u:next<<(32u-bit));
  uint value=joined&((1u<<bits)-1u);
  bit+=bits;
  if (bit>=32u) {
    bit-=32u;word=next;
    if (cursor<end) ++cursor;
    next=cursor<end?payload[cursor]:0u;
  }
  return value;
}
inline uint tans_funnel_pair(device const uint* payload, thread uint& cursor,
    uint end, threadgroup const uint* table, thread uint& state,
    thread uint& word, thread uint& bit, thread uint& next) {
  uint code=table[state],bits=(code>>12)&15u;
  uint low=tans_funnel_bits(payload,cursor,end,bits,word,bit,next);
  state=(code>>16)+low;
  uint pair=code&4095u;
  return pair==4095u?tans_funnel_bits(payload,cursor,end,12u,word,bit,next):pair;
}
constant uint tans_pairs_requested [[function_constant(6)]];
constant uint tans_pairs = is_function_constant_defined(tans_pairs_requested)
    ? tans_pairs_requested : 1u;
kernel void tans_detector_cuda_funnel_batch(
    device const TANSDetectorRecord* records [[buffer(0)]],
    device const uint* decoding [[buffer(1)]],
    device const uint* retainedRank [[buffer(4)]],
    constant TANSDetectorQuery& base [[buffer(7)]],
    device const uint2* groups [[buffer(9)]],
    device const uint* selected [[buffer(10)]],
    device const int* coefficients [[buffer(11)]],
    device const uint* groupOffsets [[buffer(12)]],
    device const uint* groupCounts [[buffer(13)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint threadID [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  if (group.x>=groupCounts[group.z]) return;
  uint2 descriptor=groups[groupOffsets[group.z]+group.x];
  uint model=descriptor.x,index=descriptor.y+lane;
  threadgroup uint table[1024];
  if (model!=255u)
    for (uint i=threadID;i<1024;i+=tans_shared_threads) table[i]=decoding[model*1024u+i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint q=selected[index],sign=uint(coefficients[index]);
  bool active=q!=0xffffffffu;
  uint uniformSign=simd_broadcast_first(sign);
  uint packet=group.y*(tans_shared_threads/32u)+threadID/32u;
  TANSDetectorRecord record=records[group.z];
  uint cursor=0,end=0,state=0,bit=10,word=0,next=0;
  if (active) {
    uint2 bounds=stream_bounds(record.offsets,
      packet*base.retainedColumns+retainedRank[q],base.retainedColumns,false);
    cursor=bounds.x;end=bounds.y;
    if (model!=255u) {
      word=record.payload[cursor++];state=word&1023u;
      next=cursor<end?record.payload[cursor]:0u;
    }
  }
  #pragma clang loop unroll(disable)
  for (uint batch=0;batch<512;batch+=32) {
    uint saved=0;
    #pragma clang loop unroll(disable)
    for (uint step=0;step<32;step+=2*tans_pairs) {
      uint4 aa=0,bb=0;
      // Decode several exact symbols before SIMD collectives to overlap
      // independent output arithmetic with subsequent state-table work.
      #pragma unroll
      for (uint i=0;i<tans_pairs;++i) {
        if (active) {
          if (model==255u) {
            uint pair=record.payload[cursor++];aa[i]=pair&65535u;bb[i]=pair>>16;
          } else {
            uint pair=tans_funnel_pair(record.payload,cursor,end,table,state,word,bit,next);
            aa[i]=pair&63u;bb[i]=pair>>6;
          }
        }
      }
      #pragma unroll
      for (uint i=0;i<tans_pairs;++i) {
        if (model!=255u) {
          uint packed=simd_sum(aa[i]|(bb[i]<<16));
          if ((lane>>1)==(step>>1)+i)
            saved=((packed>>((lane&1u)*16u))&65535u)*uniformSign;
        } else {
          uint sumA=simd_sum(aa[i]*sign),sumB=simd_sum(bb[i]*sign);
          if (lane==step+2*i) saved=sumA;
          if (lane==step+2*i+1) saved=sumB;
        }
      }
    }
    device atomic_uint* output=reinterpret_cast<device atomic_uint*>(record.output);
    atomic_fetch_add_explicit(output+packet*512u+batch+lane,saved,memory_order_relaxed);
  }
}

// Exact, optional small tile index. These buffers contain losslessly packed
// 2D sums, never a raw 4D source. Width is measured from GPU integer maxima.
struct TANSIndexImage { device uint* values [[id(0)]]; };
struct TANSIndexField {
  device const uint* words [[id(0)]];
  uint width [[id(1)]];
  uint wordsPerImage [[id(2)]];
  device const uint4* blocks [[id(3)]];
  uint blocked [[id(4)]];
};
// Exact block-frame-of-reference encoding. 256 values share a measured
// minimum and bit width; outliers affect only their own block. Source counts
// and output uint32 precision are unchanged. All scientific work stays on GPU.
kernel void tans_index_block_stats(device const TANSIndexImage* images [[buffer(0)]],
    device uint4* blocks [[buffer(1)]], uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  uint value=images[group.y].values[group.x*256u+tid];
  uint lo=simd_min(value),hi=simd_max(value);
  threadgroup uint lower[8],upper[8];
  if(lane==0) { lower[tid/32]=lo;upper[tid/32]=hi; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(tid==0) {
    for(uint i=1;i<8;++i) {lo=min(lo,lower[i]);hi=max(hi,upper[i]);}
    uint width=hi==lo?0u:32u-clz(hi-lo);
    blocks[group.y*1024u+group.x]=uint4(lo,width,0,width*8u);
  }
}
// Independent acquisition prefix scans run in parallel; only 66 allocation
// totals need the tiny final scan. No count data crosses to the CPU.
kernel void tans_index_block_offsets(device uint4* blocks [[buffer(0)]],
    device uint* totals [[buffer(1)]],uint acquisition [[thread_position_in_grid]]) {
  uint cursor=0;
  for(uint i=0;i<1024;++i) {
    uint index=acquisition*1024u+i;
    uint4 block=blocks[index];block.z=cursor;cursor+=block.w;blocks[index]=block;
  }
  totals[acquisition]=cursor;
}
kernel void tans_index_acquisition_offsets(device uint* totals [[buffer(0)]],
    constant uint& count [[buffer(1)]],uint tid [[thread_position_in_grid]]) {
  if(tid!=0) return;
  uint cursor=0;
  for(uint i=0;i<count;++i) {uint words=totals[i];totals[i]=cursor;cursor+=words;}
  totals[count]=cursor;
}
kernel void tans_index_block_pack(device const TANSIndexImage* images [[buffer(0)]],
    device uint4* blocks [[buffer(1)]],device const uint* offsets [[buffer(2)]],
    device uint* packed [[buffer(3)]],uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  uint index=group.y*1024u+group.x;
  uint4 block=blocks[index];uint destination=block.z+offsets[group.y];
  threadgroup_barrier(mem_flags::mem_device);
  if(tid==0) blocks[index].z=destination;
  if(tid>=block.w) return;
  uint bit=tid*32u,valueIndex=bit/block.y,skip=bit%block.y,used=0,result=0;
  while(used<32u && valueIndex<256u) {
    uint take=min(32u-used,block.y-skip);
    uint mask=take==32u?0xffffffffu:(1u<<take)-1u;
    uint delta=images[group.y].values[group.x*256u+valueIndex]-block.x;
    result|=((delta>>skip)&mask)<<used;
    used+=take;++valueIndex;skip=0;
  }
  packed[destination+tid]=result;
}
kernel void tans_index_max(device const TANSIndexImage* images [[buffer(0)]],
    device atomic_uint* maximum [[buffer(1)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  uint value=0;
  for (uint s=tid;s<262144;s+=256) value=max(value,images[group].values[s]);
  value=simd_max(value);
  threadgroup uint maxima[8];
  if (lane==0) maxima[tid/32]=value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid==0) {
    uint all=0;for(uint i=0;i<8;++i) all=max(all,maxima[i]);
    atomic_fetch_max_explicit(maximum,all,memory_order_relaxed);
  }
}
kernel void tans_index_pack(device const TANSIndexImage* images [[buffer(0)]],
    device uint* packed [[buffer(1)]],constant uint2& layout [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]) {
  uint width=layout.x,words=layout.y;
  if (gid.x>=words) return;
  uint bit=gid.x*32u,valueIndex=bit/width,skip=bit%width,used=0,result=0;
  while(used<32u && valueIndex<262144u) {
    uint take=min(32u-used,width-skip);
    uint mask=take==32u?0xffffffffu:(1u<<take)-1u;
    result|=((images[gid.y].values[valueIndex]>>skip)&mask)<<used;
    used+=take;++valueIndex;skip=0;
  }
  packed[gid.y*words+gid.x]=result;
}
kernel void tans_index_add(device const TANSIndexImage* images [[buffer(0)]],
    device const TANSIndexField* fields [[buffer(1)]],
    device const int2* selected [[buffer(2)]],
    constant uint& count [[buffer(3)]],device const uint* acquisitions [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
  if (gid.x>=262144u) return;
  uint value=images[gid.y].values[gid.x],acquisition=acquisitions[gid.y];
  for(uint i=0;i<count;++i) {
    TANSIndexField field=fields[selected[i].x];
    uint width=field.width,base=0,valueIndex=gid.x,offset=acquisition*field.wordsPerImage;
    if(field.blocked) {
      uint4 block=field.blocks[acquisition*1024u+gid.x/256u];
      base=block.x;width=block.y;offset=block.z;valueIndex=gid.x%256u;
    }
    uint decoded=0;
    if(width>0) {
      uint bit=valueIndex*width,word=bit>>5,shift=bit&31u;
      device const uint* payload=field.words+offset;
      decoded=payload[word]>>shift;
      if(shift+width>32u) decoded|=payload[word+1]<<(32u-shift);
      if(width<32u) decoded&=(1u<<width)-1u;
    }
    decoded+=base;
    value+=decoded*uint(selected[i].y);
  }
  images[gid.y].values[gid.x]=value;
}
// Same exact field sum as tans_index_add, but the starting value is the exact
// seed image (hasBase=1) or zero (hasBase=0) instead of a previously
// initialized output. This replaces a whole-image blit copy/fill followed by
// a read-modify-write of the same output with one read and one write.
kernel void tans_index_add_base(device const TANSIndexImage* images [[buffer(0)]],
    device const TANSIndexField* fields [[buffer(1)]],
    device const int2* selected [[buffer(2)]],
    constant uint& count [[buffer(3)]],device const uint* acquisitions [[buffer(4)]],
    device const TANSIndexImage* bases [[buffer(5)]],constant uint& hasBase [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]) {
  if (gid.x>=262144u) return;
  uint value=hasBase!=0u?bases[gid.y].values[gid.x]:0u,acquisition=acquisitions[gid.y];
  for(uint i=0;i<count;++i) {
    TANSIndexField field=fields[selected[i].x];
    uint width=field.width,base=0,valueIndex=gid.x,offset=acquisition*field.wordsPerImage;
    if(field.blocked) {
      uint4 block=field.blocks[acquisition*1024u+gid.x/256u];
      base=block.x;width=block.y;offset=block.z;valueIndex=gid.x%256u;
    }
    uint decoded=0;
    if(width>0) {
      uint bit=valueIndex*width,word=bit>>5,shift=bit&31u;
      device const uint* payload=field.words+offset;
      decoded=payload[word]>>shift;
      if(shift+width>32u) decoded|=payload[word+1]<<(32u-shift);
      if(width<32u) decoded&=(1u<<width)-1u;
    }
    decoded+=base;
    value+=decoded*uint(selected[i].y);
  }
  images[gid.y].values[gid.x]=value;
}
// Exact output initialization inside the decode compute pass: copy the seed
// image (hasBase=1) or write zeros. Four uint32 per thread; 65536 per image.
struct TANSInitImage { device uint4* values; };
kernel void tans_detector_init(device const TANSInitImage* outputs [[buffer(0)]],
    device const TANSInitImage* bases [[buffer(1)]],constant uint& hasBase [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]) {
  if (gid.x>=65536u) return;
  outputs[gid.y].values[gid.x]=hasBase!=0u?bases[gid.y].values[gid.x]:uint4(0u);
}

// Non-atomic batch variant. Each record/group owns a disjoint partial range,
// so the reduction lane can publish one exact uint32 sum without atomics.
// The caller selects this only when the bounded partial scratch fits its
// explicit memory budget; the atomic path remains the exact fallback.
kernel void tans_detector_partial_batch(
    device const TANSDetectorRecord* records [[buffer(0)]],
    device const uint* decoding [[buffer(1)]], device const uchar* models [[buffer(2)]],
    device const int* cacheMap [[buffer(3)]], device const uint* retainedRank [[buffer(4)]],
    device const uint* selected [[buffer(5)]], device const int* coefficients [[buffer(6)]],
    constant TANSDetectorQuery& base [[buffer(7)]], device const uint* modelOffsets [[buffer(8)]],
    device uint* partials [[buffer(9)]],
    uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  TANSDetectorRecord record=records[group.z];
  TANSDetectorQuery query=base;
  query.modelOffset=modelOffsets[group.z];
  device uint* recordPartials = partials + group.z * 32u * query.groups * 512u;
  tans_detector_reduce(record.payload,record.offsets,record.events,record.sparseOffsets,
    decoding,models,cacheMap,retainedRank,recordPartials,query,selected,lane,group.xy,
    coefficients,false,false);
}

// Sparse detector columns contain explicit nonzero events. Scatter only those
// events; do not walk all 512 scans or force dense entropy lanes to diverge.
constant bool tans_sparse_prefix_requested [[function_constant(22)]];
constant bool tans_sparse_prefix = is_function_constant_defined(tans_sparse_prefix_requested)
  ? tans_sparse_prefix_requested : false;
kernel void tans_detector_sparse_batch(
    device const TANSDetectorRecord* records [[buffer(0)]],
    device const int* cacheMap [[buffer(1)]], device const uint* selected [[buffer(2)]],
    device const int* coefficients [[buffer(3)]], constant uint2& parameters [[buffer(4)]],
    uint2 position [[thread_position_in_grid]]) {
  uint count=parameters.x;
  if (position.x>=count*32u) return;
  uint packet=position.x/count,index=position.x%count;
  uint cached=uint(cacheMap[selected[index]]);
  TANSDetectorRecord record=records[position.y];
  uint2 bounds=stream_bounds(record.sparseOffsets,packet*parameters.y+cached,parameters.y,true);
  device atomic_uint* output=reinterpret_cast<device atomic_uint*>(record.output);
  uint sign=uint(coefficients[index]);
  // A stream visits consecutive events, so only its first flagged event needs
  // the global rank prefix. Later flagged events consume consecutive count
  // bytes, even across 32-bit flag words and 256-event rank checkpoints.
  uint nextRank=0;
  bool haveRank=false;
  for (uint cursor=bounds.x;cursor<bounds.y;++cursor) {
    uint event;
    if (tans_sparse_prefix) {
      device const uint* data=record.events;
      uint pwords=data[1], fwords=data[2], rwords=data[3];
      device const uint* positions=data+4;
      device const uint* flags=positions+pwords;
      device const uint* ranks=flags+fwords;
      device const uchar* values=reinterpret_cast<device const uchar*>(ranks+rwords);
      uint bit=cursor*9u,word=bit>>5,shift=bit&31u;
      ulong joined=ulong(positions[word]);
      if (word+1<pwords) joined|=ulong(positions[word+1])<<32;
      uint countValue=1,flag=flags[cursor>>5];
      if ((flag>>(cursor&31u))&1u) {
        if (!haveRank) {
          nextRank=ranks[cursor>>8];
          for(uint j=(cursor>>8)*8;j<(cursor>>5);++j) nextRank+=popcount(flags[j]);
          nextRank+=popcount(flag&((1u<<(cursor&31u))-1u));
          haveRank=true;
        }
        countValue=values[nextRank++];
      }
      event=(uint(joined>>shift)&511u)|(countValue<<9);
    } else {
      event=sparse_event(record.events,cursor);
    }
    atomic_fetch_add_explicit(output+packet*512+(event&511u),(event>>9)*sign,memory_order_relaxed);
  }
}
kernel void tans_detector_finish(device const uint* partials [[buffer(0)]],
    device uint* output [[buffer(1)]], constant uint& groups [[buffer(2)]],
    device const uint* previous [[buffer(3)]], constant uint& seed [[buffer(4)]],
    uint scan [[thread_position_in_grid]]) {
  if (scan>=16384) return;
  uint packet=scan/512, position=scan%512, sum=seed ? previous[scan] : 0u;
  for (uint group=0; group<groups; ++group) sum+=partials[(packet*groups+group)*512+position];
  output[scan]=sum;
}

// Finish all chunk records in one dispatch. The argument table already holds
// the exact per-chunk output pointer, so no CPU loop or intermediate copy is
// needed between the producer and this ordered consumer.
kernel void tans_detector_finish_batch(
    device const uint* partials [[buffer(0)]],
    device const TANSDetectorRecord* records [[buffer(1)]],
    constant uint& groups [[buffer(2)]],
    uint3 position [[thread_position_in_grid]]) {
  if (position.x >= 16384u) return;
  uint record = position.z;
  uint sum = 0u;
  uint packet = position.x / 512u, scan = position.x % 512u;
  uint base = (record * 32u + packet) * groups * 512u;
  for (uint group=0; group<groups; ++group)
    sum += partials[base + group * 512u + scan];
  records[record].output[position.x] += sum;
}
