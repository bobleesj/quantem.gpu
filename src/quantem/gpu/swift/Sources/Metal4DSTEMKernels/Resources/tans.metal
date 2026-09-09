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

kernel void tans_detector_shared_model_batch(
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
  uint packet=group.y*(tans_shared_threads/32u)+threadID/32u;
  TANSDetectorRecord record=records[group.z];
  uint cursor=0,end=0,state=0,available=0,lookahead=0;
  ulong reservoir=0;
  uint reservoir32=0;
  if (active) {
    uint2 bounds=stream_bounds(record.offsets,
      packet*base.retainedColumns+retainedRank[q],base.retainedColumns,false);
    cursor=bounds.x;end=bounds.y;
    if (model!=255u) {
      uint header=record.payload[cursor++];
      state=header&1023u;available=22;reservoir=header>>10;reservoir32=header>>10;
      lookahead=cursor<end?record.payload[cursor]:0;
    }
  }
  #pragma clang loop unroll(disable)
  for (uint batch=0;batch<512;batch+=32) {
    uint saved=0;
    #pragma clang loop unroll(disable)
    for (uint step=0;step<32;step+=2) {
      uint a=0,b=0;
      if (active) {
        if (model==255u) {
          uint pair=record.payload[cursor++];
          a=pair&65535u;b=pair>>16;
        } else if (tans_shared_word32) {
          uint code=table[state],bits=(code>>12)&15u;
          uint low=read_bits32(record.payload,cursor,end,bits,reservoir32,available);
          state=(code>>16)+low;
          uint pair=code&4095u;
          if (pair==4095u)
            pair=read_bits32(record.payload,cursor,end,12,reservoir32,available);
          a=pair&63u;b=pair>>6;
        } else {
          // Refill can begin independently of the next state-table lookup.
          if (available<12) {
            reservoir|=ulong(lookahead)<<available;available+=32;
            if (cursor<end) ++cursor;
            lookahead=cursor<end?record.payload[cursor]:0;
          }
          uint code=table[state],bits=(code>>12)&15u;
          uint low=uint(reservoir)&((1u<<bits)-1u);
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
      if (tans_shared_pair && model!=255u) {
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
    device atomic_uint* output=reinterpret_cast<device atomic_uint*>(record.output);
    atomic_fetch_add_explicit(output+packet*512u+batch+lane,saved,memory_order_relaxed);
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
  for (uint cursor=bounds.x;cursor<bounds.y;++cursor) {
    uint event=sparse_event(record.events,cursor);
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
