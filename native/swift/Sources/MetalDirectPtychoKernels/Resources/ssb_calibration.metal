#include <metal_stdlib>
using namespace metal;

kernel void ssb_disk_seed(device const ulong *raw [[buffer(0)]], device ulong *ordered [[buffer(1)]],
                          constant uint4 &shape [[buffer(3)]], uint i [[thread_position_in_grid]]) {
  ordered[i] = i < shape.x * shape.y ? raw[i] : 0;
}

kernel void ssb_disk_sort(device ulong *ordered [[buffer(1)]], constant uint2 &step [[buffer(4)]],
                          uint i [[thread_position_in_grid]]) {
  uint other = i ^ step.y;
  if (other <= i) return;
  ulong a = ordered[i], b = ordered[other];
  bool ascending = (i & step.x) == 0;
  ordered[i] = ascending ? min(a, b) : max(a, b);
  ordered[other] = ascending ? max(a, b) : min(a, b);
}

kernel void ssb_disk_measure(device const ulong *raw [[buffer(0)]],
                             device const ulong *ordered [[buffer(1)]],
                             device ulong *result [[buffer(2)]], constant uint4 &shape [[buffer(3)]],
                             uint t [[thread_index_in_threadgroup]]) {
  threadgroup float2 moments[256];
  threadgroup ulong4 totals[256];
  threadgroup float threshold;
  threadgroup ulong halfPlateau;
  uint pixels = shape.x * shape.y;
  float2 m = 0;
  // Normalize before squaring so every uint64 input is finite in float32.
  for (uint i = t; i < pixels; i += 256) {
    float x = float(raw[i]) * 0x1p-32f;
    m += float2(x, x*x);
  }
  moments[t] = m;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint n = 128; n; n >>= 1) {
    if (t < n) moments[t] += moments[t+n];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (!t) {
    float2 mean = moments[0] / float(pixels);
    threshold = mean.x + sqrt(max(0.0f, mean.y - mean.x*mean.x));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  ulong4 sum = 0;
  for (uint i = t; i < pixels; i += 256) {
    if (float(raw[i]) * 0x1p-32f > threshold) sum += ulong4(1, i / shape.y, i % shape.y, 0);
  }
  totals[t] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint n = 128; n; n >>= 1) {
    if (t < n) totals[t] += totals[t+n];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (!t) {
    result[0] = totals[0].x; result[1] = totals[0].y; result[2] = totals[0].z;
    ulong count = totals[0].x;
    if (count) {
      ulong start = shape.z - count;
      ulong a = ordered[start + (count-1)/2], b = ordered[start + count/2];
      // floor((a+b)/4), without uint64 overflow or loss of count precision.
      halfPlateau = a/4 + b/4 + ((a%4 + b%4)/4);
    } else halfPlateau = ULONG_MAX;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  sum = 0;
  for (uint i = t; i < pixels; i += 256) {
    if (raw[i] > halfPlateau) {
      bool clipped = i < shape.y || i >= (shape.x-1)*shape.y || i%shape.y == 0 || i%shape.y == shape.y-1;
      sum += ulong4(1, clipped ? 1 : 0, 0, 0);
    }
  }
  totals[t] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint n = 128; n; n >>= 1) {
    if (t < n) totals[t] += totals[t+n];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (!t) { result[3] = totals[0].x; result[4] = totals[0].y; }
}
