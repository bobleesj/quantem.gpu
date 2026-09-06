// Serial integer-kernel harness only. This does not emulate GPU scheduling,
// memory ownership, atomics under contention, driver behavior, or performance.
#include <algorithm>
#include <cstdint>
using std::min;
#define __device__
#define __global__
#define __clz __builtin_clz
struct Index { unsigned int x; };
static Index blockIdx = {0}, blockDim = {1}, threadIdx = {0};
static unsigned int atomicOr(unsigned int* target, unsigned int value) {
    unsigned int previous = *target;
    *target |= value;
    return previous;
}
static unsigned long long atomicAdd(unsigned long long* target, unsigned long long value) {
    unsigned long long previous = *target;
    *target += value;
    return previous;
}
extern "C" void set_thread(unsigned int index) { blockIdx.x = index; }
#include "_ans.cu"
