#include "CMetal4DSTEMInteractions.h"

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

static void apply_entry(
    const uint32_t *values,
    uint32_t *output,
    size_t count,
    uint32_t coefficients) {
  size_t scan = 0;

#if defined(__aarch64__)
  const uint32x4_t mask = vdupq_n_u32(UINT32_C(0xffff));
  for (; scan + 16 <= count; scan += 16) {
    for (size_t lane = 0; lane < 16; lane += 4) {
      uint32x4_t packed = vld1q_u32(values + scan + lane);
      uint32x4_t destination = vld1q_u32(output + scan + lane);
      uint32x4_t low = vandq_u32(packed, mask);
      uint32x4_t high = vshrq_n_u32(packed, 16);
      switch (coefficients) {
      case 1:
        destination = vaddq_u32(destination, low);
        break;
      case 2:
        destination = vsubq_u32(destination, low);
        break;
      case 4:
        destination = vaddq_u32(destination, high);
        break;
      case 5:
        destination = vaddq_u32(destination, vaddq_u32(low, high));
        break;
      case 6:
        destination = vaddq_u32(vsubq_u32(destination, low), high);
        break;
      case 8:
        destination = vsubq_u32(destination, high);
        break;
      case 9:
        destination = vsubq_u32(vaddq_u32(destination, low), high);
        break;
      case 10:
        destination = vsubq_u32(destination, vaddq_u32(low, high));
        break;
      default:
        return;
      }
      vst1q_u32(output + scan + lane, destination);
    }
  }
#endif

  for (; scan < count; ++scan) {
    uint32_t packed = values[scan];
    uint32_t low = packed & UINT32_C(0xffff);
    uint32_t high = packed >> 16;
    switch (coefficients) {
    case 1:
      output[scan] += low;
      break;
    case 2:
      output[scan] -= low;
      break;
    case 4:
      output[scan] += high;
      break;
    case 5:
      output[scan] += low + high;
      break;
    case 6:
      output[scan] = output[scan] - low + high;
      break;
    case 8:
      output[scan] -= high;
      break;
    case 9:
      output[scan] = output[scan] + low - high;
      break;
    case 10:
      output[scan] -= low + high;
      break;
    default:
      return;
    }
  }
}

void q_update_virtual_detector_u16(
    const uint32_t *data,
    uint32_t *output,
    size_t scan_count,
    const QDetectorWordEntry *entries,
    size_t entry_count) {
  for (size_t entry = 0; entry < entry_count; ++entry) {
    const QDetectorWordEntry spec = entries[entry];
    apply_entry(
        data + (size_t)spec.word * scan_count,
        output,
        scan_count,
        spec.coefficients);
  }
}
