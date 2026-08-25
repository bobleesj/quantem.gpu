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

static void apply_u8_single_lane(
    const uint32_t *values,
    uint32_t *output,
    size_t count,
    uint32_t lane,
    int subtract) {
  size_t scan = 0;

#if defined(__aarch64__)
  const uint32x4_t mask = vdupq_n_u32(UINT32_C(0xff));
  for (; scan + 4 <= count; scan += 4) {
    uint32x4_t packed = vld1q_u32(values + scan);
    uint32x4_t sample;
    switch (lane) {
    case 0:
      sample = vandq_u32(packed, mask);
      break;
    case 1:
      sample = vandq_u32(vshrq_n_u32(packed, 8), mask);
      break;
    case 2:
      sample = vandq_u32(vshrq_n_u32(packed, 16), mask);
      break;
    default:
      sample = vshrq_n_u32(packed, 24);
      break;
    }
    uint32x4_t destination = vld1q_u32(output + scan);
    vst1q_u32(
        output + scan,
        subtract ? vsubq_u32(destination, sample)
                 : vaddq_u32(destination, sample));
  }
#endif

  const uint32_t shift = lane * 8;
  for (; scan < count; ++scan) {
    uint32_t sample = (values[scan] >> shift) & UINT32_C(0xff);
    if (subtract) output[scan] -= sample;
    else output[scan] += sample;
  }
}

static void apply_u8_two_lanes(
    const uint32_t *values,
    uint32_t *output,
    size_t count,
    uint32_t first_lane,
    int subtract_first,
    uint32_t second_lane,
    int subtract_second) {
  size_t scan = 0;

#if defined(__aarch64__)
  const uint32x4_t mask = vdupq_n_u32(UINT32_C(0xff));
  for (; scan + 4 <= count; scan += 4) {
    uint32x4_t packed = vld1q_u32(values + scan);
    uint32x4_t first = vandq_u32(vshlq_u32(
        packed, vdupq_n_s32(-((int32_t)first_lane * 8))), mask);
    uint32x4_t second = vandq_u32(vshlq_u32(
        packed, vdupq_n_s32(-((int32_t)second_lane * 8))), mask);
    uint32x4_t delta =
        subtract_first ? vsubq_u32(vdupq_n_u32(0), first) : first;
    delta = subtract_second ? vsubq_u32(delta, second) : vaddq_u32(delta, second);
    uint32x4_t destination = vld1q_u32(output + scan);
    vst1q_u32(output + scan, vaddq_u32(destination, delta));
  }
#endif

  const uint32_t first_shift = first_lane * 8;
  const uint32_t second_shift = second_lane * 8;
  for (; scan < count; ++scan) {
    uint32_t packed = values[scan];
    uint32_t first = (packed >> first_shift) & UINT32_C(0xff);
    uint32_t second = (packed >> second_shift) & UINT32_C(0xff);
    uint32_t delta = subtract_first ? 0 - first : first;
    delta = subtract_second ? delta - second : delta + second;
    output[scan] += delta;
  }
}

static void apply_u8_entry(
    const uint32_t *values,
    uint32_t *output,
    size_t count,
    uint32_t coefficients) {
  switch (coefficients) {
  case 1:
    apply_u8_single_lane(values, output, count, 0, 0);
    return;
  case 2:
    apply_u8_single_lane(values, output, count, 0, 1);
    return;
  case 4:
    apply_u8_single_lane(values, output, count, 1, 0);
    return;
  case 8:
    apply_u8_single_lane(values, output, count, 1, 1);
    return;
  case 16:
    apply_u8_single_lane(values, output, count, 2, 0);
    return;
  case 32:
    apply_u8_single_lane(values, output, count, 2, 1);
    return;
  case 64:
    apply_u8_single_lane(values, output, count, 3, 0);
    return;
  case 128:
    apply_u8_single_lane(values, output, count, 3, 1);
    return;
  case 5:
    apply_u8_two_lanes(values, output, count, 0, 0, 1, 0);
    return;
  case 9:
    apply_u8_two_lanes(values, output, count, 0, 0, 1, 1);
    return;
  case 6:
    apply_u8_two_lanes(values, output, count, 0, 1, 1, 0);
    return;
  case 10:
    apply_u8_two_lanes(values, output, count, 0, 1, 1, 1);
    return;
  case 17:
    apply_u8_two_lanes(values, output, count, 0, 0, 2, 0);
    return;
  case 33:
    apply_u8_two_lanes(values, output, count, 0, 0, 2, 1);
    return;
  case 18:
    apply_u8_two_lanes(values, output, count, 0, 1, 2, 0);
    return;
  case 34:
    apply_u8_two_lanes(values, output, count, 0, 1, 2, 1);
    return;
  case 65:
    apply_u8_two_lanes(values, output, count, 0, 0, 3, 0);
    return;
  case 129:
    apply_u8_two_lanes(values, output, count, 0, 0, 3, 1);
    return;
  case 66:
    apply_u8_two_lanes(values, output, count, 0, 1, 3, 0);
    return;
  case 130:
    apply_u8_two_lanes(values, output, count, 0, 1, 3, 1);
    return;
  case 20:
    apply_u8_two_lanes(values, output, count, 1, 0, 2, 0);
    return;
  case 36:
    apply_u8_two_lanes(values, output, count, 1, 0, 2, 1);
    return;
  case 24:
    apply_u8_two_lanes(values, output, count, 1, 1, 2, 0);
    return;
  case 40:
    apply_u8_two_lanes(values, output, count, 1, 1, 2, 1);
    return;
  case 68:
    apply_u8_two_lanes(values, output, count, 1, 0, 3, 0);
    return;
  case 132:
    apply_u8_two_lanes(values, output, count, 1, 0, 3, 1);
    return;
  case 72:
    apply_u8_two_lanes(values, output, count, 1, 1, 3, 0);
    return;
  case 136:
    apply_u8_two_lanes(values, output, count, 1, 1, 3, 1);
    return;
  case 80:
    apply_u8_two_lanes(values, output, count, 2, 0, 3, 0);
    return;
  case 144:
    apply_u8_two_lanes(values, output, count, 2, 0, 3, 1);
    return;
  case 96:
    apply_u8_two_lanes(values, output, count, 2, 1, 3, 0);
    return;
  case 160:
    apply_u8_two_lanes(values, output, count, 2, 1, 3, 1);
    return;
  default:
    break;
  }
  size_t scan = 0;

#if defined(__aarch64__)
  const uint32x4_t mask = vdupq_n_u32(UINT32_C(0xff));
  for (; scan + 4 <= count; scan += 4) {
    uint32x4_t packed = vld1q_u32(values + scan);
    uint32x4_t delta = vdupq_n_u32(0);
    for (uint32_t lane = 0; lane < 4; ++lane) {
      uint32_t coefficient = (coefficients >> (lane * 2)) & UINT32_C(3);
      uint32x4_t sample;
      switch (lane) {
      case 0:
        sample = vandq_u32(packed, mask);
        break;
      case 1:
        sample = vandq_u32(vshrq_n_u32(packed, 8), mask);
        break;
      case 2:
        sample = vandq_u32(vshrq_n_u32(packed, 16), mask);
        break;
      default:
        sample = vshrq_n_u32(packed, 24);
        break;
      }
      if (coefficient == 1) delta = vaddq_u32(delta, sample);
      else if (coefficient == 2) delta = vsubq_u32(delta, sample);
    }
    uint32x4_t destination = vld1q_u32(output + scan);
    vst1q_u32(output + scan, vaddq_u32(destination, delta));
  }
#endif

  for (; scan < count; ++scan) {
    uint32_t packed = values[scan];
    uint32_t delta = 0;
    for (uint32_t lane = 0; lane < 4; ++lane) {
      uint32_t coefficient = (coefficients >> (lane * 2)) & UINT32_C(3);
      uint32_t sample = (packed >> (lane * 8)) & UINT32_C(0xff);
      if (coefficient == 1) delta += sample;
      else if (coefficient == 2) delta -= sample;
    }
    output[scan] += delta;
  }
}

void q_update_virtual_detector_u8_range(
    const uint32_t *data,
    uint32_t *output,
    size_t data_scan_count,
    size_t scan_offset,
    size_t scan_count,
    const QDetectorWordEntry *entries,
    size_t entry_count) {
  for (size_t entry = 0; entry < entry_count; ++entry) {
    const QDetectorWordEntry spec = entries[entry];
    apply_u8_entry(
        data + (size_t)spec.word * data_scan_count + scan_offset,
        output + scan_offset,
        scan_count,
        spec.coefficients);
  }
}
