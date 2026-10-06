// Internal exact-integer center-of-mass support for the WebGPU detector engine.
//
// This module is intentionally not re-exported as a public QuantEM.GPU API.
// The detector backend owns selection and GPU resource lifecycle; this file
// owns only the portable integer-width proof and WGSL paired-word arithmetic.

const MAX_U32 = BigInt(0xffff_ffff);
const MAX_U64 = BigInt("18446744073709551615");

type ExactIntegerCoMPlan = {
  totalBound: bigint;
  rowMomentBound: bigint;
  columnMomentBound: bigint;
  narrowInteger: 0 | 1;
  narrowProducts: 0 | 1;
};

function valueMaximum(residentMode: number): bigint {
  if (residentMode === 0) return BigInt(0xffff); // packed uint16
  if (residentMode === 1) return BigInt(0xff); // packed uint8
  if (residentMode === 3) return MAX_U32; // native uint32
  throw new RangeError(
    `WebGPU exact integer CoM does not support resident mode ${residentMode}; expected uint16, uint8, or uint32.`,
  );
}

/**
 * Prove whether one active integer detector selection needs u32 or paired-u32
 * accumulation. Coordinates and counts are nonnegative, so the full-selection
 * bounds also prove that every per-thread and reduction-tree partial fits.
 *
 * @internal
 */
export function planExactIntegerCoM(
  detectorIndices: Uint32Array,
  activeCount: number,
  detectorColumns: number,
  residentMode: number,
): ExactIntegerCoMPlan {
  if (
    !Number.isSafeInteger(activeCount) ||
    activeCount < 0 ||
    activeCount > detectorIndices.length
  ) {
    throw new RangeError(
      `WebGPU exact integer CoM active count must be within the detector-index array; got ${activeCount} for length ${detectorIndices.length}.`,
    );
  }
  if (
    !Number.isSafeInteger(detectorColumns) ||
    detectorColumns < 1 ||
    detectorColumns > 0xffff_ffff
  ) {
    throw new RangeError(
      `WebGPU exact integer CoM detectorColumns must be representable as a positive u32; got ${detectorColumns}.`,
    );
  }
  const maximum = valueMaximum(residentMode);

  // Keep the common 192 x 192 path in exact Number arithmetic, then convert
  // once. Escalate to BigInt only for unusually large index sets that could
  // cross Number.MAX_SAFE_INTEGER. This avoids per-index BigInt conversion and
  // addition in every drag request while retaining a fail-closed geometry proof.
  let rowCoordinateSum = 0;
  let columnCoordinateSum = 0;
  let maximumRow = 0;
  let maximumColumn = 0;
  let rowCoordinateSumBig: bigint | null = null;
  let columnCoordinateSumBig: bigint | null = null;
  for (let indexOffset = 0; indexOffset < activeCount; indexOffset++) {
    const detectorIndex = detectorIndices[indexOffset];
    const detectorRow = Math.floor(detectorIndex / detectorColumns);
    const detectorColumn = detectorIndex % detectorColumns;
    maximumRow = Math.max(maximumRow, detectorRow);
    maximumColumn = Math.max(maximumColumn, detectorColumn);

    if (
      rowCoordinateSumBig === null &&
      rowCoordinateSum <= Number.MAX_SAFE_INTEGER - detectorRow
    ) {
      rowCoordinateSum += detectorRow;
    } else {
      rowCoordinateSumBig =
        (rowCoordinateSumBig ?? BigInt(rowCoordinateSum)) + BigInt(detectorRow);
    }
    if (
      columnCoordinateSumBig === null &&
      columnCoordinateSum <= Number.MAX_SAFE_INTEGER - detectorColumn
    ) {
      columnCoordinateSum += detectorColumn;
    } else {
      columnCoordinateSumBig =
        (columnCoordinateSumBig ?? BigInt(columnCoordinateSum)) +
        BigInt(detectorColumn);
    }
  }

  const rowCoordinates = rowCoordinateSumBig ?? BigInt(rowCoordinateSum);
  const columnCoordinates =
    columnCoordinateSumBig ?? BigInt(columnCoordinateSum);
  const totalBound = BigInt(activeCount) * maximum;
  const rowMomentBound = rowCoordinates * maximum;
  const columnMomentBound = columnCoordinates * maximum;
  if (
    totalBound > MAX_U64 ||
    rowMomentBound > MAX_U64 ||
    columnMomentBound > MAX_U64
  ) {
    throw new RangeError(
      `WebGPU exact integer CoM exceeds its two-u32 accumulation contract: total<=${totalBound}, rowMoment<=${rowMomentBound}, columnMoment<=${columnMomentBound}, maximum=${MAX_U64}. ` +
        "QuantEM.GPU will not downcast, crop, or approximate this detector selection.",
    );
  }

  return {
    totalBound,
    rowMomentBound,
    columnMomentBound,
    narrowInteger:
      totalBound <= MAX_U32 &&
      rowMomentBound <= MAX_U32 &&
      columnMomentBound <= MAX_U32
        ? 1
        : 0,
    // Moment totals may need paired words even when each coordinate * count
    // term fits u32. This is the common full 192 x 192 uint16 case, and lets
    // the shader avoid two 32 x 32 -> 64 limb multiplications per pixel.
    narrowProducts:
      BigInt(maximumRow) * maximum <= MAX_U32 &&
      BigInt(maximumColumn) * maximum <= MAX_U32
        ? 1
        : 0,
  };
}

// WGSL has no portable u64 scalar. Integer detector counts therefore use an
// explicit (lo, hi) pair. planExactIntegerCoM proves that admitted totals and
// moments are at most 2^64 - 1 before this code is dispatched.
export const EXACT_INTEGER_COM_WGSL = /* wgsl */ `
struct U64Words { lo: u32, hi: u32 }
struct U64Step { remainder: U64Words, bit: u32 }
struct U32Step { remainder: u32, bit: u32 }

fn u32DoubleStep(remainder: u32, denominator: u32) -> U32Step {
  let overflow = (remainder & 0x80000000u) != 0u;
  let doubled = remainder << 1u;
  if (overflow || doubled >= denominator) {
    return U32Step(doubled - denominator, 1u);
  }
  return U32Step(doubled, 0u);
}

fn u32CompareTwice(remainder: u32, denominator: u32) -> i32 {
  if ((remainder & 0x80000000u) != 0u) { return 1i; }
  let doubled = remainder << 1u;
  if (doubled < denominator) { return -1i; }
  if (doubled > denominator) { return 1i; }
  return 0i;
}

// Exact u32 specialization of ratioU64 for masks whose complete total and
// moments fit u32. It avoids paired-word operations in the narrow fast path.
fn ratioU32Exact(numerator: u32, denominator: u32) -> f32 {
  if (numerator == 0u || denominator == 0u) { return 0.0; }
  var exponent = 0i;
  var remainder: u32;
  var normalizedDenominator: u32;
  if (numerator >= denominator) {
    var scaledDenominator = denominator;
    loop {
      if ((scaledDenominator & 0x80000000u) != 0u) { break; }
      let next = scaledDenominator << 1u;
      if (next > numerator) { break; }
      scaledDenominator = next;
      exponent = exponent + 1i;
    }
    remainder = numerator - scaledDenominator;
    normalizedDenominator = scaledDenominator;
  } else {
    remainder = numerator;
    loop {
      let step = u32DoubleStep(remainder, denominator);
      remainder = step.remainder;
      exponent = exponent - 1i;
      if (step.bit != 0u) { break; }
    }
    normalizedDenominator = denominator;
  }
  var significand = 1u << 23u;
  for (var bit = 23u; bit > 0u; bit = bit - 1u) {
    let step = u32DoubleStep(remainder, normalizedDenominator);
    remainder = step.remainder;
    significand = significand | (step.bit << (bit - 1u));
  }
  let roundComparison = u32CompareTwice(remainder, normalizedDenominator);
  if (roundComparison > 0i || (roundComparison == 0i && (significand & 1u) != 0u)) {
    significand = significand + 1u;
  }
  if (significand == (1u << 24u)) {
    significand = significand >> 1u;
    exponent = exponent + 1i;
  }
  let exponentBits = u32(exponent + 127i) << 23u;
  return bitcast<f32>(exponentBits | (significand & 0x7fffffu));
}

fn u64IsZero(value: U64Words) -> bool {
  return (value.lo | value.hi) == 0u;
}

fn u64Compare(a: U64Words, b: U64Words) -> i32 {
  if (a.hi < b.hi) { return -1i; }
  if (a.hi > b.hi) { return 1i; }
  if (a.lo < b.lo) { return -1i; }
  if (a.lo > b.lo) { return 1i; }
  return 0i;
}

fn u64Add(a: U64Words, b: U64Words) -> U64Words {
  let lo = a.lo + b.lo;
  let carry = select(0u, 1u, lo < a.lo);
  return U64Words(lo, a.hi + b.hi + carry);
}

fn u64Subtract(a: U64Words, b: U64Words) -> U64Words {
  let borrow = select(0u, 1u, a.lo < b.lo);
  return U64Words(a.lo - b.lo, a.hi - b.hi - borrow);
}

fn u64ShiftLeftOne(value: U64Words) -> U64Words {
  return U64Words(value.lo << 1u, (value.hi << 1u) | (value.lo >> 31u));
}

// Exact 32 x 32 -> 64 multiplication using 16-bit limbs. No intermediate
// product exceeds u32 and u64Add propagates both cross-term carries.
fn u64MultiplyU32(a: u32, b: u32) -> U64Words {
  let a0 = a & 0xffffu;
  let a1 = a >> 16u;
  let b0 = b & 0xffffu;
  let b1 = b >> 16u;
  let cross01 = a0 * b1;
  let cross10 = a1 * b0;
  var product = U64Words(a0 * b0, a1 * b1);
  product = u64Add(product, U64Words(cross01 << 16u, cross01 >> 16u));
  return u64Add(product, U64Words(cross10 << 16u, cross10 >> 16u));
}

// Return (2 * remainder) mod denominator and its exact quotient bit. If the
// high bit is set, the mathematical doubling exceeds 2^64; subtracting in
// paired-word modular arithmetic still yields the exact value because
// remainder < denominator implies 2 * remainder - denominator < 2^64.
fn u64DoubleStep(remainder: U64Words, denominator: U64Words) -> U64Step {
  let overflow = (remainder.hi & 0x80000000u) != 0u;
  let doubled = u64ShiftLeftOne(remainder);
  if (overflow || u64Compare(doubled, denominator) >= 0i) {
    return U64Step(u64Subtract(doubled, denominator), 1u);
  }
  return U64Step(doubled, 0u);
}

fn u64CompareTwice(remainder: U64Words, denominator: U64Words) -> i32 {
  if ((remainder.hi & 0x80000000u) != 0u) { return 1i; }
  return u64Compare(u64ShiftLeftOne(remainder), denominator);
}

// Correctly rounded, round-to-nearest-even conversion of one exact positive
// rational to float32. A CoM quotient is a detector coordinate, so every
// admitted nonzero result is normal (2^-64 <= ratio < 2^32). Only integer
// comparisons, subtraction, and shifts determine the IEEE-754 result.
fn ratioU64(numerator: U64Words, denominator: U64Words) -> f32 {
  if (u64IsZero(numerator) || u64IsZero(denominator)) { return 0.0; }
  var exponent = 0i;
  var remainder: U64Words;
  var normalizedDenominator: U64Words;
  if (u64Compare(numerator, denominator) >= 0i) {
    var scaledDenominator = denominator;
    loop {
      if ((scaledDenominator.hi & 0x80000000u) != 0u) { break; }
      let next = u64ShiftLeftOne(scaledDenominator);
      if (u64Compare(next, numerator) > 0i) { break; }
      scaledDenominator = next;
      exponent = exponent + 1i;
    }
    remainder = u64Subtract(numerator, scaledDenominator);
    normalizedDenominator = scaledDenominator;
  } else {
    remainder = numerator;
    loop {
      let step = u64DoubleStep(remainder, denominator);
      remainder = step.remainder;
      exponent = exponent - 1i;
      if (step.bit != 0u) { break; }
    }
    normalizedDenominator = denominator;
  }
  var significand = 1u << 23u;
  for (var bit = 23u; bit > 0u; bit = bit - 1u) {
    let step = u64DoubleStep(remainder, normalizedDenominator);
    remainder = step.remainder;
    significand = significand | (step.bit << (bit - 1u));
  }
  let roundComparison = u64CompareTwice(remainder, normalizedDenominator);
  if (roundComparison > 0i || (roundComparison == 0i && (significand & 1u) != 0u)) {
    significand = significand + 1u;
  }
  if (significand == (1u << 24u)) {
    significand = significand >> 1u;
    exponent = exponent + 1i;
  }
  let exponentBits = u32(exponent + 127i) << 23u;
  return bitcast<f32>(exponentBits | (significand & 0x7fffffu));
}
`;
