/** Canonical display statistics shared by quantem.gpu browser clients. */

/** Find min/max range of a Float32Array, filtering out NaN and Infinity. */
export function findDataRange(data: Float32Array): { min: number; max: number } {
  let min = Infinity, max = -Infinity;
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (!isFinite(v)) continue;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  // If no finite values found, return zeros
  if (min === Infinity) return { min: 0, max: 0 };
  return { min, max };
}

/** Signed log1p. For non-negative inputs identical to log1p(x); for negatives
 *  returns -log1p(|x|) so diff_mode frames don't collapse to zero. */
export function signedLog1p(value: number): number {
  return value >= 0 ? Math.log1p(value) : -Math.log1p(-value);
}

export function applyLogScale(data: Float32Array): Float32Array {
  const result = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) result[i] = signedLog1p(data[i]);
  return result;
}

/** Apply signed log1p scale into a pre-allocated buffer. Avoids per-frame allocation. */
export function applyLogScaleInPlace(data: Float32Array, out: Float32Array): Float32Array {
  for (let i = 0; i < data.length; i++) {
    out[i] = signedLog1p(data[i]);
  }
  return out;
}

type NumericArray = ArrayLike<number>;

/** Percentile-based clipping using O(n) histogram approach.
 *  Also returns data min/max so callers can skip a redundant findDataRange scan. */
export function percentileClip(
  data: NumericArray, pLow: number, pHigh: number,
): { vmin: number; vmax: number; min: number; max: number } {
  const len = data.length;
  if (len === 0) return { vmin: 0, vmax: 0, min: 0, max: 0 };

  // Pass 1: find min/max
  let min = Infinity, max = -Infinity;
  let finiteCount = 0;
  for (let i = 0; i < len; i++) {
    const v = data[i];
    if (!isFinite(v)) continue;
    finiteCount++;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  if (finiteCount === 0) return { vmin: 0, vmax: 0, min: 0, max: 0 };
  if (min === max) return { vmin: min, vmax: max, min, max };

  // Pass 2: build histogram
  const NUM_BINS = 1024;
  const bins = new Uint32Array(NUM_BINS);
  const range = max - min;
  const scale = (NUM_BINS - 1) / range;
  for (let i = 0; i < len; i++) {
    const value = data[i];
    if (isFinite(value)) bins[Math.floor((value - min) * scale)]++;
  }

  // Walk cumulative histogram to find percentile values. Linear-interpolate
  // between bin edges where the target count is crossed so the result is
  // continuous in the data, not snapped to 1024 discrete bin midpoints.
  const lowCount = finiteCount * (pLow / 100);
  const highCount = finiteCount * (pHigh / 100);
  let cumSum = 0;
  let vmin = min, vmax = max;
  let prevSum = 0;
  for (let i = 0; i < NUM_BINS; i++) {
    prevSum = cumSum;
    cumSum += bins[i];
    if (cumSum >= lowCount) {
      const frac = (lowCount - prevSum) / Math.max(1, cumSum - prevSum);
      vmin = min + ((i + frac) / NUM_BINS) * range;
      break;
    }
  }
  cumSum = 0;
  prevSum = 0;
  for (let i = 0; i < NUM_BINS; i++) {
    prevSum = cumSum;
    cumSum += bins[i];
    if (cumSum >= highCount) {
      const frac = (highCount - prevSum) / Math.max(1, cumSum - prevSum);
      vmax = min + ((i + frac) / NUM_BINS) * range;
      break;
    }
  }
  return { vmin, vmax, min, max };
}

/** Compute mean, min, max, and standard deviation of a Float32Array. */
export function computeStats(data: Float32Array): { mean: number; min: number; max: number; std: number } {
  if (data.length === 0) return { mean: 0, min: 0, max: 0, std: 0 };
  let sum = 0, min = Infinity, max = -Infinity;
  let count = 0;
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (!isFinite(v)) continue;
    count++;
    sum += v;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  if (count === 0) return { mean: 0, min: 0, max: 0, std: 0 };
  const mean = sum / count;
  let variance = 0;
  for (let i = 0; i < data.length; i++) {
    if (isFinite(data[i])) variance += (data[i] - mean) ** 2;
  }
  const std = Math.sqrt(variance / count);
  return { mean, min, max, std };
}

/** Convert histogram slider percentages (0-100) to vmin/vmax in data space. */
export function sliderRange(
  dataMin: number, dataMax: number, vminPct: number, vmaxPct: number,
): { vmin: number; vmax: number } {
  const range = dataMax - dataMin;
  return {
    vmin: dataMin + (vminPct / 100) * range,
    vmax: dataMin + (vmaxPct / 100) * range,
  };
}

/** Compute normalized histogram bins from Float32Array.
 *  fixedMin/fixedMax pin bin edges to a global range (so scrubbing through
 *  a stack doesn't rescale per-frame). Defaults to per-array min/max. */
export function computeHistogramFromBytes(
  data: NumericArray | null,
  numBins = 256,
  fixedMin?: number,
  fixedMax?: number,
): number[] {
  if (!data || data.length === 0) return new Array(numBins).fill(0);
  const bins = new Array(numBins).fill(0);
  let min: number, max: number;
  if (fixedMin !== undefined && fixedMax !== undefined && isFinite(fixedMin) && isFinite(fixedMax)) {
    min = fixedMin;
    max = fixedMax;
  } else {
    min = Infinity; max = -Infinity;
    for (let i = 0; i < data.length; i++) {
      const v = data[i];
      if (isFinite(v)) { if (v < min) min = v; if (v > max) max = v; }
    }
    if (!isFinite(min) || !isFinite(max)) return bins;
  }
  if (!(max > min)) {
    let count = 0;
    for (let i = 0; i < data.length; i++) if (isFinite(data[i])) count++;
    if (count > 0) bins[Math.floor(numBins / 2)] = 1;
    return bins;
  }
  const range = max - min;
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (isFinite(v)) {
      // Clamp into last bin so max-value pixels aren't silently dropped.
      let idx = Math.floor(((v - min) / range) * numBins);
      if (idx === numBins) idx = numBins - 1;
      if (idx >= 0 && idx < numBins) bins[idx]++;
    }
  }
  const maxCount = Math.max(...bins);
  if (maxCount > 0) for (let i = 0; i < numBins; i++) bins[i] /= maxCount;
  return bins;
}
