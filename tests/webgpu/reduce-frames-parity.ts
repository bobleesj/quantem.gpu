/// <reference types="@webgpu/types" />
import { DetectorCompute } from '../../src/quantem/gpu/detector/webgpu/backend';

type Case = { name: string; mode: number; words: Uint32Array; values: number[]; scanCount: number; detSize: number };

/** Exact scan-ROI patterns on a physical adapter for every resident mode, across two chunks.

The integer sums exceed 2^32 per pixel, which a 32-bit accumulator wraps: uint16 counts of
65535 over 70001 frames, and uint32 counts near 2^32 that wrap one thread's slice after two
frames. Float32 data must be read as IEEE-754 bits, not as packed uint16 counts; its values
are quarter-integers whose sums stay exact in float32, so every case compares exactly against
`Math.fround(sum)` and `Math.fround(sum / selected)` computed in float64.
*/
export async function runReduceFramesParity(device: GPUDevice) {
  const detSize = 6;
  const cases: Case[] = [];
  const counts16 = (scanCount: number) => Array.from({length: scanCount * detSize}, (_, i) => (i % detSize === 0 ? 65535 : (i * 7919) % 65536));
  const packed16 = (values: number[]) => {
    const words = new Uint32Array(Math.ceil(values.length / 2));
    values.forEach((value, i) => { words[i >> 1] |= value << ((i & 1) * 16); });
    return words;
  };
  const uint16 = counts16(70001);
  cases.push({name: 'uint16', mode: 0, words: packed16(uint16), values: uint16, scanCount: 70001, detSize});
  const uint32 = Array.from({length: 300 * detSize}, (_, i) => 4294967295 - (i % 1013));
  cases.push({name: 'uint32', mode: 3, words: Uint32Array.from(uint32), values: uint32, scanCount: 300, detSize});
  const float32 = Array.from({length: 300 * detSize}, (_, i) => ((i * 37) % 200 - 100) * 0.25);
  cases.push({name: 'float32', mode: 2, words: new Uint32Array(Float32Array.from(float32).buffer), values: float32, scanCount: 300, detSize});
  const results: Record<string, {checked: number}> = {};
  for (const item of cases) {
    // Two chunks split at a non-multiple of the frame blocks, so slices and chunk dispatches both accumulate.
    const split = Math.floor(item.scanCount / 3) + 1;
    const wordsPerFrame = item.words.length / item.scanCount;
    const chunkBuffer = (first: number, stop: number) => {
      const words = item.words.slice(Math.round(first * wordsPerFrame), Math.round(stop * wordsPerFrame));
      const buffer = device.createBuffer({size: Math.max(16, words.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST});
      device.queue.writeBuffer(buffer, 0, words);
      return {buffer, startScan: first, nScan: stop - first};
    };
    const chunks = [chunkBuffer(0, split), chunkBuffer(split, item.scanCount)];
    const compute = DetectorCompute.fromGpuChunks(device, chunks, item.scanCount, item.detSize, item.mode);
    let checked = 0;
    try {
      for (const selection of ['all', 'odd']) {
        const scanMask = Uint32Array.from({length: item.scanCount}, (_, scan) => (selection === 'all' || scan % 2 === 1 ? 1 : 0));
        const selected = scanMask.reduce((total, value) => total + value, 0);
        const sums = new Float64Array(item.detSize);
        for (let scan = 0; scan < item.scanCount; scan++) {
          if (!scanMask[scan]) continue;
          for (let pixel = 0; pixel < item.detSize; pixel++) sums[pixel] += item.values[scan * item.detSize + pixel];
        }
        if (item.mode !== 2 && selection === 'all' && Math.max(...sums) < 2 ** 32) throw Error(`${item.name} sums must exceed 2^32`);
        const total = await compute.reduceFrames(scanMask, false);
        const mean = await compute.reduceFrames(scanMask, true);
        for (let pixel = 0; pixel < item.detSize; pixel++) {
          const expectedTotal = Math.fround(sums[pixel]);
          const expectedMean = Math.fround(sums[pixel] / selected);
          if (total[pixel] !== expectedTotal) throw Error(`${item.name} ${selection} sum pixel ${pixel}: ${total[pixel]} != ${expectedTotal}`);
          if (mean[pixel] !== expectedMean) throw Error(`${item.name} ${selection} mean pixel ${pixel}: ${mean[pixel]} != ${expectedMean}`);
          checked += 2;
        }
      }
    } finally {
      await device.queue.onSubmittedWorkDone();
      chunks.forEach((chunk) => chunk.buffer.destroy());
    }
    results[item.name] = {checked};
  }
  return {results, allExact: true};
}
