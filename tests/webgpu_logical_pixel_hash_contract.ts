import assert from "node:assert/strict";
import { createHash } from "node:crypto";

import {
  StreamingSha256,
  correctedLogicalBytesForHash,
} from "../src/quantem/gpu/io/backends/webgpu/logical-pixel-hash.ts";

function trustedSha256(bytes: Uint8Array): string {
  return createHash("sha256").update(bytes).digest("hex");
}

function streamingSha256(bytes: Uint8Array, chunkSizes: readonly number[]): string {
  const digest = new StreamingSha256();
  let offset = 0;
  let chunkIndex = 0;
  while (offset < bytes.byteLength) {
    const size = chunkSizes[chunkIndex % chunkSizes.length];
    digest.update(bytes.subarray(offset, Math.min(bytes.byteLength, offset + size)));
    offset += size;
    chunkIndex += 1;
  }
  if (bytes.byteLength === 0) digest.update(bytes);
  return digest.digestHex();
}

const encoder = new TextEncoder();
const knownVectors = [
  new Uint8Array(0),
  encoder.encode("abc"),
  encoder.encode("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
  new Uint8Array(1_000_000).fill("a".charCodeAt(0)),
];
const chunkPlans = [
  [1],
  [2, 3, 5, 7, 11, 13],
  [55, 56, 63, 64, 65],
  [1024, 17, 4096],
];

for (const bytes of knownVectors) {
  const expected = trustedSha256(bytes);
  for (const chunks of chunkPlans) {
    assert.equal(streamingSha256(bytes, chunks), expected);
  }
}

for (const length of [1, 55, 56, 63, 64, 65, 127, 128, 129, 4097, 65_537]) {
  const bytes = Uint8Array.from({ length }, (_, index) => (index * 131 + 17) & 0xff);
  const expected = trustedSha256(bytes);
  for (const chunks of chunkPlans) {
    assert.equal(streamingSha256(bytes, chunks), expected);
  }
}

function encodePixels(values: readonly number[], pixelBytes: 1 | 2 | 4): Uint8Array {
  const bytes = new Uint8Array(values.length * pixelBytes);
  const view = new DataView(bytes.buffer);
  values.forEach((value, index) => {
    if (pixelBytes === 1) view.setUint8(index, value);
    else if (pixelBytes === 2) view.setUint16(index * 2, value, true);
    else view.setUint32(index * 4, value, true);
  });
  return bytes;
}

for (const pixelBytes of [1, 2, 4] as const) {
  const detectorSize = 7;
  const badPixels = [0, 3, 6];
  const values = Array.from({ length: detectorSize * 5 }, (_, index) => index + 1);
  const source = encodePixels(values, pixelBytes);
  const monolithic = correctedLogicalBytesForHash(
    source,
    source.byteLength,
    0,
    detectorSize,
    pixelBytes,
    badPixels,
  );

  const correctedWindows: Uint8Array[] = [];
  const pixelWindows = [2, 6, 4, 8, 3, 12];
  let firstPixel = 0;
  for (const requestedPixels of pixelWindows) {
    if (firstPixel >= values.length) break;
    const pixels = Math.min(requestedPixels, values.length - firstPixel);
    const byteStart = firstPixel * pixelBytes;
    const window = source.slice(byteStart, byteStart + pixels * pixelBytes);
    const original = window.slice();
    correctedWindows.push(correctedLogicalBytesForHash(
      window,
      window.byteLength,
      firstPixel,
      detectorSize,
      pixelBytes,
      badPixels,
    ));
    assert.deepEqual(window, original, "correction must not mutate resident readback bytes");
    firstPixel += pixels;
  }
  const joined = new Uint8Array(monolithic.byteLength);
  let byteOffset = 0;
  for (const window of correctedWindows) {
    joined.set(window, byteOffset);
    byteOffset += window.byteLength;
  }
  assert.equal(byteOffset, monolithic.byteLength);
  assert.deepEqual(joined, monolithic, "window boundaries must preserve canonical correction");
  assert.equal(streamingSha256(joined, [63, 64, 65]), trustedSha256(monolithic));
}

assert.throws(
  () => correctedLogicalBytesForHash(new Uint8Array(4), 5, 0, 2, 2, []),
  /Logical byte count/,
);
assert.throws(
  () => correctedLogicalBytesForHash(new Uint8Array(4), 3, 0, 2, 2, []),
  /complete/,
);
assert.throws(
  () => correctedLogicalBytesForHash(new Uint8Array(4), 4, 0, 2, 2, [2]),
  /Bad-pixel index/,
);

console.log(JSON.stringify({
  status: "passed",
  trustedOracle: "node:crypto SHA-256",
  knownVectors: knownVectors.length,
  updateChunkPlans: chunkPlans.length,
  boundaryLengths: 11,
  correctionPixelWidths: [1, 2, 4],
}));
