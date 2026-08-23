/// <reference types="@webgpu/types" />

export const LOGICAL_PIXEL_HASH_SCHEMA = "quantem.gpu.4dstem-logical-pixels/v1";

export type LogicalPixelDtype = "uint8" | "uint16" | "uint32" | "float32";

export interface LogicalPixelGpuChunk {
  buffer: GPUBuffer;
  startScan: number;
  nScan: number;
}

export interface LogicalPixelHashResult {
  sha256: string;
  bytes: number;
  readbackBytes: number;
  badPixels: number;
  elapsedMs: number;
}

export interface LogicalPixelHashOptions {
  maximumReadbackBytes?: number;
  badPixels?: Uint32Array;
}

export interface LogicalPixelHashProfile {
  fullOutputHashState?: "ready" | "pending" | "complete" | "failed";
  fullOutputHashError?: string;
  fullOutputHashMs?: number;
  fullOutputHashBytes?: number;
  fullOutputHashReadbackBytes?: number;
  fullOutputHashBadPixels?: number;
  fullOutputHashDomain?: "corrected-logical-pixels";
  fullOutputSha256?: string;
  logicalPixelHashSchema?: string;
}

const SHA256_INITIAL = new Uint32Array([
  0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
  0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
]);

const SHA256_ROUND = new Uint32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
  0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
  0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
  0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
  0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
  0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
  0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
  0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
  0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]);

function rotateRight(value: number, count: number): number {
  return (value >>> count) | (value << (32 - count));
}

export class StreamingSha256 {
  private readonly state = new Uint32Array(SHA256_INITIAL);
  private readonly block = new Uint8Array(64);
  private readonly words = new Uint32Array(64);
  private blockLength = 0;
  private bytesHashed = 0;
  private finished = false;

  update(bytes: Uint8Array): void {
    if (this.finished) throw new Error("SHA-256 cannot accept bytes after digestHex().");
    this.bytesHashed += bytes.byteLength;
    let offset = 0;
    if (this.blockLength > 0) {
      const take = Math.min(64 - this.blockLength, bytes.byteLength);
      this.block.set(bytes.subarray(0, take), this.blockLength);
      this.blockLength += take;
      offset += take;
      if (this.blockLength === 64) {
        this.transform(this.block, 0);
        this.blockLength = 0;
      }
    }
    while (offset + 64 <= bytes.byteLength) {
      this.transform(bytes, offset);
      offset += 64;
    }
    if (offset < bytes.byteLength) {
      this.block.set(bytes.subarray(offset), 0);
      this.blockLength = bytes.byteLength - offset;
    }
  }

  digestHex(): string {
    if (!this.finished) this.finish();
    return Array.from(this.state)
      .map((word) => word.toString(16).padStart(8, "0"))
      .join("");
  }

  private finish(): void {
    const messageBytes = this.bytesHashed;
    this.block[this.blockLength++] = 0x80;
    if (this.blockLength > 56) {
      this.block.fill(0, this.blockLength);
      this.transform(this.block, 0);
      this.blockLength = 0;
    }
    this.block.fill(0, this.blockLength, 56);
    const bitLengthHigh = Math.floor(messageBytes / 0x20000000) >>> 0;
    const bitLengthLow = (messageBytes * 8) >>> 0;
    this.writeUint32BigEndian(56, bitLengthHigh);
    this.writeUint32BigEndian(60, bitLengthLow);
    this.transform(this.block, 0);
    this.finished = true;
  }

  private writeUint32BigEndian(offset: number, value: number): void {
    this.block[offset] = value >>> 24;
    this.block[offset + 1] = value >>> 16;
    this.block[offset + 2] = value >>> 8;
    this.block[offset + 3] = value;
  }

  private transform(bytes: Uint8Array, offset: number): void {
    const words = this.words;
    for (let index = 0; index < 16; index++) {
      const byte = offset + index * 4;
      words[index] = (
        (bytes[byte] << 24)
        | (bytes[byte + 1] << 16)
        | (bytes[byte + 2] << 8)
        | bytes[byte + 3]
      ) >>> 0;
    }
    for (let index = 16; index < 64; index++) {
      const left = words[index - 15];
      const right = words[index - 2];
      const sigma0 = rotateRight(left, 7) ^ rotateRight(left, 18) ^ (left >>> 3);
      const sigma1 = rotateRight(right, 17) ^ rotateRight(right, 19) ^ (right >>> 10);
      words[index] = (words[index - 16] + sigma0 + words[index - 7] + sigma1) >>> 0;
    }

    let a = this.state[0];
    let b = this.state[1];
    let c = this.state[2];
    let d = this.state[3];
    let e = this.state[4];
    let f = this.state[5];
    let g = this.state[6];
    let h = this.state[7];
    for (let index = 0; index < 64; index++) {
      const sum1 = rotateRight(e, 6) ^ rotateRight(e, 11) ^ rotateRight(e, 25);
      const choose = (e & f) ^ (~e & g);
      const temporary1 = (h + sum1 + choose + SHA256_ROUND[index] + words[index]) >>> 0;
      const sum0 = rotateRight(a, 2) ^ rotateRight(a, 13) ^ rotateRight(a, 22);
      const majority = (a & b) ^ (a & c) ^ (b & c);
      const temporary2 = (sum0 + majority) >>> 0;
      h = g;
      g = f;
      f = e;
      e = (d + temporary1) >>> 0;
      d = c;
      c = b;
      b = a;
      a = (temporary1 + temporary2) >>> 0;
    }
    this.state[0] = (this.state[0] + a) >>> 0;
    this.state[1] = (this.state[1] + b) >>> 0;
    this.state[2] = (this.state[2] + c) >>> 0;
    this.state[3] = (this.state[3] + d) >>> 0;
    this.state[4] = (this.state[4] + e) >>> 0;
    this.state[5] = (this.state[5] + f) >>> 0;
    this.state[6] = (this.state[6] + g) >>> 0;
    this.state[7] = (this.state[7] + h) >>> 0;
  }
}

function bytesPerPixel(dtype: LogicalPixelDtype): number {
  if (dtype === "uint8") return 1;
  if (dtype === "uint16") return 2;
  return 4;
}

/** @internal Pure correction seam used by the streaming hash and its oracle tests. */
export function correctedLogicalBytesForHash(
  bytes: Uint8Array,
  logicalBytes: number,
  firstPixel: number,
  detectorSize: number,
  pixelBytes: number,
  badPixels: readonly number[],
): Uint8Array {
  if (!Number.isSafeInteger(logicalBytes) || logicalBytes < 0 || logicalBytes > bytes.byteLength) {
    throw new Error(
      `Logical byte count must be an integer in [0, ${bytes.byteLength}]; got ${logicalBytes}.`,
    );
  }
  if (!Number.isSafeInteger(firstPixel) || firstPixel < 0) {
    throw new Error(`First logical pixel must be a non-negative integer; got ${firstPixel}.`);
  }
  if (!Number.isSafeInteger(detectorSize) || detectorSize <= 0) {
    throw new Error(`Detector size must be a positive integer; got ${detectorSize}.`);
  }
  if (![1, 2, 4].includes(pixelBytes) || logicalBytes % pixelBytes !== 0) {
    throw new Error(
      `Logical bytes ${logicalBytes} must contain complete 1-, 2-, or 4-byte pixels; got ${pixelBytes}.`,
    );
  }
  const invalidBadPixel = badPixels.find(
    (pixel) => !Number.isSafeInteger(pixel) || pixel < 0 || pixel >= detectorSize,
  );
  if (invalidBadPixel !== undefined) {
    throw new Error(
      `Bad-pixel index must be in [0, ${detectorSize}); got ${invalidBadPixel}.`,
    );
  }
  if (badPixels.length === 0) return bytes.subarray(0, logicalBytes);
  const corrected = bytes.slice(0, logicalBytes);
  const firstFrame = Math.floor(firstPixel / detectorSize);
  const lastPixel = firstPixel + logicalBytes / pixelBytes;
  const lastFrame = Math.ceil(lastPixel / detectorSize);
  for (let frame = firstFrame; frame < lastFrame; frame++) {
    const frameStart = frame * detectorSize;
    for (const badPixel of badPixels) {
      const pixel = frameStart + badPixel;
      if (pixel < firstPixel || pixel >= lastPixel) continue;
      const localByte = (pixel - firstPixel) * pixelBytes;
      corrected.fill(0, localByte, localByte + pixelBytes);
    }
  }
  return corrected;
}

export async function hashGpuResidentLogicalPixels(
  device: GPUDevice,
  chunks: readonly LogicalPixelGpuChunk[],
  scanCount: number,
  detectorSize: number,
  dtype: LogicalPixelDtype,
  options: LogicalPixelHashOptions = {},
): Promise<LogicalPixelHashResult> {
  const maximumReadbackBytes = options.maximumReadbackBytes ?? 32 * 1024 * 1024;
  const readbackBytes = Math.max(4, Math.floor(maximumReadbackBytes / 4) * 4);
  const pixelBytes = bytesPerPixel(dtype);
  const badPixels = Array.from(new Set(Array.from(options.badPixels ?? [])))
    .sort((left, right) => left - right);
  const invalidBadPixel = badPixels.find((pixel) => pixel >= detectorSize);
  if (invalidBadPixel !== undefined) {
    throw new Error(
      `Bad-pixel index must be below detector size ${detectorSize}; got ${invalidBadPixel}.`,
    );
  }
  const expectedBytes = scanCount * detectorSize * pixelBytes;
  if (!Number.isSafeInteger(expectedBytes) || expectedBytes <= 0) {
    throw new Error(
      `Logical pixel byte count must be a positive safe integer; got ${expectedBytes}.`,
    );
  }
  let expectedStartScan = 0;
  for (const chunk of chunks) {
    if (chunk.startScan !== expectedStartScan) {
      throw new Error(
        `Logical pixel chunks must be contiguous in scan order; expected scan ${expectedStartScan}, got ${chunk.startScan}.`,
      );
    }
    expectedStartScan += chunk.nScan;
  }
  if (expectedStartScan !== scanCount) {
    throw new Error(
      `Logical pixel chunks cover ${expectedStartScan} scan positions, expected ${scanCount}.`,
    );
  }

  const readback = device.createBuffer({
    size: readbackBytes,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const digest = new StreamingSha256();
  const started = performance.now();
  let mapped = false;
  try {
    for (const chunk of chunks) {
      const logicalChunkBytes = chunk.nScan * detectorSize * pixelBytes;
      const packedChunkBytes = Math.ceil(logicalChunkBytes / 4) * 4;
      if (packedChunkBytes > chunk.buffer.size) {
        throw new Error(
          `Logical pixel chunk needs ${packedChunkBytes} packed bytes, but its GPU buffer has ${chunk.buffer.size}.`,
        );
      }
      let sourceOffset = 0;
      while (sourceOffset < logicalChunkBytes) {
        const logicalCopyBytes = Math.min(readbackBytes, logicalChunkBytes - sourceOffset);
        const alignedCopyBytes = Math.ceil(logicalCopyBytes / 4) * 4;
        const encoder = device.createCommandEncoder();
        encoder.copyBufferToBuffer(
          chunk.buffer,
          sourceOffset,
          readback,
          0,
          alignedCopyBytes,
        );
        device.queue.submit([encoder.finish()]);
        await readback.mapAsync(GPUMapMode.READ, 0, alignedCopyBytes);
        mapped = true;
        const bytes = new Uint8Array(readback.getMappedRange(0, alignedCopyBytes));
        const firstPixel = chunk.startScan * detectorSize + sourceOffset / pixelBytes;
        digest.update(correctedLogicalBytesForHash(
          bytes,
          logicalCopyBytes,
          firstPixel,
          detectorSize,
          pixelBytes,
          badPixels,
        ));
        readback.unmap();
        mapped = false;
        sourceOffset += logicalCopyBytes;
      }
    }
  } finally {
    if (mapped) readback.unmap();
    readback.destroy();
  }
  return {
    sha256: digest.digestHex(),
    bytes: expectedBytes,
    readbackBytes,
    badPixels: badPixels.length,
    elapsedMs: performance.now() - started,
  };
}

export function clearGpuResidentLogicalPixelHash(): void {
  delete (
    globalThis as {
      __QT_H5_RUN_FULL_OUTPUT_HASH?: () => Promise<LogicalPixelHashProfile>;
    }
  ).__QT_H5_RUN_FULL_OUTPUT_HASH;
}

export function exposeGpuResidentLogicalPixelHash(
  device: GPUDevice,
  chunks: readonly LogicalPixelGpuChunk[],
  scanCount: number,
  detectorSize: number,
  dtype: LogicalPixelDtype,
  profile: LogicalPixelHashProfile,
  badPixels: Uint32Array,
): void {
  profile.fullOutputHashState = "ready";
  let hashPromise: Promise<LogicalPixelHashProfile> | null = null;
  const runFullOutputHash = (): Promise<LogicalPixelHashProfile> => {
    if (hashPromise) return hashPromise;
    profile.fullOutputHashState = "pending";
    hashPromise = hashGpuResidentLogicalPixels(
      device,
      chunks,
      scanCount,
      detectorSize,
      dtype,
      { badPixels },
    ).then((result) => {
      profile.fullOutputSha256 = result.sha256;
      profile.logicalPixelHashSchema = LOGICAL_PIXEL_HASH_SCHEMA;
      profile.fullOutputHashMs = result.elapsedMs;
      profile.fullOutputHashBytes = result.bytes;
      profile.fullOutputHashReadbackBytes = result.readbackBytes;
      profile.fullOutputHashBadPixels = result.badPixels;
      profile.fullOutputHashDomain = "corrected-logical-pixels";
      profile.fullOutputHashState = "complete";
      return profile;
    }).catch((error: unknown) => {
      profile.fullOutputHashError = error instanceof Error ? error.message : String(error);
      profile.fullOutputHashState = "failed";
      return profile;
    });
    return hashPromise;
  };
  (
    globalThis as {
      __QT_H5_RUN_FULL_OUTPUT_HASH?: () => Promise<LogicalPixelHashProfile>;
    }
  ).__QT_H5_RUN_FULL_OUTPUT_HASH = runFullOutputHash;
}
