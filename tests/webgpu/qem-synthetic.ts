/** Synthetic authenticated .qem containers for CPU-only admission tests. */
import { createHash } from "node:crypto";

const sha256 = (bytes: Uint8Array) => createHash("sha256").update(bytes).digest();

/** One-pixel detector, one 512-scan entropy-coded block per chunk with the given payload sizes.
 *
 * Admission accepts these streams; their bytes are zeros and are never decoded.
 * Large payloads cross 64 MiB authentication chunks and GPU group limits cheaply.
 */
export function syntheticQem(payloadBytes: number[]): Uint8Array {
  const widths = [1, 4, 1, 4, 8, 1];
  let previousEnd = 0;
  const chunks = payloadBytes.map((payload, index) => {
    const counts = [payload, 2, 1, 0, 0, 0];
    const arrays = counts.map((count, array) => {
      const offset = Math.ceil(previousEnd / 8) * 8;
      previousEnd = offset + count * widths[array];
      return { offset, count };
    });
    return { first: index * 512, scans: 512, arrays };
  });
  const payload = new Uint8Array(previousEnd);
  chunks.forEach(({ arrays }, index) => {
    new DataView(payload.buffer).setUint32(arrays[1].offset + 4, payloadBytes[index], true);
  });
  const chunkBytes = 64 << 20;
  const hashes = [];
  for (let begin = 0; begin < payload.length; begin += chunkBytes) {
    hashes.push(sha256(payload.subarray(begin, begin + chunkBytes)).toString("hex"));
  }
  const shape = [payloadBytes.length, 512, 1, 1];
  const names = ["scan_row", "scan_column", "detector_row", "detector_column"];
  const header = new TextEncoder().encode(JSON.stringify({
    container: "quantem.qem", container_version: 1,
    codec: "runtime-column-rans-spatial-v2", profile: "runtime-column-rans-spatial-v2",
    dtype: "uint16", shape, interval: 512, bytes: payload.length, sha256: hashes, valid: "80", chunks,
    metadata: {},
    scientific_metadata: {
      schema: "quantem.scientific-metadata/2",
      axes: names.map((name, axis) => ({ name, size: shape[axis] })),
    },
  }));
  const file = new Uint8Array(56 + header.length + payload.length);
  file.set(new TextEncoder().encode("QEMDATA1"));
  const prefix = new DataView(file.buffer);
  prefix.setBigUint64(8, BigInt(header.length), true);
  prefix.setBigUint64(16, BigInt(56 + header.length), true);
  file.set(sha256(header), 24);
  file.set(header, 56);
  file.set(payload, 56 + header.length);
  return file;
}
