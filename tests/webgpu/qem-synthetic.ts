/** Synthetic authenticated .qem containers for CPU-only admission tests. */
import { createHash } from "node:crypto";

const sha256 = (bytes: Uint8Array) => createHash("sha256").update(bytes).digest();

/** One-pixel detector; `chunks[c][b]` is the payload size of 512-scan block b in chunk c.
 *
 * Every block is an entropy-coded column, which admission accepts at any size of
 * at least four bytes; the bytes are a counter pattern and are never decoded.
 * Large sizes cross 64 MiB authentication chunks and GPU group limits cheaply.
 * `misdeclare` makes the first chunk's authenticated offset table claim one byte
 * fewer than its payload.
 */
export function syntheticQem(chunks: number[][], misdeclare = false): Uint8Array {
  const widths = [1, 4, 1, 4, 8, 1];
  let previousEnd = 0;
  const layout = chunks.map((blocks, index) => {
    const payload = blocks.reduce((sum, size) => sum + size, 0);
    const counts = [payload, blocks.length + 1, blocks.length, 0, 0, 0];
    const arrays = counts.map((count, array) => {
      const offset = Math.ceil(previousEnd / 8) * 8;
      previousEnd = offset + count * widths[array];
      return { offset, count };
    });
    return { first: chunks.slice(0, index).reduce((sum, previous) => sum + previous.length * 512, 0), scans: blocks.length * 512, arrays };
  });
  const payload = new Uint8Array(previousEnd);
  for (let index = 0; index < payload.length; index += 4096) payload[index] = (index / 4096) & 255;
  layout.forEach(({ arrays }, index) => {
    const table = new DataView(payload.buffer, arrays[1].offset, arrays[1].count * 4);
    let end = 0;
    table.setUint32(0, 0, true);
    chunks[index].forEach((size, block) => { end += size; table.setUint32((block + 1) * 4, end, true); });
    if (misdeclare && index === 0) table.setUint32(chunks[0].length * 4, end - 1, true);
    payload.fill(0, arrays[2].offset, arrays[2].offset + arrays[2].count);
  });
  const chunkBytes = 64 << 20;
  const hashes = [];
  for (let begin = 0; begin < payload.length; begin += chunkBytes) {
    hashes.push(sha256(payload.subarray(begin, begin + chunkBytes)).toString("hex"));
  }
  const shape = [1, layout.reduce((sum, chunk) => sum + chunk.scans, 0), 1, 1];
  const names = ["scan_row", "scan_column", "detector_row", "detector_column"];
  const header = new TextEncoder().encode(JSON.stringify({
    container: "quantem.qem", container_version: 1,
    codec: "runtime-column-rans-spatial-v2", profile: "runtime-column-rans-spatial-v2",
    dtype: "uint16", shape, interval: 512, bytes: payload.length, sha256: hashes, valid: "80", chunks: layout,
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
