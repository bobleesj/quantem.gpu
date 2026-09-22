/** Authenticated integer QEM admission; detector measurements remain encoded. */
import { SectionSHA256, parseManifest, requireANS } from "./count-ans";
import tables from "../../../io/qem-rans-tables-v1.json";
import type { RansByteSource } from "./rans-source";
import type { RansManifest } from "./rans";

type Span = { offset: number; count: number };
type Chunk = { first: number; scans: number; arrays: Span[] };
type Header = {
  container: string;
  container_version: number;
  codec: string;
  profile: string;
  dtype: string;
  shape: number[];
  interval: number;
  bytes: number;
  sha256: string[];
  valid: string;
  chunks: Chunk[];
  metadata: Record<string, unknown>;
  scientific_metadata: Record<string, unknown>;
};
const digest = (bytes: Uint8Array) => {
  const hash = new SectionSHA256();
  hash.update(bytes);
  return hash.finish();
};

export async function qemFileSource(
  file: File,
  onStatus: (text: string) => void,
): Promise<RansByteSource> {
  requireANS(file.size >= 56, "truncated envelope");
  const prefix = new Uint8Array(await file.slice(0, 56).arrayBuffer());
  requireANS(
    new TextDecoder().decode(prefix.subarray(0, 8)) === "QEMDATA1",
    "unsupported container; re-export the original acquisition as .qem",
  );
  const view = new DataView(prefix.buffer);
  const length = Number(view.getBigUint64(8, true)),
    body = Number(view.getBigUint64(16, true));
  requireANS(
    Number.isSafeInteger(length) &&
      length > 0 &&
      length <= 16 << 20 &&
      body === length + 56 &&
      body <= file.size,
    "invalid envelope bounds",
  );
  const json = new Uint8Array(await file.slice(56, body).arrayBuffer());
  requireANS(
    digest(json) ===
      [...prefix.subarray(24)]
        .map((value) => value.toString(16).padStart(2, "0"))
        .join(""),
    "header checksum mismatch",
  );
  const h = parseManifest(new TextDecoder().decode(json)) as unknown as Header;
  requireANS(
    h && typeof h === "object" && !Array.isArray(h),
    "header must be a JSON object",
  );
  requireANS(
    h.container === "quantem.qem" && h.container_version === 1,
    "unsupported container version",
  );
  requireANS(
    h.codec === "runtime-column-rans-spatial-v2" &&
      h.profile === h.codec &&
      ["uint8", "uint16"].includes(h.dtype),
    "this browser supports integer QEM only; open float32 QEM in the native application or a Python GPU session",
  );
  requireANS(
    Array.isArray(h.shape) &&
      h.shape.length === 4 &&
      h.shape.every((value) => Number.isSafeInteger(value) && value > 0),
    "invalid four-dimensional geometry",
  );
  requireANS(
    h.metadata && typeof h.metadata === "object" && !Array.isArray(h.metadata),
    "metadata must be an object",
  );
  const scientific = h.scientific_metadata;
  requireANS(
    scientific &&
      [
        "quantem.scientific-metadata/1",
        "quantem.scientific-metadata/2",
      ].includes(String(scientific.schema)),
    "unsupported scientific metadata schema",
  );
  const axes = scientific.axes as { name: string; size: number }[];
  const names = ["scan_row", "scan_column", "detector_row", "detector_column"];
  requireANS(
    Array.isArray(axes) &&
      axes.length === 4 &&
      axes.every(
        (axis, index) =>
          axis && axis.name === names[index] && axis.size === h.shape[index],
      ),
    "scientific axes disagree with stored geometry",
  );
  for (const section of ["calibration_overrides", "electron_microscope"]) {
    const quantities = scientific[section] ?? {};
    requireANS(
      quantities &&
        typeof quantities === "object" &&
        !Array.isArray(quantities),
      `${section} must contain named quantities`,
    );
    requireANS(
      !Object.keys(quantities).some((key) =>
        /(?:pixel_size|reciprocal_pixel_size)_[xy]$/.test(key),
      ),
      "retired x/y calibration names; re-export with row/column metadata",
    );
  }
  requireANS(
    h.interval === 512 &&
      Number.isSafeInteger(h.bytes) &&
      body + h.bytes === file.size,
    "invalid interval or payload bounds",
  );
  const [rows, cols, detRows, detCols] = h.shape,
    scans = rows * cols,
    K = detRows * detCols;
  requireANS(
    Number.isSafeInteger(scans) &&
      scans < 2 ** 32 &&
      K < 2 ** 24 &&
      K * (h.dtype === "uint8" ? 255 : 65535) < 2 ** 32,
    "geometry exceeds browser integer-product capacity; use the native GPU application",
  );
  const chunkBytes = 64 << 20;
  requireANS(
    Array.isArray(h.sha256) &&
      h.sha256.length === Math.ceil(h.bytes / chunkBytes),
    "missing payload checksums",
  );
  for (let index = 0; index < h.sha256.length; index++) {
    onStatus(`Verifying .qem ${index + 1}/${h.sha256.length}`);
    const hash = new SectionSHA256(),
      end = Math.min(h.bytes, (index + 1) * chunkBytes);
    for (let offset = index * chunkBytes; offset < end; offset += 8 << 20)
      hash.update(
        new Uint8Array(
          await file
            .slice(body + offset, body + Math.min(end, offset + (8 << 20)))
            .arrayBuffer(),
        ),
      );
    requireANS(hash.finish() === h.sha256[index], "payload checksum mismatch");
  }
  requireANS(
    typeof h.valid === "string" &&
      /^[0-9a-f]*$/.test(h.valid) &&
      h.valid.length === Math.ceil(K / 8) * 2,
    "invalid detector validity mask",
  );
  const badPixels: number[] = [];
  for (let k = 0; k < K; k++)
    if (
      !(
        parseInt(h.valid.slice((k >> 3) * 2, (k >> 3) * 2 + 2), 16) &
        (128 >> (k & 7))
      )
    )
      badPixels.push(k);
  const entries = new Uint32Array(64 * 33 * 2);
  tables.frequencies.forEach((frequencies, model) => {
    let cumulative = 0;
    frequencies.forEach((frequency, symbol) => {
      const at = (model * 33 + symbol) * 2;
      entries[at] = (cumulative << 16) | symbol;
      entries[at + 1] = frequency;
      cumulative += frequency;
    });
    requireANS(cumulative === 1024, "invalid fixed probability table");
  });
  const blockMeta: {
    index: number;
    bytes: number;
    model: number;
    byte_start: number;
    byte_end: number;
    frames: number;
  }[] = [];
  const offsets: Uint32Array<ArrayBuffer>[] = [],
    columns: Uint32Array<ArrayBuffer>[] = [];
  let nextScan = 0,
    previousEnd = 0;
  requireANS(
    Array.isArray(h.chunks) && h.chunks.length > 0,
    "missing encoded chunks",
  );
  for (const chunk of h.chunks) {
    requireANS(
      chunk.first === nextScan &&
        Number.isSafeInteger(chunk.scans) &&
        chunk.scans > 0 &&
        chunk.first + chunk.scans <= scans &&
        (chunk.first + chunk.scans === scans || chunk.scans % 512 === 0),
      "noncontiguous or unaligned scan chunks",
    );
    requireANS(
      Array.isArray(chunk.arrays) && chunk.arrays.length === 6,
      "missing typed chunk arrays",
    );
    const widths = [1, 4, 1, 4, 8, 1];
    chunk.arrays.forEach((span, index) => {
      requireANS(
        Number.isSafeInteger(span.offset) &&
          Number.isSafeInteger(span.count) &&
          span.count >= 0 &&
          span.offset === Math.ceil(previousEnd / 8) * 8 &&
          span.offset + span.count * widths[index] <= h.bytes,
        "invalid chunk array bounds",
      );
      previousEnd = span.offset + span.count * widths[index];
    });
    const [payload, offsetSpan, modelSpan] = chunk.arrays,
      blocks = Math.ceil(chunk.scans / 512);
    requireANS(
      offsetSpan.count === blocks * K + 1 && modelSpan.count === blocks * K,
      "invalid stream table dimensions",
    );
    const read = (span: Span, size: number) =>
      file
        .slice(body + span.offset, body + span.offset + span.count * size)
        .arrayBuffer();
    const local = new Uint32Array(await read(offsetSpan, 4)),
      models = new Uint8Array(await read(modelSpan, 1));
    requireANS(
      local[0] === 0 && local[local.length - 1] === payload.count,
      "stream table does not partition payload",
    );
    for (let block = 0; block < blocks; block++) {
      const first = block * K,
        start = local[first],
        end = local[first + K],
        frames = Math.min(512, chunk.scans - block * 512);
      const relative = new Uint32Array(K + 1),
        metadata = new Uint32Array(K * 3);
      for (let k = 0; k < K; k++) {
        const model = models[first + k],
          size = local[first + k + 1] - local[first + k];
        requireANS(
          local[first + k + 1] >= local[first + k] &&
            (model < 64
              ? size >= 4
              : model === 252
                ? size % 2 === 0 && size <= frames * 2
                : model === 253
                  ? size === 0
                  : model === 254
                    ? size === frames * 2
                    : model === 255 && size === 2),
          "invalid encoded stream mode or length",
        );
        metadata.set(
          model < 64
            ? [model * 33, (model + 1) * 33, 2]
            : [
                0,
                0,
                model === 252 ? 5 : model === 253 ? 3 : model === 254 ? 1 : 4,
              ],
          k * 3,
        );
        relative[k] = local[first + k] - start;
      }
      relative[K] = end - start;
      const index = blockMeta.length;
      blockMeta.push({
        index,
        bytes: end - start,
        model: index,
        byte_start: body + payload.offset + start,
        byte_end: body + payload.offset + end,
        frames,
      });
      offsets.push(relative);
      columns.push(metadata);
    }
    nextScan += chunk.scans;
  }
  requireANS(
    nextScan === scans && previousEnd === h.bytes,
    "incomplete scan coverage or undeclared payload",
  );
  const mapped: RansManifest = {
    scan_shape: [rows, cols],
    detector_shape: [detRows, detCols],
    native_dtype: h.dtype as "uint8" | "uint16",
    bad_pixels: badPixels,
    source_metadata: {
      ...h.metadata,
      scientific_metadata: h.scientific_metadata,
    },
    tilts: [
      {
        tilt: 0,
        K,
        frames: 512,
        blocks: blockMeta.length,
        scale: 10,
        model_frames: 512,
        binary_lookup: true,
        payload_url: "payload",
        blocks_meta: blockMeta,
        models: blockMeta.map((block) => ({
          index: block.index,
          symbols: 64 * 33,
          colmeta_url: `columns-${block.index}`,
          entries_url: "entries",
          lut_url: "lookup",
        })),
      },
    ],
  };
  return {
    mode: "local-folder",
    async read(name, start, end) {
      if (name === "manifest.json")
        return new TextEncoder().encode(JSON.stringify(mapped)).buffer;
      if (name === "entries") return entries.buffer;
      if (name === "lookup") return new ArrayBuffer(4);
      if (name === "payload") {
        requireANS(
          start !== undefined &&
            end !== undefined &&
            blockMeta.some(
              (block) =>
                start >= block.byte_start &&
                end <= block.byte_end &&
                end >= start,
            ),
          "invalid payload request",
        );
        return file.slice(start, end).arrayBuffer();
      }
      const column = /^columns-(\d+)$/.exec(name);
      if (column) return columns[Number(column[1])].buffer;
      const offset = /^t0-offsets-(\d+)\.u32$/.exec(name);
      requireANS(
        offset && offsets[Number(offset[1])],
        "invalid stream table request",
      );
      return offsets[Number(offset[1])].buffer;
    },
  };
}
