/** Admit a self-contained QEM file to the existing resident rANS engine.
 * Only encoded tables are transformed on the host. Native counts are decoded
 * by the shared GPU recurrence, including literal streams and tail blocks.
 */
import type { RansByteSource } from "./rans-source";
import type { RansManifest } from "./rans";
import { qemFileSource } from "./qem-source";

// Incremental SHA-256 keeps payload verification bounded; WebCrypto.digest
// requires an entire multi-gigabyte section in one ArrayBuffer.
export class SectionSHA256 {
  private state = new Uint32Array([0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]);
  private block = new Uint8Array(64);
  private words = new Uint32Array(64);
  private used = 0;
  private length = 0;
  private static readonly constants = new Uint32Array([
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
  ]);
  private transform(bytes: Uint8Array, offset: number): void {
    const rotate = (value: number, shift: number) => (value >>> shift) | (value << (32 - shift));
    const words = this.words;
    for (let i = 0; i < 16; i++) words[i] = (bytes[offset + i * 4] << 24) | (bytes[offset + i * 4 + 1] << 16) | (bytes[offset + i * 4 + 2] << 8) | bytes[offset + i * 4 + 3];
    for (let i = 16; i < 64; i++) {
      const a = words[i - 15], b = words[i - 2];
      words[i] = words[i - 16] + (rotate(a, 7) ^ rotate(a, 18) ^ (a >>> 3)) + words[i - 7] + (rotate(b, 17) ^ rotate(b, 19) ^ (b >>> 10));
    }
    let [a,b,c,d,e,f,g,h] = this.state;
    for (let i = 0; i < 64; i++) {
      const first = (h + (rotate(e, 6) ^ rotate(e, 11) ^ rotate(e, 25)) + ((e & f) ^ (~e & g)) + SectionSHA256.constants[i] + words[i]) >>> 0;
      const second = ((rotate(a, 2) ^ rotate(a, 13) ^ rotate(a, 22)) + ((a & b) ^ (a & c) ^ (b & c))) >>> 0;
      h=g; g=f; f=e; e=(d+first)>>>0; d=c; c=b; b=a; a=(first+second)>>>0;
    }
    const next = [a,b,c,d,e,f,g,h];
    for (let i = 0; i < 8; i++) this.state[i] += next[i];
  }
  update(bytes: Uint8Array): void {
    this.length += bytes.length;
    let offset = 0;
    if (this.used) {
      const count = Math.min(64 - this.used, bytes.length);
      this.block.set(bytes.subarray(0, count), this.used); this.used += count; offset += count;
      if (this.used === 64) { this.transform(this.block, 0); this.used = 0; }
    }
    for (; offset + 64 <= bytes.length; offset += 64) this.transform(bytes, offset);
    this.block.set(bytes.subarray(offset), this.used); this.used += bytes.length - offset;
  }
  finish(): string {
    const bitLength = BigInt(this.length) * 8n;
    this.update(new Uint8Array([128]));
    const padding = new Uint8Array((56 - this.used + 64) % 64 + 8);
    new DataView(padding.buffer).setBigUint64(padding.length - 8, bitLength);
    this.update(padding);
    return [...this.state].map(value => value.toString(16).padStart(8, "0")).join("");
  }
}

export function requireANS(condition: unknown, detail: string): asserts condition {
  if (!condition) throw new Error(`Invalid QEM source: ${detail}. Select an intact .qem acquisition.`);
}
export function parseManifest(text: string): unknown {
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    throw new Error("Invalid QEM JSON metadata; recopy or re-export the original acquisition.");
  }
  // JSON.parse alone would silently accept conflicting duplicate schema keys.
  const tokens = text.match(/"(?:\\.|[^"\\])*"|[{}\[\]:,]/g) ?? [];
  const stack: (Set<string> | null)[] = [];
  for (let i = 0; i < tokens.length; i++) {
    const token = tokens[i];
    if (token === "{") stack.push(new Set());
    else if (token === "[") stack.push(null);
    else if (token === "}" || token === "]") stack.pop();
    else if (token.startsWith('"') && tokens[i + 1] === ":") {
      const keys = stack[stack.length - 1]!; const key = JSON.parse(token);
      requireANS(!keys.has(key), `duplicate manifest field ${key}`); keys.add(key);
    }
  }
  return value;
}

/** Admit the current QEM container without expanding detector counts. */
export async function countAnsFileSource(file: File, onStatus: (text: string) => void = () => {}): Promise<RansByteSource> {
  return qemFileSource(file, onStatus);
}


/** Join compatible encoded files without decoding counts or changing their order. */
export async function countAnsFilesSource(files: ArrayLike<File>, onStatus: (text: string) => void = () => {}, badPixels: number[] = []): Promise<RansByteSource> {
  const ordered = Array.from(files);
  requireANS(ordered.length > 0, "select at least one count-ANS file");
  const sources: RansByteSource[] = [];
  const manifests: RansManifest[] = [];
  for (let index = 0; index < ordered.length; index++) {
    const source = await countAnsFileSource(ordered[index], text => onStatus(`${index + 1}/${ordered.length} ${ordered[index].name}: ${text}`));
    const manifest = JSON.parse(new TextDecoder().decode(await source.read("manifest.json"))) as RansManifest;
    if (index > 0) {
      const first = manifests[0];
      requireANS(JSON.stringify(manifest.bad_pixels) === JSON.stringify(first.bad_pixels), "series detector validity masks differ; open each acquisition separately");
      const shape = [...manifest.scan_shape!, ...manifest.detector_shape!];
      const expected = [...first.scan_shape!, ...first.detector_shape!];
      if (JSON.stringify(shape) !== JSON.stringify(expected) || manifest.native_dtype !== first.native_dtype) {
        throw new Error(`Count-ANS file ${ordered[index].name} has shape ${shape.join("x")} and dtype ${manifest.native_dtype}; ${ordered[0].name} has ${expected.join("x")} ${first.native_dtype}. Select acquisitions with matching native geometry and dtype.`);
      }
      const profile = manifest.tilts[0], firstProfile = first.tilts[0];
      if (profile.frames !== firstProfile.frames || profile.scale !== firstProfile.scale) {
        throw new Error(`Count-ANS file ${ordered[index].name} uses block_frames=${profile.frames}, scale=${profile.scale}; ${ordered[0].name} uses block_frames=${firstProfile.frames}, scale=${firstProfile.scale}. Re-encode the series with the same block_frames and scale before loading it together.`);
      }
    }
    sources.push(source); manifests.push(manifest);
  }
  const detectorPixels = manifests[0].tilts[0].K;
  requireANS(badPixels.every(index => Number.isInteger(index) && index >= 0 && index < detectorPixels), `badPixels must contain detector indices from 0 to ${detectorPixels - 1}`);
  const combined: RansManifest = {
    ...manifests[0],
    bad_pixels: [...new Set([...(manifests[0].bad_pixels ?? []), ...badPixels])],
    source_metadata: ordered.length === 1 ? manifests[0].source_metadata : {
      acquisitions: ordered.map((file, index) => ({ file: file.name, metadata: manifests[index].source_metadata })),
    },
    tilts: manifests.map((manifest, index) => {
      const tilt = manifest.tilts[0]; const prefix = `acquisition-${index}/`;
      return { ...tilt, tilt: index, payload_url: prefix + tilt.payload_url,
        models: tilt.models.map(model => ({ ...model, colmeta_url: prefix + model.colmeta_url,
          entries_url: prefix + model.entries_url, lut_url: prefix + model.lut_url })),
      };
    }),
  };
  return { mode: "local-folder", async read(name, start, end) {
    if (name === "manifest.json") return new TextEncoder().encode(JSON.stringify(combined)).buffer;
    const namespaced = /^acquisition-(\d+)\/(.+)$/.exec(name);
    if (namespaced) return sources[Number(namespaced[1])].read(namespaced[2], start, end);
    const offsets = /^t(\d+)-offsets-(\d+)\.u32$/.exec(name);
    requireANS(offsets, `unknown series table ${name}`);
    return sources[Number(offsets[1])].read(`t0-offsets-${offsets[2]}.u32`);
  } };
}
