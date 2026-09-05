import { StreamingSha256 } from "./logical-pixel-hash";
import type { DataRepresentation } from "./local-h5";

/** Canonical camel-case form of resident_contract.schema.json. */
export interface CompactH5ResidentReceipt {
  readonly schema: "quantem.gpu.4dstem-resident-receipt/v2";
  readonly representation: DataRepresentation;
  readonly sourceIdentitySHA256: string;
  readonly sourceShape: readonly [number, number, number, number];
  readonly workingShape: readonly [number, number, number, number];
  readonly sourceDtype: "uint16";
  readonly workingDtype: "uint8" | "uint16";
  readonly sourceLogicalTensorBytes: number;
  readonly workingLogicalTensorBytes: number;
  /** Packed data and prepared products; other auxiliary buffers are separate. */
  readonly physicalResidentBytes: number;
  readonly containerBytes: number;
  readonly storageSchema: string;
  readonly losslessExact: true;
  readonly scanBin: 1;
  readonly detectorBin: 1;
  readonly crop: null;
  readonly detectorMaskCount: number;
  readonly detectorMaskSHA256: string | null;
  readonly detectorMaskSchema: string | null;
  readonly calibrationSchema: string | null;
  readonly calibrationSHA256: string | null;
  readonly provenanceSchema: "quantem.gpu.packed-detector-h5-manifest/v1";
  readonly provenanceSHA256: string;
  readonly sourceRawLogicalSHA256: string;
  readonly workingLogicalSHA256: string | null;
  readonly implementationRevision: string | null;
}

function asciiJson(value: string | boolean | null): string {
  return JSON.stringify(value).replace(/[\u007f-\uffff]/g, character => (
    `\\u${character.charCodeAt(0).toString(16).padStart(4, "0")}`
  ));
}

function compareCodePoints(left: string, right: string): number {
  const a = Array.from(left, character => character.codePointAt(0)!);
  const b = Array.from(right, character => character.codePointAt(0)!);
  for (let index = 0; index < Math.min(a.length, b.length); index++) {
    if (a[index] !== b[index]) return a[index] - b[index];
  }
  return a.length - b.length;
}

function canonicalMetadataJson(value: unknown): string {
  if (typeof value === "number") {
    if (!Number.isFinite(value)) throw new Error("Receipt metadata numbers must be finite.");
    const encoded = new Uint8Array(8);
    new DataView(encoded.buffer).setFloat64(0, value, false);
    const hex = Array.from(encoded, byte => byte.toString(16).padStart(2, "0")).join("");
    return asciiJson(`f64be:${hex}`);
  }
  if (value === null) return "null";
  if (typeof value === "boolean" || typeof value === "string") {
    return asciiJson(value);
  }
  if (Array.isArray(value)) return `[${value.map(canonicalMetadataJson).join(",")}]`;
  if (typeof value === "object" && Object.getPrototypeOf(value) === Object.prototype) {
    const object = value as Record<string, unknown>;
    return `{${Object.keys(object).sort(compareCodePoints).map(key => (
      `${asciiJson(key)}:${canonicalMetadataJson(object[key])}`
    )).join(",")}}`;
  }
  throw new Error("Receipt metadata must contain only JSON-compatible values.");
}

/** Match Python metadata_sha256, including binary64 numbers and Unicode. */
export function metadataSha256(value: unknown): string {
  const sha256 = new StreamingSha256();
  sha256.update(new TextEncoder().encode(canonicalMetadataJson(value)));
  return sha256.digestHex();
}

/** Reject a different source, plan, calibration, storage size, or build identity. */
export function requireMatchingResidentReceipt(
  observed: CompactH5ResidentReceipt | null,
  expected: CompactH5ResidentReceipt,
): void {
  if (observed === null) {
    throw new Error("This compact source cannot supply an exact raw resident receipt. Rebuild from the original uint16 source with recoverable excluded pixels.");
  }
  if (expected === null || typeof expected !== "object" || Array.isArray(expected)) {
    throw new Error("Expected resident receipt must be a trusted complete producer receipt.");
  }
  for (const key of new Set([...Object.keys(observed), ...Object.keys(expected)])) {
    const field = key as keyof CompactH5ResidentReceipt;
    if (!(field in observed) || !(field in expected)
        || canonicalMetadataJson(observed[field]) !== canonicalMetadataJson(expected[field])) {
      throw new Error(`Resident receipt field ${field} disagrees with the selected source or implementation. Select the qualified source and matching build.`);
    }
  }
}
