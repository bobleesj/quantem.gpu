/** CPU-only checks that compare-grid batches integrate every resident set they show. */
import assert from "node:assert/strict";
import test from "node:test";
import { RansDetectorCompute, ransMaskedSumBuffersBatch, ransMaskedSumDeltaBuffersBatch, type RansResidentSet } from "../../src/quantem/gpu/detector/webgpu/rans";

/** Records what one resident set is asked to do; display copies are labelled set + acquisition. */
function recordingSet(name: string, device: GPUDevice, log: string[]): RansResidentSet {
  const encoderUse = (encoder?: GPUCommandEncoder) => encoder ? "shared" : "own";
  return {
    device, scanCount: 4, detSize: 3, badPx: new Uint32Array(0),
    resetImages(tilts: Set<number>) { log.push(`${name} reset ${[...tilts]}`); },
    integrate(tilts: Set<number>, added: Uint8Array | null, removed: Uint8Array | null, encoder?: GPUCommandEncoder) {
      log.push(`${name} integrate ${[...tilts]} +${added?.join("") ?? ""} -${removed?.join("") ?? ""} ${encoderUse(encoder)}`);
      return 1;
    },
    imageBuffersF32(tilts: number[], previous?: GPUBuffer[], encoder?: GPUCommandEncoder) {
      log.push(`${name} copy ${tilts} ${encoderUse(encoder)}`);
      return tilts.map((tilt, index) => previous?.[index] ?? { label: `${name}${tilt}` });
    },
  } as unknown as RansResidentSet;
}

function fixture() {
  const log: string[] = [];
  let submissions = 0;
  const device = { createCommandEncoder: () => ({ finish: () => ({}) }), queue: { submit() { submissions++; } } } as unknown as GPUDevice;
  return { log, device, submissions: () => submissions };
}
const labels = (buffers: GPUBuffer[]) => buffers.map(buffer => buffer.label);

test("separately loaded acquisitions each integrate their own image in one submission", () => {
  const { log, device, submissions } = fixture();
  const first = recordingSet("A", device, log), second = recordingSet("B", device, log);
  const computes = [new RansDetectorCompute(first, 0), new RansDetectorCompute(second, 0)];
  const full = ransMaskedSumBuffersBatch(computes, new Uint32Array([1, 0, 1]));
  assert.deepEqual(labels(full.buffers), ["A0", "B0"]);
  assert.deepEqual(log, ["A reset 0", "A integrate 0 +101 - shared", "A copy 0 shared", "B reset 0", "B integrate 0 +101 - shared", "B copy 0 shared"]);
  assert.equal(submissions(), 1);
  log.length = 0;
  const delta = ransMaskedSumDeltaBuffersBatch(computes, new Uint32Array([0, 1, 0]), new Uint32Array([1, 0, 0]), full.buffers);
  assert.deepEqual(delta.buffers, full.buffers, "delta copies refresh the previous display buffers");
  assert.deepEqual(log, ["A integrate 0 +010 -100 shared", "A copy 0 shared", "B integrate 0 +010 -100 shared", "B copy 0 shared"]);
  assert.equal(submissions(), 2);
  assert.deepEqual(computes.map(compute => [...compute.currentMask!]), [[0, 1, 1], [0, 1, 1]]);
});

test("acquisitions of one set keep one integrate when another set is mixed in", () => {
  const { log, device, submissions } = fixture();
  const first = recordingSet("A", device, log), second = recordingSet("B", device, log);
  const computes = [new RansDetectorCompute(first, 0), new RansDetectorCompute(second, 0), new RansDetectorCompute(first, 1)];
  const full = ransMaskedSumBuffersBatch(computes, new Uint32Array([0, 1, 1]));
  assert.deepEqual(labels(full.buffers), ["A0", "B0", "A1"]);
  assert.deepEqual(log.filter(entry => entry.includes("integrate")), ["A integrate 0,1 +011 - shared", "B integrate 0 +011 - shared"]);
  assert.equal(submissions(), 1);
});

test("a delta that cannot apply to every panel changes no set", () => {
  const { log, device, submissions } = fixture();
  const first = recordingSet("A", device, log), second = recordingSet("B", device, log);
  const computes = [new RansDetectorCompute(first, 0), new RansDetectorCompute(second, 0)];
  computes[0].currentMask = new Uint8Array([1, 0, 1]);
  assert.throws(() => ransMaskedSumDeltaBuffersBatch(computes, new Uint32Array([0, 1, 0]), new Uint32Array([1, 0, 0])), /before the first full mask/);
  assert.deepEqual([...computes[0].currentMask], [1, 0, 1]);
  assert.deepEqual(log, []);
  assert.equal(submissions(), 0);
});
