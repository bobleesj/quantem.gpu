/** CPU-only checks that .qem admission stages authenticated stream bytes on the GPU once, and owns them until a resident set does. */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fakeDevice, type FakeBuffer } from "./fake-gpu";
import { qemFileSource, qemFilesSource } from "../../src/quantem/gpu/detector/webgpu/qem-source";
import { RansResidentSet } from "../../src/quantem/gpu/detector/webgpu/rans";
import type { RansByteSource } from "../../src/quantem/gpu/detector/webgpu/rans-source";
import { syntheticQem } from "./qem-synthetic";

const fixture = new Uint8Array(readFileSync("tests/data/qem-v2/u16-multiple-chunks.qem"));
// Several blocks per chunk, so most blocks start at an unaligned byte inside their group.
const unaligned = syntheticQem([[1001, 6, 70001], [4, 999]]);

/** A local file that counts the payload bytes (everything after the header) and all bytes read through it. */
function countingFile(bytes: Uint8Array, name = "acquisition.qem") {
  const body = Number(new DataView(bytes.buffer).getBigUint64(16, true));
  const file = { name, size: bytes.length, payload: bytes.length - body, read: 0, total: 0, slice(start = 0, end = bytes.length) {
    return { arrayBuffer: async () => {
      file.read += Math.max(0, end - Math.max(start, body));
      file.total += end - start;
      return bytes.slice(start, end).buffer;
    } };
  } };
  return file;
}

type Block = { index: number; byte_start: number; byte_end: number };
const blocksOf = async (source: RansByteSource) =>
  JSON.parse(new TextDecoder().decode(await source.read("manifest.json"))).tilts[0].blocks_meta as Block[];
/** Distinct buffers at one binding of the decoder's eight-entry bind groups (0 payload, 6 units). */
const boundBuffers = (groups: GPUBindGroupDescriptor[], binding: number) => [...new Set(groups
  .filter(group => [...group.entries].length === 8)
  .map(group => ([...group.entries].find(entry => entry.binding === binding)!.resource as GPUBufferBinding).buffer as unknown as FakeBuffer))];
const stagedBytes = (segment: { buffer: GPUBuffer; offset: number }, length: number) =>
  new Uint8Array((segment.buffer as unknown as FakeBuffer).bytes, segment.offset, length);

test("every block's stream bytes are staged once, where the resident payload reports them", async () => {
  for (const bytes of [fixture, unaligned]) {
    const gpu = fakeDevice();
    const file = countingFile(bytes);
    const source = await qemFileSource(file, () => {}, gpu.device);
    assert.equal(file.read, file.payload, "every payload byte is read exactly once");
    const blocks = await blocksOf(source);
    for (const block of blocks) {
      const segment = source.residentPayload!("payload", block.byte_start, block.byte_end)!;
      assert.deepEqual(stagedBytes(segment, block.byte_end - block.byte_start), bytes.subarray(block.byte_start, block.byte_end));
    }
    assert.ok(gpu.buffers.length > 0 && gpu.buffers.every(buffer => !buffer.mapped), "staged groups are unmapped");
    const plain = await qemFileSource(new File([bytes], "plain.qem"));
    for (const block of blocks) {
      for (const name of [`columns-${block.index}`, `t0-offsets-${block.index}.u32`]) {
        assert.deepEqual(new Uint8Array(await source.read(name)), new Uint8Array(await plain.read(name)), name);
      }
    }
    source.dispose!();
    assert.ok(gpu.buffers.every(buffer => buffer.destroyed));
  }
});

test("the loader binds staged groups with each block's byte offset instead of reading the payload again", async () => {
  const gpu = fakeDevice();
  const file = countingFile(unaligned);
  const set = await RansResidentSet.loadQemFile(gpu.device, file);
  assert.equal(file.read, file.payload, "the payload is not read a second time");
  const payloads = boundBuffers(gpu.bindGroups, 0), units = boundBuffers(gpu.bindGroups, 6);
  assert.equal(payloads.length, 1, "both chunks share one staged group");
  const table = new Uint32Array(units[0].bytes);
  const blocks = await blocksOf(await qemFileSource(new File([unaligned], "plain.qem")));
  const offsets: number[] = [];
  blocks.forEach((block, unit) => {
    const pad = table[unit * 8 + 7], byte = table[unit * 8] * 4 + ((pad >>> 28) & 3);
    assert.equal(pad & 0x0fffffff, 512);
    assert.deepEqual(new Uint8Array(payloads[0].bytes, byte, block.byte_end - block.byte_start), unaligned.subarray(block.byte_start, block.byte_end));
    offsets.push(byte % 4);
  });
  assert.ok(offsets.some(offset => offset !== 0), "unaligned blocks are exercised");
  set.dispose();
  assert.ok(gpu.buffers.every(buffer => buffer.destroyed));
});

test("failed admissions and loads release every staged buffer", async () => {
  const twoChunks = syntheticQem([[40 << 20], [40 << 20]]);
  const corrupted = twoChunks.slice();
  corrupted[corrupted.length - 1000] ^= 1;
  for (const [bytes, message] of [[corrupted, /payload checksum mismatch/], [syntheticQem([[4096, 4096]], true), /does not partition payload/]] as const) {
    const gpu = fakeDevice();
    await assert.rejects(qemFileSource(countingFile(bytes), () => {}, gpu.device), message);
    assert.ok(gpu.buffers.length > 0, "the failure happened after staging began");
    assert.ok(gpu.buffers.every(buffer => buffer.destroyed));
  }
  const faulty = fakeDevice(undefined, 1);
  await assert.rejects(RansResidentSet.loadQemFile(faulty.device, countingFile(unaligned)), /did not terminate exactly/);
  const staged = boundBuffers(faulty.bindGroups, 0);
  assert.ok(staged.length === 1 && staged[0].destroyed, "a decoder fault releases the staged payload");
  const mismatch = fakeDevice();
  await assert.rejects(RansResidentSet.loadQemFiles(mismatch.device, [countingFile(unaligned), countingFile(syntheticQem([[64]]))]), /matching native geometry/);
  assert.equal(mismatch.buffers.length, 0, "a mismatched series is rejected from its headers, before staging");
  const laterFailure = fakeDevice();
  const corruptedSecond = unaligned.slice();
  corruptedSecond[corruptedSecond.length - 100] ^= 1;
  await assert.rejects(RansResidentSet.loadQemFiles(laterFailure.device, [countingFile(unaligned), countingFile(corruptedSecond)]), /payload checksum mismatch/);
  assert.ok(laterFailure.buffers.length > 0 && laterFailure.buffers.every(buffer => buffer.destroyed), "the first file's staged payload is released");
});

test("a chunk larger than one GPU group keeps the per-block upload path", async () => {
  const gpu = fakeDevice({ maxBufferSize: 4096, maxStorageBufferBindingSize: 4096 });
  const bytes = syntheticQem([[3000, 3000]]);
  const source = await qemFileSource(countingFile(bytes), () => {}, gpu.device);
  const [block] = await blocksOf(source);
  assert.equal(source.residentPayload!("payload", block.byte_start, block.byte_end), undefined);
  assert.equal(gpu.buffers.length, 0);
  const file = countingFile(bytes);
  const set = await RansResidentSet.loadQemFile(gpu.device, file);
  assert.equal(file.read, file.payload + 6000, "each block is uploaded from the file");
  set.dispose();
});

test("a series forwards each acquisition's staged payload", async () => {
  const gpu = fakeDevice();
  const second = syntheticQem([[1001, 6, 70001], [5, 998]]);
  const files = [countingFile(unaligned, "first.qem"), countingFile(second, "second.qem")];
  const source = await qemFilesSource(files, () => {}, [], gpu.device);
  assert.deepEqual(files.map(file => file.total), files.map(file => file.size), "each file, header included, is read once");
  const manifest = JSON.parse(new TextDecoder().decode(await source.read("manifest.json")));
  manifest.tilts.forEach((tilt: { payload_url: string; blocks_meta: Block[] }, acquisition: number) => {
    for (const block of tilt.blocks_meta) {
      const segment = source.residentPayload!(tilt.payload_url, block.byte_start, block.byte_end)!;
      assert.deepEqual(stagedBytes(segment, block.byte_end - block.byte_start), [unaligned, second][acquisition].subarray(block.byte_start, block.byte_end));
    }
  });
  source.dispose!();
  assert.ok(gpu.buffers.every(buffer => buffer.destroyed));
});
