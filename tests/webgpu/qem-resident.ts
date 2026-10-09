/** CPU-only checks that .qem admission stages authenticated stream bytes on the GPU once, and owns them until a resident set does. */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fakeDevice, type FakeBuffer } from "./fake-gpu";
import { qemFileSource, qemFilesSource, qemHttpFiles } from "../../src/quantem/gpu/detector/webgpu/qem-source";
import { RansResidentSet } from "../../src/quantem/gpu/detector/webgpu/rans";
import { RansResidentSeries } from "../../src/quantem/gpu/detector/webgpu/rans-series";
import { validateUint32ImageView } from "../../src/quantem/gpu/detector/webgpu/borrowed-image";
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
/** Distinct buffers at one binding of bind groups with `entries` entries:
 * the decoder's eight (0 payload, 6 units) or the image update's four (1 uint32 images). */
const boundBuffers = (groups: GPUBindGroupDescriptor[], binding: number, entries = 8) => [...new Set(groups
  .filter(group => [...group.entries].length === entries)
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
  assert.ok(faulty.buffers.every(buffer => buffer.destroyed), "a decoder fault releases every buffer of the load");
  const mismatch = fakeDevice();
  await assert.rejects(RansResidentSet.loadQemFiles(mismatch.device, [countingFile(unaligned), countingFile(syntheticQem([[64]]))]), /matching native geometry/);
  assert.equal(mismatch.buffers.length, 0, "a mismatched series is rejected from its headers, before staging");
  const laterFailure = fakeDevice();
  const corruptedSecond = unaligned.slice();
  corruptedSecond[corruptedSecond.length - 100] ^= 1;
  await assert.rejects(RansResidentSet.loadQemFiles(laterFailure.device, [countingFile(unaligned), countingFile(corruptedSecond)]), /payload checksum mismatch/);
  assert.ok(laterFailure.buffers.length > 0 && laterFailure.buffers.every(buffer => buffer.destroyed), "the first file's staged payload is released");
});

test("a chunk larger than a group is split and its cut blocks copied on the GPU; nothing is read after authentication", async () => {
  // 4096-byte groups. Expected groups: staged groups holding whole blocks plus groups of copied blocks;
  // a staged piece that holds only the tail of a cut block is released once the copies are made.
  const layouts: [number[][], number][] = [
    [[[1001, 3006, 2005], [4, 999]], 3],
    [[[1000, 3000, 2000]], 2],
    [[[3000, 3000, 3000, 3000]], 4],
  ];
  Object.assign(globalThis, { location: { href: "http://localhost/" } });
  for (const [layout, expectedGroups] of layouts) {
    const bytes = syntheticQem(layout);
    const body = Number(new DataView(bytes.buffer).getBigUint64(16, true));
    let authenticated = false;
    const later: string[] = [];
    globalThis.fetch = (async (_url: URL, init: RequestInit = {}) => {
      if (init.method === "HEAD") return new Response(null, { headers: { "Content-Length": String(bytes.length) } });
      const range = (init.headers as Record<string, string>).Range;
      const [first, last] = /bytes=(\d+)-(\d+)/.exec(range)!.slice(1).map(Number);
      const served = bytes.slice(first, last + 1);
      // Anything served after the authentication stream is tampered and must never reach the decoder.
      if (authenticated) { later.push(range); served.fill(0xa5); }
      if (first === body && last === bytes.length - 1) authenticated = true;
      return new Response(served, { status: 206 });
    }) as typeof fetch;
    const gpu = fakeDevice({ maxBufferSize: 4096, maxStorageBufferBindingSize: 4096 });
    const [file] = await qemHttpFiles("/", ["split.qem"]);
    const set = await RansResidentSet.loadQemFile(gpu.device, file);
    assert.deepEqual(later, [], "no byte is read after the authentication stream");
    const blocks = await blocksOf(await qemFileSource(new File([bytes], "plain.qem")));
    const groups = (set as unknown as { groups: { payload: FakeBuffer; units: { block: number; payload_word: number; frameFlags: number }[] }[] }).groups;
    assert.equal(groups.length, expectedGroups, JSON.stringify(layout));
    let bound = 0;
    for (const group of groups) {
      for (const unit of group.units) {
        const block = blocks[unit.block], at = unit.payload_word * 4 + ((unit.frameFlags >>> 28) & 3);
        assert.deepEqual(new Uint8Array(group.payload.bytes, at, block.byte_end - block.byte_start), bytes.subarray(block.byte_start, block.byte_end), `block ${unit.block}`);
        bound++;
      }
    }
    assert.equal(bound, blocks.length);
    set.dispose();
    assert.ok(gpu.buffers.every(buffer => buffer.destroyed), `every buffer is released for ${JSON.stringify(layout)}`);
  }
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

test("mapped payload groups stay within 256 MiB even when the device allows more", async () => {
  const largest = (buffers: FakeBuffer[]) => Math.max(...buffers.filter(buffer => buffer.mappedAtCreation).map(buffer => buffer.size));
  // Two chunks that fit one group each but not one group together: staged in two groups.
  const staged = fakeDevice();
  const twoChunks = syntheticQem([[129 << 20], [128 << 20]]);
  const source = await qemFileSource(countingFile(twoChunks), () => {}, staged.device);
  for (const block of await blocksOf(source)) {
    const segment = source.residentPayload!("payload", block.byte_start, block.byte_end)!;
    assert.deepEqual(stagedBytes(segment, 1 << 16), twoChunks.subarray(block.byte_start, block.byte_start + (1 << 16)));
  }
  assert.ok(largest(staged.buffers) <= 256 << 20, `largest staged group ${largest(staged.buffers)} bytes`);
  source.dispose!();
  // One chunk larger than a group: split at 256 MiB. Its second block (256 MiB - 2 bytes,
  // starting at byte 7) is cut; its word-aligned copy needs 256 MiB + 4 bytes, which only
  // an unmapped copy group may hold. The 5-byte piece left after the split holds no block.
  const uploaded = fakeDevice();
  const edge = syntheticQem([[7, (256 << 20) - 2]]);
  const set = await RansResidentSet.loadQemFile(uploaded.device, countingFile(edge));
  assert.ok(largest(uploaded.buffers) <= 256 << 20, `largest mapped group ${largest(uploaded.buffers)} bytes`);
  const [, block] = await blocksOf(await qemFileSource(new File([edge], "plain.qem")));
  const cutGroup = (set as unknown as { groups: { payload: FakeBuffer; units: { block: number; payload_word: number; frameFlags: number }[] }[] }).groups
    .find(group => group.units.some(unit => unit.block === 1))!;
  const unit = cutGroup.units.find(unit => unit.block === 1)!;
  const at = unit.payload_word * 4 + ((unit.frameFlags >>> 28) & 3);
  assert.ok(Buffer.from(cutGroup.payload.bytes, at, block.byte_end - block.byte_start).equals(edge.subarray(block.byte_start, block.byte_end)));
  set.dispose();
  assert.ok(uploaded.buffers.every(buffer => buffer.destroyed), "the piece holding only a block's tail is released too");
});

test("borrowed count views address each acquisition's exact uint32 image", async () => {
  const gpu = fakeDevice();
  const set = await RansResidentSet.loadQemFiles(gpu.device, [countingFile(unaligned, "a.qem"), countingFile(unaligned, "b.qem")]);
  const [images] = boundBuffers(gpu.bindGroups, 1, 4);
  const views = set.imageViewsU32([1, 0], 7);
  assert.deepEqual(views.map(view => [view.buffer, view.byteOffset, view.count, view.divisor]),
    [[images, set.scanCount * 4, set.scanCount, 7], [images, 0, set.scanCount, 7]]);
  for (const view of views) validateUint32ImageView(view, gpu.device, set.scanCount);
  assert.throws(() => set.imageViewsU32([2], 1), /within 0\.\.1/);
  set.dispose();
  assert.throws(() => set.imageViewsU32([0], 1), /disposed/);
});

test("unit rows follow their groups when per-block uploads and staged segments interleave", async () => {
  // 4096-byte groups. Acquisitions 0 and 2 take the per-block upload path and
  // share a group around the staged acquisition 1, as a mixed series would.
  const gpu = fakeDevice({ maxBufferSize: 4096, maxStorageBufferBindingSize: 4096 });
  const bytes = [syntheticQem([[1500, 1500, 1500]]), syntheticQem([[1000, 1000, 1000]]), syntheticQem([[1500, 1500, 1500]])];
  const staged = await qemFilesSource(bytes.map((file, index) => countingFile(file, `t${index}.qem`)), () => {}, [], gpu.device);
  const uploaded = /^acquisition-([02])\/payload$/;
  const mixed: RansByteSource = { ...staged,
    residentPayload: (name, start, end) => uploaded.test(name) ? undefined : staged.residentPayload!(name, start, end),
    read: async (name, start, end) => {
      const acquisition = uploaded.exec(name);
      return acquisition ? bytes[Number(acquisition[1])].slice(start, end).buffer : staged.read(name, start, end);
    },
  };
  const loadSource = (RansResidentSet as unknown as { loadSource(device: GPUDevice, source: RansByteSource, status: () => void): Promise<RansResidentSet> }).loadSource;
  const set = await loadSource.call(RansResidentSet, gpu.device, mixed, () => {});
  const groups = (set as unknown as { groups: { unit0: number; units: { out_base: number; payload_word: number }[] }[] }).groups;
  const table = new Uint32Array((set as unknown as { unitsBuf: FakeBuffer }).unitsBuf.bytes);
  assert.ok(groups.some(group => group.units.length > 1 && new Set(group.units.map(unit => Math.floor(unit.out_base / 1536))).size > 1), "one upload group holds blocks of acquisitions 0 and 2");
  for (const group of groups) {
    group.units.forEach((unit, index) => {
      const row = group.unit0 + index;
      assert.deepEqual([table[row * 8], table[row * 8 + 6]], [unit.payload_word, unit.out_base], `row ${row}`);
    });
  }
  set.dispose();
  staged.dispose!();
});

test("a count view that a display cannot bind is refused when it is lent", async () => {
  // 537 scans: the second acquisition's image starts at byte 2148, not on a 256-byte boundary.
  const fixture = new Uint8Array(readFileSync("tests/data/qem-v2/u16-multiple-chunks.qem"));
  const gpu = fakeDevice();
  const set = await RansResidentSet.loadQemFiles(gpu.device, [countingFile(fixture, "a.qem"), countingFile(fixture, "b.qem")]);
  assert.equal(set.scanCount, 537);
  for (const view of set.imageViewsU32([0], 1)) validateUint32ImageView(view, gpu.device, set.scanCount);
  assert.throws(() => set.imageViewsU32([0, 1], 1), /Acquisition 1 cannot be lent as a count view: its image starts at byte 2148/);
  set.dispose();
});

test("cancellation during the decoder build releases the set instead of returning it", async () => {
  const gpu = fakeDevice();
  const controller = new AbortController();
  // The checkpoint build is the load's first wait on submitted GPU work.
  gpu.device.queue.onSubmittedWorkDone = async () => controller.abort();
  await assert.rejects(RansResidentSet.loadQemFile(gpu.device, countingFile(unaligned), () => {}, [], controller.signal), { name: "AbortError" });
  assert.ok(gpu.buffers.length > 0 && gpu.buffers.every(buffer => buffer.destroyed));
});

test("a series names each acquisition once in its status, and alone when its loader reports nothing", async () => {
  const gpu = fakeDevice();
  const statuses: string[] = [];
  const files = [countingFile(unaligned, "t0.qem"), countingFile(unaligned, "t1.qem")];
  const series = await RansResidentSeries.load(gpu.device, files, text => statuses.push(text), () => {}, new AbortController().signal);
  await series.completion;
  const report = statuses.join("\n");
  for (const status of statuses.slice(0, -1)) {
    // One "position/count name" prefix; the inner loader text never repeats it.
    assert.match(status, /^[12]\/2 t[01]\.qem(: (?!\d+\/\d+ )\S.*)?$/, report);
  }
  assert.equal(statuses.at(-1), "", "the finished series clears its status");
  for (const [position, name] of [["1/2", "t0.qem"], ["2/2", "t1.qem"]]) {
    assert.ok(statuses.some(status => new RegExp(`^${position} ${name}: Verifying \\.qem \\d+/\\d+$`).test(status)), report);
    assert.ok(statuses.includes(`${position} ${name}`), report);
  }
  series.dispose();
  const single: string[] = [];
  (await RansResidentSet.loadQemFile(gpu.device, countingFile(unaligned, "t0.qem"), text => single.push(text))).dispose();
  assert.ok(single.some(status => /^Verifying \.qem \d+\/\d+$/.test(status)), "a single file needs no position prefix");
  assert.ok(gpu.buffers.every(buffer => buffer.destroyed));
});
