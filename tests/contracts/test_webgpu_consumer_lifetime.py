from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/quantem/gpu/io/backends/webgpu/compact-h5.ts"


@pytest.fixture(scope="session")
def compact_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the compact WebGPU module for a Node-hosted lifetime test."""
    output = tmp_path_factory.mktemp("webgpu-consumer-lifetime") / "compact-h5.mjs"
    subprocess.run(
        [
            "npx",
            "--no-install",
            "esbuild",
            str(SOURCE),
            "--bundle",
            "--platform=browser",
            "--format=esm",
            f"--outfile={output}",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return output


@pytest.mark.parametrize("collect_timings", [False, True])
def test_display_conversion_blocks_release_until_its_queue_fence_resolves(
    compact_bundle: Path, collect_timings: bool,
) -> None:
    """A borrowed display buffer must outlive its queued u32-to-f32 write."""
    script = r"""
globalThis.GPUBufferUsage = {
  MAP_READ: 1,
  COPY_SRC: 2,
  COPY_DST: 4,
  STORAGE: 8,
  UNIFORM: 16,
  QUERY_RESOLVE: 32,
};

const module = await import(process.argv[1]);
let finishConversion;
let fenceRequests = 0;
let submissions = 0;
const destroyed = [];
let timestampQueries = 0;

function fakeBuffer(label) {
  const mapped = new ArrayBuffer(16);
  return {
    size: 16,
    destroy() { destroyed.push(label); },
    getMappedRange() { return mapped; },
    unmap() {},
  };
}

const queue = {
  submit() { submissions += 1; },
  onSubmittedWorkDone() {
    fenceRequests += 1;
    if (fenceRequests === 1) {
      return new Promise((resolve) => { finishConversion = resolve; });
    }
    return Promise.resolve();
  },
};
const device = {
  features: { has() { return true; } },
  createQuerySet() {
    timestampQueries++;
    return { destroy() { destroyed.push("timestamp-query"); } };
  },
  queue,
  createBuffer(descriptor) {
    return fakeBuffer(descriptor.label ?? "anonymous");
  },
  createBindGroup() { return {}; },
  createCommandEncoder() {
    return {
      beginComputePass() {
        return {
          setPipeline() {},
          setBindGroup() {},
          dispatchWorkgroups() {},
          end() {},
        };
      },
      finish() { return {}; },
    };
  },
};
const pipeline = { getBindGroupLayout() { return {}; } };
const source = new module.WebGPUCompactH5ResidentSource({
  collectDetectorTimings: process.argv[2] === "true",
  metadata: {
    residentBytes: 32,
    manifest: {},
    shape: [1, 1, 1, 1],
    excludedDetectorPixels: [],
    sourceIdentitySha256: "a".repeat(64),
  },
  loadProfile: {},
  device,
  shards: [{ payload: fakeBuffer("payload"), descriptors: fakeBuffer("descriptors") }],
  excluded: fakeBuffer("excluded"),
  maximumWidths: new Uint8Array([1]),
  detectorOutputs: [fakeBuffer("detector-0"), fakeBuffer("detector-1")],
  diffractionOutput: fakeBuffer("diffraction"),
  selectedPipeline: pipeline,
  detectorPipeline: pipeline,
  detectorResolveV3Pipeline: pipeline,
  detectorResolveV3Layout: {},
  detectorResolvedV3Pipeline: pipeline,
  u32ToF32Pipeline: pipeline,
  dpcMomentPipeline: pipeline,
  dpcMomentLayout: {},
  momentsToComPipeline: pipeline,
  dpcMeanPipeline: pipeline,
  dpcPairPipeline: pipeline,
  dpcOutputMeanPipeline: pipeline,
  dpcOutputUlpCorrectPipeline: pipeline,
  detectorLayout: {},
});
source.submitVirtualDetector = async () => ({
  mode: "rebase",
  changedDetectorPixels: 1,
  addedPixels: 1,
  removedPixels: 0,
});

let duplicateBatchError = "";
try {
  module.WebGPUCompactH5ResidentSource.maskedSumDisplayBuffersBatch(
    [source, source], new Uint32Array([1]),
  );
} catch (error) {
  duplicateBatchError = String(error);
}
const display = source.maskedSumDisplayBuffer(new Uint32Array([1]));
let releaseError = "";
try {
  source.releaseResidentStorage();
} catch (error) {
  releaseError = String(error);
}

let quiesced = false;
const pendingQuiesce = source.quiesce().then(() => { quiesced = true; });
await Promise.resolve();
await Promise.resolve();
const beforeFence = {
  quiesced,
  released: source.isReleased,
  destroyed: destroyed.length,
};

finishConversion();
await pendingQuiesce;
source.releaseResidentStorage();

console.log(JSON.stringify({
  display,
  duplicateBatchError,
  releaseError,
  beforeFence,
  afterFence: {
    quiesced,
    released: source.isReleased,
    destroyed: destroyed.length,
  },
  fenceRequests,
  submissions,
  timestampQueries,
}));
"""
    completed = subprocess.run(
        ["node", "--input-type=module", "-e", script, compact_bundle.as_uri(),
         str(collect_timings).lower()],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)

    assert "same resident source twice" in result["duplicateBatchError"]

    assert result["display"] == {
        "buffer": {"size": 16},
        "n": 1,
        "path": "compact-rebase",
        "addedPixels": 0,
        "removedPixels": 0,
        "borrowed": True,
    }
    assert "Await quiesce() before release" in result["releaseError"]
    assert result["beforeFence"] == {
        "quiesced": False,
        "released": False,
        "destroyed": 0,
    }
    assert result["afterFence"] == {
        "quiesced": True,
        "released": True,
        "destroyed": 11 if collect_timings else 8,
    }
    assert result["fenceRequests"] == 2
    assert result["submissions"] == 1
    assert result["timestampQueries"] == int(collect_timings)


@pytest.mark.parametrize("failure", ["none", "allocation", "binding", "encoding"])
def test_batched_detector_updates_reuse_storage_and_clear_empty_selection(
    compact_bundle: Path, failure: str,
) -> None:
    """Rapid queued masks retain their parameters and release only after completion.

    This records host submission order, not numerical shader execution. The
    hardware trajectory fixture separately compares every output count.
    """
    script = r"""
import assert from 'node:assert/strict';
const {WebGPUCompactH5ResidentSource: Source} = await import(process.argv[1]);
globalThis.GPUBufferUsage = {STORAGE:1,COPY_SRC:2,COPY_DST:4,MAP_READ:8,UNIFORM:16};
const buffers = [], updates = [];
let groups = 0, submissions = 0, clears = 0;
let failure = process.argv[2], retainedParameters = 0, retainedGroups = 0, passes = 0;
const pipeline = {getBindGroupLayout: () => ({})};
const device = {
  features: {has: () => false},
  limits: {minUniformBufferOffsetAlignment:256,maxBufferSize:1e8,maxStorageBufferBindingSize:1e8},
  createBuffer({size, label = ''}) {
    if (label === 'compact retained detector parameters'
        && ++retainedParameters === 2 && failure === 'allocation') {
      throw new Error('injected second-source allocation failure');
    }
    const data = new ArrayBuffer(size);
    const buffer = {size, label, data, destroys:0, getMappedRange: () => data,
      unmap() {}, destroy() {this.destroys++;}};
    buffers.push(buffer); return buffer;
  },
  createBindGroup({entries}) {
    if (entries.length === 6 && ++retainedGroups === 3 && failure === 'binding') {
      throw new Error('injected second-source binding failure');
    }
    groups++; return entries;
  },
  createCommandEncoder() {
    const commands = [];
    return {
      beginComputePass() {
        if (++passes === 3 && failure === 'encoding') {
          throw new Error('injected second-source encoding failure');
        }
        let bindings, offset = 0;
        return {
          setPipeline() {},
          setBindGroup(_, entries, offsets = [0]) {bindings = entries; offset = offsets[0];},
          dispatchWorkgroups() {
            const captured = bindings, capturedOffset = offset;
            commands.push(() => {
              if (captured.length !== 6) return;
              const config = new Uint32Array(captured[5].resource.buffer.data, capturedOffset, 8);
              const entries = new Uint32Array(captured[2].resource.buffer.data, 0, config[2] * 2);
              updates.push({rebase:config[4], entries:[...entries]});
            });
          }, end() {},
        };
      },
      clearBuffer(buffer) {commands.push(() => {new Uint8Array(buffer.data).fill(0); clears++;});},
      finish: () => commands,
    };
  },
  queue: {
    writeBuffer(buffer, offset, data) {
      assert.equal(buffer.destroys, 0);
      new Uint8Array(buffer.data, offset, data.byteLength).set(
        ArrayBuffer.isView(data)
          ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
          : new Uint8Array(data),
      );
    },
    submit(batches) {submissions++; batches.flat().forEach(command => command());},
    onSubmittedWorkDone: async () => {},
  },
};
const makeSource = (schemaVersion = 1, detectorShape = [1, 4]) => {
  const buffer = () => device.createBuffer({size:512});
  return new Source({
    metadata:{schemaVersion,scanTile:128,scansPerShard:128,residentBytes:1024,
      shape:[8,16,...detectorShape],manifest:{},excludedDetectorPixels:new Uint32Array([3])},
    loadProfile:{},device,shards:[{payload:buffer(),descriptors:buffer()}],
    excluded:buffer(),maximumWidths:new Uint8Array([16,16,16,16]),
    detectorOutputs:[buffer(),buffer()],diffractionOutput:buffer(),
    selectedPipeline:pipeline,detectorPipeline:pipeline,detectorLayout:{},
    detectorResolveV3Pipeline:pipeline,detectorResolveV3Layout:{},
    detectorResolvedV3Pipeline:pipeline,u32ToF32Pipeline:pipeline,
    dpcMomentPipeline:pipeline,dpcMomentLayout:{},momentsToComPipeline:pipeline,
    dpcMeanPipeline:pipeline,dpcPairPipeline:pipeline,
    dpcOutputMeanPipeline:pipeline,dpcOutputUlpCorrectPipeline:pipeline,
  });
};
const sources = [makeSource(), makeSource()];
const v3 = makeSource(3);
assert.throws(() => Source.maskedSumDisplayBuffersBatch([sources[0], v3], new Uint32Array([1,1,0,1])),
  /require lossless pack format v1/);
assert.equal(submissions, 0, 'unsupported batches are rejected before any source encodes');
v3.releaseResidentStorage();
const transposed = makeSource(1, [4, 1]);
assert.throws(() => Source.maskedSumDisplayBuffersBatch([sources[0], transposed], new Uint32Array([1,1,0,1])),
  /detector shape/);
assert.equal(submissions, 0, 'equal element counts do not imply equal row/column geometry');
transposed.releaseResidentStorage();
if (failure !== 'none') {
  let published;
  assert.throws(() => {
    published = Source.maskedSumDisplayBuffersBatch(sources, new Uint32Array([1,1,0,1]));
  }, /injected second-source/);
  assert.equal(published, undefined, 'no partial display generation is published');
  assert.equal(submissions, 0, 'no partial batch is submitted');
  await Promise.all(sources.map(source => source.quiesce()));
  sources.forEach(source => assert.throws(() => source.virtualDetectorBuffer(), /Run updateVirtualDetector/));
  failure = 'none';
}
let warmBuffers, warmGroups;
for (const [index, values] of [[1,1,0,1],[1,1,1,1],[0,1,1,1],[0,0,0,1]].entries()) {
  if (index === 3) {
    sources.forEach(source => source.detectorOutputs.forEach(buffer => new Uint8Array(buffer.data).fill(255)));
  }
  const displays = Source.maskedSumDisplayBuffersBatch(sources, Uint32Array.from(values));
  assert.equal(displays.length, 2);
  assert.ok(displays.every(display => display.borrowed));
  if (index === 0) {warmBuffers = buffers.length; warmGroups = groups;}
  assert.equal(buffers.length, warmBuffers, 'reuse detector entries and uniforms');
  assert.equal(groups, warmGroups, 'reuse shard bindings for both output buffers');
}
await Promise.all(sources.map(source => source.quiesce()));
assert.equal(submissions, 4, 'one submission for each complete two-source update');
assert.equal(clears, 2, 'empty masks clear every resident output');
assert.deepEqual(updates, [
  {rebase:1, entries:[0,1,1,1]}, {rebase:1, entries:[0,1,1,1]},
  {rebase:0, entries:[2,1]}, {rebase:0, entries:[2,1]},
  {rebase:0, entries:[0,0xffffffff]}, {rebase:0, entries:[0,0xffffffff]},
]);
for (const source of sources) {
  assert.ok(new Uint8Array(source.detectorOutputs[source.activeDetectorOutput].data).every(v => v === 0));
  source.releaseResidentStorage();
}
assert.ok(buffers.every(buffer => buffer.destroys === 1), 'all resident and dispatch storage freed once');
"""
    completed = subprocess.run(
        ["node", "--input-type=module", "-e", script, compact_bundle.as_uri(), failure],
        cwd=ROOT, check=False, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
