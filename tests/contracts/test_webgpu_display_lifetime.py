"""Exercise resident display ownership without acquiring a GPU device."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def display_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Bundle the real display implementation for the CPU-only contract test."""
    output = tmp_path_factory.mktemp("display-lifetime") / "colormaps.mjs"
    subprocess.run(
        [
            "npx", "--no-install", "esbuild",
            str(ROOT / "src/quantem/gpu/display/backends/webgpu/colormaps.ts"),
            "--bundle", "--platform=browser", "--format=esm",
            f"--outfile={output}",
        ],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    return output


@pytest.mark.parametrize("journey", ["resident-switch", "resize", "upload", "owned"])
def test_display_buffer_lifetime(display_bundle: Path, journey: str) -> None:
    """Switching and replacing images preserves the source owner's storage."""
    script = r"""
import assert from 'node:assert/strict';
const { GPUColormapEngine } = await import(process.argv[1]);
globalThis.GPUBufferUsage = {STORAGE:1,COPY_SRC:2,COPY_DST:4,MAP_READ:8,UNIFORM:16};
const buffers = [];
const writes = [];
const fences = [];
const device = {
  createBuffer({size, label = ''}) {
    const buffer = {size, label, destroys:0, destroy() {this.destroys++;}};
    buffers.push(buffer);
    return buffer;
  },
  createShaderModule() {return {};},
  createComputePipeline() {return {};},
  queue: {
    writeBuffer(buffer) {
      assert.equal(buffer.destroys, 0, 'write to destroyed display storage');
      writes.push(buffer);
    },
    onSubmittedWorkDone() {return new Promise(resolve => fences.push(resolve));},
  },
};
const settle = async () => {
  fences.splice(0).forEach(resolve => resolve());
  await Promise.resolve(); await Promise.resolve(); await Promise.resolve();
};
const sourceA = device.createBuffer({size:64,label:'resident-A'});
const sourceB = device.createBuffer({size:64,label:'resident-B'});
const engine = new GPUColormapEngine(device);
switch (process.argv[2]) {
  case 'resident-switch':
    engine.adoptBuffer(41, sourceA, 4, 4, 'borrowed');
    engine.adoptBuffer(41, sourceB, 4, 4, 'borrowed');
    engine.adoptBuffer(41, sourceA, 4, 4, 'borrowed');
    await settle();
    // Simulate the next resident update after A-B-A and retired-slot cleanup.
    device.queue.writeBuffer(sourceA);
    device.queue.writeBuffer(sourceB);
    engine.releaseSlot(41);
    engine.adoptBuffer(41, sourceB, 4, 4, 'borrowed');
    engine.adoptBuffer(41, sourceA, 4, 4, 'borrowed');
    engine.destroy();
    await settle();
    break;
  case 'resize':
    engine.adoptBuffer(41, sourceA, 4, 4, 'borrowed');
    engine.adoptBuffer(41, sourceA, 8, 2, 'borrowed');
    await settle();
    device.queue.writeBuffer(sourceA);
    engine.destroy();
    break;
  case 'upload':
    engine.adoptBuffer(41, sourceA, 4, 4, 'borrowed');
    engine.uploadData(41, new Float32Array(16), 4, 4);
    assert.ok(!writes.includes(sourceA), 'upload must not overwrite a borrowed resident image');
    engine.destroy();
    break;
  case 'owned':
    engine.adoptBuffer(41, sourceA, 4, 4);
    engine.adoptBuffer(41, sourceA, 8, 2);
    await settle();
    device.queue.writeBuffer(sourceA);
    engine.adoptBuffer(41, sourceB, 4, 4);
    assert.equal(sourceA.destroys, 0, 'replaced owned image must survive its queue fence');
    await settle();
    assert.equal(sourceA.destroys, 1);
    engine.destroy();
    assert.equal(sourceB.destroys, 1);
    break;
}
if (process.argv[2] !== 'owned') {
  assert.equal(sourceA.destroys, 0);
  assert.equal(sourceB.destroys, 0);
  sourceA.destroy(); sourceB.destroy();
}
assert.ok(buffers.every(buffer => buffer.destroys === 1), 'release every owned buffer exactly once');
"""
    completed = subprocess.run(
        ["node", "--input-type=module", "-e", script, display_bundle.as_uri(), journey],
        cwd=ROOT, check=False, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("failure", ["none", "allocation", "binding"])
def test_live_range_reuses_and_releases_storage(
    display_bundle: Path, failure: str,
) -> None:
    """Current-frame normalization allocates once and retains source ownership."""
    script = r"""
import assert from 'node:assert/strict';
const {GPUColormapEngine, COLORMAPS} = await import(process.argv[1]);
globalThis.GPUBufferUsage = {STORAGE:1,COPY_SRC:2,COPY_DST:4,MAP_READ:8,UNIFORM:16};
Object.defineProperty(globalThis, 'navigator', {value: {
  gpu: {getPreferredCanvasFormat: () => 'rgba8unorm'},
}, configurable: true});
const buffers = [];
let bindGroups = 0, submits = 0, computes = 0, draws = 0;
let failure = 'none', rangeGroups = 0;
const pipeline = {getBindGroupLayout: () => ({})};
const device = {
  limits: {maxComputeWorkgroupsPerDimension: 65535},
  createBuffer({size}) {
    if (failure === 'allocation' && size === 16) {
      failure = 'none';
      throw new Error('injected live-range allocation failure');
    }
    const buffer = {size, destroys:0, destroy() {this.destroys++;}};
    buffers.push(buffer); return buffer;
  },
  createShaderModule: () => ({}),
  createComputePipeline: () => pipeline,
  createRenderPipeline: () => pipeline,
  createBindGroup() {
    if (failure === 'binding' && ++rangeGroups === 2) {
      failure = 'none';
      throw new Error('injected live-range binding failure');
    }
    bindGroups++; return {};
  },
  createCommandEncoder: () => ({
    beginComputePass: () => ({setPipeline() {}, setBindGroup() {},
      dispatchWorkgroups() {computes++;}, end() {}}),
    beginRenderPass: () => ({setPipeline() {}, setBindGroup() {},
      draw() {draws++;}, end() {}}),
    finish: () => ({}),
  }),
  queue: {
    writeBuffer(buffer) {assert.equal(buffer.destroys, 0);},
    submit() {submits++;},
    onSubmittedWorkDone: async () => {},
  },
};
const engine = new GPUColormapEngine(device);
engine.uploadLUT('inferno', COLORMAPS.inferno);
const source = device.createBuffer({size:512 * 512 * 4});
engine.adoptBuffer(0, source, 512, 512, 'borrowed');
const panel = {slot:0, width:512, height:512, logScale:false,
  range:{vminPct:0, vmaxPct:100},
  context:{getCurrentTexture: () => ({createView: () => ({})})}};
failure = process.argv[2];
if (failure !== 'none') {
  assert.throws(() => engine.renderSlotsDirectToCanvases([panel]), /injected live-range/);
  assert.equal(submits, 0, 'never submit an incomplete range pass');
  assert.equal(source.destroys, 0, 'failed display setup must retain the borrowed source');
}
assert.equal(engine.renderSlotsDirectToCanvases([panel]), 1);
const warmBuffers = buffers.length, warmGroups = bindGroups;
for (let frame = 0; frame < 20; frame++) {
  panel.logScale = frame % 2 === 0;
  assert.equal(engine.renderSlotsDirectToCanvases([panel]), 1);
}
assert.equal(buffers.length, warmBuffers, 'no per-frame GPU buffers');
assert.equal(bindGroups, warmGroups, 'no per-frame bind groups, even when scale changes');
assert.equal(submits, 21, 'one combined range-and-render submission per frame');
assert.equal(computes, 42, 'two current-frame range passes for every render');
assert.equal(draws, 21);
engine.uploadUint8Data(1, new Uint8Array(16), 4, 4);
const beforeUnsupported = {buffers:buffers.length, bindGroups, submits, computes, draws};
assert.equal(engine.renderSlotsDirectToCanvases([{...panel, slot:1, width:4, height:4}]), 0);
assert.deepEqual({buffers:buffers.length, bindGroups, submits, computes, draws}, beforeUnsupported,
  'packed uint8 frames must stay on their uint8-aware renderer');
engine.destroy();
assert.equal(source.destroys, 0, 'the resident source remains borrowed');
source.destroy();
assert.ok(buffers.every(buffer => buffer.destroys === 1), 'release range storage exactly once');
"""
    completed = subprocess.run(
        ["node", "--input-type=module", "-e", script, display_bundle.as_uri(), failure],
        cwd=ROOT, check=False, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
