/** CPU-only checks of the SSB engine's option packing: higher-order keys and the sample tilt. */
import assert from "node:assert/strict";
import test from "node:test";
import "./fake-gpu";
import { makeParams, packAberrations } from "../../src/quantem/gpu/ssb/webgpu/backend";

const calibration = {
  num_bright_field_pixels: 4, wavelength_A: 0.019687, semiangle_rad: 0.03, angular_sampling_rad: [0.002, 0.002],
  dc_value: [1, 0],
} as unknown as Parameters<typeof makeParams>[0];

test("a higher-order key the engine does not know is refused with the accepted keys", () => {
  // the spellings before the unit suffix: they read as 0, the image of no aberration
  for (const earlier of [{ C30: 5e4 }, { C21_mag: 300, C21_angle: 20 }, { C30_nm: 5e3 }]) {
    assert.throws(() => packAberrations(-120, 35, 30, earlier), (error: Error) => {
      for (const key of Object.keys(earlier)) assert.ok(error.message.includes(key), error.message);
      for (const accepted of ["C21_A", "phi21_deg", "C30_A", "C50_A", "phi56_deg"]) assert.ok(error.message.includes(accepted), error.message);
      return true;
    });
  }
});

test("known higher-order keys pack their magnitude and azimuth, and none leaves the standard path", () => {
  assert.equal(packAberrations(-120, 35, 30).active, false);
  assert.equal(packAberrations(-120, 35, 30, {}).active, false);
  const packed = packAberrations(-120, 35, 30, { C30_A: 5e4, C21_A: 300, phi21_deg: 90 });
  assert.equal(packed.active, true);
  // slot 4 is C30 (no azimuth): magnitude / (n + 1) with cos 1, sin 0
  assert.deepEqual(Array.from(packed.data.subarray(16, 19)), [Math.fround(5e4 / 4), 1, 0]);
  // slot 2 is C21, m = 1: the azimuth enters as cos and sin of 90 degrees
  assert.equal(packed.data[8], Math.fround(300 / 3));
  assert.ok(Math.abs(packed.data[9]) < 1e-7 && Math.abs(packed.data[10] - 1) < 1e-7);
});

test("a depth spread without a tilt is an untilted crystal", () => {
  const words = (sample?: Parameters<typeof makeParams>[12]) =>
    Array.from(new Float32Array(makeParams(calibration, 128, -120, 35, 30, 4, 0, 4, true, 4, false, sample)).subarray(22, 25));
  // omitting the tilt threw a TypeError; it is the same request as a zero tilt
  assert.deepEqual(words({ depthSpreadA: 100 }), [100, 0, 0]);
  assert.deepEqual(words({ depthSpreadA: 100 }), words({ depthSpreadA: 100, sampleTiltMrad: [0, 0] }));
  assert.deepEqual(words({ depthSpreadA: 100, sampleTiltMrad: [2, -3] }), [100, Math.fround(2e-3), Math.fround(-3e-3)]);
  // no depth spread: the standard SSB gamma, whatever the tilt
  assert.deepEqual(words({ depthSpreadA: 0, sampleTiltMrad: [2, -3] }), [0, 0, 0]);
  assert.deepEqual(words(), [0, 0, 0]);
});
