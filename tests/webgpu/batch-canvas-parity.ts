/// <reference types="@webgpu/types" />
import { GPUColormapEngine } from "../../src/quantem/gpu/display/webgpu/colormaps";

type Surface = { texture: GPUTexture; context: GPUCanvasContext };

/** Real texture parity for batched compare rendering versus serial one-slot submits.
 * Supply the already authenticated hardware device; no adapter or canvas is selected.
 */
export async function runBatchCanvasParity(device: GPUDevice): Promise<Record<string, number | boolean>> {
  device.pushErrorScope("validation");
  let submits = 0;
  const queue = new Proxy(device.queue, { get(target, key) {
    if (key === "submit") return (commands: GPUCommandBuffer[]) => { submits++; target.submit(commands); };
    const value = Reflect.get(target, key, target);
    return typeof value === "function" ? value.bind(target) : value;
  } });
  const countedDevice = new Proxy(device, { get(target, key) {
    if (key === "queue") return queue;
    const value = Reflect.get(target, key, target);
    return typeof value === "function" ? value.bind(target) : value;
  } });
  const engine = new GPUColormapEngine(countedDevice);
  const format = navigator.gpu.getPreferredCanvasFormat();
  const width = 91, height = 69, bytesPerRow = Math.ceil(width * 4 / 256) * 256;
  const shapes = [[7, 11], [32, 33], [65, 19], [13, 83], [97, 71], [1, 9], [48, 67]];
  const indices = [0, 1, 2, 3, 4, 5, 6];
  const surface = (): Surface => {
    const texture = device.createTexture({ size: [width, height], format, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    return { texture, context: { getCurrentTexture: () => texture } as unknown as GPUCanvasContext };
  };
  const batch = indices.map(surface), serial = indices.map(surface);
  const read = device.createBuffer({ size: bytesPerRow * height * 14, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const results: Record<string, number | boolean> = { cases: 0, exact_pixels: 0, batch_submits_per_frame: 1, serial_submits_per_frame: 7 };
  try {
    // Distinct input dimensions, signed ranges, constants, sparse/large values,
    // and different smoothing thresholds catch cross-slot uniform contamination.
    const lut = Uint8Array.from({ length: 768 }, (_, index) => {
      const bin = Math.floor(index / 3), channel = index % 3;
      return channel === 0 ? bin : channel === 1 ? 255 - bin : (bin * 37) % 256;
    });
    engine.uploadLUT("batch-parity", lut);
    const cases = [
      { log: false, zoom: 1, panX: 0, panY: 0, smooth: false },
      { log: true, zoom: 1, panX: 0, panY: 0, smooth: false },
      { log: false, zoom: 0.63, panX: -3.25, panY: 1.5, smooth: true },
      { log: true, zoom: 0.63, panX: -3.25, panY: 1.5, smooth: true },
      { log: false, zoom: 2.37, panX: 4.75, panY: -2.5, smooth: true },
      { log: true, zoom: 2.37, panX: 4.75, panY: -2.5, smooth: true },
    ];
    for (let iteration = 0; iteration < cases.length; iteration++) {
      const scenario = cases[iteration];
      shapes.forEach(([cols, rows], index) => {
        const values = Float32Array.from({ length: cols * rows }, (_, pixel) => {
          if (index === 5) return iteration - 3;
          const value = ((pixel * (13 + index * 2) + iteration * 29) % 1031) - 517;
          return index === 4 ? value * 65535 : index === 2 ? Math.max(0, value) / 17 : value * (index + 1) + index / 7;
        });
        engine.uploadData(index, values, cols, rows);
      });
      const opts = { width, height, bgRgb: 0x193B71,
        transform: { zoom: scenario.zoom, panX: scenario.panX, panY: scenario.panY }, smooth: scenario.smooth };
      const beforeBatch = submits;
      const painted = engine.renderSlotsDirectWithGpuRangeToCanvases(indices, batch.map(item => item.context), 7, 93, scenario.log, opts);
      if (painted !== 7 || submits - beforeBatch !== 1) throw new Error("Seven-slot frame did not use exactly one submission");
      const beforeSerial = submits;
      indices.forEach(index => {
        if (!engine.renderSlotDirectWithGpuRangeToCanvas(index, 7, 93, scenario.log, serial[index].context, opts)) throw new Error(`Serial slot ${index} was not rendered`);
      });
      if (submits - beforeSerial !== 7) throw new Error("Serial reference did not submit once per slot");
      // No completion wait between submissions: this also exercises slot uniform
      // reuse and temporary-parameter destruction while frames are queued.
      const encoder = device.createCommandEncoder();
      [...batch, ...serial].forEach((item, index) => encoder.copyTextureToBuffer(
        { texture: item.texture }, { buffer: read, bytesPerRow, offset: index * bytesPerRow * height }, [width, height],
      ));
      device.queue.submit([encoder.finish()]); await read.mapAsync(GPUMapMode.READ);
      try {
        const pixels = new Uint8Array(read.getMappedRange());
        for (const slot of indices) for (let row = 0; row < height; row++) for (let channel = 0; channel < width * 4; channel++) {
          const offset = slot * bytesPerRow * height + row * bytesPerRow + channel;
          const reference = offset + 7 * bytesPerRow * height;
          if (pixels[offset] !== pixels[reference]) throw new Error(`Case ${iteration}, slot ${slot}, row ${row}, channel ${channel}: batch ${pixels[offset]} != serial ${pixels[reference]}`);
        }
      } finally { read.unmap(); }
      results.cases = Number(results.cases) + 1;
      results.exact_pixels = Number(results.exact_pixels) + 7 * width * height;
    }
    const options = { width, height, bgRgb: 0 };
    const beforeInvalid = submits;
    for (const [slots, contexts] of [
      [[0, 0], [batch[0].context, batch[1].context]], [[0, 1], [batch[0].context]],
    ] as [number[], GPUCanvasContext[]][]) {
      let rejected = false;
      try { engine.renderSlotsDirectWithGpuRangeToCanvases(slots, contexts, 0, 100, false, options); } catch { rejected = true; }
      if (!rejected) throw new Error("Invalid slot/context pairing was accepted");
    }
    if (engine.renderSlotsDirectWithGpuRangeToCanvases([], [], 0, 100, false, options) !== 0) throw new Error("Empty batch rendered");
    if (engine.renderSlotsDirectWithGpuRangeToCanvases([99], [batch[0].context], 0, 100, false, options) !== 0) throw new Error("Missing slot rendered");
    if (submits !== beforeInvalid) throw new Error("Empty or rejected input submitted GPU work");
    // Public record helpers may leave another encoder unsubmitted. An empty
    // batch or failed presentation must not destroy that encoder's uniforms.
    const held = device.createCommandEncoder();
    if (!engine.recordComputeRangeRegion(held, 0)) throw new Error("Could not record held range");
    engine.renderSlotsDirectWithGpuRangeToCanvases([], [], 0, 100, false, options);
    let failedPresentation = false;
    const unavailable = { getCurrentTexture() { throw new Error("Synthetic unavailable surface"); } } as unknown as GPUCanvasContext;
    const beforeFailure = submits;
    try { engine.renderSlotsDirectWithGpuRangeToCanvases([0, 1], [batch[0].context, unavailable], 0, 100, false, options); }
    catch (error) { failedPresentation = String(error).includes("Synthetic unavailable surface"); }
    if (!failedPresentation || submits !== beforeFailure) throw new Error("Failed presentation submitted a partial frame");
    device.queue.submit([held.finish()]);
    // The already-recorded range is now submitted; this normal convenience
    // call may safely release its retained temporary buffers too.
    engine.computeRangeRegion(0);
    if (engine.renderSlotsDirectWithGpuRangeToCanvases(indices, batch.map(item => item.context), 0, 100, false, options) !== 7) throw new Error("Batch did not recover after a failed surface");
    await device.queue.onSubmittedWorkDone();
    results.held_encoder_survives_empty_and_failed_batch = true;
    results.invalid_input_rejected = true;
    results.all_passed = true;
    return results;
  } finally {
    read.destroy(); [...batch, ...serial].forEach(item => item.texture.destroy()); engine.destroy();
    const validation = await device.popErrorScope();
    if (validation) throw new Error(validation.message);
  }
}
