/// <reference types="@webgpu/types" />
import { GPUColormapEngine } from "../../src/quantem/gpu/display/webgpu/colormaps";

/** All 66 native display images, one swapchain versus canonical separate canvases.
 * Optional inputs are borrowed resident float images; owned copies protect their lifetime.
 * This is a diagnostic parity test, not end-to-end widget throughput evidence.
 */
export async function runSharedCanvasParity(device: GPUDevice, inputs?: GPUBuffer[]) {
  const count = 66, size = 512, sourceBytes = size * size * 4;
  if (inputs && inputs.length !== count) throw new Error("Supply all 66 native float display buffers.");
  device.pushErrorScope("validation");
  const engine = new GPUColormapEngine(device);
  const reference = new GPUColormapEngine(device);
  const indices = Array.from({ length: count }, (_, i) => i);
  const contexts: GPUCanvasContext[] = [];
  const format = navigator.gpu.getPreferredCanvasFormat();
  const results: Record<string, unknown>[] = [];
  const lut = Uint8Array.from({ length: 768 }, (_, i) => Math.floor(i / 3));
  try {
    engine.uploadLUT("shared-parity-gray", lut); reference.uploadLUT("shared-parity-gray", lut);
    for (const index of indices) {
      const buffers = [engine, reference].map(owner => {
        const buffer = device.createBuffer({ size: sourceBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
        owner.adoptBuffer(index, buffer, size, size); return buffer;
      });
      if (inputs) {
        const encoder = device.createCommandEncoder();
        for (const buffer of buffers) encoder.copyBufferToBuffer(inputs[index], 0, buffer, 0, sourceBytes);
        device.queue.submit([encoder.finish()]);
      } else {
        const values = Float32Array.from({ length: size * size }, (_, p) => ((p * (index + 3) + index * 701) % 65521) / (index + 1));
        for (const buffer of buffers) device.queue.writeBuffer(buffer, 0, values);
      }
    }
    for (const test of [
      { name: "native-linear", tile: 512, log: false, zoom: 1, panX: 0, panY: 0, smooth: false },
      { name: "native-log", tile: 512, log: true, zoom: 1, panX: 0, panY: 0, smooth: false },
      { name: "native-zoom-pan", tile: 512, log: false, zoom: 1.7, panX: -31, panY: 19, smooth: false },
      { name: "scaled-smooth-log", tile: 192, log: true, zoom: 1.3, panX: -7.25, panY: 11.5, smooth: true },
    ]) {
      const gap = 7, cols = 11, rows = 6;
      const width = cols * test.tile + (cols - 1) * gap, height = rows * test.tile + (rows - 1) * gap;
      const rectangles = indices.map(i => ({ x: (i % cols) * (test.tile + gap), y: Math.floor(i / cols) * (test.tile + gap), width: test.tile, height: test.tile }));
      const makeCanvas = (owner: GPUColormapEngine, w: number, h: number) => {
        const canvas = document.createElement("canvas");
        const ctx = owner.configureCanvas(canvas, w, h);
        if (!ctx) throw new Error("WebGPU canvas configuration failed.");
        ctx.configure({ device, format, alphaMode: "opaque", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
        contexts.push(ctx); return ctx;
      };
      const atlas = makeCanvas(engine, width, height);
      const serial = indices.map(() => makeCanvas(reference, test.tile, test.tile));
      const rowBytes = Math.ceil(test.tile * 4 / 256) * 256, panelBytes = rowBytes * test.tile;
      const read = device.createBuffer({ size: panelBytes * count * 2 + 256, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      try {
        await device.queue.onSubmittedWorkDone();
        const begin = performance.now();
        let acquisitions = 0;
        const countedContext = { getCurrentTexture: () => { acquisitions++; return atlas.getCurrentTexture(); } } as unknown as GPUCanvasContext;
        const painted = engine.renderSlotsDirectWithGpuRangeToCanvas(indices, rectangles, countedContext, 3, 94, test.log,
          { width, height, bgRgb: 0, transform: test, smooth: test.smooth });
        // Capture current textures before awaiting/presentation invalidates them.
        const encoder = device.createCommandEncoder();
        const atlasTexture = atlas.getCurrentTexture();
        indices.forEach(i => encoder.copyTextureToBuffer({ texture: atlasTexture, origin: [rectangles[i].x, rectangles[i].y] },
          { buffer: read, offset: i * panelBytes, bytesPerRow: rowBytes }, [test.tile, test.tile]));
        // A gap must stay black even while all tiles draw into one render pass.
        encoder.copyTextureToBuffer({ texture: atlasTexture, origin: [test.tile + 1, 0] },
          { buffer: read, offset: panelBytes * count * 2, bytesPerRow: 256 }, [1, 1]);
        const recordedMs = performance.now() - begin;
        const scaledTransform = { zoom: test.zoom, panX: test.panX * test.tile / size, panY: test.panY * test.tile / size };
        reference.renderSlotsDirectWithGpuRangeToCanvases(indices, serial, 3, 94, test.log,
          { width: test.tile, height: test.tile, bgRgb: 0, transform: scaledTransform, smooth: test.smooth });
        indices.forEach(i => encoder.copyTextureToBuffer({ texture: serial[i].getCurrentTexture() },
          { buffer: read, offset: (i + count) * panelBytes, bytesPerRow: rowBytes }, [test.tile, test.tile]));
        device.queue.submit([encoder.finish()]);
        await read.mapAsync(GPUMapMode.READ);
        const pixels = new Uint8Array(read.getMappedRange());
        let mismatches = 0, maxChannelError = 0;
        for (let i = 0; i < count * panelBytes; i++) {
          const error = Math.abs(pixels[i] - pixels[i + count * panelBytes]);
          if (error) mismatches++;
          maxChannelError = Math.max(maxChannelError, error);
        }
        const gapOffset = panelBytes * count * 2;
        const blackGap = pixels[gapOffset] === 0 && pixels[gapOffset + 1] === 0 && pixels[gapOffset + 2] === 0 && pixels[gapOffset + 3] === 255;
        const tolerance = test.smooth ? 1 : 0;
        if (painted !== count || acquisitions !== 1 || !blackGap || maxChannelError > tolerance) {
          throw new Error(`${test.name}: shared parity failed: painted=${painted}, acquisitions=${acquisitions}, gap=${blackGap}, maxRGBA=${maxChannelError}, mismatches=${mismatches}`);
        }
        results.push({ ...test, painted, acquisitions, blackGap, mismatches, maxChannelError, tolerance, recordedMs,
          sourceShape: [66, 512, 512], targetShape: [height, width] });
      } finally {
        read.destroy();
        contexts.splice(0).forEach(context => context.unconfigure());
      }
    }
    return { diagnosticOnly: true, allPassed: true, results };
  } finally {
    contexts.forEach(context => context.unconfigure()); engine.destroy(); reference.destroy();
    const error = await device.popErrorScope();
    if (error) throw new Error(error.message);
  }
}
