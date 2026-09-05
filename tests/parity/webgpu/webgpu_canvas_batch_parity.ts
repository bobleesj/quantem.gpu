import { GPUColormapEngine, COLORMAPS } from "../../../src/quantem/gpu/display/backends/webgpu/colormaps";

/** Compare every displayed RGBA channel against the existing single-canvas API. */
export async function verifyCanvasBatchParity(device: GPUDevice) {
  const engine = new GPUColormapEngine(device);
  const canvases = Array.from({ length: 7 }, () => new OffscreenCanvas(16, 16));
  const contexts = canvases.map(canvas => {
    const context = canvas.getContext("webgpu")!;
    context.configure({ device, format: navigator.gpu.getPreferredCanvasFormat(), alphaMode: "opaque" });
    return context;
  });
  const read = async () => {
    await device.queue.onSubmittedWorkDone();
    return canvases.map(canvas => {
      const bitmap = canvas.transferToImageBitmap();
      const copy = new OffscreenCanvas(16, 16);
      const context = copy.getContext("2d")!;
      context.drawImage(bitmap, 0, 0);
      bitmap.close();
      return context.getImageData(0, 0, 16, 16).data;
    });
  };
  const results = [];
  try {
    engine.uploadLUT("inferno", COLORMAPS.inferno);
    for (let generation = 0; generation < 3; generation++) {
      const panels = canvases.map((_, slot) => {
        const data = Float32Array.from({ length: 256 }, (__, pixel) =>
          (pixel * (slot + 1) + generation * 17) % 251);
        engine.uploadData(slot, data, 16, 16);
        return {
          slot, context: contexts[slot], width: 16, height: 16,
          range: { vmin: 0, vmax: slot % 2 ? Math.log1p(250) : 250 },
          logScale: slot % 2 === 1, smooth: false,
        };
      });
      for (const panel of panels) {
        engine.renderPanelSlotsDirectToCanvas([panel.slot], panel.range, panel.logScale, panel.context, {
          width: 16, height: 16, panelCount: 1, cols: 1, rows: 1, gap: 0, bgRgb: 0, smooth: false,
        });
      }
      const expected = await read();
      const rendered = engine.renderSlotsDirectToCanvases(panels);
      const actual = await read();
      results.push({ generation, rendered, panels: actual.map((pixels, slot) => ({
        slot, channels: pixels.length,
        mismatches: pixels.reduce((count, value, index) => count + Number(value !== expected[slot][index]), 0),
        nonzeroChannels: pixels.reduce((count, value, index) => count + Number(index % 4 !== 3 && value !== 0), 0),
      })) });
    }
    return results;
  } finally {
    await device.queue.onSubmittedWorkDone();
    contexts.forEach(context => context.unconfigure());
    engine.destroy();
  }
}

/** Compare changing GPU-derived display ranges with independently computed limits. */
export async function verifyLiveRangeParity(device: GPUDevice) {
  const engine = new GPUColormapEngine(device);
  const width = 64, height = 65;
  const canvases = Array.from({ length: 7 }, () => new OffscreenCanvas(width, height));
  const contexts = canvases.map(canvas => {
    const context = canvas.getContext("webgpu")!;
    context.configure({ device, format: navigator.gpu.getPreferredCanvasFormat(), alphaMode: "opaque" });
    return context;
  });
  const read = async () => {
    await device.queue.onSubmittedWorkDone();
    return canvases.map(canvas => {
      const bitmap = canvas.transferToImageBitmap();
      const copy = new OffscreenCanvas(width, height);
      const context = copy.getContext("2d")!;
      context.drawImage(bitmap, 0, 0);
      bitmap.close();
      return context.getImageData(0, 0, width, height).data;
    });
  };
  const results = [];
  try {
    for (let generation = 0; generation < 5; generation++) {
      const lutName = generation % 2 ? "gray" : "inferno";
      engine.uploadLUT(lutName, COLORMAPS[lutName]);
      const panels = contexts.map((context, slot) => {
        const factor = [1000, 0.01, 50, 0, 1][generation];
        const data = Float32Array.from({ length: width * height }, (_, pixel) =>
          factor * (((pixel * (slot + 1)) % 251) - (slot % 3 === 0 ? 125 : 0)));
        if (generation !== 3) { data[0] = NaN; data[1] = Infinity; data[2] = -Infinity; }
        engine.uploadData(slot, data, width, height);
        const finite = data.filter(Number.isFinite);
        const logScale = slot % 2 === 1;
        const scaled = (value: number) => logScale ? Math.sign(value) * Math.log1p(Math.abs(value)) : value;
        const low = scaled(Math.min(...finite)), high = scaled(Math.max(...finite));
        const vminPct = slot % 2 ? 3 : 0, vmaxPct = slot % 2 ? 97 : 100;
        const range = { vmin: low + (high - low) * vminPct / 100, vmax: low + (high - low) * vmaxPct / 100 };
        engine.renderPanelSlotsDirectToCanvas([slot], range, logScale, context, {
          width, height, panelCount: 1, cols: 1, rows: 1, gap: 0, bgRgb: 0,
        });
        return { slot, context, width, height, range: { vminPct, vmaxPct }, logScale };
      });
      const expected = await read();
      const rendered = engine.renderSlotsDirectToCanvases(panels);
      const actual = await read();
      results.push({ generation, rendered, panels: actual.map((pixels, slot) => ({
        slot, channels: pixels.length,
        mismatches: pixels.reduce((count, value, index) => count + Number(value !== expected[slot][index]), 0),
        maxError: pixels.reduce((largest, value, index) => Math.max(largest, Math.abs(value - expected[slot][index])), 0),
        nonzeroChannels: pixels.reduce((count, value, index) => count + Number(index % 4 !== 3 && value !== 0), 0),
      })) });
    }
    return results;
  } finally {
    await device.queue.onSubmittedWorkDone();
    contexts.forEach(context => context.unconfigure());
    engine.destroy();
  }
}
