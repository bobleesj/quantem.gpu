/// <reference types="@webgpu/types" />
import { SOURCE112_WGSL, SOURCE112_OFFSET_RESTART_WGSL, SOURCE112_OFFSET_RESTART_BUILD_WGSL } from '../../src/quantem/gpu/detector/compute/webgpu/source112-kernels';

/** Real-GPU synthetic parity: literals, escaped pairs, packet/segment boundaries,
 * zero-bit transitions at stream end, signed deltas and corruption rejection.
 * The host constructs a synthetic encoded fixture; no real payload is decoded here.
 */
export async function runRestartSyntheticParity(device: GPUDevice) {
  const columns = 20, packets = 32, frames = 512 * packets;
  const expected = new Uint16Array(frames * columns);
  const tables = new Uint32Array(3 * 1024);
  for (let state = 0; state < 1024; state++) {
    tables[state] = 4095; // Every symbol takes a 12-bit escaped pair.
    tables[1024 + state] = (5 << 12) | (state % 7 === 0 ? 4095 : state * 97 % 4095);
    tables[2048 + state] = (((state + 1) % 1024) << 16) | state;
  }
  tables[2048 + 20] |= 11 << 12; tables[2048 + 21] |= 11 << 12;
  const modelIds = Uint8Array.from({ length: columns }, (_, q) => q % 4 === 0 ? 255 : q % 4 - 1);
  const streams: number[][] = [];
  for (let packet = 0; packet < packets; packet++) for (let q = 0; q < columns; q++) {
    let rng = ((packet + 1) * 701 + q * 313) >>> 0;
    const random = () => { rng = (Math.imul(rng, 1664525) + 1013904223) >>> 0; return rng; };
    const words: number[] = []; let cursor = 0;
    const put = (value: number, count: number) => { for (let bit = 0; bit < count; bit++, cursor++) words[cursor >>> 5] = ((words[cursor >>> 5] ?? 0) | (((value >>> bit) & 1) << (cursor & 31))) >>> 0; };
    const model = modelIds[q];
    if (model === 255) {
      for (let pair = 0; pair < 256; pair++) {
        const low = pair % 64 === 63 || pair % 64 === 0 ? 65535 : random() & 65535;
        const high = pair % 64 === 63 ? 65535 : random() & 65535;
        words.push((low | (high << 16)) >>> 0);
        expected[((packet * 512 + pair * 2) * columns) + q] = low;
        expected[((packet * 512 + pair * 2 + 1) * columns) + q] = high;
      }
    } else {
      let state = model === 1 ? 31 : 0; put(state, 10);
      for (let pairIndex = 0; pairIndex < 256; pairIndex++) {
        const code = tables[model * 1024 + state], count = (code >>> 12) & 15;
        const low = model === 2 ? 0 : random() & ((1 << count) - 1);
        put(low, count); state = (code >>> 16) + low;
        let pair = code & 4095;
        if (pair === 4095) { pair = pairIndex % 64 === 63 || pairIndex % 64 === 0 ? 4095 : random() & 4095; put(pair, 12); }
        expected[((packet * 512 + pairIndex * 2) * columns) + q] = pair & 63;
        expected[((packet * 512 + pairIndex * 2 + 1) * columns) + q] = pair >>> 6;
      }
    }
    streams.push(words);
  }
  const payloadPrefix = 16;
  const absoluteStarts: number[] = [];
  const denseWords = streams.reduce((sum, stream) => sum + stream.length, 0);
  const offsets = new Uint32Array(columns + Math.ceil(streams.length / 4));
  const packed = new Uint32Array(payloadPrefix + denseWords + offsets.length); let at = 0;
  streams.forEach((stream, i) => {
    if (!(i % 32)) offsets[i / 32] = at;
    offsets[columns + (i >>> 2)] |= (stream.length - 1) << ((i & 3) * 8);
    absoluteStarts.push(payloadPrefix + at);
    packed.set(stream, payloadPrefix + at); at += stream.length;
  });
  packed.set(offsets, payloadPrefix + denseWords);
  const descriptor = new Uint32Array([payloadPrefix, payloadPrefix + denseWords, 0, 0, 0, 0, denseWords, 0, 0, 3, 12, 12 + streams.length * 3]);
  const owned: GPUBuffer[] = [];
  const buffer = (size: number, usage: GPUBufferUsageFlags, data?: ArrayBuffer) => {
    const result = device.createBuffer({ size, usage: usage | GPUBufferUsage.COPY_DST }); owned.push(result);
    if (data) device.queue.writeBuffer(result, 0, data); return result;
  };
  device.pushErrorScope('validation');
  try {
    const payload = buffer(packed.byteLength, GPUBufferUsage.STORAGE, packed.buffer);
    const records = buffer((12 + streams.length * 4) * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, descriptor.buffer);
    const cols = buffer(columns * 4, GPUBufferUsage.STORAGE, Uint32Array.from({ length: columns }, (_, i) => i).buffer);
    const ids = buffer(modelIds.byteLength, GPUBufferUsage.STORAGE, modelIds.buffer);
    const decoding = buffer(tables.byteLength, GPUBufferUsage.STORAGE, tables.buffer);
    const selected = buffer(columns * 4, GPUBufferUsage.STORAGE);
    const output = buffer(frames * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const errors = buffer(4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const params = buffer(64, GPUBufferUsage.UNIFORM);
    const read = buffer(frames * 4 + 4, GPUBufferUsage.MAP_READ);
    const layout = (writeRecords: boolean) => device.createBindGroupLayout({ entries: Array.from({ length: 9 }, (_, binding) => ({
      binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: binding === 8 ? 'uniform' : binding >= 6 || (binding === 1 && writeRecords) ? 'storage' : 'read-only-storage' },
    })) as GPUBindGroupLayoutEntry[] });
    const normalLayout = layout(false), buildLayout = layout(true);
    const pipeline = async (code: string, entryPoint: string, bindLayout: GPUBindGroupLayout) => {
      const module = device.createShaderModule({ code });
      const errors = (await module.getCompilationInfo()).messages.filter(m => m.type === 'error');
      if (errors.length) throw new Error(errors.map(m => m.message).join('\n'));
      return device.createComputePipelineAsync({ layout: device.createPipelineLayout({ bindGroupLayouts: [bindLayout] }), compute: { module, entryPoint } });
    };
    const baseline = await pipeline(SOURCE112_WGSL, 'decode_dense', normalLayout);
    const restart = await pipeline(SOURCE112_OFFSET_RESTART_WGSL.split('p.mode').join('0u'), 'decode_dense_restart', normalLayout);
    const build = await pipeline(SOURCE112_OFFSET_RESTART_BUILD_WGSL, 'build_dense_restarts', buildLayout);
    const bind = (layout: GPUBindGroupLayout) => device.createBindGroup({ layout, entries: [payload, records, cols, ids, decoding, selected, output, errors, params]
      .map((buffer, binding) => ({ binding, resource: { buffer } })) });
    const normalBind = bind(normalLayout), buildBind = bind(buildLayout);
    const dispatch = async (pipeline: GPUComputePipeline, group: GPUBindGroup, ranks: number[], width: number, clear: boolean, packetCount = packets) => {
      if (ranks.length) device.queue.writeBuffer(selected, 0, new Uint32Array(ranks));
      device.queue.writeBuffer(params, 0, new Uint32Array([0, 1, columns, columns, ranks.length, 0, 0, columns, 0, columns, 0, 0, 0, 0, 0, 0]));
      const encoder = device.createCommandEncoder(); encoder.clearBuffer(errors); if (clear) encoder.clearBuffer(output);
      const pass = encoder.beginComputePass(); pass.setPipeline(pipeline); pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(Math.ceil(ranks.length / width), packetCount, 1); pass.end();
      encoder.copyBufferToBuffer(output, 0, read, 0, frames * 4); encoder.copyBufferToBuffer(errors, 0, read, frames * 4, 4);
      device.queue.submit([encoder.finish()]); await read.mapAsync(GPUMapMode.READ);
      const values = new Uint32Array(read.getMappedRange()).slice(); read.unmap();
      return { values: values.subarray(0, frames), error: values[frames] };
    };
    const all = Array.from({ length: columns }, (_, i) => i);
    const preparation = await dispatch(build, buildBind, all, 64, true);
    if (preparation.error) throw new Error(`Synthetic restart preprocessing error ${preparation.error}.`);
    const offsetRead = buffer(streams.length * 4, GPUBufferUsage.MAP_READ);
    const offsetsEncoder = device.createCommandEncoder(); offsetsEncoder.copyBufferToBuffer(records, descriptor[11] * 4, offsetRead, 0, streams.length * 4);
    device.queue.submit([offsetsEncoder.finish()]); await offsetRead.mapAsync(GPUMapMode.READ);
    const actualOffsets = new Uint32Array(offsetRead.getMappedRange()).slice(); offsetRead.unmap();
    if (actualOffsets.some((first, stream) => first !== absoluteStarts[stream])) throw new Error('GPU absolute stream starts differ from independent serialized fixture offsets.');
    const results: Record<string, unknown>[] = [{ name: 'GPU-absolute-stream-starts', checked: streams.length, mismatches: 0 }];
    const compare = (name: string, actual: { values: Uint32Array; error: number }, ranks: number[]) => {
      let mismatches = 0;
      for (let frame = 0; frame < frames; frame++) {
        let sum = 0; for (const rank of ranks) sum += expected[frame * columns + rank];
        if (actual.values[frame] !== sum) mismatches++;
      }
      if (actual.error || mismatches) throw new Error(`${name}: decoderError=${actual.error}, mismatches=${mismatches}.`);
      results.push({ name, checked: frames, mismatches, decoderStatus: actual.error });
    };
    compare('baseline-all-columns', await dispatch(baseline, normalBind, all, 64, true), all);
    compare('restart-all-columns', await dispatch(restart, normalBind, all, 16, true), all);
    const removed = all.filter(q => q % 3 === 0), retained = all.filter(q => q % 3 !== 0);
    compare('restart-subtract-delta', await dispatch(restart, normalBind, removed.map(q => q | 0x1000000), 16, false), retained);
    compare('restart-add-delta', await dispatch(restart, normalBind, removed, 16, false), all);
    for (const rank of [0, 1, 2, 3, 17]) compare(`restart-single-column-${rank}`, await dispatch(restart, normalBind, [rank], 16, true), [rank]);
    for (const [name, first] of [['before-dense-base', 0], ['after-dense-end', payloadPrefix + denseWords], ['length-past-end', payloadPrefix + denseWords - 1]] as const) {
      device.queue.writeBuffer(records, (descriptor[11] + 1) * 4, new Uint32Array([first]));
      const malformed = await dispatch(restart, normalBind, [1], 16, true, 1);
      if (!(malformed.error & 1)) throw new Error(`Invalid cached offset was not rejected: ${name}`);
      results.push({ name: `offset-${name}-rejected`, decoderStatus: malformed.error });
    }
    device.queue.writeBuffer(records, (descriptor[11] + 1) * 4, new Uint32Array([absoluteStarts[1]]));
    // Rank1 packet0, segment1 checkpoint: absence must be rejected, never treated as a valid state0 restart.
    device.queue.writeBuffer(records, (12 + 1 * 3) * 4, new Uint32Array([0]));
    const missing = await dispatch(restart, normalBind, [1], 16, true, 1);
    if (!(missing.error & 64)) throw new Error('Missing restart checkpoint was not rejected.');
    results.push({ name: 'missing-checkpoint-rejected', decoderStatus: missing.error });
    device.queue.writeBuffer(records, (12 + 1 * 3) * 4, new Uint32Array([0x80000000 | (8191 << 10)]));
    const outside = await dispatch(restart, normalBind, [1], 16, true, 1);
    if (!(outside.error & 64)) throw new Error('Out-of-stream checkpoint cursor was not rejected.');
    results.push({ name: 'checkpoint-cursor-rejected', decoderStatus: outside.error });
    // Truncate the escaped rank1 stream to its header word in synthetic compact metadata.
    const lengthWord = payloadPrefix + denseWords + columns;
    device.queue.writeBuffer(payload, lengthWord * 4, new Uint32Array([packed[lengthWord] & ~(255 << 8)]));
    const truncated = await dispatch(build, buildBind, all, 64, true, 1);
    if (!(truncated.error & 4)) throw new Error('Truncated escape stream was not rejected.');
    results.push({ name: 'truncated-bits-rejected', decoderStatus: truncated.error });
    return { allPassed: true, syntheticOnly: true, sourceValues: expected.length, packets, columns, results };
  } finally {
    await device.queue.onSubmittedWorkDone().catch(() => {}); owned.forEach(buffer => buffer.destroy());
    const error = await device.popErrorScope(); if (error) throw new Error(error.message);
  }
}
