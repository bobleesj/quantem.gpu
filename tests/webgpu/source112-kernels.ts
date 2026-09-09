/// <reference types="@webgpu/types" />
import { SOURCE112_WGSL, SOURCE112_SUM_WGSL } from "../../src/quantem/gpu/detector/compute/webgpu/source112-kernels";

const D = 67, C = 35, K = D + C, N = 16384, RECORDS = 2;
function compactOffsets(lengths: number[], dense: boolean): Uint32Array {
  const coarse = lengths.length / 32;
  const words = new Uint32Array(coarse + Number(!dense) + Math.ceil(lengths.length / 4));
  const bytes = new Uint8Array(words.buffer, (coarse + Number(!dense)) * 4);
  let cursor = 0;
  lengths.forEach((length, i) => {
    if (i % 32 === 0) words[i / 32] = cursor;
    bytes[i] = length - Number(dense); cursor += length;
  });
  if (!dense) words[coarse] = cursor;
  return words;
}
/** Independent fixture writer: forward table transitions, original compact offsets. */
function fixture() {
  const native = new Uint16Array(RECORDS * N * K);
  const decoding = new Uint32Array(2048);
  for (let state = 0; state < 1024; state++) {
    decoding[state] = (state === 0 ? 5 | (9 << 6) : 4095) | (1 << 12);
    decoding[1024 + state] = ((state * 29) % 4095) | (10 << 12);
  }
  const ids = new Uint8Array(RECORDS * K);
  const descriptors = new Uint32Array(RECORDS * 12);
  const parts: Uint32Array[] = [new Uint32Array(13)]; let wordCursor = 13;
  const append = (input: number[] | Uint32Array) => {
    const data = input instanceof Uint32Array ? input : new Uint32Array(input);
    const start = wordCursor; parts.push(data); wordCursor += data.length; return start;
  };
  for (let record = 0; record < RECORDS; record++) {
    for (let q = 0; q < K; q++) ids[record * K + q] = q >= D ? 254 : q % 5 === 0 ? 255 : (q + record) % 2;
    const dense: number[] = [], denseLengths: number[] = [];
    for (let packet = 0; packet < 32; packet++) for (let q = 0; q < D; q++) {
      const model = ids[record * K + q]; const words: number[] = [];
      let state = model === 0 ? (packet + q) % 2 : (packet * 17 + q) % 1024;
      let bit = 10; words[0] = state;
      const write = (value: number, count: number) => {
        // Independent bit-by-bit encoder avoids duplicating GPU reservoir math.
        for (let b = 0; b < count; b++, bit++) if (value & (1 << b)) words[bit >>> 5] = (words[bit >>> 5] ?? 0) | (1 << (bit & 31));
        while (words.length < Math.ceil(bit / 32)) words.push(0);
      };
      for (let pair = 0; pair < 256; pair++) {
        let a: number, b: number;
        if (model === 255) {
          a = pair % 17 === 0 ? 65535 : (pair * 271 + q * 31 + record) & 65535;
          b = pair % 19 === 0 ? 32768 : (65535 - pair * 17 - packet) & 65535;
          words[pair] = (a | (b << 16)) >>> 0;
        } else {
          const code = decoding[model * 1024 + state];
          const next = model === 0 ? (pair + q + packet) % 2 : (pair * 37 + q * 11 + packet) % 1024;
          write(next, (code >>> 12) & 15); state = next;
          let symbol = code & 4095;
          if (symbol === 4095) { symbol = (pair * 173 + q * 61 + packet) & 4095; write(symbol, 12); }
          a = symbol & 63; b = symbol >>> 6;
        }
        const scan = packet * 512 + pair * 2;
        native[(record * N + scan) * K + q] = a;
        native[(record * N + scan + 1) * K + q] = b;
      }
      denseLengths.push(words.length); dense.push(...words);
    }
    const sparseLengths: number[] = [], positions: number[] = [], values: number[] = [];
    for (let packet = 0; packet < 32; packet++) for (let rank = 0; rank < C; rank++) {
      let count = 0;
      for (let frame = 0; rank % 7 !== 0 && frame < 512; frame++) if ((frame + packet + rank) % 31 === 0) {
        const value = frame % 3 === 0 ? 1 : 2 + (frame * 7 + rank + record) % 126;
        positions.push(frame); values.push(value); count++;
        native[(record * N + packet * 512 + frame) * K + D + rank] = value;
      }
      sparseLengths.push(count);
    }
    const n = positions.length, pwords = Math.ceil(n * 9 / 32), fwords = Math.ceil(n / 32), rwords = Math.ceil(n / 256);
    const pos = new Uint32Array(pwords), flags = new Uint32Array(fwords), ranks = new Uint32Array(rwords);
    const explicit: number[] = [];
    for (let i = 0; i < n; i++) {
      for (let b = 0; b < 9; b++) if (positions[i] & (1 << b)) pos[(i * 9 + b) >>> 5] |= 1 << ((i * 9 + b) & 31);
      if (i % 256 === 0) ranks[i / 256] = explicit.length;
      if (values[i] !== 1) { flags[i >>> 5] |= 1 << (i & 31); explicit.push(values[i]); }
    }
    const valueWords = new Uint32Array(Math.ceil(explicit.length / 4)); new Uint8Array(valueWords.buffer).set(explicit);
    const sparse = new Uint32Array(4 + pwords + fwords + rwords + valueWords.length);
    sparse.set([n, pwords, fwords, rwords]); sparse.set(pos, 4); sparse.set(flags, 4 + pwords); sparse.set(ranks, 4 + pwords + fwords); sparse.set(valueWords, 4 + pwords + fwords + rwords);
    const denseBase = append(dense), denseOffsets = append(compactOffsets(denseLengths, true));
    const sparseBase = append(sparse), sparseOffsets = append(compactOffsets(sparseLengths, false));
    descriptors.set([denseBase, denseOffsets, sparseBase, sparseOffsets, record * K, record * N, dense.length, sparse.length, 0, 2, 0, 0], record * 12);
  }
  const packed = new Uint32Array(wordCursor); let at = 0;
  for (const part of parts) { packed.set(part, at); at += part.length; }
  return { native, packed, descriptors, ids, decoding };
}

export async function runSource112KernelParity(device: GPUDevice): Promise<Record<string, boolean>> {
  const f = fixture(); const owned: GPUBuffer[] = [];
  const upload = (view: ArrayBufferView, usage = GPUBufferUsage.STORAGE) => {
    const buffer = device.createBuffer({ size: Math.max(4, Math.ceil(view.byteLength / 4) * 4), usage: usage | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    device.queue.writeBuffer(buffer, 0, view.buffer, view.byteOffset, view.byteLength); owned.push(buffer); return buffer;
  };
  device.pushErrorScope("validation");
  try {
    const module = device.createShaderModule({ code: SOURCE112_WGSL });
    const messages = (await module.getCompilationInfo()).messages.filter(message => message.type === "error");
    if (messages.length) throw new Error(messages.map(message => `${message.lineNum}: ${message.message}`).join("\n"));
    const layout = device.createBindGroupLayout({ entries: [...Array.from({ length: 8 }, (_, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: binding < 6 ? "read-only-storage" as const : "storage" as const } })), { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } }] });
    const pl = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const pipelines = ["decode_dense", "decode_sparse"].map(entryPoint => device.createComputePipeline({ layout: pl, compute: { module, entryPoint } }));
    const sumModule = device.createShaderModule({ code: SOURCE112_SUM_WGSL });
    const sumMessages = (await sumModule.getCompilationInfo()).messages.filter(message => message.type === "error");
    if (sumMessages.length) throw new Error(sumMessages.map(message => `${message.lineNum}: ${message.message}`).join("\n"));
    const sumPipeline = device.createComputePipeline({ layout: pl, compute: { module: sumModule, entryPoint: "decode_dense" } });
    const payload = upload(f.packed), records = upload(f.descriptors), ids = upload(f.ids), decoding = upload(f.decoding);
    const output = upload(new Uint32Array(RECORDS * N)), errors = upload(new Uint32Array(1));
    const columns = [upload(Uint32Array.from({ length: D }, (_, q) => q)), upload(Uint32Array.from({ length: C }, (_, rank) => D + rank))];
    const selections = [upload(new Uint32Array(D)), upload(new Uint32Array(C))];
    const params = [upload(new Uint32Array(16), GPUBufferUsage.UNIFORM), upload(new Uint32Array(16), GPUBufferUsage.UNIFORM)];
    const groups = [0, 1].map(i => device.createBindGroup({ layout, entries: [payload, records, columns[i], ids, decoding, selections[i], output, errors, params[i]].map((buffer, binding) => ({ binding, resource: { buffer } })) }));
    const read = async (checkFaults = true) => {
      const rb = device.createBuffer({ size: (RECORDS * N + 1) * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const enc = device.createCommandEncoder(); enc.copyBufferToBuffer(output, 0, rb, 0, RECORDS * N * 4); enc.copyBufferToBuffer(errors, 0, rb, RECORDS * N * 4, 4); device.queue.submit([enc.finish()]);
      try { await rb.mapAsync(GPUMapMode.READ); const data = new Uint32Array(rb.getMappedRange().slice(0)); if (checkFaults && data[RECORDS * N]) throw new Error(`Source112 fault bits ${data[RECORDS * N]}`); return data; } finally { rb.destroy(); }
    };
    const dispatch = (lists: Uint32Array[], mode: number, frame = 0, recordFirst = 0, recordCount = RECORDS) => {
      const enc = device.createCommandEncoder(); const pass = enc.beginComputePass();
      for (let i = 0; i < 2; i++) {
        if (!lists[i].length) continue;
        device.queue.writeBuffer(selections[i], 0, lists[i].slice());
        device.queue.writeBuffer(params[i], 0, new Uint32Array([recordFirst, recordCount, i ? C : D, i ? C + 1 : D, lists[i].length, mode, frame, K, 0, K, RECORDS * K, 0, 0, 0, 0, 0]));
        pass.setPipeline(mode === 0 && i === 0 ? sumPipeline : pipelines[i]); pass.setBindGroup(0, groups[i]); pass.dispatchWorkgroups(Math.ceil(lists[i].length / 64), mode === 0 ? 32 : 1, recordCount);
      }
      pass.end(); device.queue.submit([enc.finish()]);
    };
    const full = [Uint32Array.from({ length: D }, (_, q) => q), Uint32Array.from({ length: C }, (_, q) => q)];
    let mask = new Uint32Array(K).fill(1); dispatch(full, 0);
    const checkMask = async () => {
      const actual = await read();
      for (let scan = 0; scan < RECORDS * N; scan++) {
        let expected = 0; for (let q = 0; q < K; q++) if (mask[q]) expected += f.native[scan * K + q];
        if (actual[scan] !== expected) throw new Error(`Mask mismatch ${scan}: ${actual[scan]} != ${expected}`);
      }
    };
    await checkMask();
    for (let step = 0; step < 3; step++) {
      const next = Uint32Array.from({ length: K }, (_, q) => Number((q + step) % 3 === 0));
      const lists: number[][] = [[], []];
      for (let q = 0; q < K; q++) if (next[q] !== mask[q]) lists[Number(q >= D)].push((q >= D ? q - D : q) | (next[q] ? 0 : 0x1000000));
      dispatch(lists.map(values => new Uint32Array(values)), 0); mask = next; await checkMask();
    }
    for (const frame of [0, 1, 31, 32, 511, 512, 513, 8191, 16383]) {
      device.queue.writeBuffer(output, 0, new Uint32Array(RECORDS * N)); dispatch(full, 1, frame);
      const actual = await read();
      for (let record = 0; record < RECORDS; record++) for (let q = 0; q < K; q++) if (actual[record * K + q] !== f.native[(record * N + frame) * K + q]) throw new Error(`Pattern mismatch ${record}:${frame}:${q}`);
    }
    device.queue.writeBuffer(output, 0, new Uint32Array(RECORDS * N));
    const frames = [0, 511, 512, 16383];
    for (const frame of frames) dispatch(full, 2, frame);
    const gathered = await read();
    for (let record = 0; record < RECORDS; record++) for (let q = 0; q < K; q++) {
      const expected = frames.reduce((sum, frame) => sum + f.native[(record * N + frame) * K + q], 0);
      if (gathered[record * K + q] !== expected || gathered[RECORDS * K + record * K + q] !== 0) throw new Error("Gather mismatch");
    }
    // Force carry without decoding an impractically large number of patterns.
    const initial = new Uint32Array(RECORDS * K * 2); initial.fill(0xfffffff0, 0, RECORDS * K);
    device.queue.writeBuffer(output, 0, initial); dispatch(full, 2, 0);
    const carried = await read();
    for (let record = 0; record < RECORDS; record++) for (let q = 0; q < K; q++) {
      const expected = 0xfffffff0 + f.native[record * N * K + q];
      if (carried[record * K + q] !== (expected >>> 0) || carried[RECORDS * K + record * K + q] !== Math.floor(expected / 2 ** 32)) throw new Error("Wide gather carry mismatch");
    }
    const expectFault = async (bit: number, lists: Uint32Array[]) => {
      device.queue.writeBuffer(errors, 0, new Uint32Array(1));
      dispatch(lists, 0, 0, 0, 1);
      const result = await read(false);
      if (!(result[RECORDS * N] & bit)) throw new Error(`Malformed stream did not report fault ${bit}`);
    };
    // Truncate the first packet's first compressed column to its header only.
    const truncated = f.packed.slice();
    const lengthWord = f.descriptors[1] + D;
    truncated[lengthWord] &= ~(255 << 8);
    device.queue.writeBuffer(payload, 0, truncated);
    await expectFault(4, [new Uint32Array([1]), new Uint32Array(0)]);
    device.queue.writeBuffer(payload, 0, f.packed);
    const invalidModels = f.descriptors.slice(); invalidModels[9] = 0;
    device.queue.writeBuffer(records, 0, invalidModels);
    await expectFault(2, [new Uint32Array([1]), new Uint32Array(0)]);
    device.queue.writeBuffer(records, 0, f.descriptors);
    const badSparse = f.packed.slice();
    const sparseBase = f.descriptors[2];
    const valuesBase = sparseBase + 4 + f.packed[sparseBase + 1] + f.packed[sparseBase + 2] + f.packed[sparseBase + 3];
    badSparse[valuesBase] &= ~255;
    device.queue.writeBuffer(payload, 0, badSparse);
    await expectFault(32, [new Uint32Array(0), full[1]]);
    device.queue.writeBuffer(payload, 0, f.packed);
    device.queue.writeBuffer(errors, 0, new Uint32Array(1));
    return { dense_tans_and_uint16_literals_exact: true, sparse_events_exact: true, full_and_signed_delta_sums_exact: true, batched_record_patterns_exact: true, wide_gather_exact: true, malformed_streams_rejected: true, all_passed: true };
  } finally {
    owned.forEach(buffer => buffer.destroy());
    const error = await device.popErrorScope(); if (error) throw new Error(error.message);
  }
}
