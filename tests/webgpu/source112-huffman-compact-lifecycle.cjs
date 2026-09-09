const assert = require('node:assert/strict'), { test: nodeTest } = require('node:test'), { Source112ResidentSet } = require('../../src/quantem/gpu/detector/compute/webgpu/source112.ts');
global.GPUBufferUsage = { STORAGE: 1, COPY_SRC: 2, COPY_DST: 4, UNIFORM: 8, MAP_READ: 16 };
global.GPUShaderStage = { COMPUTE: 1 };
global.GPUMapMode = { READ: 1 };
async function test(mode) {
    let scopes = 0, guarded = 0, dispatches = 0, validationInjected = false;
    const all = [];
    let source;
    const mk = (size, data, borrowed = false) => { const b = { size, data, borrowed, destroyed: 0, async mapAsync() { guard(); }, getMappedRange() { return this.data.buffer; }, unmap() { }, destroy() { assert.equal(this.destroyed, 0, 'double destroy'); this.destroyed++; } }; all.push(b); return b; };
    const guard = () => { if (source.preparing) {
        assert.throws(() => source.check(), source.disposed ? /disposed/ : /preparing/);
        guarded++;
    } };
    const device = { limits: { maxBufferSize: 4294967292, maxStorageBufferBindingSize: 2147483644 }, lost: new Promise(() => { }), pushErrorScope() { scopes++; }, async popErrorScope() { guard(); scopes--; if (mode === 'gpu-validation' && dispatches && !validationInjected) {
            validationInjected = true;
            return { message: 'Injected command validation' };
        } return null; }, createBuffer({ size, usage }) { const b = mk(size); b.usage = usage; return b; }, createShaderModule({ code }) { return { async getCompilationInfo() { guard(); return { messages: mode === 'compile' ? [{ type: 'error', message: 'compile-failure' }] : [] }; } }; }, createBindGroupLayout(x) { return x; }, createPipelineLayout(x) { return x; }, async createComputePipelineAsync(x) { guard(); return x; }, createBindGroup({ entries }) { return { buffers: entries.map(e => e.resource.buffer) }; }, queue: { writeBuffer(b, o, data, off = 0, length) { assert(!b.borrowed); assert.equal(o, 0); b.data = new Uint32Array((data instanceof ArrayBuffer ? new Uint8Array(data, off, length) : new Uint8Array(data.buffer, data.byteOffset, data.byteLength)).slice().buffer); }, submit(commands) { for (const ops of commands)
                for (const op of ops)
                    op(); }, async onSubmittedWorkDone() { guard(); } }, createCommandEncoder() { const ops = []; return { clearBuffer(b) { assert(b.usage & GPUBufferUsage.COPY_DST, 'clearBuffer requires COPY_DST'); ops.push(() => b.data = new Uint32Array(b.size / 4)); }, beginComputePass() { let bind; return { setPipeline() { }, setBindGroup(i, b) { bind = b; }, dispatchWorkgroups() { ops.push(() => { dispatches++; bind.buffers[6].data = new Uint32Array([558912, 128, 278400, 280384]); if (mode === 'late-builder' && dispatches === 4)
                    bind.buffers[7].data = new Uint32Array([128]); }); }, end() { } }; }, copyBufferToBuffer(a, ao, b, bo, n) { ops.push(() => { assert.equal(bo, 0); b.data = a.data.slice(ao / 4, (ao + n) / 4); }); }, finish() { return ops; } }; } };
    const header = new Uint32Array(20);
    header[10] = 12;
    header.set([8700, 20, 100, 200, 11, 64, 82, 1], 12);
    source = { device, storageRepresentation: 'huffman64', disposed: false, preparing: false, restartPreparing: false, restartEnabled: true, restartProfile: { segmentValues: 64 }, pipelines: [{}, {}], previousMask: new Uint32Array([123]), groups: Array.from({ length: 2 }, (_, i) => ({ payload: mk(1024, new Uint32Array(256), true), descriptors: mk(8800000, header.slice(), true), records: [{}], params: [mk(64), mk(64)], bind: [{}, {}] })), globals: Array.from({ length: 4 }, () => mk(4)), selected: [mk(4), mk(4)], output: mk(4), errors: mk(4), layout: {}, check() { if (this.disposed)
            throw Error('disposed'); if (this.preparing)
            throw Error('preparing'); }, destroy() { if (this.disposed)
            return; this.disposed = true; for (const b of this.owned)
            b.destroy(); this.owned = []; this.groups = []; } };
    source.owned = all.slice();
    source.profile = { restartPreparationMs: 10, residentBytes: all.reduce((n, b) => n + b.size, 0), peakResidentBytes: 0, restartCacheBytes: 17600000 };
    const old = source.groups.map(g => g.descriptors), payloads = source.groups.map(g => g.payload);
    const controller = new AbortController();
    let hooks = 0;
    const options = { signal: controller.signal, beforeGroup: () => { guard(); hooks++; if (mode === 'abort-before' && hooks === 1)
            controller.abort(); if (mode === 'abort-partial' && hooks === 2)
            controller.abort(); if (mode === 'dispose-after-await' && hooks === 2)
            source.destroy(); } };
    if (mode === 'success') {
        await Source112ResidentSet.prototype.prepareCompactHuffman.call(source, options.beforeGroup, options.signal);
        assert.equal(source.profile.restartCacheLayout, 'huffman64-compact');
        assert(source.profile.restartPreparationMs >= 10);
        assert.equal(source.restartProfile.segmentValues, 64);
        assert.equal(source.previousMask, null);
        assert.equal(source.profile.residentBytes, source.owned.reduce((n, b) => n + b.size, 0));
        assert(old.every(b => b.destroyed === 1));
        assert(payloads.every(b => !b.destroyed));
        assert.equal(source.storageRepresentation, 'huffman64');
        await assert.rejects(Source112ResidentSet.prototype.prepareCompactHuffman.call(source, options.beforeGroup, options.signal), /unmodified/);
        source.destroy();
    }
    else {
        await assert.rejects(Source112ResidentSet.prototype.prepareCompactHuffman.call(source, options.beforeGroup, options.signal), mode === 'gpu-validation' ? /Injected command validation/ : undefined);
        if (['abort-partial', 'dispose-after-await', 'late-builder'].includes(mode))
            assert(source.disposed);
        else {
            assert(!source.disposed);
            assert(old.every(b => !b.destroyed));
            source.check();
            source.destroy();
        }
    }
    assert.equal(scopes, 0);
    assert(all.every(b => b.destroyed === 1));
    return { mode, guarded, dispatches, allOwnedDestroyedOnce: true };
}
for (const mode of ['success', 'compile', 'gpu-validation', 'abort-before', 'abort-partial', 'dispose-after-await', 'late-builder'])
    nodeTest(`compact Huffman cache lifecycle: ${mode}`, () => test(mode));

nodeTest('compact layout preserves global output offsets and literal checkpoint slots', () => {
    const {compactGroupLayout} = require('../../src/quantem/gpu/detector/compute/webgpu/source112-huffman-compact.ts');
    const old = new Uint32Array(40), D = 17466, streams = D * 32;
    for (let r = 0; r < 2; r++) {
        old[r * 12 + 5] = 524288 + r * 16384;
        old[r * 12 + 10] = 24 + r * 8;
        old.set([r ? D : 0, 40, 50, 60, 11, 64, 82, 1], 24 + r * 8);
    }
    const plan = compactGroupLayout(old, 2, 32);
    assert.equal(plan.headers[5], old[5]); assert.equal(plan.headers[17], old[17]);
    assert.equal(plan.headers[31], 2); assert.equal(plan.headers[39], 2);
    const expected = 40 * 4 + 2 * (D * 4 + streams * 2 + streams / 32 * 4) + streams * 8 + D * 170 * 4;
    assert.equal(plan.bytes, expected);
    const malformed = old.slice(); malformed[28] = 12;
    assert.throws(() => compactGroupLayout(malformed, 2, 32), /production Huffman64/);
});
