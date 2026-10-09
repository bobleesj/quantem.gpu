/** A CPU stand-in for a WebGPU device: real mapped memory and recorded allocations, no shader execution. */

Object.assign(globalThis, {
  GPUBufferUsage: { MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, INDEX: 16, VERTEX: 32, UNIFORM: 64, STORAGE: 128, INDIRECT: 256, QUERY_RESOLVE: 512 },
  GPUMapMode: { READ: 1, WRITE: 2 },
  GPUShaderStage: { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 },
});

export class FakeBuffer {
  readonly size: number;
  readonly usage: number;
  readonly mappedAtCreation: boolean;
  bytes: ArrayBuffer;
  mapped: boolean;
  destroyed = false;
  constructor(descriptor: GPUBufferDescriptor) {
    this.size = descriptor.size;
    this.usage = descriptor.usage;
    this.mappedAtCreation = Boolean(descriptor.mappedAtCreation);
    this.mapped = this.mappedAtCreation;
    this.bytes = new ArrayBuffer(descriptor.size);
  }
  getMappedRange() { return this.bytes; }
  unmap() { this.mapped = false; }
  async mapAsync() { this.mapped = true; }
  destroy() { this.destroyed = true; }
}

/** `faults` is what the decoder's exact-termination readback reports. */
export function fakeDevice(limits = { maxBufferSize: 2 ** 32, maxStorageBufferBindingSize: 2 ** 32 }, faults = 0) {
  const buffers: FakeBuffer[] = [];
  const bindGroups: GPUBindGroupDescriptor[] = [];
  let submissions = 0;
  const pass = { setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() {}, end() {} };
  const device = {
    limits: { ...limits, maxComputeWorkgroupsPerDimension: 65535, minStorageBufferOffsetAlignment: 256 },
    createBuffer(descriptor: GPUBufferDescriptor) {
      const buffer = new FakeBuffer(descriptor);
      // The decoder's termination check reads one 16-byte word block back.
      if (descriptor.usage & GPUBufferUsage.MAP_READ && descriptor.size === 16) {
        buffer.mapAsync = async () => { buffer.bytes = new Uint32Array([faults, 0, 0, 0]).buffer; };
      }
      buffers.push(buffer);
      return buffer;
    },
    createShaderModule: () => ({}),
    createBindGroupLayout: () => ({}),
    createPipelineLayout: () => ({}),
    createComputePipeline: () => ({ getBindGroupLayout: () => ({}) }),
    createBindGroup(descriptor: GPUBindGroupDescriptor) { bindGroups.push(descriptor); return {}; },
    // Copies run at once: the tests only read results after the matching submit.
    createCommandEncoder: () => ({ beginComputePass: () => pass, finish: () => ({}),
      copyBufferToBuffer(source: FakeBuffer, sourceOffset: number, target: FakeBuffer, targetOffset: number, size: number) {
        new Uint8Array(target.bytes, targetOffset, size).set(new Uint8Array(source.bytes, sourceOffset, size));
      } }),
    queue: { writeBuffer() {}, submit() { submissions++; }, onSubmittedWorkDone: async () => {} },
  };
  return { device: device as unknown as GPUDevice, buffers, bindGroups, submissions: () => submissions };
}
