/** Immutable programs shared by acquisitions on one device; never source buffers. */
type FirstWritableBinding = 5 | 6;
interface Program {
  module: Promise<GPUShaderModule>;
  pipelines: Map<string, Promise<GPUComputePipeline>>;
}
interface Programs {
  layouts: Map<FirstWritableBinding, {bind: GPUBindGroupLayout; pipeline: GPUPipelineLayout}>;
  shaders: Map<string, Program>;
}
const devices = new WeakMap<GPUDevice, Programs>();
function programs(device: GPUDevice): Programs {
  let value = devices.get(device);
  if (!value) {
    value = {layouts: new Map(), shaders: new Map()};
    devices.set(device, value);
  }
  return value;
}
function layout(device: GPUDevice, firstWritable: FirstWritableBinding) {
  const cache = programs(device).layouts;
  let value = cache.get(firstWritable);
  if (!value) {
    const bind = device.createBindGroupLayout({entries: Array.from({length: 9}, (_, binding) => ({
      binding, visibility: GPUShaderStage.COMPUTE,
      buffer: {type: binding === 8 ? 'uniform' : binding >= firstWritable ? 'storage' : 'read-only-storage'},
    })) as GPUBindGroupLayoutEntry[]});
    value = {bind, pipeline: device.createPipelineLayout({bindGroupLayouts: [bind]})};
    cache.set(firstWritable, value);
  }
  return value;
}
export function source112Layout(device: GPUDevice, firstWritable: FirstWritableBinding = 6): GPUBindGroupLayout {
  return layout(device, firstWritable).bind;
}
/** Coalesce compilation while preserving device and binding-layout identity. */
export function source112Pipeline(
  device: GPUDevice, code: string, entryPoint: string, firstWritable: FirstWritableBinding = 6,
): Promise<GPUComputePipeline> {
  const owner = programs(device), cache = owner.shaders;
  // A failed pipeline can contain an invalid layout after device allocation failure.
  const retryFresh = () => { if (devices.get(device) === owner) devices.delete(device); };
  let program = cache.get(code);
  if (!program) {
    const shader = device.createShaderModule({code});
    const module = shader.getCompilationInfo().then(info => {
      const errors = info.messages.filter(message => message.type === 'error');
      if (errors.length) throw Error(errors.map(message => `${message.lineNum ?? 0}:${message.linePos ?? 0} ${message.message}`).join('\n'));
      return shader;
    });
    program = {module, pipelines: new Map()};
    cache.set(code, program);
    void module.catch(retryFresh);
  }
  const key = `${firstWritable}:${entryPoint}`, entries = program.pipelines;
  let pipeline = entries.get(key);
  if (!pipeline) {
    const pipelineLayout = layout(device, firstWritable).pipeline;
    pipeline = program.module.then(module => device.createComputePipelineAsync({
      layout: pipelineLayout, compute: {module, entryPoint},
    }));
    entries.set(key, pipeline);
    void pipeline.catch(retryFresh);
  }
  return pipeline;
}
