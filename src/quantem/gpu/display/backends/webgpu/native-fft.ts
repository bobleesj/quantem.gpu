/// <reference types="@webgpu/types" />
/** Native-grid 2D forward DFT via batched Bluestein convolutions.
 * Input has already been mean-centered/windowed by the caller. No image
 * resizing or zero-padding of the scientific frequency grid is performed.
 * The convolution padding is internal and removed before the next axis.
 */
const SHADER = /* wgsl */`
struct Params { n: u32, m: u32, batches: u32, axis: u32,
  width: u32, height: u32, stage: u32, inverse: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> work: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> source: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> chirp: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> dest: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read_write> magnitude: array<f32>;
@group(0) @binding(6) var<storage, read> twiddles: array<vec2<f32>>;
fn mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x);
}
fn conjugate(a: vec2<f32>) -> vec2<f32> { return vec2<f32>(a.x, -a.y); }
fn index(batch: u32, pos: u32) -> u32 {
  if (p.axis == 0u) { return batch*p.width+pos; }
  return pos*p.width+batch;
}
@compute @workgroup_size(256) fn prepare(@builtin(global_invocation_id) gid: vec3<u32>) {
  let j=gid.x; if (j>=p.m*(p.batches+1u)) { return; }
  let batch=j/p.m; let k=j%p.m;
  var value=vec2<f32>(0.0);
  if (batch<p.batches && k<p.n) { value=mul(source[index(batch,k)],chirp[k]); }
  if (batch==p.batches) {
    if (k<p.n) { value=conjugate(chirp[k]); }
    else if (k>p.m-p.n) { value=conjugate(chirp[p.m-k]); }
  }
  work[j]=value;
}
@compute @workgroup_size(256) fn reverse(@builtin(global_invocation_id) gid: vec3<u32>) {
  let j=gid.x; if (j>=p.m*(p.batches+1u)) { return; }
  let k=j%p.m; let rev=reverseBits(k)>>(32u-p.stage);
  if (k<rev) { let other=j-k+rev; let temp=work[j]; work[j]=work[other]; work[other]=temp; }
}
@compute @workgroup_size(256) fn butterfly(@builtin(global_invocation_id) gid: vec3<u32>) {
  let j=gid.x; if (j>=p.m*(p.batches+1u)/2u) { return; }
  let batch=j/(p.m/2u); let k=j%(p.m/2u);
  let half=1u<<p.stage; let pos=k%half; let i=batch*p.m+(k/half)*2u*half+pos;
  var twiddle=twiddles[pos*(p.m/(2u*half))];
  if(p.inverse==1u) { twiddle=conjugate(twiddle); }
  let a=work[i]; let b=mul(work[i+half],twiddle);
  work[i]=a+b; work[i+half]=a-b;
}
@compute @workgroup_size(256) fn product(@builtin(global_invocation_id) gid: vec3<u32>) {
  let j=gid.x; if (j>=p.m*p.batches) { return; }
  work[j]=mul(work[j],work[p.m*p.batches+j%p.m]);
}
@compute @workgroup_size(256) fn finish(@builtin(global_invocation_id) gid: vec3<u32>) {
  let j=gid.x; if (j>=p.n*p.batches) { return; }
  let batch=j/p.n; let k=j%p.n;
  dest[index(batch,k)]=mul(work[batch*p.m+k]/f32(p.m),chirp[k]);
}
@compute @workgroup_size(256) fn logMagnitude(@builtin(global_invocation_id) gid: vec3<u32>) {
  let j=gid.x; if (j>=p.width*p.height) { return; }
  let row=(j/p.width+(p.height+1u)/2u)%p.height;
  let col=(j%p.width+(p.width+1u)/2u)%p.width;
  magnitude[j]=log(1.0+length(source[row*p.width+col]));
}`;

export class NativeGridFFT {
  private pipelines: Record<string, GPUComputePipeline> = {};
  private tail: Promise<unknown> = Promise.resolve();

  constructor(private device: GPUDevice) {
    const module = device.createShaderModule({code: SHADER});
    for (const entryPoint of ["prepare", "reverse", "butterfly", "product", "finish", "logMagnitude"]) {
      this.pipelines[entryPoint] = device.createComputePipeline({layout: "auto", compute: {module, entryPoint}});
    }
  }

  /** Serialize scratch allocation; discard obsolete requests before GPU submission. */
  magnitude(data: Float32Array, rows: number, cols: number, cancelled: () => boolean = () => false): Promise<Float32Array | null> {
    const job = this.tail.then(() => cancelled() ? null : this.compute(data, rows, cols));
    this.tail = job.catch(() => undefined);
    return job;
  }

  private async compute(data: Float32Array, rows: number, cols: number): Promise<Float32Array> {
    if (!Number.isInteger(rows) || !Number.isInteger(cols) || rows<1 || cols<1 || data.length!==rows*cols) {
      throw new Error("Native FFT requires a nonempty rows × columns input.");
    }
    const d=this.device, buffers: GPUBuffer[]=[];
    const make=(size: number, usage=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST) => {
      if (size>d.limits.maxStorageBufferBindingSize || size>d.limits.maxBufferSize) {
        throw new Error("Native FFT exceeds the browser GPU buffer limit.");
      }
      const b=d.createBuffer({size,usage}); buffers.push(b); return b;
    };
    try {
      const complex=new Float32Array(data.length*2);
      for(let i=0;i<data.length;i++) complex[2*i]=data[i];
      let source=make(complex.byteLength); d.queue.writeBuffer(source,0,complex);
      const encoder=d.createCommandEncoder();
      const dispatch=(name: string, values: number[], count: number, bindings: [number,GPUBuffer][]) => {
        const params=make(32,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST);
        d.queue.writeBuffer(params,0,new Uint32Array(values));
        const pipeline=this.pipelines[name];
        const group=d.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:
          [{binding:0,resource:{buffer:params}},...bindings.map(([binding,buffer])=>({binding,resource:{buffer}}))]});
        const pass=encoder.beginComputePass(); pass.setPipeline(pipeline); pass.setBindGroup(0,group);
        if (Math.ceil(count/256)>d.limits.maxComputeWorkgroupsPerDimension) {
          throw new Error("Native FFT exceeds the browser GPU dispatch limit.");
        }
        pass.dispatchWorkgroups(Math.ceil(count/256)); pass.end();
      };
      for(let axis=0;axis<2;axis++) {
        const n=axis===0?cols:rows, batches=axis===0?rows:cols;
        const m=2**Math.ceil(Math.log2(Math.max(2,2*n-1))), levels=Math.log2(m);
        const work=make(m*(batches+1)*8), dest=make(complex.byteLength);
        // Double-precision host generation avoids large-angle f32 chirp error.
        const chirps=new Float32Array(2*n);
        for(let k=0;k<n;k++) {
          const angle=-Math.PI*((k*k)%(2*n))/n;
          chirps[2*k]=Math.cos(angle); chirps[2*k+1]=Math.sin(angle);
        }
        const chirp=make(chirps.byteLength); d.queue.writeBuffer(chirp,0,chirps);
        const twiddleValues=new Float32Array(m);
        for(let k=0;k<m/2;k++) {
          twiddleValues[2*k]=Math.cos(-2*Math.PI*k/m);
          twiddleValues[2*k+1]=Math.sin(-2*Math.PI*k/m);
        }
        const twiddles=make(twiddleValues.byteLength); d.queue.writeBuffer(twiddles,0,twiddleValues);
        const params=(stage=0,inverse=0)=>[n,m,batches,axis,cols,rows,stage,inverse];
        dispatch("prepare",params(),m*(batches+1),[[1,work],[2,source],[3,chirp]]);
        for(let inverse=0;inverse<2;inverse++) {
          dispatch("reverse",params(levels,inverse),m*(batches+1),[[1,work]]);
          for(let stage=0;stage<levels;stage++) {
            dispatch("butterfly",params(stage,inverse),m*(batches+1)/2,[[1,work],[6,twiddles]]);
          }
          if(inverse===0) dispatch("product",params(),m*batches,[[1,work]]);
        }
        dispatch("finish",params(),n*batches,[[1,work],[3,chirp],[4,dest]]);
        source=dest;
      }
      const output=make(data.byteLength,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC);
      dispatch("logMagnitude",[0,0,0,0,cols,rows,0,0],data.length,[[2,source],[5,output]]);
      const read=make(data.byteLength,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ);
      encoder.copyBufferToBuffer(output,0,read,0,data.byteLength);
      d.queue.submit([encoder.finish()]);
      await read.mapAsync(GPUMapMode.READ);
      const result=new Float32Array(read.getMappedRange().slice(0)); read.unmap();
      return result;
    } finally { for(const b of buffers) b.destroy(); }
  }
}
