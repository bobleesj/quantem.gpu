import assert from 'node:assert/strict';
import test from 'node:test';
import { Source112Ingress } from '../../src/quantem/gpu/detector/compute/webgpu/source112-ingress';
Object.assign(globalThis,{GPUBufferUsage:{MAP_WRITE:1,COPY_SRC:2,COPY_DST:4},GPUMapMode:{WRITE:1}});
const MiB=1024*1024;
function mock(failure='none'){
 const buffers:any[]=[],copies:number[]=[];let maps=0,fences=0,used=0,peak=0,lost!:(info:GPUDeviceLostInfo)=>void;
 const device:any={limits:{maxBufferSize:2**31},lost:new Promise(r=>{lost=r;}),createBuffer({size,usage,mappedAtCreation}:any){
  if(failure==='allocation'&&buffers.length===1)throw Error('allocation failure');
  assert.equal(usage,GPUBufferUsage.MAP_WRITE|GPUBufferUsage.COPY_SRC);assert(mappedAtCreation);
  let rejectMap:((error:Error)=>void)|undefined;const bytes=new Uint8Array(size),b={size,bytes,mapped:true,destroyed:0,getMappedRange(){assert(b.mapped);return bytes.buffer},unmap(){b.mapped=false},mapAsync(mode:number){assert.equal(mode,GPUMapMode.WRITE);maps++;if(failure==='pending')return new Promise<void>((_resolve,reject)=>{rejectMap=reject;});if(failure==='map'&&maps===2)return Promise.reject(Error('map failure'));if(failure==='lost'){lost({message:'injected loss'} as GPUDeviceLostInfo);return Promise.reject(Error('device lost'));}return Promise.resolve().then(()=>{if(b.destroyed)throw Error('map cancelled');b.mapped=true;});},destroy(){assert.equal(b.destroyed,0);b.destroyed++;rejectMap?.(Error('map cancelled'))}};buffers.push(b);return b;
 },createCommandEncoder(){const commands:Array<()=>void>=[];return{copyBufferToBuffer(a:any,ao:number,b:any,bo:number,n:number){assert(!a.mapped);assert(n<=32*MiB);commands.push(()=>{b.bytes.set(a.bytes.subarray(ao,ao+n),bo);copies.push(n);})},finish(){return commands}}},queue:{submit(commands:Array<Array<()=>void>>){for(const list of commands)for(const op of list)op()},async onSubmittedWorkDone(){fences++;if(failure==='fence')throw Error('fence failure')}}};
 const profile={payloadHostCopyMs:0,payloadMapPendingMs:0,payloadMapBlockedMs:0,payloadCopySubmitMs:0,payloadFenceWaitMs:0,payloadChunks:0};
 return{device:device as GPUDevice,profile,buffers,copies,owned:(n:number)=>{used=n;peak=Math.max(peak,n)},get used(){return used},get peak(){return peak},get fences(){return fences}};
}
test('mapped ring copies complete records and tails in order with bounded staging',async()=>{
 const m=mock(),ring=new Source112Ingress(m.device,m.profile,m.owned),input=new Uint8Array(33*MiB+12),destination={size:input.length+20,bytes:new Uint8Array(input.length+20)};
 for(let i=0;i<input.length;i++)input[i]=(Math.imul(i,37)+11)&255;
 const original=input.slice();await ring.write(destination as unknown as GPUBuffer,8,input.buffer);await ring.finish();
 assert.deepEqual(destination.bytes.subarray(8,8+input.length),original);assert.deepEqual(input,original);assert.deepEqual(m.copies,[32*MiB,MiB+12]);assert.equal(m.profile.payloadChunks,2);assert.equal(m.peak,128*MiB);
 await ring.write(destination as unknown as GPUBuffer,0,new Uint8Array([1,2,3,4]).buffer);assert.deepEqual(destination.bytes.subarray(0,4),new Uint8Array([1,2,3,4]));
 await ring.dispose();await ring.dispose();assert.equal(m.used,0);assert(m.buffers.every(b=>b.destroyed===1));assert(m.fences>=2);
});
for(const failure of ['map','fence','lost'])test(`mapped ingress drains and destroys staging after ${failure}`,async()=>{
 const m=mock(failure),ring=new Source112Ingress(m.device,m.profile,m.owned),destination={size:256*MiB,bytes:new Uint8Array(256*MiB)};
 await assert.rejects((async()=>{await ring.write(destination as unknown as GPUBuffer,0,new ArrayBuffer(256*MiB));await ring.finish()})(),failure==='map'?/map failure/:failure==='fence'?/fence failure/:/lost/);
 await ring.dispose().catch(()=>{});assert.equal(m.used,0);assert(m.buffers.every(b=>b.destroyed===1));
});
test('abort and invalid bounds never submit a partial new record',async()=>{
 const m=mock(),abort=new AbortController(),ring=new Source112Ingress(m.device,m.profile,m.owned,abort.signal),destination={size:8,bytes:new Uint8Array(8)};
 await assert.rejects(ring.write(destination as unknown as GPUBuffer,2,new ArrayBuffer(4)),/aligned/);abort.abort();await assert.rejects(ring.write(destination as unknown as GPUBuffer,0,new ArrayBuffer(4)),{name:'AbortError'});assert.equal(m.copies.length,0);await ring.dispose();assert.equal(m.used,0);
});
test('second staging allocation failure destroys the first immediately',()=>{
 const m=mock('allocation');assert.throws(()=>new Source112Ingress(m.device,m.profile,m.owned),/allocation failure/);assert.equal(m.used,0);assert.equal(m.buffers[0].destroyed,1);
});

test('disposal cancels and drains both pending maps without detached rejections',async()=>{
 const m=mock('pending'),ring=new Source112Ingress(m.device,m.profile,m.owned),destination={size:256*MiB,bytes:new Uint8Array(256*MiB)};
 const writing=ring.write(destination as unknown as GPUBuffer,0,new ArrayBuffer(256*MiB));
 const failure=assert.rejects(writing,/cancelled|closed/);
 while(m.copies.length<2)await Promise.resolve();
 await ring.dispose();await failure;assert.equal(m.used,0);assert(m.buffers.every(b=>b.destroyed===1));
});


test('record boundaries do not fence or wait for unrelated ring slots',async()=>{
 const m=mock('pending'),ring=new Source112Ingress(m.device,m.profile,m.owned),destination={size:8,bytes:new Uint8Array(8)};
 for(let i=0;i<4;i++)await ring.write(destination as unknown as GPUBuffer,0,new Uint8Array([i,2,3,4]).buffer);
 assert.equal(m.copies.length,4);assert.equal(m.fences,0);
 const pending=ring.write(destination as unknown as GPUBuffer,0,new Uint8Array([5,2,3,4]).buffer);
 const failure=assert.rejects(pending,/cancelled|closed/);
 await ring.dispose();await failure;assert.equal(m.used,0);assert(m.buffers.every(b=>b.destroyed===1));
});
