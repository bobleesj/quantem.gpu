import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import test from 'node:test';
import {source112AcquisitionFiles, Source112ResidentSet} from '../../src/quantem/gpu/detector/compute/webgpu/source112';
const digest=(raw:ArrayBuffer)=>createHash('sha256').update(new Uint8Array(raw)).digest('hex');
test('acquisition views preserve exact interleaved records without copying payload files',async()=>{
 const reads:Array<[number,number,number]>=[],bytes=[new Uint8Array(128),new Uint8Array(128)];
 bytes.forEach((a,shard)=>a.forEach((_,i)=>{a[i]=(i+shard*73)&255}));
 const files=bytes.map((a,shard)=>({name:`data-${shard}.bin`,size:a.length,slice(lo:number,hi:number){reads.push([shard,lo,hi]);return new Blob([a.subarray(lo,hi)])}} as File));
 const modelIds=new Uint8Array(8*36864);modelIds.fill(17,4*36864);
 const globals={model_ids:modelIds.buffer,valid:new Uint8Array(36864).fill(1).buffer};
 const manifest:any={format:'source112-tans1024-pair-v1',dtype:'<u2',shape:[2,512,512,192,192],
  globals:Object.fromEntries(Object.entries(globals).map(([name,raw])=>[name,{file:name+'.bin',dtype:'|u1',shape:name==='model_ids'?[8,36864]:[36864],nbytes:raw.byteLength,sha256:digest(raw)}])),
  layout:{files:files.map(f=>({name:f.name,nbytes:f.size})),chunks:Array.from({length:32},(_,chunk)=>({chunk,acquisition:Math.floor(chunk/16),first_scan:chunk%16*16384,scan_count:16384,shard:chunk%2,file_offset:Math.floor(chunk/2)*8,record_bytes:8,sha256:digest(bytes[chunk%2].slice(Math.floor(chunk/2)*8,Math.floor(chunk/2)*8+8).buffer),components:[]}))}};
 const frozen=structuredClone(manifest),views=await source112AcquisitionFiles(files,manifest,globals,1);
 assert.deepEqual(reads,[],'preparing views must not read payload bytes');assert.deepEqual(manifest,frozen);
 const local=JSON.parse(await views.find(f=>f.name==='manifest.json')!.text());assert.equal(local.shape[0],1);assert.deepEqual(local.globals.model_ids.shape,[4,36864]);
 for(const row of local.layout.chunks){const file=views.find(f=>f.name===local.layout.files[row.shard].name)!;const raw=await file.slice(row.file_offset,row.file_offset+row.record_bytes).arrayBuffer();assert.equal(digest(raw),row.sha256);assert.equal(row.acquisition,0);}
 assert.equal(reads.length,16);assert(reads.every(([,lo,hi])=>lo>=64&&hi<=128));
 const ids=new Uint8Array(await views.find(f=>f.name==='model_ids.bin')!.arrayBuffer());assert(ids.every(n=>n===17));
});
test('progressive products reject unloaded acquisitions and release only admitted owners',async()=>{
 const calls:number[]=[],child={integrate:()=>({added:1,removed:0,full:false}),imageViewsU32:()=>[{buffer:'child',divisor:1}],destroy:()=>calls.push(0)};
 const source:any=Object.assign(Object.create(Source112ResidentSet.prototype),{acquisitionCount:3,partitions:[child],disposed:false,preparing:false,owned:[],groups:[],profile:{},loadingAbort:new AbortController()});
 assert.equal(source.loadedAcquisitions,1);assert.equal(source.imageViewsU32([0],1)[0].buffer,'child');assert.throws(()=>source.imageViewsU32([1],1),/still loading/);
 source.destroy();source.destroy();assert.deepEqual(calls,[0]);assert.equal(source.loadingAbort.signal.aborted,true);
});

for (const failure of ['checksum', 'abort']) test(`background ${failure} preserves ready ownership and drains completion`,async()=>{
 const Class:any=Source112ResidentSet,original=Class.loadFiles,globalBytes=new Uint8Array(8*36864),payload=new File([new Uint8Array(128)],'payload.bin');
 const manifest:any={shape:[2,512,512,192,192],globals:{model_ids:{file:'models.bin',dtype:'|u1',shape:[8,36864],nbytes:globalBytes.length,sha256:digest(globalBytes.buffer)}},layout:{files:[{name:'payload.bin',nbytes:128}],chunks:Array.from({length:32},(_,chunk)=>({chunk,acquisition:Math.floor(chunk/16),shard:0,file_offset:chunk*4,record_bytes:4}))}};
 let destroyed=0,calls=0,parent:any;const published:number[]=[];
 const profile=new Proxy({representation:'huffman64',restartCacheLayout:'huffman64-compact'} as Record<string,unknown>,{get:(target,key:string)=>target[key]??0});
 const child={profile,representation:'huffman64',destroy(){destroyed++}};
 Class.loadFiles=async (_device:unknown,_files:unknown,_status:unknown,signal:AbortSignal)=>{
  if(calls++===0)return child;
  if(failure==='checksum')throw Error('Preserved SHA-256 mismatch');
  signal.throwIfAborted();
  await new Promise<void>((_resolve,reject)=>signal.addEventListener('abort',()=>reject(signal.reason),{once:true}));
 };
 try{
  parent=await Class.loadAcquisitions({},[payload],manifest,{model_ids:globalBytes.buffer},()=>{},undefined,
   {progressive:true,onProgress:(source:any)=>published.push(source.loadedAcquisitions)},performance.now(),new Uint32Array());
  assert.equal(parent.loadedAcquisitions,1);assert.equal(destroyed,0);assert.deepEqual(published,[1]);
  const rejected=assert.rejects(parent.completion,failure==='checksum'?/SHA-256/:{name:'AbortError'});
  if(failure==='abort')parent.destroy();
  await rejected;
  if(failure==='checksum'){assert.equal(parent.loadedAcquisitions,1);assert.equal(destroyed,0);}
  parent.destroy();assert.equal(destroyed,1);assert.deepEqual(published,[1]);
 }finally{parent?.destroy();Class.loadFiles=original;}
});
