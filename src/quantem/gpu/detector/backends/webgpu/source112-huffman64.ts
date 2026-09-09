import { SOURCE112_WGSL } from './source112-kernels';

/** Explicit hybrid generation. The unchanged nine bindings contain eight storage
 * buffers and one uniform. All scientific source buffers are borrowed read-only.
 * Record.pad0 addresses meta8; pad1 addresses absolute starts for all dense streams.
 * meta8 = [hotColumns, rankMap, hotCP, coldCP, bits, 64, 7*bits+5, 1];
 * bits is11..13 (the admitted current source uses11).
 * Hot rank maps store the hot prefix BEFORE each column. Cold slots include
 * literal holes. The dense length bytes must describe the repacked payload.
 */
const HYBRID_DENSE = /* wgsl */ `
struct HybridMeta {
  hotColumns:u32, rankMap:u32, hotCP:u32, coldCP:u32,
  bits:u32, nativeStep:u32, slotBits:u32, version:u32,
}
fn load_record(index:u32)->Record {
  if(index>=arrayLength(&recordWords)/12u){fault(64u);return Record();}
  let b=index*12u;
  return Record(recordWords[b],recordWords[b+1u],recordWords[b+2u],recordWords[b+3u],
    recordWords[b+4u],recordWords[b+5u],recordWords[b+6u],recordWords[b+7u],
    recordWords[b+8u],recordWords[b+9u],recordWords[b+10u],recordWords[b+11u]);
}
fn hybrid_meta(rec:Record)->HybridMeta {
  let n=arrayLength(&recordWords);let b=rec.pad0;
  if(b>n||n-b<8u){fault(64u);return HybridMeta();}
  let m=HybridMeta(recordWords[b],recordWords[b+1u],recordWords[b+2u],recordWords[b+3u],
    recordWords[b+4u],recordWords[b+5u],recordWords[b+6u],recordWords[b+7u]);
  if(m.hotColumns>p.columnCount||(m.bits<11u||m.bits>13u)||m.nativeStep!=64u||m.slotBits!=7u*m.bits+5u||m.version!=1u){fault(64u);return HybridMeta();}
  return m;
}
fn hybrid_range(rec:Record,stream:u32)->vec2u {
  let n=arrayLength(&recordWords);
  if(rec.pad1>n||stream>=n-rec.pad1){fault(1u);return vec2u(0u);}
  let first=recordWords[rec.pad1+stream];
  let lengthIndex=rec.denseOffsetsBase+p.lengthWordBase+(stream>>2u);
  if(lengthIndex>=arrayLength(&packed)){fault(1u);return vec2u(0u);}
  let words=1u+((packed[lengthIndex]>>((stream&3u)*8u))&255u);
  if(first<rec.denseBase){fault(1u);return vec2u(0u);}
  let relative=first-rec.denseBase;
  if(relative>=rec.denseWords||words>rec.denseWords-relative||first>=arrayLength(&packed)||words>arrayLength(&packed)-first){fault(1u);return vec2u(0u);}
  return vec2u(first,first+words);
}
fn hybrid_prefix(m:HybridMeta,rank:u32)->u32 {
  let n=arrayLength(&recordWords);
  if(m.rankMap>n||rank>=n-m.rankMap){fault(64u);return 0xffffffffu;}
  let prefix=recordWords[m.rankMap+rank];
  if(prefix>rank||prefix>m.hotColumns){fault(64u);return 0xffffffffu;}
  return prefix;
}
fn hot_field(m:HybridMeta,ordinal:u32,index:u32)->u32 {
  // Group admission also checks this bound; retain it before u32 bit arithmetic.
  if(index>7u||ordinal>(0xffffffffu-(m.slotBits-1u))/m.slotBits){fault(64u);return 0xffffffffu;}
  let bit=ordinal*m.slotBits+index*m.bits;let word=bit>>5u;let shift=bit&31u;
  let bits=select(m.bits,5u,index==7u);let n=arrayLength(&recordWords);
  if(m.hotCP>n||word>=n-m.hotCP){fault(64u);return 0xffffffffu;}
  var value=recordWords[m.hotCP+word]>>shift;
  if(shift+bits>32u){if(word+1u>=n-m.hotCP){fault(64u);return 0xffffffffu;}value|=recordWords[m.hotCP+word+1u]<<(32u-shift);}
  return value&((1u<<bits)-1u);
}
fn cold_checkpoint(m:HybridMeta,ordinal:u32,segment:u32)->u32 {
  let n=arrayLength(&recordWords);
  if(segment<1u||segment>3u||ordinal>(0xffffffffu-2u)/3u){fault(64u);return 0u;}
  let word=ordinal*3u+segment-1u;
  if(m.coldCP>n||word>=n-m.coldCP){fault(64u);return 0u;}
  return recordWords[m.coldCP+word];
}
struct HuffBits {word:u32,end:u32,lo:u32,hi:u32,available:u32,cursor:u32,limit:u32,bad:bool}
fn open_huff(first:u32,end:u32,cursor:u32,limit:u32)->HuffBits {
  var r=HuffBits(first+(cursor>>5u),end,0u,0u,0u,cursor,limit,false);
  if(cursor>limit||limit>(end-first)*32u){r.bad=true;return r;}
  if(r.word<end){r.lo=packed[r.word]>>(cursor&31u);r.available=32u-(cursor&31u);r.word++;}
  return r;
}
fn fill_huff(r:ptr<function,HuffBits>,bits:u32){
  if((*r).available<bits&&(*r).word<(*r).end){
    let value=packed[(*r).word];(*r).word++;
    (*r).lo|=value<<(*r).available;
    (*r).hi=select(0u,value>>(32u-(*r).available),(*r).available!=0u);(*r).available+=32u;
  }
}
fn consume_huff(r:ptr<function,HuffBits>,bits:u32)->u32 {
  if((*r).bad||bits>(*r).available||(*r).cursor>(*r).limit||bits>(*r).limit-(*r).cursor){(*r).bad=true;return 0u;}
  let value=(*r).lo&((1u<<bits)-1u);
  (*r).lo=((*r).lo>>bits)|((*r).hi<<(32u-bits));(*r).hi>>=bits;
  (*r).available-=bits;(*r).cursor+=bits;return value;
}
fn decode_huff(r:ptr<function,HuffBits>,model:u32)->u32 {
  fill_huff(r,10u);
  let at=82944u+(model-78u)*1024u+((*r).lo&1023u);
  if(at>=arrayLength(&decoding)){(*r).bad=true;return 0u;}
  let code=decoding[at];let bits=(code>>12u)&15u;
  if(bits<1u||bits>10u){(*r).bad=true;return 0u;}
  consume_huff(r,bits);var pair=code&4095u;
  if(pair==4095u){fill_huff(r,12u);pair=consume_huff(r,12u);}
  return pair;
}
// One complete segment for sums, or the prefix through one gatherTarget pair for gather.
fn hybrid_segment(rec:Record,packet:u32,rank:u32,lane:u32,subtract:bool,relativeRecord:u32,gatherTarget:bool){
  let q=columns[rank];if(q>=p.detectorPixels){fault(1u);return;}
  let idIndex=rec.modelIdsBase+q;
  if((idIndex>>2u)>=arrayLength(&ids)){fault(2u);return;}
  let model=(ids[idIndex>>2u]>>((idIndex&3u)*8u))&255u;
  let hot=model==78u||model==79u;
  if(!gatherTarget&&!hot&&lane>=4u){return;}
  let range=hybrid_range(rec,packet*p.columnCount+rank);if(range.y<=range.x){return;}
  let frame=p.patternFrame&511u;let nativeStep=select(128u,64u,hot);
  let segment=select(lane,frame/nativeStep,gatherTarget);let begin=segment*(nativeStep/2u);
  let stop=select(begin+nativeStep/2u,(frame>>1u)+1u,gatherTarget);
  if(model==255u){
    if(range.y-range.x!=256u){fault(1u);return;}
    let first=select(begin,frame>>1u,gatherTarget);
    for(var i=first;i<stop;i++){let word=packed[range.x+i];emit(q,i*2u,word&65535u,subtract,relativeRecord);emit(q,i*2u+1u,word>>16u,subtract,relativeRecord);}return;
  }
  if(model>=rec.modelCount||model>=81u){fault(2u);return;}
  let m=hybrid_meta(rec);if(m.version!=1u){return;}
  let prefix=hybrid_prefix(m,rank);if(prefix==0xffffffffu){return;}
  if(hot){
    if(prefix>=m.hotColumns){fault(64u);return;}
    let ordinal=packet*m.hotColumns+prefix;let padding=hot_field(m,ordinal,7u);let physical=(range.y-range.x)*32u;
    if(padding>31u||padding>=physical){fault(64u);return;}
    let limit=physical-padding;var cursor=0u;if(segment>0u){cursor=hot_field(m,ordinal,segment-1u);}
    var expectedEnd=limit;if(segment<7u){expectedEnd=hot_field(m,ordinal,segment);}
    if(cursor>=expectedEnd||expectedEnd>limit){fault(64u);return;}
    var r=open_huff(range.x,range.y,cursor,limit);
    for(var i=begin;i<stop;i++){let pair=decode_huff(&r,model);if(r.bad||r.cursor>expectedEnd){fault(4u);return;}emit(q,i*2u,pair&63u,subtract,relativeRecord);emit(q,i*2u+1u,pair>>6u,subtract,relativeRecord);}
    if((!gatherTarget||stop==begin+nativeStep/2u)&&r.cursor!=expectedEnd){fault(64u);}return;
  }
  let coldCount=p.columnCount-m.hotColumns;let coldRank=rank-prefix;
  if(coldRank>=coldCount){fault(64u);return;}
  let first=range.x;let end=range.y;let header=packed[first];var state=header&1023u;
  var bits=Bits(first+1u,end,header>>10u,22u,false);
  if(segment>0u){let checkpoint=cold_checkpoint(m,packet*coldCount+coldRank,segment);let cursor=(checkpoint>>10u)&8191u;
    if((checkpoint&0x80000000u)==0u||cursor<10u||cursor>(end-first)*32u){fault(64u);return;}
    state=checkpoint&1023u;let word=first+(cursor>>5u);let shift=cursor&31u;bits=Bits(word,end,0u,0u,false);
    if(word<end){bits=Bits(word+1u,end,packed[word]>>shift,32u-shift,false);}
  }
  for(var i=begin;i<stop;i++){
    if(state>=1024u){fault(2u);return;}let code=decoding[rec.decodingBase+model*1024u+state];
    state=(code>>16u)+take_bits(&bits,(code>>12u)&15u);var pair=code&4095u;if(pair==4095u){pair=take_bits(&bits,12u);}
    if(bits.bad){fault(4u);return;}emit(q,i*2u,pair&63u,subtract,relativeRecord);emit(q,i*2u+1u,pair>>6u,subtract,relativeRecord);
  }
  if(state>=1024u){fault(2u);}
}
fn dense_column(rec:Record,packet:u32,rank:u32,subtract:bool,relativeRecord:u32){
  if(p.mode!=0u){hybrid_segment(rec,packet,rank,0u,subtract,relativeRecord,true);}
  else{for(var lane=0u;lane<8u;lane++){hybrid_segment(rec,packet,rank,lane,subtract,relativeRecord,false);}}
}
@compute @workgroup_size(128)
fn decode_dense_huffman64(@builtin(workgroup_id) wg:vec3u,@builtin(local_invocation_index)t:u32){
  for(var frame=t;frame<512u;frame+=128u){atomicStore(&sums[sum_index(frame)],0u);}workgroupBarrier();
  if(wg.z<p.recordCount&&wg.y<32u){
    let rec=load_record(p.recordFirst+wg.z);let index=wg.x*16u+t/8u;
    if(index<p.selectedCount){let entry=selected[index];let rank=entry&0xffffffu;
      if(rank<p.columnCount){hybrid_segment(rec,wg.y,rank,t&7u,(entry&0x1000000u)!=0u,wg.z,false);}else{fault(1u);}}
    workgroupBarrier();
    for(var frame=t;frame<512u;frame+=128u){let value=atomicLoad(&sums[sum_index(frame)]);if(value!=0u){atomicAdd(&output[rec.outBase+wg.y*512u+frame],value);}}
  }
}
`;

const start = SOURCE112_WGSL.indexOf('fn dense_column(');
const end = SOURCE112_WGSL.indexOf('fn sparse_column(');
if (start < 0 || end <= start) throw Error('Expected the canonical source112 dense/sparse boundary.');
const base = (SOURCE112_WGSL.slice(0, start) + SOURCE112_WGSL.slice(end))
  .replace('var<storage, read> records: array<Record>;', 'var<storage, read> recordWords: array<u32>;')
  .split('records[p.recordFirst + wg.z]').join('load_record(p.recordFirst + wg.z)')
  .replace('var<workgroup> sums: array<atomic<u32>, 512>;', 'var<workgroup> sums: array<atomic<u32>, 528>;\nfn sum_index(frame:u32)->u32{return frame+(frame>>5u);}')
  .split('&sums[frame]').join('&sums[sum_index(frame)]');

/** decode_dense: existing mode1 pattern/mode2 wide gather, WG64 one column/lane.
 * decode_dense_huffman64: sum/delta, WG128 sixteen columns/group.
 * decode_sparse: unchanged scientific decoder, using the new record headers.
 */
export const SOURCE112_HUFFMAN64_WGSL = base + HYBRID_DENSE;
export const SOURCE112_HUFFMAN64_SUM_WGSL = SOURCE112_HUFFMAN64_WGSL.replace(/\bp\.mode\b/g, '0u');
