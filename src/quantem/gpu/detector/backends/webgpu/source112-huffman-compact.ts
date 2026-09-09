/// <reference types="@webgpu/types" />
import { SOURCE112_HUFFMAN64_WGSL } from './source112-huffman64';
/** Compact version2 descriptors: starts=u16 offsets +u32 block bases; coldCP=3×21bits in2u32. */
export function compactHuffmanShader(nativeStep: 32 | 64) {
    if (nativeStep !== 32 && nativeStep !== 64)
        throw Error('Native step must be32 or64.');
    const segments = 512 / nativeStep, checkpointCount = segments - 1, slotBits = checkpointCount * 11 + 5, width = segments * 16;
    let s = SOURCE112_HUFFMAN64_WGSL;
    const change = (old: string, next: string) => { if (!s.includes(old))
        throw Error(`Canonical shader changed at ${old.slice(0, 60)}`); s = s.replace(old, next); };
    change('(m.bits<11u||m.bits>13u)||m.nativeStep!=64u||m.slotBits!=7u*m.bits+5u||m.version!=1u', `m.bits!=11u||m.nativeStep!=${nativeStep}u||m.slotBits!=${slotBits}u||m.version!=2u`);
    change('if(m.version!=1u)', 'if(m.version!=2u)');
    change('if(rec.pad1>n||stream>=n-rec.pad1){fault(1u);return vec2u(0u);}\n  let first=recordWords[rec.pad1+stream];', `let streams=p.columnCount*32u;
  if(stream>=streams||rec.pad1>n||streams/2u+streams/32u>n-rec.pad1){fault(1u);return vec2u(0u);}
  let blockBase=recordWords[rec.pad1+streams/2u+(stream>>5u)];
  let relative16=(recordWords[rec.pad1+(stream>>1u)]>>((stream&1u)*16u))&65535u;
  if(relative16>7936u||blockBase>0xffffffffu-relative16){fault(1u);return vec2u(0u);}
  let first=blockBase+relative16;`);
    const coldStart = s.indexOf('fn cold_checkpoint('), coldEnd = s.indexOf('struct HuffBits', coldStart);
    if (coldStart < 0 || coldEnd < coldStart)
        throw Error('Missing cold checkpoint function.');
    s = s.slice(0, coldStart) + `fn cold_checkpoint(m:HybridMeta,ordinal:u32,segment:u32)->u32 {
  let n=arrayLength(&recordWords);if(segment<1u||segment>3u||ordinal>0x7fffffffu){fault(64u);return 0u;}
  let word=ordinal*2u;if(m.coldCP>n||word>=n-m.coldCP||n-m.coldCP-word<2u){fault(64u);return 0u;}
  let bit=(segment-1u)*21u;let shift=bit&31u;let at=m.coldCP+word+(bit>>5u);
  var value=recordWords[at]>>shift;if(shift+21u>32u){value|=recordWords[at+1u]<<(32u-shift);}value&=0x1fffffu;
  let cursor=(value>>10u)+9u;if(cursor<10u||cursor>2056u){fault(64u);return 0u;}
  return 0x80000000u|(cursor<<10u)|(value&1023u);
}
` + s.slice(coldEnd);
    change('if(index>7u||', `if(index>${checkpointCount}u||`);
    change('index==7u', `index==${checkpointCount}u`);
    change('select(128u,64u,hot)', `select(128u,${nativeStep}u,hot)`);
    change('hot_field(m,ordinal,7u)', `hot_field(m,ordinal,${checkpointCount}u)`);
    change('if(segment<7u)', `if(segment<${checkpointCount}u)`);
    change('lane<8u', `lane<${segments}u`);
    change('@workgroup_size(128)\nfn decode_dense_huffman64', `@workgroup_size(${width})\nfn decode_dense_huffman64`);
    s = s.replace(/frame\+=128u/g, `frame+=${width}u`);
    change('wg.x*16u+t/8u', `wg.x*16u+t/${segments}u`);
    change('t&7u', `t&${segments - 1}u`);
    return s;
}
/** Build or independently verify compact descriptors from a borrowed checked Huffman64 generation.
 * Binding5 owns only candidate metadata; pad0 selects native step, pad1 selects verify. */
export const COMPACT_BUILD_WGSL = SOURCE112_HUFFMAN64_WGSL.replace('var<storage, read> selected: array<u32>;', 'var<storage, read_write> compact: array<atomic<u32>>;')
    .replace(/selected\[index\]/g, 'atomicLoad(&compact[index])') + `
fn candidate(at:u32)->u32{return atomicLoad(&compact[at]);}
fn exact_word(at:u32,value:u32){if(p.pad1==0u){atomicStore(&compact[at],value);}else if(candidate(at)!=value){fault(128u);}}
fn exact_field(base:u32,bit:u32,n:u32,value:u32){
 if(value>=(1u<<n)){fault(64u);return;}let at=base+(bit>>5u);let shift=bit&31u;
 if(at>=arrayLength(&compact)||(shift+n>32u&&at+1u>=arrayLength(&compact))){fault(64u);return;}
 if(p.pad1==0u){atomicOr(&compact[at],value<<shift);if(shift+n>32u){atomicOr(&compact[at+1u],value>>(32u-shift));}}
 else{var actual=candidate(at)>>shift;if(shift+n>32u){actual|=candidate(at+1u)<<(32u-shift);}if((actual&((1u<<n)-1u))!=value){fault(128u);}}
}
@compute @workgroup_size(64)
fn build_compact(@builtin(workgroup_id) wg:vec3u,@builtin(local_invocation_index)t:u32){
 let stream=wg.x*64u+t;let streams=p.columnCount*32u;if(stream>=streams||wg.z>=p.recordCount){return;}
 let rec=load_record(wg.z);let m=hybrid_meta(rec);if(m.version!=1u||m.bits!=11u){fault(64u);return;}
 let rank=stream%p.columnCount;let packet=stream/p.columnCount;let prefix=hybrid_prefix(m,rank);if(prefix==0xffffffffu){return;}
 let b=wg.z*12u;let cm=candidate(b+10u);let starts=candidate(b+11u);let range=hybrid_range(rec,stream);if(range.y<=range.x){return;}
 let blockFirst=recordWords[rec.pad1+(stream&~31u)];if(range.x<blockFirst||range.x-blockFirst>7936u){fault(1u);return;}
 exact_field(starts,stream*16u,16u,range.x-blockFirst);
 if((stream&31u)==0u){exact_word(starts+streams/2u+(stream>>5u),blockFirst);}
 if(packet==0u){exact_word(candidate(cm+1u)+rank,prefix);}
 let q=columns[rank];let address=rec.modelIdsBase+q;let model=(ids[address>>2u]>>((address&3u)*8u))&255u;
 if(model==255u){if(range.y-range.x!=256u){fault(1u);return;}atomicAdd(&output[1],1u);}
 else if(model==78u||model==79u){
  let step=p.pad0;let checkpointCount=512u/step-1u;let slot=checkpointCount*11u+5u;let ordinal=packet*m.hotColumns+prefix;
  let oldPadding=hot_field(m,ordinal,7u);let physical=(range.y-range.x)*32u;if(oldPadding>31u||oldPadding>=physical){fault(64u);return;}
  let limit=physical-oldPadding;var reader=open_huff(range.x,range.y,0u,limit);
  for(var pair=0u;pair<256u;pair++){decode_huff(&reader,model);if(reader.bad){fault(4u);return;}
   if(pair<255u&&(pair+1u)%32u==0u&&hot_field(m,ordinal,pair/32u)!=reader.cursor){fault(128u);return;}
   if(pair<255u&&(pair+1u)%(step/2u)==0u){exact_field(candidate(cm+2u),ordinal*slot+(pair/(step/2u))*11u,11u,reader.cursor);}}
  if(reader.cursor!=limit){fault(128u);return;}
  exact_field(candidate(cm+2u),ordinal*slot+checkpointCount*11u,5u,oldPadding);atomicAdd(&output[2],1u);
 }else{
  if(model>=81u){fault(2u);return;}let ordinal=packet*(p.columnCount-m.hotColumns)+rank-prefix;
  for(var segment=1u;segment<=3u;segment++){let cp=cold_checkpoint(m,ordinal,segment);let cursor=(cp>>10u)&8191u;
   if((cp&0x80000000u)==0u||cursor<10u||cursor>2056u||cursor>(range.y-range.x)*32u){fault(64u);return;}
   exact_field(candidate(cm+3u),ordinal*64u+(segment-1u)*21u,21u,((cursor-9u)<<10u)|(cp&1023u));}
  atomicAdd(&output[3],1u);
 }
 atomicAdd(&output[0],1u);
}`;
const D = 17466, S = D * 32;
export function compactGroupLayout(old: Uint32Array, records: number, step: 32 | 64) {
    if (old.length !== records * 20)
        throw Error('Expected complete12-word records and8-word metadata.');
    const headers = old.slice();
    let words = records * 20;
    for (let r = 0; r < records; r++) {
        const b = r * 12, m = old[b + 10];
        if (m < records * 12 || m + 8 > old.length || old[m + 4] !== 11 || old[m + 5] !== 64 || old[m + 6] !== 82 || old[m + 7] !== 1)
            throw Error('Expected production Huffman64/11-bit metadata.');
        const hot = old[m];
        if (hot > D)
            throw Error('Hot-column count exceeds geometry.');
        headers[b + 10] = m;
        headers[m + 1] = words;
        words += D;
        headers[b + 11] = words;
        words += S / 2 + S / 32;
        headers[m + 2] = words;
        words += hot * (512 / step - 1) * 11 + hot * 5;
        headers[m + 3] = words;
        words += (D - hot) * 32 * 2;
        headers[m + 5] = step;
        headers[m + 6] = (512 / step - 1) * 11 + 5;
        headers[m + 7] = 2;
    }
    return { headers, bytes: words * 4 };
}
