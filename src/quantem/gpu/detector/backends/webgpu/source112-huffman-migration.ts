import { MIGRATION_READERS } from './source112-huffman-readers';
/** Two-phase transcode. Eight storage bindings and one dispatch uniform.
 * Original payload and descriptors are immutable until all per-group gates pass. */
export const HUFFMAN_MIGRATION_WGSL = /* wgsl */ `
struct Params { recordsN:u32,columns:u32,streams:u32,capacity:u32,
 prefixBase:u32,sparseOffsetWords:u32,phase:u32,unused:u32 }
@group(0) @binding(0) var<storage,read> original:array<u32>;
@group(0) @binding(1) var<storage,read> controls:array<u32>;
@group(0) @binding(2) var<storage,read> columns:array<u32>;
@group(0) @binding(3) var<storage,read> models:array<u32>;
@group(0) @binding(4) var<storage,read> tables:array<u32>;
@group(0) @binding(5) var<storage,read_write> metadata:array<atomic<u32>>;
@group(0) @binding(6) var<storage,read_write> encoded:array<u32>;
@group(0) @binding(7) var<storage,read_write> result:array<atomic<u32>>;
@group(0) @binding(8) var<uniform> p:Params;
fn fault(code:u32){atomicOr(&result[0],code);}
fn rd(i:u32)->u32{return atomicLoad(&metadata[i]);}
fn wr(i:u32,v:u32){atomicStore(&metadata[i],v);}
fn hot(model:u32)->bool{return model==78u||model==79u;}
fn model_id(r:u32,rank:u32)->u32{let at=controls[r*12u+4u]+columns[rank];return(models[at>>2u]>>((at&3u)*8u))&255u;}
fn source_range(r:u32,s:u32)->vec2u{
 let b=r*12u;let o=controls[b+1u];var first=original[o+(s>>5u)];
 for(var j=s&~31u;j<s;j++){first+=1u+((original[o+p.columns+(j>>2u)]>>((j&3u)*8u))&255u);}
 let n=1u+((original[o+p.columns+(s>>2u)]>>((s&3u)*8u))&255u);
 if(first>controls[b+6u]||n>controls[b+6u]-first){fault(1u);return vec2u(0u);}
 first+=controls[b];if(first>arrayLength(&original)||n>arrayLength(&original)-first){fault(1u);return vec2u(0u);}return vec2u(first,first+n);
}
fn destination_range(r:u32,s:u32)->vec2u{let b=r*12u;let a=rd(b+11u);let first=rd(a+s);var end=rd(b)+rd(b+6u);if(s+1u<p.streams){end=rd(a+s+1u);}if(first>=end||end>p.capacity){fault(16u);return vec2u(0u);}return vec2u(first,end);}
fn encode_code(model:u32,value:u32)->u32{return tables[82944u+(model-78u)*4096u+value];}
` + MIGRATION_READERS + /* wgsl */ `
@compute @workgroup_size(64)
fn copy_words(@builtin(global_invocation_id) id:vec3u){if(id.x<p.capacity){encoded[id.x]=original[id.x];}}
@compute @workgroup_size(1)
fn rank_columns(@builtin(workgroup_id) wg:vec3u){let r=wg.x;let m=rd(r*12u+10u);var count=0u;
 for(var q=0u;q<p.columns;q++){wr(rd(m+1u)+q,count);if(hot(model_id(r,q))){count++;}}
 wr(m,count);atomicStore(&result[16u+r*9u],count*32u);
}
@compute @workgroup_size(64)
fn plan(@builtin(global_invocation_id) id:vec3u,@builtin(num_workgroups) grid:vec3u){
 let s=id.x+id.y*grid.x*64u;let r=id.z;if(s>=p.streams){return;}let bounds=source_range(r,s);if(bounds.y<=bounds.x){return;}let model=model_id(r,s%p.columns);var words=bounds.y-bounds.x;
 if(hot(model)){var reader=original_reader(bounds.x,bounds.y,10u);var state=original[bounds.x]&1023u;var bits=0u;
  for(var pair=0u;pair<256u;pair++){let value=ans_pair(&reader,&state,model);let code=encode_code(model,value);let n=(code>>10u)&15u;if(n==0u||n>10u){fault(2u);return;}bits+=n+select(0u,12u,(code&16384u)!=0u);}
  if(reader.bad||state>=1024u){fault(4u);return;}words=(bits+31u)/32u;
 }else{if(model!=255u&&model>=81u){fault(2u);return;}if(model==255u){if(words!=256u){fault(1u);return;}atomicAdd(&result[24u+r*9u],1u);}atomicAdd(&result[18u+r*9u],words);}
 if(words==0u||words>256u){fault(16u);return;}wr(rd(r*12u+11u)+s,words);
}
@compute @workgroup_size(64)
fn prefix_blocks(@builtin(global_invocation_id) id:vec3u){let block=id.x;let r=id.z;if(block>=p.columns){return;}let a=rd(r*12u+11u);var sum=0u;for(var j=0u;j<32u;j++){let k=block*32u+j;let n=rd(a+k);wr(a+k,sum);sum+=n;}wr(p.prefixBase+r*p.columns+block,sum);}
@compute @workgroup_size(1)
fn prefix_records(@builtin(workgroup_id) wg:vec3u){let r=wg.x;var sum=0u;for(var b=0u;b<p.columns;b++){let i=p.prefixBase+r*p.columns+b;let n=rd(i);wr(i,sum);sum+=n;}wr(r*12u+6u,sum);}
@compute @workgroup_size(64)
fn finish_starts(@builtin(global_invocation_id) id:vec3u,@builtin(num_workgroups) grid:vec3u){let s=id.x+id.y*grid.x*64u;let r=id.z;if(s>=p.streams){return;}let a=rd(r*12u+11u);wr(a+s,rd(a+s)+rd(p.prefixBase+r*p.columns+(s>>5u))+rd(r*12u));}
@compute @workgroup_size(64)
fn write_lengths(@builtin(global_invocation_id) id:vec3u){let word=id.x;let r=id.z;if(word>=p.streams/4u){return;}var packed=0u;for(var j=0u;j<4u;j++){let b=destination_range(r,word*4u+j);let n=b.y-b.x;if(n==0u||n>256u){fault(16u);return;}packed|=(n-1u)<<(j*8u);}encoded[rd(r*12u+1u)+p.columns+word]=packed;if(word<p.columns){encoded[rd(r*12u+1u)+word]=rd(p.prefixBase+r*p.columns+word);}}
@compute @workgroup_size(64)
fn encode(@builtin(global_invocation_id) id:vec3u,@builtin(num_workgroups) grid:vec3u){let s=id.x+id.y*grid.x*64u;let r=id.z;if(s>=p.streams){return;}let model=model_id(r,s%p.columns);let src=source_range(r,s);let dst=destination_range(r,s);if(src.y<=src.x||dst.y<=dst.x){return;}
 if(!hot(model)){if(dst.y-dst.x!=src.y-src.x){fault(16u);return;}for(var i=0u;i<src.y-src.x;i++){encoded[dst.x+i]=original[src.x+i];}return;}
 for(var word=dst.x;word<dst.y;word++){encoded[word]=0u;}var reader=original_reader(src.x,src.y,10u);var state=original[src.x]&1023u;var cursor=0u;
 for(var pair=0u;pair<256u;pair++){let v=ans_pair(&reader,&state,model);let code=encode_code(model,v);let n=(code>>10u)&15u;put(dst.x,cursor,n,code&1023u);cursor+=n;if((code&16384u)!=0u){put(dst.x,cursor,12u,v);cursor+=12u;}}
 if(reader.bad||state>=1024u||(cursor+31u)/32u!=dst.y-dst.x){fault(4u);}
}
@compute @workgroup_size(64)
fn copy_sparse(@builtin(global_invocation_id) id:vec3u,@builtin(num_workgroups) grid:vec3u){let i=id.x+id.y*grid.x*64u;let r=id.z;let b=r*12u;let words=controls[b+7u];if(i<words){encoded[rd(b+2u)+i]=original[controls[b+2u]+i];}if(i<p.sparseOffsetWords){encoded[rd(b+3u)+i]=original[controls[b+3u]+i];}}
@compute @workgroup_size(64)
fn verify(@builtin(global_invocation_id) id:vec3u,@builtin(num_workgroups) grid:vec3u){let s=id.x+id.y*grid.x*64u;let r=id.z;if(s>=p.streams){return;}let model=model_id(r,s%p.columns);let src=source_range(r,s);let dst=destination_range(r,s);if(src.y<=src.x||dst.y<=dst.x){return;}
 if(!hot(model)){if(src.y-src.x!=dst.y-dst.x){fault(16u);return;}for(var i=0u;i<src.y-src.x;i++){if(original[src.x+i]!=encoded[dst.x+i]){fault(128u);}}atomicAdd(&result[19u+r*9u],src.y-src.x);return;}
 var reference=original_reader(src.x,src.y,10u);var state=original[src.x]&1023u;var reader=huff_reader(dst.x,dst.y-dst.x,0u,(dst.y-dst.x)*32u);var bits=0u;
 for(var pair=0u;pair<256u;pair++){let expected=ans_pair(&reference,&state,model);let actual=huff_pair(&reader,model);if(actual!=expected){fault(32u);}let code=encode_code(model,expected);bits+=((code>>10u)&15u)+select(0u,12u,(code&16384u)!=0u);}
 if(reader.bad||reference.bad||state>=1024u||reader.cursor!=bits||(bits+31u)/32u!=dst.y-dst.x){fault(32u);}atomicAdd(&result[17u+r*9u],512u);
}
@compute @workgroup_size(64)
fn verify_sparse(@builtin(global_invocation_id) id:vec3u,@builtin(num_workgroups) grid:vec3u){let i=id.x+id.y*grid.x*64u;let r=id.z;let b=r*12u;let n=controls[b+7u];if(i<n){if(encoded[rd(b+2u)+i]!=original[controls[b+2u]+i]){fault(128u);}atomicAdd(&result[20u+r*9u],1u);}if(i<p.sparseOffsetWords){if(encoded[rd(b+3u)+i]!=original[controls[b+3u]+i]){fault(128u);}atomicAdd(&result[21u+r*9u],1u);}}
fn cp_write(base:u32,bit:u32,bits:u32,value:u32){if(value>=(1u<<bits)){fault(64u);return;}let at=base+(bit>>5u);let shift=bit&31u;atomicOr(&metadata[at],value<<shift);if(shift+bits>32u){atomicOr(&metadata[at+1u],value>>(32u-shift));}}
@compute @workgroup_size(64)
fn build_checkpoints(@builtin(global_invocation_id) id:vec3u,@builtin(num_workgroups) grid:vec3u){let s=id.x+id.y*grid.x*64u;let r=id.z;if(s>=p.streams){return;}let b=r*12u;let m=rd(b+10u);let rank=s%p.columns;let model=model_id(r,rank);let first=rd(rd(b+11u)+s);let n=1u+((original[rd(b+1u)+p.columns+(s>>2u)]>>((s&3u)*8u))&255u);
 if(first<rd(b)||first+n>rd(b)+rd(b+6u)||first+n>arrayLength(&original)){fault(1u);return;}if(model==255u){if(n!=256u){fault(1u);}return;}
 let prefix=rd(rd(m+1u)+rank);let packet=s/p.columns;
 if(hot(model)){let ordinal=packet*rd(m)+prefix;let bitBase=ordinal*rd(m+6u);var reader=huff_reader(first,n,0u,n*32u);
  for(var pair=0u;pair<256u;pair++){huff_pair(&reader,model);if(pair<255u&&(pair+1u)%32u==0u){cp_write(rd(m+2u),bitBase+(pair/32u)*rd(m+4u),rd(m+4u),reader.cursor);}}
  if(reader.bad||(reader.cursor+31u)/32u!=n){fault(32u);return;}cp_write(rd(m+2u),bitBase+7u*rd(m+4u),5u,n*32u-reader.cursor);
 }else{if(model>=81u){fault(2u);return;}let ordinal=packet*(p.columns-rd(m))+rank-prefix;var reader=original_reader(first,first+n,10u);var state=original[first]&1023u;
  for(var pair=0u;pair<256u;pair++){ans_pair(&reader,&state,model);if(pair<255u&&(pair+1u)%64u==0u){if(reader.cursor>8191u||state>=1024u){fault(64u);return;}wr(rd(m+3u)+ordinal*3u+pair/64u,0x80000000u|(reader.cursor<<10u)|state);}}
  if(reader.bad||state>=1024u){fault(4u);}
 }
 atomicAdd(&result[22u+r*9u],1u);
}
fn cp_read(base:u32,bit:u32,bits:u32)->u32{let at=base+(bit>>5u);let shift=bit&31u;var v=rd(at)>>shift;if(shift+bits>32u){v|=rd(at+1u)<<(32u-shift);}return v&((1u<<bits)-1u);}
@compute @workgroup_size(64)
fn verify_checkpoints(@builtin(global_invocation_id) id:vec3u,@builtin(num_workgroups) grid:vec3u){let s=id.x+id.y*grid.x*64u;let r=id.z;if(s>=p.streams){return;}let b=r*12u;let m=rd(b+10u);let rank=s%p.columns;let model=model_id(r,rank);if(model==255u){return;}let first=rd(rd(b+11u)+s);let n=1u+((original[rd(b+1u)+p.columns+(s>>2u)]>>((s&3u)*8u))&255u);let prefix=rd(rd(m+1u)+rank);let packet=s/p.columns;
 if(hot(model)){let ordinal=packet*rd(m)+prefix;let bitBase=ordinal*rd(m+6u);var reader=huff_reader(first,n,0u,n*32u);
  for(var pair=0u;pair<256u;pair++){huff_pair(&reader,model);if(pair<255u&&(pair+1u)%32u==0u){if(cp_read(rd(m+2u),bitBase+pair/32u*rd(m+4u),rd(m+4u))!=reader.cursor){fault(64u);}}}
  if(reader.bad||cp_read(rd(m+2u),bitBase+7u*rd(m+4u),5u)!=n*32u-reader.cursor){fault(64u);}
 }else{let ordinal=packet*(p.columns-rd(m))+rank-prefix;var reader=original_reader(first,first+n,10u);var state=original[first]&1023u;
  for(var pair=0u;pair<256u;pair++){ans_pair(&reader,&state,model);if(pair<255u&&(pair+1u)%64u==0u){if(rd(rd(m+3u)+ordinal*3u+pair/64u)!=(0x80000000u|(reader.cursor<<10u)|state)){fault(64u);}}}if(reader.bad||state>=1024u){fault(4u);}
 }atomicAdd(&result[23u+r*9u],1u);
}

`;
// Checkpoint pass only reads the new payload through binding0. Binding6 is a
// separate dummy buffer, avoiding read/read-write binding aliases.
export const HUFFMAN_CHECKPOINT_WGSL = HUFFMAN_MIGRATION_WGSL
  .split('r.lo=encoded[r.word]').join('r.lo=original[r.word]')
  .split('let word=encoded[(*r).word]').join('let word=original[(*r).word]');
