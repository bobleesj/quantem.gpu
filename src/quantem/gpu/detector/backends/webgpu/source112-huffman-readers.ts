/** Private checked pair readers shared by GPU migration and its native parity gate. */
export const MIGRATION_READERS = /* wgsl */ `
struct Reader{first:u32,end:u32,word:u32,lo:u32,hi:u32,available:u32,cursor:u32,limit:u32,bad:bool}
fn original_reader(first:u32,end:u32,cursor:u32)->Reader{
 var r=Reader(first,end,first+(cursor>>5u),0u,0u,0u,cursor,(end-first)*32u,false);
 if(cursor>r.limit){r.bad=true;return r;}
 if(r.word<end){r.lo=original[r.word]>>(cursor&31u);r.available=32u-(cursor&31u);r.word++;}
 return r;
}
fn huff_reader(first:u32,words:u32,cursor:u32,limit:u32)->Reader{
 var r=Reader(first,first+words,first+(cursor>>5u),0u,0u,0u,cursor,limit,false);
 if(cursor>limit||limit>words*32u||first+words>p.capacity){r.bad=true;return r;}
 if(r.word<r.end){r.lo=encoded[r.word]>>(cursor&31u);r.available=32u-(cursor&31u);r.word++;}
 return r;
}
fn consume(r:ptr<function,Reader>,n:u32)->u32{
 if(n==0u){return 0u;}
 if((*r).cursor+n>(*r).limit){(*r).bad=true;return 0u;}
 let v=(*r).lo&((1u<<n)-1u);
 (*r).lo=((*r).lo>>n)|((*r).hi<<(32u-n));(*r).hi>>=n;(*r).available-=n;(*r).cursor+=n;return v;
}
fn original_take(r:ptr<function,Reader>,n:u32)->u32{
 if(n==0u){return 0u;}
 if((*r).available<n){
  if((*r).word>=(*r).end){(*r).bad=true;return 0u;}
  let word=original[(*r).word];(*r).word++;
  (*r).lo|=word<<(*r).available;
  (*r).hi=select(0u,word>>(32u-(*r).available),(*r).available!=0u);(*r).available+=32u;
 }
 return consume(r,n);
}
fn huff_fill(r:ptr<function,Reader>,n:u32){
 if((*r).available<n && (*r).word<(*r).end){
  let word=encoded[(*r).word];(*r).word++;
  (*r).lo|=word<<(*r).available;
  (*r).hi=select(0u,word>>(32u-(*r).available),(*r).available!=0u);(*r).available+=32u;
 }
}
fn ans_pair(r:ptr<function,Reader>,state:ptr<function,u32>,model:u32)->u32{
 if((*state)>=1024u){(*r).bad=true;return 0u;}
 let code=tables[model*1024u+(*state)];(*state)=(code>>16u)+original_take(r,(code>>12u)&15u);
 var pair=code&4095u;if(pair==4095u){pair=original_take(r,12u);}return pair;
}
fn huff_pair(r:ptr<function,Reader>,model:u32)->u32{
 huff_fill(r,10u);let code=tables[82944u+8192u+(model-78u)*1024u+((*r).lo&1023u)];let n=(code>>12u)&15u;
 if(n==0u||n>10u){(*r).bad=true;return 0u;}consume(r,n);
 var pair=code&4095u;if(pair==4095u){huff_fill(r,12u);pair=consume(r,12u);}return pair;
}
fn put(first:u32,cursor:u32,n:u32,value:u32){
 if(n==0u){return;}
 let at=first+(cursor>>5u);let shift=cursor&31u;
 if(at>=p.capacity){fault(16u);return;}encoded[at]|=value<<shift;
 if(shift+n>32u){if(at+1u>=p.capacity){fault(16u);return;}encoded[at+1u]|=value>>(32u-shift);}
}
`;
