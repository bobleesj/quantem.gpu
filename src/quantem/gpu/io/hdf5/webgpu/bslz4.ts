/// <reference types="@webgpu/types" />
// Offline GPU decompression of HDF5 bitshuffle+LZ4 (bslz4) 4D-STEM data.
//
// Ships the NATIVE detector codec (Dectris/Arina write bslz4) so the browser
// downloads ~6x less than raw uint16 and decompresses on WebGPU - no Python, no
// server. Fused kernels LZ4-decode each block into workgroup shared memory and
// invert the bit transpose from there; native uint32 output keeps a two-pass
// decode through a global buffer. Output is packed in [scanPos][detPixel] order -
// the exact layout DetectorCompute reads in uint8/uint16/uint32/float32 mode, so
// it feeds masked_sum / reduce_frames with no copy.
//
// Per-frame chunk layout (one HDF5 chunk = one diffraction pattern):
//   blockMeta gives, per (frame,block), the absolute byte offset + compressed
//   length of that block's LZ4 stream within the concatenated `compressed`. Each
//   block decompresses to blockElemBytes = blockElems * elemBytes, holds
//   nbits = elemBytes*8 bit-planes of planeBytes = blockElems/8 each.

import { getGPUDevice, onGPULost } from "../../../device/webgpu";

type Bslz4Globals = {
  __BSLZ4_LOW8_ONLY?: boolean;
  __BSLZ4_FRAME_WG?: number;
  __BSLZ4_PIPELINE_STAGING?: boolean;
};

function bslz4Global(): Bslz4Globals {
  return typeof globalThis === "undefined" ? {} : globalThis as Bslz4Globals;
}

// Low-8 decode for lossless-valid uint8 browse data. This is NOT the general
// uint16->uint8 clip path: it is only scientifically valid when an upstream
// count-range audit proves all unmasked detector counts are <=255. In that case the
// high bit-planes can be ignored, and the LZ4 stream only needs to be decoded until
// the low 8 bit-planes are materialized.
function bslz4Low8Only(): boolean {
  return bslz4Global().__BSLZ4_LOW8_ONLY === true;
}

// Threads per workgroup of the frame-cooperative kernels (one workgroup per diffraction pattern).
function bslz4FrameWorkgroupSize(): 8 | 16 | 32 | 64 | 128 {
  const raw = Number(bslz4Global().__BSLZ4_FRAME_WG);
  if (raw === 8 || raw === 16 || raw === 32 || raw === 64 || raw === 128) return raw;
  return 32;
}

// Overlap the host copy of the next group into the staging pool with the current group's decode.
function bslz4PipelineStaging(): boolean {
  return bslz4Global().__BSLZ4_PIPELINE_STAGING !== false;
}

// Pass1: one thread per block, LZ4-decode into the `inter` (bitshuffled) buffer.
// Blocks are independent -> embarrassingly parallel. Byte-addressed RMW because
// WGSL storage is u32-only.
const PASS1_WGSL = `
@group(0) @binding(0) var<storage,read> raw: array<u32>;
@group(0) @binding(1) var<storage,read_write> inter: array<u32>;
@group(0) @binding(2) var<storage,read> blkMeta: array<u32>;   // coff,clen per block
@group(0) @binding(3) var<uniform> cfg: vec4<u32>;             // totalBlocks, blockBytes
fn rraw(i:u32)->u32{return (raw[i>>2u]>>((i&3u)*8u))&0xffu;}
fn rout(i:u32)->u32{return (inter[i>>2u]>>((i&3u)*8u))&0xffu;}
fn wout(i:u32,v:u32){let w=i>>2u;let s=(i&3u)*8u;inter[w]=(inter[w]&(~(0xffu<<s)))|((v&0xffu)<<s);}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3<u32>){
  let g=gid.x; if(g>=cfg.x){return;}
  let coff=blkMeta[g*2u]; let cend=coff+blkMeta[g*2u+1u]; let base=g*cfg.y;
  var ci=coff; var di=0u;
  loop{ if(ci>=cend){break;}
    let tok=rraw(ci); ci=ci+1u; var nlit=tok>>4u;
    if(nlit==15u){loop{let bb=rraw(ci);ci=ci+1u;nlit=nlit+bb;if(bb!=255u){break;}}}
    var k=0u; loop{if(k>=nlit){break;} wout(base+di+k,rraw(ci+k)); k=k+1u;} ci=ci+nlit; di=di+nlit;
    if(ci>=cend){break;}
    let off=rraw(ci)|(rraw(ci+1u)<<8u); ci=ci+2u; var ml=4u+(tok&0xfu);
    if((tok&0xfu)==15u){loop{let bb=rraw(ci);ci=ci+1u;ml=ml+bb;if(bb!=255u){break;}}}
    var j=0u; loop{if(j>=ml){break;} wout(base+di+j,rout(base+di+j-off)); j=j+1u;} di=di+ml;
  }
}`;

// Pass2 (native uint32 output): inverse bitshuffle of 32 bit-planes. One thread per GROUP
// of 8 consecutive pixels - they share the same 32 plane-bytes (one byte per bit-plane), so
// 32 reads feed 8 outputs. Plane-major, LSB-first: pixel (group*8 + i) bit b = bit i of
// plane-byte (b*planeBytes + group_byte); each full 32-bit value is stored as one u32/pixel.
// cfg = nGroups, strideX, blockElems, planeBytes.
const PASS2_U32_WGSL = `
@group(0) @binding(0) var<storage,read> inter: array<u32>;
@group(0) @binding(1) var<storage,read_write> stack: array<u32>;  // float32 bit-pattern, 1/pixel
@group(0) @binding(2) var<uniform> cfg: vec4<u32>;  // nGroups, strideX, blockElems, planeBytes
fn byteAt(o:u32)->u32{return (inter[o>>2u]>>((o&3u)*8u))&0xffu;}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3<u32>){
  let grp=gid.y*cfg.y + gid.x; if(grp>=cfg.x){return;}
  let blockElems=cfg.z; let planeBytes=cfg.w; let blockBytes=blockElems*4u;
  let nBlk=__NBLK__; let framePix=__FRAMEPIX__;
  let e0=grp*8u; let frm=e0/framePix; let inFrame=e0%framePix;
  let blk=inFrame/blockElems; let groupByte=(inFrame%blockElems)>>3u;
  let pb=(frm*nBlk+blk)*blockBytes + groupByte;
  var v0:u32=0u; var v1:u32=0u; var v2:u32=0u; var v3:u32=0u; var v4:u32=0u; var v5:u32=0u; var v6:u32=0u; var v7:u32=0u;
  for(var b:u32=0u;b<32u;b=b+1u){
    let byte=byteAt(pb+b*planeBytes); let bit=1u<<b;
    if((byte&1u)!=0u){v0=v0|bit;} if((byte&2u)!=0u){v1=v1|bit;}
    if((byte&4u)!=0u){v2=v2|bit;} if((byte&8u)!=0u){v3=v3|bit;}
    if((byte&16u)!=0u){v4=v4|bit;} if((byte&32u)!=0u){v5=v5|bit;}
    if((byte&64u)!=0u){v6=v6|bit;} if((byte&128u)!=0u){v7=v7|bit;}
  }
  let o=grp*8u;
  stack[o]=v0; stack[o+1u]=v1; stack[o+2u]=v2; stack[o+3u]=v3;
  stack[o+4u]=v4; stack[o+5u]=v5; stack[o+6u]=v6; stack[o+7u]=v7;
}`;

const MAX_WG = 65535;

// Copy bytes into a mapped GPU range using the widest aligned element view. V8's
// Uint8Array.set is byte-granular (~3.75 GB/s); a Float64Array view stores 8 bytes/op and is
// ~2-3x faster for multi-GB uploads. `src` is a Uint8Array over a whole file ArrayBuffer at
// byteOffset 0 and the mapped range is a fresh 4-aligned ArrayBuffer, so copy the bulk as
// f64 and the <8 trailing bytes as u8.
function copyWide(dst: ArrayBuffer, src: Uint8Array, dstOffset = 0): void {
  const len = src.byteLength, off = src.byteOffset;
  if (((off | dstOffset) & 7) === 0) {
    const n8 = len >>> 3;
    if (n8 > 0) new Float64Array(dst, dstOffset, n8).set(new Float64Array(src.buffer, off, n8));
    const tail = n8 << 3;
    if (tail < len) new Uint8Array(dst, dstOffset + tail, len - tail).set(src.subarray(tail));
    return;
  }
  new Uint8Array(dst, dstOffset, len).set(src);
}

// FUSED decode (uint8 or uint32 source -> uint8 output; uint16 sources use the
// frame-cooperative clip8 kernel below): ONE WORKGROUP per block. Thread 0 LZ4-decodes the
// block's bytes into workgroup-SHARED `sh` (byte RMW on shared is ~100x a global RMW, and
// there is no intermediate global buffer), barrier, then all 64 threads inverse-bitshuffle
// the blockElems/8 eight-pixel groups straight from `sh` and write uint8-packed output
// coalesced to `stack`. __NPB__ = blockElems/8 = planeBytes = groups per block; __BE__ =
// blockElems; __NBITS__ = 8/32 source bit planes.
const FUSED_U16U8_WGSL = `
@group(0) @binding(0) var<storage,read> raw: array<u32>;
@group(0) @binding(1) var<storage,read> blkMeta: array<u32>;   // coff,clen per block
@group(0) @binding(2) var<storage,read_write> stack: array<u32>;  // uint8-packed (4/u32)
@group(0) @binding(3) var<uniform> cfg: vec4<u32>;   // totalBlocks, gridX, nBlk, framePix
var<workgroup> sh: array<u32, 2048>;   // up to 8192 decoded (bitshuffled) bytes / block
fn rraw(i:u32)->u32{return (raw[i>>2u]>>((i&3u)*8u))&0xffu;}
fn rsh(i:u32)->u32{return (sh[i>>2u]>>((i&3u)*8u))&0xffu;}
fn wsh(i:u32,v:u32){let w=i>>2u;let s=(i&3u)*8u;sh[w]=(sh[w]&(~(0xffu<<s)))|((v&0xffu)<<s);}
fn clip8(v:u32)->u32{return select(v,255u,v>255u);}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>){
  let g = wid.y*cfg.y + wid.x; let lx = lid.x;
  if(lx==0u && g<cfg.x){
    let coff=blkMeta[g*2u]; let cend=coff+blkMeta[g*2u+1u];
    var ci=coff; var di=0u;
    loop{ if(ci>=cend){break;}
      let tok=rraw(ci); ci=ci+1u; var nlit=tok>>4u;
      if(nlit==15u){loop{let bb=rraw(ci);ci=ci+1u;nlit=nlit+bb;if(bb!=255u){break;}}}
      var k=0u; loop{if(k>=nlit){break;} wsh(di+k,rraw(ci+k)); k=k+1u;} ci=ci+nlit; di=di+nlit;
      if(ci>=cend){break;}
      let off=rraw(ci)|(rraw(ci+1u)<<8u); ci=ci+2u; var ml=4u+(tok&0xfu);
      if((tok&0xfu)==15u){loop{let bb=rraw(ci);ci=ci+1u;ml=ml+bb;if(bb!=255u){break;}}}
      var j=0u; loop{if(j>=ml){break;} wsh(di+j,rsh(di+j-off)); j=j+1u;} di=di+ml;
    }
  }
  workgroupBarrier();
  if(g>=cfg.x){return;}
  let frm = g/cfg.z; let blk = g%cfg.z;
  let pixBase = frm*cfg.w + blk*__BE__;   // first detector pixel of this block
  let oBase = (pixBase>>3u)*2u;            // 2 u32 per 8-pixel group
  // uint8 output == clip8(value): 255 iff ANY bit >=8 is set, else the low byte. So only the
  // 8 LOW planes need a bit-transpose; planes 8..nbits-1 collapse to one OR ("any high bit set"
  // per pixel). Bit-exact for any input, and 2x (uint16) / 4x (uint32) less bitshuffle work.
  for(var lg=lx; lg<__NPB__; lg=lg+64u){
    var hi:u32=0u;
    for(var b:u32=8u;b<__NBITS__;b=b+1u){ hi = hi | rsh(lg + b*__NPB__); }
    var v0:u32=0u; var v1:u32=0u; var v2:u32=0u; var v3:u32=0u; var v4:u32=0u; var v5:u32=0u; var v6:u32=0u; var v7:u32=0u;
    for(var b:u32=0u;b<8u;b=b+1u){
      let byte=rsh(lg + b*__NPB__); let bit=1u<<b;
      if((byte&1u)!=0u){v0=v0|bit;} if((byte&2u)!=0u){v1=v1|bit;}
      if((byte&4u)!=0u){v2=v2|bit;} if((byte&8u)!=0u){v3=v3|bit;}
      if((byte&16u)!=0u){v4=v4|bit;} if((byte&32u)!=0u){v5=v5|bit;}
      if((byte&64u)!=0u){v6=v6|bit;} if((byte&128u)!=0u){v7=v7|bit;}
    }
    let o=oBase + lg*2u;
    stack[o]=select(v0,255u,(hi&1u)!=0u)|(select(v1,255u,(hi&2u)!=0u)<<8u)|(select(v2,255u,(hi&4u)!=0u)<<16u)|(select(v3,255u,(hi&8u)!=0u)<<24u);
    stack[o+1u]=select(v4,255u,(hi&16u)!=0u)|(select(v5,255u,(hi&32u)!=0u)<<8u)|(select(v6,255u,(hi&64u)!=0u)<<16u)|(select(v7,255u,(hi&128u)!=0u)<<24u);
  }
}`;

// FUSED frame-cooperative low-8 decode: one workgroup per diffraction pattern, looping
// over the frame's bitshuffle blocks; all lanes parse each LZ4 token and copy its bytes
// cooperatively, stopping once the first __BE__ bytes (the low 8 bit-planes) are decoded.
// Overlapping LZ4 matches use the period identity value[j] = output[start - offset + (j % offset)].
const FUSED_FRAME_COOP_LOW8_WGSL = `
@group(0) @binding(0) var<storage,read> raw: array<u32>;
@group(0) @binding(1) var<storage,read> blkMeta: array<u32>;   // coff,clen per block
@group(0) @binding(2) var<storage,read_write> stack: array<u32>;  // uint8-packed (4/u32)
@group(0) @binding(3) var<uniform> cfg: vec4<u32>;   // nFrames, gridX, nBlk, framePix
var<workgroup> sh: array<atomic<u32>, __SH_WORDS__>;
fn rraw(i:u32)->u32{return (raw[i>>2u]>>((i&3u)*8u))&0xffu;}
fn rsh(i:u32)->u32{return (atomicLoad(&sh[i>>2u])>>((i&3u)*8u))&0xffu;}
fn wsh(i:u32,v:u32){atomicOr(&sh[i>>2u], (v&0xffu)<<((i&3u)*8u));}
@compute @workgroup_size(__WG__)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>){
  let frm = wid.y*cfg.y + wid.x; let lx = lid.x;
  if(frm>=cfg.x){return;}
  for(var blk=0u; blk<cfg.z; blk=blk+1u){
    for(var w=lx; w<__SH_WORDS__; w=w+__WG__){ atomicStore(&sh[w], 0u); }
    workgroupBarrier();
    let g = frm*cfg.z + blk;
    let coff=blkMeta[g*2u]; let cend=coff+blkMeta[g*2u+1u];
    var ci=coff; var di=0u; var working=true;
    loop{
      if(!working){break;}
      let tok=rraw(ci); ci=ci+1u; var nlit=tok>>4u;
      if(nlit==15u){loop{let bb=rraw(ci);ci=ci+1u;nlit=nlit+bb;if(bb!=255u){break;}}}
      let litCopy=min(nlit,__BE__-di);
      for(var k=lx; k<litCopy; k=k+__WG__){ wsh(di+k,rraw(ci+k)); }
      ci=ci+nlit; di=di+nlit;
      workgroupBarrier();
      if(ci>=cend || di>=__BE__){
        working=false;
      } else {
        let off=rraw(ci)|(rraw(ci+1u)<<8u); ci=ci+2u; var ml=4u+(tok&0xfu);
        if((tok&0xfu)==15u){loop{let bb=rraw(ci);ci=ci+1u;ml=ml+bb;if(bb!=255u){break;}}}
        let matchCopy=min(ml,__BE__-di);
        for(var j=lx; j<matchCopy; j=j+__WG__){ wsh(di+j,rsh(di-off+(j%off))); }
        di=di+ml;
        workgroupBarrier();
        if(ci>=cend || di>=__BE__){ working=false; }
      }
    }
    workgroupBarrier();
    let pixBase = frm*cfg.w + blk*__BE__;
    let blockPix = min(__BE__, cfg.w - min(cfg.w, blk*__BE__));
    let groupsThis = (blockPix + 7u) >> 3u;
    let oBase = pixBase >> 2u;
    for(var lg=lx; lg<groupsThis; lg=lg+__WG__){
      var v0:u32=0u; var v1:u32=0u; var v2:u32=0u; var v3:u32=0u; var v4:u32=0u; var v5:u32=0u; var v6:u32=0u; var v7:u32=0u;
      for(var b:u32=0u;b<8u;b=b+1u){
        let byte=rsh(lg + b*__NPB__); let bit=1u<<b;
        if((byte&1u)!=0u){v0=v0|bit;} if((byte&2u)!=0u){v1=v1|bit;}
        if((byte&4u)!=0u){v2=v2|bit;} if((byte&8u)!=0u){v3=v3|bit;}
        if((byte&16u)!=0u){v4=v4|bit;} if((byte&32u)!=0u){v5=v5|bit;}
        if((byte&64u)!=0u){v6=v6|bit;} if((byte&128u)!=0u){v7=v7|bit;}
      }
      let o=oBase + lg*2u;
      stack[o]=v0|(v1<<8u)|(v2<<16u)|(v3<<24u);
      stack[o+1u]=v4|(v5<<8u)|(v6<<16u)|(v7<<24u);
    }
    workgroupBarrier();
  }
}`;

// FUSED frame-cooperative NATIVE uint16 decode: the lossless counterpart of
// FUSED_FRAME_COOP_LOW8. Same one-workgroup-per-frame LZ4 loop, but it decodes the
// FULL bitshuffle block (__BB__ = 2x blockElems bytes: low8 stops at the low 8
// planes, which silently WRAPS counts >255 - fatal for high-count data), then
// reconstructs all 16 bit planes and packs 2 pixels per u32 (mode 0). Inherent
// cost vs low8: ~2x the LZ4 bytes resolved and 2x the output stores; that is the
// dtype's floor, not kernel inefficiency.
const FUSED_FRAME_U16_WGSL = `
@group(0) @binding(0) var<storage,read> raw: array<u32>;
@group(0) @binding(1) var<storage,read> blkMeta: array<u32>;   // coff,clen per block
@group(0) @binding(2) var<storage,read_write> stack: array<u32>;  // uint16-packed (2/u32)
@group(0) @binding(3) var<uniform> cfg: vec4<u32>;   // nFrames, gridX, nBlk, framePix
var<workgroup> sh: array<atomic<u32>, __SH_WORDS__>;
fn rraw(i:u32)->u32{return (raw[i>>2u]>>((i&3u)*8u))&0xffu;}
fn rsh(i:u32)->u32{return (atomicLoad(&sh[i>>2u])>>((i&3u)*8u))&0xffu;}
fn wsh(i:u32,v:u32){atomicOr(&sh[i>>2u], (v&0xffu)<<((i&3u)*8u));}
@compute @workgroup_size(__WG__)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>){
  let frm = wid.y*cfg.y + wid.x; let lx = lid.x;
  if(frm>=cfg.x){return;}
  for(var blk=0u; blk<cfg.z; blk=blk+1u){
    for(var w=lx; w<__SH_WORDS__; w=w+__WG__){ atomicStore(&sh[w], 0u); }
    workgroupBarrier();
    let g = frm*cfg.z + blk;
    let coff=blkMeta[g*2u]; let cend=coff+blkMeta[g*2u+1u];
    var ci=coff; var di=0u; var working=true;
    loop{
      if(!working){break;}
      let tok=rraw(ci); ci=ci+1u; var nlit=tok>>4u;
      if(nlit==15u){loop{let bb=rraw(ci);ci=ci+1u;nlit=nlit+bb;if(bb!=255u){break;}}}
      let litCopy=min(nlit,__BB__-di);
      for(var k=lx; k<litCopy; k=k+__WG__){ wsh(di+k,rraw(ci+k)); }
      ci=ci+nlit; di=di+nlit;
      workgroupBarrier();
      if(ci>=cend || di>=__BB__){
        working=false;
      } else {
        let off=rraw(ci)|(rraw(ci+1u)<<8u); ci=ci+2u; var ml=4u+(tok&0xfu);
        if((tok&0xfu)==15u){loop{let bb=rraw(ci);ci=ci+1u;ml=ml+bb;if(bb!=255u){break;}}}
        let matchCopy=min(ml,__BB__-di);
        for(var j=lx; j<matchCopy; j=j+__WG__){ wsh(di+j,rsh(di-off+(j%off))); }
        di=di+ml;
        workgroupBarrier();
        if(ci>=cend || di>=__BB__){ working=false; }
      }
    }
    workgroupBarrier();
    let pixBase = frm*cfg.w + blk*__BE__;
    let blockPix = min(__BE__, cfg.w - min(cfg.w, blk*__BE__));
    let groupsThis = (blockPix + 7u) >> 3u;
    let oBase = pixBase >> 1u;
    for(var lg=lx; lg<groupsThis; lg=lg+__WG__){
      var v0:u32=0u; var v1:u32=0u; var v2:u32=0u; var v3:u32=0u; var v4:u32=0u; var v5:u32=0u; var v6:u32=0u; var v7:u32=0u;
      for(var b:u32=0u;b<16u;b=b+1u){
        let byte=rsh(lg + b*__NPB__); let bit=1u<<b;
        if((byte&1u)!=0u){v0=v0|bit;} if((byte&2u)!=0u){v1=v1|bit;}
        if((byte&4u)!=0u){v2=v2|bit;} if((byte&8u)!=0u){v3=v3|bit;}
        if((byte&16u)!=0u){v4=v4|bit;} if((byte&32u)!=0u){v5=v5|bit;}
        if((byte&64u)!=0u){v6=v6|bit;} if((byte&128u)!=0u){v7=v7|bit;}
      }
      let o=oBase + lg*4u;
      stack[o]=v0|(v1<<16u);
      stack[o+1u]=v2|(v3<<16u);
      stack[o+2u]=v4|(v5<<16u);
      stack[o+3u]=v6|(v7<<16u);
    }
    workgroupBarrier();
  }
}`;

// FUSED frame-cooperative EXACT clip-u8 decode: same full-block frame-cooperative
// LZ4 loop as FUSED_FRAME_U16 but clips to 255 at the store (4 px/u32, mode 1).
// This replaces the one-workgroup-per-BLOCK FUSED_U16U8 path for uint16 sources:
// measured 2.7x faster on real Arina data purely from the dispatch shape. Unlike
// low8 (which stops at 8 planes and WRAPS >255), this decodes all 16 planes and
// saturates - exact clip semantics with no count audit required.
const FUSED_FRAME_CLIP8_WGSL = `
@group(0) @binding(0) var<storage,read> raw: array<u32>;
@group(0) @binding(1) var<storage,read> blkMeta: array<u32>;   // coff,clen per block
@group(0) @binding(2) var<storage,read_write> stack: array<u32>;  // uint8-packed (4/u32)
@group(0) @binding(3) var<uniform> cfg: vec4<u32>;   // nFrames, gridX, nBlk, framePix
var<workgroup> sh: array<atomic<u32>, __SH_WORDS__>;
fn rraw(i:u32)->u32{return (raw[i>>2u]>>((i&3u)*8u))&0xffu;}
fn rsh(i:u32)->u32{return (atomicLoad(&sh[i>>2u])>>((i&3u)*8u))&0xffu;}
fn wsh(i:u32,v:u32){atomicOr(&sh[i>>2u], (v&0xffu)<<((i&3u)*8u));}
fn clip8(v:u32)->u32{return select(v,255u,v>255u);}
@compute @workgroup_size(__WG__)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>){
  let frm = wid.y*cfg.y + wid.x; let lx = lid.x;
  if(frm>=cfg.x){return;}
  for(var blk=0u; blk<cfg.z; blk=blk+1u){
    for(var w=lx; w<__SH_WORDS__; w=w+__WG__){ atomicStore(&sh[w], 0u); }
    workgroupBarrier();
    let g = frm*cfg.z + blk;
    let coff=blkMeta[g*2u]; let cend=coff+blkMeta[g*2u+1u];
    var ci=coff; var di=0u; var working=true;
    loop{
      if(!working){break;}
      let tok=rraw(ci); ci=ci+1u; var nlit=tok>>4u;
      if(nlit==15u){loop{let bb=rraw(ci);ci=ci+1u;nlit=nlit+bb;if(bb!=255u){break;}}}
      let litCopy=min(nlit,__BB__-di);
      for(var k=lx; k<litCopy; k=k+__WG__){ wsh(di+k,rraw(ci+k)); }
      ci=ci+nlit; di=di+nlit;
      workgroupBarrier();
      if(ci>=cend || di>=__BB__){
        working=false;
      } else {
        let off=rraw(ci)|(rraw(ci+1u)<<8u); ci=ci+2u; var ml=4u+(tok&0xfu);
        if((tok&0xfu)==15u){loop{let bb=rraw(ci);ci=ci+1u;ml=ml+bb;if(bb!=255u){break;}}}
        let matchCopy=min(ml,__BB__-di);
        for(var j=lx; j<matchCopy; j=j+__WG__){ wsh(di+j,rsh(di-off+(j%off))); }
        di=di+ml;
        workgroupBarrier();
        if(ci>=cend || di>=__BB__){ working=false; }
      }
    }
    workgroupBarrier();
    let pixBase = frm*cfg.w + blk*__BE__;
    let blockPix = min(__BE__, cfg.w - min(cfg.w, blk*__BE__));
    let groupsThis = (blockPix + 7u) >> 3u;
    let oBase = pixBase >> 2u;
    for(var lg=lx; lg<groupsThis; lg=lg+__WG__){
      var v0:u32=0u; var v1:u32=0u; var v2:u32=0u; var v3:u32=0u; var v4:u32=0u; var v5:u32=0u; var v6:u32=0u; var v7:u32=0u;
      for(var b:u32=0u;b<16u;b=b+1u){
        let byte=rsh(lg + b*__NPB__); let bit=1u<<b;
        if((byte&1u)!=0u){v0=v0|bit;} if((byte&2u)!=0u){v1=v1|bit;}
        if((byte&4u)!=0u){v2=v2|bit;} if((byte&8u)!=0u){v3=v3|bit;}
        if((byte&16u)!=0u){v4=v4|bit;} if((byte&32u)!=0u){v5=v5|bit;}
        if((byte&64u)!=0u){v6=v6|bit;} if((byte&128u)!=0u){v7=v7|bit;}
      }
      let o=oBase + lg*2u;
      stack[o]=clip8(v0)|(clip8(v1)<<8u)|(clip8(v2)<<16u)|(clip8(v3)<<24u);
      stack[o+1u]=clip8(v4)|(clip8(v5)<<8u)|(clip8(v6)<<16u)|(clip8(v7)<<24u);
    }
    workgroupBarrier();
  }
}`;

// FUSED decode (float32 source -> float32 output, full precision): ONE WORKGROUP per block.
// Thread 0 LZ4-decodes the block into workgroup-SHARED sh (the fast path, no interBuf global
// round-trip), barrier, then all 64 threads inverse-bitshuffle the blockElems/8 eight-pixel
// groups over ALL 32 planes, reconstruct the full 32-bit value and write it as 1 u32/pixel
// (the IEEE-754 bit pattern - reduce/frame shaders bitcast<f32>). __NPB__ = blockElems/8 =
// planeBytes = groups/block; __BE__ = blockElems.
const FUSED_F32_WGSL = `
@group(0) @binding(0) var<storage,read> raw: array<u32>;
@group(0) @binding(1) var<storage,read> blkMeta: array<u32>;   // coff,clen per block
@group(0) @binding(2) var<storage,read_write> stack: array<u32>;  // float32 bit pattern (1/pixel)
@group(0) @binding(3) var<uniform> cfg: vec4<u32>;   // totalBlocks, gridX, nBlk, framePix
var<workgroup> sh: array<u32, 2048>;   // up to 8192 decoded (bitshuffled) bytes / block
fn rraw(i:u32)->u32{return (raw[i>>2u]>>((i&3u)*8u))&0xffu;}
fn rsh(i:u32)->u32{return (sh[i>>2u]>>((i&3u)*8u))&0xffu;}
fn wsh(i:u32,v:u32){let w=i>>2u;let s=(i&3u)*8u;sh[w]=(sh[w]&(~(0xffu<<s)))|((v&0xffu)<<s);}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>){
  let g = wid.y*cfg.y + wid.x; let lx = lid.x;
  if(lx==0u && g<cfg.x){
    let coff=blkMeta[g*2u]; let cend=coff+blkMeta[g*2u+1u];
    var ci=coff; var di=0u;
    loop{ if(ci>=cend){break;}
      let tok=rraw(ci); ci=ci+1u; var nlit=tok>>4u;
      if(nlit==15u){loop{let bb=rraw(ci);ci=ci+1u;nlit=nlit+bb;if(bb!=255u){break;}}}
      var k=0u; loop{if(k>=nlit){break;} wsh(di+k,rraw(ci+k)); k=k+1u;} ci=ci+nlit; di=di+nlit;
      if(ci>=cend){break;}
      let off=rraw(ci)|(rraw(ci+1u)<<8u); ci=ci+2u; var ml=4u+(tok&0xfu);
      if((tok&0xfu)==15u){loop{let bb=rraw(ci);ci=ci+1u;ml=ml+bb;if(bb!=255u){break;}}}
      var j=0u; loop{if(j>=ml){break;} wsh(di+j,rsh(di+j-off)); j=j+1u;} di=di+ml;
    }
  }
  workgroupBarrier();
  if(g>=cfg.x){return;}
  let frm = g/cfg.z; let blk = g%cfg.z;
  let pixBase = frm*cfg.w + blk*__BE__;
  let oBase = pixBase;   // 1 u32 per pixel
  for(var lg=lx; lg<__NPB__; lg=lg+64u){
    var v0:u32=0u; var v1:u32=0u; var v2:u32=0u; var v3:u32=0u; var v4:u32=0u; var v5:u32=0u; var v6:u32=0u; var v7:u32=0u;
    for(var b:u32=0u;b<32u;b=b+1u){
      let byte=rsh(lg + b*__NPB__); let bit=1u<<b;
      if((byte&1u)!=0u){v0=v0|bit;} if((byte&2u)!=0u){v1=v1|bit;}
      if((byte&4u)!=0u){v2=v2|bit;} if((byte&8u)!=0u){v3=v3|bit;}
      if((byte&16u)!=0u){v4=v4|bit;} if((byte&32u)!=0u){v5=v5|bit;}
      if((byte&64u)!=0u){v6=v6|bit;} if((byte&128u)!=0u){v7=v7|bit;}
    }
    let o=oBase + lg*8u;
    stack[o]=v0; stack[o+1u]=v1; stack[o+2u]=v2; stack[o+3u]=v3;
    stack[o+4u]=v4; stack[o+5u]=v5; stack[o+6u]=v6; stack[o+7u]=v7;
  }
}`;

// Compiled pipelines per device, by WGSL source. A pipeline belongs to the device that built it:
// after a device is lost, the replacement device must compile its own, so the cache is keyed by
// device first (a WeakMap, so a lost device's pipelines are collected with it).
const PIPELINES = new WeakMap<GPUDevice, Map<string, GPUComputePipeline>>();

/** Compile a specialized WGSL source once per device and reuse the pipeline for every later chunk. */
export function cachedPipeline(device: GPUDevice, code: string): GPUComputePipeline {
  let pipelines = PIPELINES.get(device);
  if (!pipelines) {
    pipelines = new Map();
    PIPELINES.set(device, pipelines);
  }
  let pipeline = pipelines.get(code);
  if (!pipeline) {
    pipeline = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code }), entryPoint: "main" } });
    pipelines.set(code, pipeline);
  }
  return pipeline;
}

function getFusedF32Pipe(device: GPUDevice, blockElems: number): GPUComputePipeline {
  const code = FUSED_F32_WGSL.replace(/__NPB__/g, `${blockElems / 8}u`).replace(/__BE__/g, `${blockElems}u`);
  return cachedPipeline(device, code);
}

// Raw bytes come either pre-uploaded through the staging pool (batch path) or, for the
// single decode path, through a mappedAtCreation buffer here; the block table is always uploaded.
function uploadFusedInputs(device: GPUDevice, spec: Bslz4Spec, preRaw?: GPUBuffer | RawInput): { rawBuf: GPUBuffer | RawInput; metaBuf: GPUBuffer } {
  const { compressed, blockMeta } = spec;
  let rawBuf: GPUBuffer | RawInput;
  if (preRaw) { rawBuf = preRaw; }
  else {
    const rawSize = Math.ceil(compressed.byteLength / 4) * 4;
    const buffer = device.createBuffer({ size: rawSize, usage: GPUBufferUsage.STORAGE, mappedAtCreation: true });
    copyWide(buffer.getMappedRange(), compressed);
    buffer.unmap();
    rawBuf = rawInput(buffer, rawSize);
  }
  const metaBuf = device.createBuffer({ size: blockMeta.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(metaBuf, 0, blockMeta.buffer as ArrayBuffer, blockMeta.byteOffset, blockMeta.byteLength);
  return { rawBuf, metaBuf };
}

// Every fused kernel binds raw bytes, block table, output stack and dims at 0-3 and runs one 2D grid.
function fusedJob(
  device: GPUDevice, pipe: GPUComputePipeline, rawBuf: GPUBuffer | RawInput, metaBuf: GPUBuffer,
  stack: GPUBuffer, cfg: GPUBuffer, gx: number, gy: number, mode: number,
): DecodeJob {
  const bg = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [
    { binding: 0, resource: rawBinding(rawBuf) }, { binding: 1, resource: { buffer: metaBuf } },
    { binding: 2, resource: { buffer: stack } }, { binding: 3, resource: { buffer: cfg } } ] });
  return {
    stack, mode,
    record(enc) { const pass = enc.beginComputePass(); pass.setPipeline(pipe); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(gx, gy); pass.end(); },
    releaseTemps() { releaseRaw(rawBuf); metaBuf.destroy(); cfg.destroy(); },
  };
}

// Fused float32 decode job: one workgroup/block, float32 stack out (1 u32/pixel, mode 2).
function buildFusedJobF32(device: GPUDevice, spec: Bslz4Spec, preRaw?: GPUBuffer | RawInput): DecodeJob {
  const { nFrames, nBlocksPerFrame, blockElems, detSize } = spec;
  const totalBlocks = nFrames * nBlocksPerFrame;
  const stackWords = nFrames * detSize;   // 1 u32/pixel
  const { rawBuf, metaBuf } = uploadFusedInputs(device, spec, preRaw);
  const stack = device.createBuffer({ size: stackWords * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const gx = Math.min(totalBlocks, MAX_WG), gy = Math.ceil(totalBlocks / MAX_WG);
  const cfg = uniform(device, [totalBlocks, gx, nBlocksPerFrame, detSize]);
  const pipe = getFusedF32Pipe(device, blockElems);
  return fusedJob(device, pipe, rawBuf, metaBuf, stack, cfg, gx, gy, 2);
}

function getFusedPipe(device: GPUDevice, blockElems: number, nbits: number): GPUComputePipeline {
  const npb = blockElems / 8;
  const code = FUSED_U16U8_WGSL.replace(/__NPB__/g, `${npb}u`).replace(/__BE__/g, `${blockElems}u`).replace(/__NBITS__/g, `${nbits}u`);
  return cachedPipeline(device, code);
}

// Frame-cooperative kernels: "u16" and "clip8" decode the full uint16 block (__BB__ bytes),
// "low8" only its low 8 bit-planes (the first __BE__ bytes).
function getFusedFramePipe(device: GPUDevice, blockElems: number, out: "u16" | "clip8" | "low8"): GPUComputePipeline {
  const blockBytes = out === "low8" ? blockElems : blockElems * 2;
  const template = out === "u16" ? FUSED_FRAME_U16_WGSL : out === "clip8" ? FUSED_FRAME_CLIP8_WGSL : FUSED_FRAME_COOP_LOW8_WGSL;
  const code = template
    .replace(/__NPB__/g, `${blockElems / 8}u`)
    .replace(/__SH_WORDS__/g, `${Math.ceil(blockBytes / 4)}u`)
    .replace(/__WG__/g, `${bslz4FrameWorkgroupSize()}u`)
    .replace(/__BB__/g, `${blockBytes}u`)
    .replace(/__BE__/g, `${blockElems}u`);
  return cachedPipeline(device, code);
}

// Per-block fused job for uint8 and uint32 sources -> uint8 output (mode 1).
function buildFusedJob(device: GPUDevice, spec: Bslz4Spec, srcDtype: "uint8" | "uint32", preRaw?: GPUBuffer | RawInput): DecodeJob {
  const { nFrames, nBlocksPerFrame, blockElems, detSize } = spec;
  const totalBlocks = nFrames * nBlocksPerFrame;
  const { rawBuf, metaBuf } = uploadFusedInputs(device, spec, preRaw);
  const stack = device.createBuffer({ size: Math.ceil(nFrames * detSize / 4) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const gx = Math.min(totalBlocks, MAX_WG), gy = Math.ceil(totalBlocks / MAX_WG);
  const cfg = uniform(device, [totalBlocks, gx, nBlocksPerFrame, detSize]);
  const pipe = getFusedPipe(device, blockElems, srcDtype === "uint32" ? 32 : 8);
  return fusedJob(device, pipe, rawBuf, metaBuf, stack, cfg, gx, gy, 1);
}

// Frame-cooperative job, one workgroup per frame: "u16" packs 2 px/u32 (mode 0, lossless);
// "clip8" saturates to 255 and "low8" keeps the low byte, both 4 px/u32 (mode 1).
function buildFusedJobFrame(device: GPUDevice, spec: Bslz4Spec, out: "u16" | "clip8" | "low8", preRaw?: GPUBuffer | RawInput): DecodeJob {
  const { nFrames, nBlocksPerFrame, blockElems, detSize } = spec;
  const stackWords = Math.ceil(nFrames * detSize / (out === "u16" ? 2 : 4));
  const { rawBuf, metaBuf } = uploadFusedInputs(device, spec, preRaw);
  const stack = device.createBuffer({ size: stackWords * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const gx = Math.min(nFrames, MAX_WG), gy = Math.ceil(nFrames / MAX_WG);
  const cfg = uniform(device, [nFrames, gx, nBlocksPerFrame, detSize]);
  const pipe = getFusedFramePipe(device, blockElems, out);
  return fusedJob(device, pipe, rawBuf, metaBuf, stack, cfg, gx, gy, out === "u16" ? 0 : 1);
}

// Two-pass job for uint32 sources -> native uint32 output (mode 3): PASS1 LZ4-decodes every
// block into a global buffer, PASS2_U32 inverts the bit transpose of all 32 planes.
function buildTwoPassJobU32(device: GPUDevice, spec: Bslz4Spec, preRaw?: GPUBuffer | RawInput): DecodeJob {
  const { nFrames, nBlocksPerFrame, blockElems, detSize } = spec;
  const blockBytes = blockElems * 4;
  const totalBlocks = nFrames * nBlocksPerFrame;
  const totalElems = nFrames * detSize;
  const { rawBuf, metaBuf } = uploadFusedInputs(device, spec, preRaw);
  const interBuf = device.createBuffer({ size: totalBlocks * blockBytes, usage: GPUBufferUsage.STORAGE });
  const stack = device.createBuffer({ size: totalElems * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const cfg1 = uniform(device, [totalBlocks, blockBytes, 0, 0]);
  const nGroups = totalElems / 8;
  const p2wg = Math.ceil(nGroups / 64), gx = Math.min(p2wg, MAX_WG), gy = Math.ceil(p2wg / MAX_WG);
  const cfg2 = uniform(device, [nGroups, gx * 64, blockElems, blockElems / 8]);
  const p1 = cachedPipeline(device, PASS1_WGSL);
  const p2 = cachedPipeline(device, PASS2_U32_WGSL.replace("__NBLK__", `${nBlocksPerFrame}u`).replace("__FRAMEPIX__", `${detSize}u`));
  const bg1 = device.createBindGroup({ layout: p1.getBindGroupLayout(0), entries: [
    { binding: 0, resource: rawBinding(rawBuf) }, { binding: 1, resource: { buffer: interBuf } },
    { binding: 2, resource: { buffer: metaBuf } }, { binding: 3, resource: { buffer: cfg1 } } ] });
  const bg2 = device.createBindGroup({ layout: p2.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: interBuf } }, { binding: 1, resource: { buffer: stack } },
    { binding: 2, resource: { buffer: cfg2 } } ] });
  return {
    stack, mode: 3,
    record(enc) {
      const pa = enc.beginComputePass(); pa.setPipeline(p1); pa.setBindGroup(0, bg1); pa.dispatchWorkgroups(Math.ceil(totalBlocks / 64)); pa.end();
      const pb = enc.beginComputePass(); pb.setPipeline(p2); pb.setBindGroup(0, bg2); pb.dispatchWorkgroups(gx, gy); pb.end();
    },
    releaseTemps() { releaseRaw(rawBuf); interBuf.destroy(); metaBuf.destroy(); cfg1.destroy(); cfg2.destroy(); },
  };
}

// The decode job for one dtype request, and its profile name.
function decodeJobBuilder(dtype: DecodeDtype, srcDtype: SourceDtype): {
  variant: string;
  build: (device: GPUDevice, spec: Bslz4Spec, preRaw?: GPUBuffer | RawInput) => DecodeJob;
} {
  const wg = bslz4FrameWorkgroupSize();
  if (srcDtype === "float32") return { variant: "fused-f32", build: buildFusedJobF32 };
  if (dtype === "uint32") return { variant: "two-pass", build: buildTwoPassJobU32 };
  if (dtype === "uint16") return { variant: `fused-frame-coop-u16-wg${wg}`, build: (device, spec, raw) => buildFusedJobFrame(device, spec, "u16", raw) };
  if (bslz4Low8Only()) return { variant: `fused-frame-coop-low8-wg${wg}`, build: (device, spec, raw) => buildFusedJobFrame(device, spec, "low8", raw) };
  if (srcDtype === "uint16") return { variant: `fused-frame-coop-clip8-wg${wg}`, build: (device, spec, raw) => buildFusedJobFrame(device, spec, "clip8", raw) };
  return { variant: "fused-clip-u8", build: (device, spec, raw) => buildFusedJob(device, spec, srcDtype, raw) };
}

export interface Bslz4Spec {
  compressed: Uint8Array;        // concatenated per-frame bslz4 chunks (frame-padded to 4B)
  blockMeta: Uint32Array;        // [coff,clen] per (frame,block), absolute byte offsets
  nFrames: number;
  nBlocksPerFrame: number;
  blockElems: number;            // elements per bitshuffle block (e.g. 4096 for uint16/8192B)
  detSize: number;               // detector pixels per frame (e.g. 192*192)
}

type SourceDtype = "uint8" | "uint16" | "uint32" | "float32";
type DecodeDtype = "uint8" | "uint16" | "uint32" | "float32";

function validateDecodeDtypes(dtype: DecodeDtype, srcDtype: SourceDtype): void {
  if (dtype === "uint16" && srcDtype !== "uint16") {
    throw new Error("WebGPU dtype='uint16' requires a uint16 source.");
  }
  if (dtype === "uint32" && srcDtype !== "uint32") {
    throw new Error("WebGPU dtype='uint32' requires a uint32 source.");
  }
  if (dtype === "float32" && srcDtype !== "float32") {
    throw new Error("WebGPU dtype='float32' requires a float32 source.");
  }
  if (srcDtype === "float32" && dtype !== "float32") {
    throw new Error("WebGPU float32 source decode must request dtype='float32'.");
  }
}

interface RawInput {
  buffer: GPUBuffer;
  offset?: number;
  size?: number;
  owned?: boolean;
}

function rawInput(buffer: GPUBuffer, size?: number): RawInput {
  return { buffer, offset: 0, size, owned: true };
}

function isRawInput(raw: GPUBuffer | RawInput): raw is RawInput {
  return typeof (raw as RawInput).buffer !== "undefined";
}

function rawBinding(raw: GPUBuffer | RawInput): GPUBufferBinding {
  if (!isRawInput(raw)) return { buffer: raw };
  const binding: GPUBufferBinding = { buffer: raw.buffer };
  if (raw.offset) binding.offset = raw.offset;
  if (raw.size) binding.size = raw.size;
  return binding;
}

function releaseRaw(raw: GPUBuffer | RawInput): void {
  if (!isRawInput(raw)) {
    raw.destroy();
  } else if (raw.owned !== false) {
    raw.buffer.destroy();
  }
}

interface DecodeJob {
  stack: GPUBuffer;
  mode: number;
  record(enc: GPUCommandEncoder): void;
  releaseTemps(): void;
}
export interface Bslz4BatchProfile {
  variant: string;
  groups: number;
  specs: number;
  compressedMB: number;
  uploadMs: number;
  buildMs: number;
  gpuWaitMs: number;
  totalMs: number;
}

export interface Bslz4MaskedSumSpec extends Bslz4Spec {
  // Output scan offset. For a full scan this is usually the same as the source
  // offset; for a scan-region product it is the compact crop output offset.
  startScan: number;
  // Optional source-global offset and contiguous source-frame window. These make
  // a row-major scan crop exact without decoding unrelated scan positions.
  sourceStartScan?: number;
  frameStart?: number;
  frameCount?: number;
  // Optional fast-source contract: compressed/blockMeta already contain exactly
  // these detector bitshuffle blocks, in this order, for every frame. Used by
  // selected-block HDF5/range/sidecar sources to skip the CPU pack pass.
  selectedBlockIds?: number[];
}

export interface Bslz4MaskedSumProfile extends Bslz4BatchProfile {
  packMs: number;
  selectedBlocks: number;
  selectedPixels: number;
  selectedGroups: number;
  selectedBlockIds: number[];
}

// Product-first virtual image: one workgroup per frame LZ4-decodes only the detector blocks the
// mask touches into shared memory, rebuilds each selected count from all __NBITS__ bit planes
// (8 for uint8, 16 for uint16 sources) and sums it. The full block (__BB__ bytes) is decoded:
// a count above 255 has bits in the high planes, which the first __BE__ bytes do not hold.
// Per-frame sums stay in u32 (the host proves they fit) and round once to f32 at the store.
const MASKED_SUM_PIXEL_WGSL = `
@group(0) @binding(0) var<storage,read> raw: array<u32>;
@group(0) @binding(1) var<storage,read> blkMeta: array<u32>;     // compact selected-block coff,clen
@group(0) @binding(2) var<storage,read> idx: array<u32>;         // block-local detector pixels
@group(0) @binding(3) var<storage,read> blkTable: array<u32>;    // originalBlock, idxOffset, idxCount
@group(0) @binding(4) var<storage,read_write> sums: array<f32>;
@group(0) @binding(5) var<uniform> cfg: vec4<u32>;               // nFrames, gridX, selectedBlocks, startScan
var<workgroup> sh: array<atomic<u32>, __SH_WORDS__>;
var<workgroup> part: array<u32, __WG__>;
fn rraw(i:u32)->u32{return (raw[i>>2u]>>((i&3u)*8u))&0xffu;}
fn rsh(i:u32)->u32{return (atomicLoad(&sh[i>>2u])>>((i&3u)*8u))&0xffu;}
fn wsh(i:u32,v:u32){atomicOr(&sh[i>>2u], (v&0xffu)<<((i&3u)*8u));}
@compute @workgroup_size(__WG__)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>){
  let frm = wid.y*cfg.y + wid.x; let lx = lid.x;
  if(frm >= cfg.x){return;}
  var frameSum = 0u;
  for(var bi=0u; bi<cfg.z; bi=bi+1u){
    for(var w=lx; w<__SH_WORDS__u; w=w+__WG__){ atomicStore(&sh[w], 0u); }
  workgroupBarrier();
  let g = frm*cfg.z + bi;
  let coff=blkMeta[g*2u]; let cend=coff+blkMeta[g*2u+1u];
  var ci=coff; var di=0u; var working=true;
  loop{
    if(!working){break;}
    let tok=rraw(ci); ci=ci+1u; var nlit=tok>>4u;
    if(nlit==15u){loop{let bb=rraw(ci);ci=ci+1u;nlit=nlit+bb;if(bb!=255u){break;}}}
    let litCopy=min(nlit,__BB__-di);
    for(var k=lx; k<litCopy; k=k+__WG__){ wsh(di+k,rraw(ci+k)); }
    ci=ci+nlit; di=di+nlit;
    workgroupBarrier();
    if(ci>=cend || di>=__BB__){
      working=false;
    } else {
      let off=rraw(ci)|(rraw(ci+1u)<<8u); ci=ci+2u; var ml=4u+(tok&0xfu);
      if((tok&0xfu)==15u){loop{let bb=rraw(ci);ci=ci+1u;ml=ml+bb;if(bb!=255u){break;}}}
      let matchCopy=min(ml,__BB__-di);
      for(var j=lx; j<matchCopy; j=j+__WG__){ wsh(di+j,rsh(di-off+(j%off))); }
      di=di+ml;
      workgroupBarrier();
      if(ci>=cend || di>=__BB__){ working=false; }
    }
  }
  workgroupBarrier();
  let row = bi*3u;
  let idxOff = blkTable[row + 1u];
  let idxN = blkTable[row + 2u];
  var sum = 0u;
  for(var j=lx; j<idxN; j=j+__WG__){
    let p = idx[idxOff + j];
    let groupByte = p >> 3u;
    let pixBit = 1u << (p & 7u);
    var v = 0u;
    for(var b=0u; b<__NBITS__; b=b+1u){
      if((rsh(groupByte + b*__NPB__) & pixBit) != 0u){ v = v | (1u << b); }
    }
    sum = sum + v;
  }
  part[lx] = sum;
  workgroupBarrier();
  for(var s=__WG_HALF__; s>0u; s=s>>1u){
    if(lx < s){ part[lx] = part[lx] + part[lx+s]; }
    workgroupBarrier();
  }
  if(lx == 0u){ frameSum = frameSum + part[0]; }
  workgroupBarrier();
  }
  if(lx == 0u){ sums[cfg.w + frm] = f32(frameSum); }
}`;

// Same decode as MASKED_SUM_PIXEL_WGSL, but each thread sums one 8-pixel group: per bit plane
// countOneBits(plane byte & selected bits) << plane adds that plane's contribution of every
// selected pixel in the group at once. Fewer, wider reads for large scans.
const MASKED_SUM_GROUPMASK_WGSL = `
@group(0) @binding(0) var<storage,read> raw: array<u32>;
@group(0) @binding(1) var<storage,read> blkMeta: array<u32>;     // compact selected-block coff,clen
@group(0) @binding(2) var<storage,read> grpByte: array<u32>;     // block-local pixel byte groups
@group(0) @binding(3) var<storage,read> grpMask: array<u32>;     // selected low bits inside each group
@group(0) @binding(4) var<storage,read> blkTable: array<u32>;    // originalBlock, groupOffset, groupCount
@group(0) @binding(5) var<storage,read_write> sums: array<f32>;
@group(0) @binding(6) var<uniform> cfg: vec4<u32>;               // nFrames, gridX, selectedBlocks, startScan
var<workgroup> sh: array<atomic<u32>, __SH_WORDS__>;
var<workgroup> part: array<u32, __WG__>;
fn rraw(i:u32)->u32{return (raw[i>>2u]>>((i&3u)*8u))&0xffu;}
fn rsh(i:u32)->u32{return (atomicLoad(&sh[i>>2u])>>((i&3u)*8u))&0xffu;}
fn wsh(i:u32,v:u32){atomicOr(&sh[i>>2u], (v&0xffu)<<((i&3u)*8u));}
@compute @workgroup_size(__WG__)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>){
  let frm = wid.y*cfg.y + wid.x; let lx = lid.x;
  if(frm >= cfg.x){return;}
  var frameSum = 0u;
  for(var bi=0u; bi<cfg.z; bi=bi+1u){
    for(var w=lx; w<__SH_WORDS__u; w=w+__WG__){ atomicStore(&sh[w], 0u); }
  workgroupBarrier();
  let g = frm*cfg.z + bi;
  let coff=blkMeta[g*2u]; let cend=coff+blkMeta[g*2u+1u];
  var ci=coff; var di=0u; var working=true;
  loop{
    if(!working){break;}
    let tok=rraw(ci); ci=ci+1u; var nlit=tok>>4u;
    if(nlit==15u){loop{let bb=rraw(ci);ci=ci+1u;nlit=nlit+bb;if(bb!=255u){break;}}}
    let litCopy=min(nlit,__BB__-di);
    for(var k=lx; k<litCopy; k=k+__WG__){ wsh(di+k,rraw(ci+k)); }
    ci=ci+nlit; di=di+nlit;
    workgroupBarrier();
    if(ci>=cend || di>=__BB__){
      working=false;
    } else {
      let off=rraw(ci)|(rraw(ci+1u)<<8u); ci=ci+2u; var ml=4u+(tok&0xfu);
      if((tok&0xfu)==15u){loop{let bb=rraw(ci);ci=ci+1u;ml=ml+bb;if(bb!=255u){break;}}}
      let matchCopy=min(ml,__BB__-di);
      for(var j=lx; j<matchCopy; j=j+__WG__){ wsh(di+j,rsh(di-off+(j%off))); }
      di=di+ml;
      workgroupBarrier();
      if(ci>=cend || di>=__BB__){ working=false; }
    }
  }
  workgroupBarrier();
  let row = bi*3u;
  let idxOff = blkTable[row + 1u];
  let idxN = blkTable[row + 2u];
  var sum = 0u;
  for(var j=lx; j<idxN; j=j+__WG__){
    let groupByte = grpByte[idxOff + j];
    let bits = grpMask[idxOff + j];
    for(var b=0u; b<__NBITS__; b=b+1u){
      sum = sum + (countOneBits(rsh(groupByte + b*__NPB__) & bits) << b);
    }
  }
  part[lx] = sum;
  workgroupBarrier();
  for(var s=__WG_HALF__; s>0u; s=s>>1u){
    if(lx < s){ part[lx] = part[lx] + part[lx+s]; }
    workgroupBarrier();
  }
  if(lx == 0u){ frameSum = frameSum + part[0]; }
  workgroupBarrier();
  }
  if(lx == 0u){ sums[cfg.w + frm] = f32(frameSum); }
}`;

function useMaskedSumGroupMask(scanCount: number): boolean {
  const forced = (globalThis as { __QT_BSLZ4_MASKED_SUM_GROUPMASK?: unknown }).__QT_BSLZ4_MASKED_SUM_GROUPMASK;
  if (forced !== undefined) return forced !== false;
  return scanCount > 256 * 256;
}

function useMaskedSumPipeline(): boolean {
  return (globalThis as { __QT_BSLZ4_MASKED_SUM_PIPELINE?: unknown }).__QT_BSLZ4_MASKED_SUM_PIPELINE !== false;
}

function maskedSumWorkgroupSize(): 64 | 128 | 256 {
  const forced = Number((globalThis as { __QT_BSLZ4_MASKED_SUM_WG?: unknown }).__QT_BSLZ4_MASKED_SUM_WG);
  if (forced === 64 || forced === 128 || forced === 256) return forced;
  return 64;
}

function getMaskedSumPipe(device: GPUDevice, blockElems: number, nbits: 8 | 16, wgSize: 64 | 128 | 256, groupMask: boolean): GPUComputePipeline {
  const blockBytes = blockElems * nbits / 8;
  const code = (groupMask ? MASKED_SUM_GROUPMASK_WGSL : MASKED_SUM_PIXEL_WGSL)
    .replace(/__NPB__/g, `${blockElems / 8}u`)
    .replace(/__BB__/g, `${blockBytes}u`)
    .replace(/__NBITS__/g, `${nbits}u`)
    .replace(/__SH_WORDS__/g, `${Math.ceil(blockBytes / 4)}`)
    .replace(/__WG_HALF__/g, `${Math.floor(wgSize / 2)}u`)
    .replace(/__WG__/g, `${wgSize}u`);
  return cachedPipeline(device, code);
}

function maskedSumBlocks(mask: Uint32Array, badPixels: Uint32Array | undefined, blockElems: number): {
  blockIds: number[];
  blockTable: Uint32Array;
  groupBlockTable: Uint32Array;
  localIdx: Uint32Array;
  groupIdx: Uint32Array;
  groupMask: Uint32Array;
  selectedPixels: number;
  selectedGroups: number;
} {
  const bad = badPixels && badPixels.length ? new Set(badPixels) : null;
  const byBlock = new Map<number, number[]>();
  const groupByBlock = new Map<number, Map<number, number>>();
  for (let k = 0; k < mask.length; k++) {
    if (mask[k] === 0 || bad?.has(k)) continue;
    const block = Math.floor(k / blockElems);
    let local = byBlock.get(block);
    if (!local) {
      local = [];
      byBlock.set(block, local);
    }
    const localPix = k - block * blockElems;
    local.push(localPix);
    let groups = groupByBlock.get(block);
    if (!groups) {
      groups = new Map<number, number>();
      groupByBlock.set(block, groups);
    }
    const groupByte = localPix >> 3;
    const bit = 1 << (localPix & 7);
    groups.set(groupByte, (groups.get(groupByte) || 0) | bit);
  }
  const blockIds = Array.from(byBlock.keys()).sort((a, b) => a - b);
  const table = new Uint32Array(Math.max(1, blockIds.length) * 3);
  const groupTable = new Uint32Array(Math.max(1, blockIds.length) * 3);
  const idxParts: number[] = [];
  const groupParts: number[] = [];
  const groupMaskParts: number[] = [];
  for (let i = 0; i < blockIds.length; i++) {
    const pixels = byBlock.get(blockIds[i]) || [];
    const groups = Array.from((groupByBlock.get(blockIds[i]) || new Map<number, number>()).entries()).sort((a, b) => a[0] - b[0]);
    table[i * 3] = blockIds[i];
    table[i * 3 + 1] = idxParts.length;
    table[i * 3 + 2] = pixels.length;
    groupTable[i * 3] = blockIds[i];
    groupTable[i * 3 + 1] = groupParts.length;
    groupTable[i * 3 + 2] = groups.length;
    idxParts.push(...pixels);
    for (const [groupByte, groupMask] of groups) {
      groupParts.push(groupByte);
      groupMaskParts.push(groupMask);
    }
  }
  return {
    blockIds,
    blockTable: table,
    groupBlockTable: groupTable,
    localIdx: new Uint32Array(idxParts.length ? idxParts : [0]),
    groupIdx: new Uint32Array(groupParts.length ? groupParts : [0]),
    groupMask: new Uint32Array(groupMaskParts.length ? groupMaskParts : [0]),
    selectedPixels: idxParts.length,
    selectedGroups: groupParts.length,
  };
}

export function maskedSumBlockIds(mask: Uint32Array, badPixels: Uint32Array | undefined, blockElems: number): number[] {
  return maskedSumBlocks(mask, badPixels, blockElems).blockIds;
}

export function selectedBlockIdsCover(available: number[] | undefined, requested: number[]): boolean {
  if (!available) return false;
  const have = new Set(available);
  return requested.every((block) => have.has(block));
}

export function sliceMaskedSumSpecsByScanRegion(
  specs: Bslz4MaskedSumSpec[],
  scanRows: number,
  scanCols: number,
  scanRegion?: readonly [number, number, number, number] | null,
): Bslz4MaskedSumSpec[] {
  if (!scanRegion) return specs;
  const rows = Math.max(1, Math.round(scanRows));
  const cols = Math.max(1, Math.round(scanCols));
  const r0 = Math.max(0, Math.min(rows, Math.round(scanRegion[0])));
  const r1 = Math.max(0, Math.min(rows, Math.round(scanRegion[1])));
  const c0 = Math.max(0, Math.min(cols, Math.round(scanRegion[2])));
  const c1 = Math.max(0, Math.min(cols, Math.round(scanRegion[3])));
  if (r1 <= r0 || c1 <= c0) {
    throw new Error(`Invalid scan_region (${scanRegion.join(", ")}); expected (row_start, row_stop, col_start, col_stop) with non-empty bounds.`);
  }
  const cropCols = c1 - c0;
  const out: Bslz4MaskedSumSpec[] = [];
  for (let row = r0; row < r1; row++) {
    const rowSourceStart = row * cols + c0;
    const rowSourceStop = row * cols + c1;
    const rowOutputStart = (row - r0) * cropCols;
    for (const spec of specs) {
      const sourceStart = spec.sourceStartScan ?? spec.startScan;
      const sourceCount = spec.frameCount ?? spec.nFrames;
      const sourceStop = sourceStart + sourceCount;
      const start = Math.max(rowSourceStart, sourceStart);
      const stop = Math.min(rowSourceStop, sourceStop);
      if (stop <= start) continue;
      out.push({
        ...spec,
        startScan: rowOutputStart + (start - rowSourceStart),
        sourceStartScan: start,
        frameStart: (spec.frameStart ?? 0) + (start - sourceStart),
        frameCount: stop - start,
      });
    }
  }
  return out;
}

function maskedFrameWindow(spec: Bslz4MaskedSumSpec): { frameStart: number; frameCount: number } {
  const frameStart = Math.max(0, Math.min(spec.nFrames, Math.round(spec.frameStart ?? 0)));
  const requested = Math.round(spec.frameCount ?? (spec.nFrames - frameStart));
  const frameCount = Math.max(0, Math.min(spec.nFrames - frameStart, requested));
  return { frameStart, frameCount };
}

function sameBlockIds(a: number[] | undefined, b: number[]): boolean {
  if (!a || a.length !== b.length) return false;
  for (let i = 0; i < b.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

function sliceCompactedBslz4Frames(spec: Bslz4MaskedSumSpec, frameStart = 0, frameCount = spec.nFrames): Bslz4MaskedSumSpec {
  const selectedBlocks = spec.nBlocksPerFrame;
  const firstFrame = Math.max(0, Math.min(spec.nFrames, Math.round(frameStart)));
  const nFrames = Math.max(0, Math.min(spec.nFrames - firstFrame, Math.round(frameCount)));
  if (firstFrame === 0 && nFrames === spec.nFrames) return spec;
  if (nFrames === 0 || selectedBlocks === 0) {
    return { ...spec, compressed: new Uint8Array(0), blockMeta: new Uint32Array(2), nFrames: 0 };
  }
  const firstBlock = firstFrame * selectedBlocks;
  const lastBlock = firstBlock + nFrames * selectedBlocks - 1;
  const rangeStart = spec.blockMeta[firstBlock * 2];
  const rangeEnd = spec.blockMeta[lastBlock * 2] + spec.blockMeta[lastBlock * 2 + 1];
  const blockMeta = new Uint32Array(nFrames * selectedBlocks * 2);
  for (let i = 0; i < nFrames * selectedBlocks; i++) {
    const src = (firstBlock + i) * 2;
    blockMeta[i * 2] = spec.blockMeta[src] - rangeStart;
    blockMeta[i * 2 + 1] = spec.blockMeta[src + 1];
  }
  return {
    ...spec,
    compressed: spec.compressed.subarray(rangeStart, rangeEnd),
    blockMeta,
    nFrames,
    frameStart: 0,
    frameCount: nFrames,
  };
}

function compactSelectedBslz4Blocks(spec: Bslz4MaskedSumSpec, blockIds: number[], frameStart = 0, frameCount = spec.nFrames): Bslz4MaskedSumSpec {
  if (sameBlockIds(spec.selectedBlockIds, blockIds)) {
    return sliceCompactedBslz4Frames(spec, frameStart, frameCount);
  }
  const positions = blockIds.map((block) => spec.selectedBlockIds!.indexOf(block));
  const missing = positions.findIndex((pos) => pos < 0);
  if (missing >= 0) {
    throw new Error(`Selected-block sidecar is missing detector block ${blockIds[missing]}; use native HDF5 or a sidecar that covers this mask.`);
  }
  const firstFrame = Math.max(0, Math.min(spec.nFrames, Math.round(frameStart)));
  const nFrames = Math.max(0, Math.min(spec.nFrames - firstFrame, Math.round(frameCount)));
  if (nFrames === 0 || blockIds.length === 0) {
    return { ...spec, compressed: new Uint8Array(0), blockMeta: new Uint32Array(2), nFrames: 0, nBlocksPerFrame: blockIds.length, selectedBlockIds: blockIds };
  }
  const { compressed, blockMeta } = gatherCompressedBlocks(spec, positions, firstFrame, nFrames);
  return {
    ...spec,
    compressed,
    blockMeta,
    nFrames,
    nBlocksPerFrame: blockIds.length,
    selectedBlockIds: blockIds,
    frameStart: 0,
    frameCount: nFrames,
  };
}

/** Index the detector blocks `blockIds` of frames [frameStart, frameStart + frameCount) in place:
 * the block table points into one zero-copy range of the source bytes. Uploading that range
 * costs less than gathering every selected block into a new array on the CPU. The range starts
 * 8-byte aligned in its buffer so the staging copy takes the wide path. */
function indexBslz4Blocks(spec: Bslz4Spec, blockIds: number[], frameStart: number, frameCount: number): Bslz4Spec {
  const blockMeta = new Uint32Array(frameCount * blockIds.length * 2);
  let rangeStart = Number.POSITIVE_INFINITY, rangeEnd = 0;
  for (let f = 0; f < frameCount; f++) {
    for (let bi = 0; bi < blockIds.length; bi++) {
      const src = ((frameStart + f) * spec.nBlocksPerFrame + blockIds[bi]) * 2;
      const dst = (f * blockIds.length + bi) * 2;
      blockMeta[dst] = spec.blockMeta[src];
      blockMeta[dst + 1] = spec.blockMeta[src + 1];
      rangeStart = Math.min(rangeStart, blockMeta[dst]);
      rangeEnd = Math.max(rangeEnd, blockMeta[dst] + blockMeta[dst + 1]);
    }
  }
  rangeStart -= Math.min(rangeStart, (spec.compressed.byteOffset + rangeStart) & 7);
  for (let i = 0; i < blockMeta.length; i += 2) blockMeta[i] -= rangeStart;
  return { ...spec, compressed: spec.compressed.subarray(rangeStart, rangeEnd), blockMeta, nFrames: frameCount, nBlocksPerFrame: blockIds.length };
}

/** Copy blocks `sourceBlocks` (positions within each source frame) of `nFrames` frames
 * into one contiguous LZ4 stream with a rebased [offset, length] table, so the decoder
 * reads only the detector blocks a product needs. */
function gatherCompressedBlocks(
  spec: Bslz4Spec, sourceBlocks: number[], firstFrame: number, nFrames: number,
): { compressed: Uint8Array; blockMeta: Uint32Array } {
  const lengths: number[] = [];
  let total = 0;
  for (let f = 0; f < nFrames; f++) {
    const sourceFrame = firstFrame + f;
    for (const block of sourceBlocks) {
      const src = (sourceFrame * spec.nBlocksPerFrame + block) * 2;
      const clen = spec.blockMeta[src + 1];
      lengths.push(clen);
      total += clen;
    }
  }
  const compressed = new Uint8Array(total);
  const blockMeta = new Uint32Array(Math.max(1, nFrames * sourceBlocks.length * 2));
  let dst = 0;
  let m = 0;
  for (let f = 0; f < nFrames; f++) {
    const sourceFrame = firstFrame + f;
    for (let bi = 0; bi < sourceBlocks.length; bi++) {
      const src = (sourceFrame * spec.nBlocksPerFrame + sourceBlocks[bi]) * 2;
      const coff = spec.blockMeta[src];
      const clen = lengths[m++];
      blockMeta[(f * sourceBlocks.length + bi) * 2] = dst;
      blockMeta[(f * sourceBlocks.length + bi) * 2 + 1] = clen;
      compressed.set(spec.compressed.subarray(coff, coff + clen), dst);
      dst += clen;
    }
  }
  return { compressed, blockMeta };
}

interface MaskedSumJob {
  record(enc: GPUCommandEncoder): void;
  releaseTemps(): void;
}

function buildMaskedSumJob(
  device: GPUDevice,
  spec: Bslz4MaskedSumSpec,
  rawBuf: GPUBuffer,
  groupIdxBuf: GPUBuffer,
  groupMaskBuf: GPUBuffer | null,
  blockTableBuf: GPUBuffer,
  sums: GPUBuffer,
  selectedBlocks: number,
  pipe: GPUComputePipeline,
  useGroupMask: boolean,
): MaskedSumJob {
  const metaBuf = device.createBuffer({ size: Math.max(8, spec.blockMeta.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(metaBuf, 0, spec.blockMeta.buffer as ArrayBuffer, spec.blockMeta.byteOffset, spec.blockMeta.byteLength);
  const dispatchUnits = spec.nFrames;
  const gx = Math.min(dispatchUnits, MAX_WG), gy = Math.ceil(dispatchUnits / MAX_WG);
  const cfg = uniform(device, [spec.nFrames, gx, selectedBlocks, spec.startScan]);
  const bg = useGroupMask
    ? device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: rawBuf } },
      { binding: 1, resource: { buffer: metaBuf } },
      { binding: 2, resource: { buffer: groupIdxBuf } },
      { binding: 3, resource: { buffer: groupMaskBuf! } },
      { binding: 4, resource: { buffer: blockTableBuf } },
      { binding: 5, resource: { buffer: sums } },
      { binding: 6, resource: { buffer: cfg } },
    ] })
    : device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: rawBuf } },
      { binding: 1, resource: { buffer: metaBuf } },
      { binding: 2, resource: { buffer: groupIdxBuf } },
      { binding: 3, resource: { buffer: blockTableBuf } },
      { binding: 4, resource: { buffer: sums } },
      { binding: 5, resource: { buffer: cfg } },
    ] });
  return {
    record(enc) {
      const pass = enc.beginComputePass();
      pass.setPipeline(pipe);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(gx, gy);
      pass.end();
    },
    releaseTemps() { rawBuf.destroy(); metaBuf.destroy(); cfg.destroy(); },
  };
}

/** Virtual image straight from bslz4 chunks: decode only the masked detector blocks of each
 * frame and sum the selected counts, without a resident stack. ``srcDtype`` is the stored
 * dtype; every count is exact, and each frame's integer sum is rounded once to float32. */
export async function decodeBslz4MaskedSumBatch(
  specs: Bslz4MaskedSumSpec[],
  mask: Uint32Array,
  scanCount: number,
  srcDtype: "uint8" | "uint16",
  badPixels?: Uint32Array,
  groupSize = 8,
): Promise<{ device: GPUDevice; buffer: GPUBuffer; profile: Bslz4MaskedSumProfile } | null> {
  const device = await getGPUDevice();
  if (!device) return null;
  const blockElems = specs[0]?.blockElems || 4096;
  const selected = maskedSumBlocks(mask, badPixels, blockElems);
  const nbits = srcDtype === "uint8" ? 8 : 16;
  // Each frame's sum is accumulated in u32: refuse a mask whose largest possible sum could wrap.
  if (selected.selectedPixels * (2 ** nbits - 1) > 0xffff_ffff) {
    throw new Error(`Product-first WebGPU masked sum over ${selected.selectedPixels} ${srcDtype} pixels can exceed 2^32 counts per frame; load the resident stack instead.`);
  }
  const groupMask = useMaskedSumGroupMask(scanCount);
  const wgSize = maskedSumWorkgroupSize();
  const pipe = getMaskedSumPipe(device, blockElems, nbits, wgSize, groupMask);
  const profile: Bslz4MaskedSumProfile = {
    variant: `product-first-frame-masked-sum-${srcDtype}/${groupMask ? `groupmask-wg${wgSize}` : `pixel-wg${wgSize}`}/${specs.some((spec) => Boolean(spec.selectedBlockIds)) ? "selected-block-sidecar/" : ""}direct-f32/${useMaskedSumPipeline() ? "stagingPipeline" : "staging"}`,
    groups: 0,
    specs: specs.length,
    compressedMB: 0,
    uploadMs: 0,
    buildMs: 0,
    gpuWaitMs: 0,
    totalMs: 0,
    packMs: 0,
    selectedBlocks: selected.blockIds.length,
    selectedPixels: selected.selectedPixels,
    selectedGroups: selected.selectedGroups,
    selectedBlockIds: selected.blockIds,
  };
  const outF32 = device.createBuffer({ size: Math.max(4, scanCount * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const tAll = performance.now();
  const clear = device.createCommandEncoder();
  clear.clearBuffer(outF32, 0, Math.max(4, scanCount * 4));
  device.queue.submit([clear.finish()]);
  if (selected.selectedPixels === 0 || selected.blockIds.length === 0) {
    await device.queue.onSubmittedWorkDone();
    profile.totalMs = performance.now() - tAll;
    return { device, buffer: outF32, profile };
  }
  const idxSource = groupMask ? selected.groupIdx : selected.localIdx;
  const groupIdxBuf = device.createBuffer({ size: Math.max(4, idxSource.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(groupIdxBuf, 0, idxSource.buffer as ArrayBuffer, idxSource.byteOffset, idxSource.byteLength);
  const groupMaskBuf = groupMask
    ? device.createBuffer({ size: Math.max(4, selected.groupMask.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST })
    : null;
  if (groupMaskBuf) {
    device.queue.writeBuffer(groupMaskBuf, 0, selected.groupMask.buffer as ArrayBuffer, selected.groupMask.byteOffset, selected.groupMask.byteLength);
  }
  const tableSource = groupMask ? selected.groupBlockTable : selected.blockTable;
  const blockTableBuf = device.createBuffer({ size: Math.max(4, tableSource.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(blockTableBuf, 0, tableSource.buffer as ArrayBuffer, tableSource.byteOffset, tableSource.byteLength);
  type PreparedMaskedSumGroup = {
    jobs: MaskedSumJob[];
    compressedMB: number;
    packMs: number;
    uploadMs: number;
    buildMs: number;
  };
  const prepareGroup = async (groupSpecsInput: Bslz4MaskedSumSpec[]): Promise<PreparedMaskedSumGroup | null> => {
    const groupSpecs = groupSpecsInput.filter((spec) => maskedFrameWindow(spec).frameCount > 0);
    if (groupSpecs.length === 0) return null;
    const tPack = performance.now();
    const compacted = groupSpecs.map((spec) => {
      const { frameStart, frameCount } = maskedFrameWindow(spec);
      if (spec.selectedBlockIds) {
        return { ...compactSelectedBslz4Blocks(spec, selected.blockIds, frameStart, frameCount), startScan: spec.startScan };
      }
      return { ...indexBslz4Blocks(spec, selected.blockIds, frameStart, frameCount), startScan: spec.startScan };
    });
    const packMs = performance.now() - tPack;
    const compressedMB = compacted.reduce((n, s) => n + s.compressed.byteLength, 0) / 1e6;
    const tUpload = performance.now();
    const raws = await uploadViaStaging(device, compacted);
    const uploadMs = performance.now() - tUpload;
    const tBuild = performance.now();
    const jobs = compacted.map((spec, i) => buildMaskedSumJob(device, spec, raws[i], groupIdxBuf, groupMaskBuf, blockTableBuf, outF32, selected.blockIds.length, pipe, groupMask));
    const buildMs = performance.now() - tBuild;
    return { jobs, compressedMB, packMs, uploadMs, buildMs };
  };
  const submitGroup = (prepared: PreparedMaskedSumGroup): Promise<number> => {
    const enc = device.createCommandEncoder();
    for (const job of prepared.jobs) job.record(enc);
    device.queue.submit([enc.finish()]);
    const tGpu = performance.now();
    return device.queue.onSubmittedWorkDone().then(() => performance.now() - tGpu);
  };
  const acceptGroup = (prepared: PreparedMaskedSumGroup, gpuMs: number): void => {
    profile.packMs += prepared.packMs;
    profile.compressedMB += prepared.compressedMB;
    profile.uploadMs += prepared.uploadMs;
    profile.buildMs += prepared.buildMs;
    profile.gpuWaitMs += gpuMs;
    for (const job of prepared.jobs) job.releaseTemps();
    profile.groups++;
  };
  if (useMaskedSumPipeline()) {
    let nextPrepared: Promise<PreparedMaskedSumGroup | null> | null = specs.length ? prepareGroup(specs.slice(0, groupSize)) : null;
    for (let g = 0; g < specs.length; g += groupSize) {
      const prepared = await nextPrepared;
      const nextStart = g + groupSize;
      if (!prepared) {
        nextPrepared = nextStart < specs.length ? prepareGroup(specs.slice(nextStart, nextStart + groupSize)) : null;
        continue;
      }
      const done = submitGroup(prepared);
      nextPrepared = nextStart < specs.length ? prepareGroup(specs.slice(nextStart, nextStart + groupSize)) : null;
      acceptGroup(prepared, await done);
    }
	  } else {
	    for (let g = 0; g < specs.length; g += groupSize) {
	      const prepared = await prepareGroup(specs.slice(g, g + groupSize));
	      if (!prepared) continue;
	      acceptGroup(prepared, await submitGroup(prepared));
	    }
	  }
	  groupIdxBuf.destroy();
  groupMaskBuf?.destroy();
  blockTableBuf.destroy();
  profile.compressedMB = +profile.compressedMB.toFixed(1);
  profile.totalMs = performance.now() - tAll;
  return { device, buffer: outF32, profile };
}

// Reusable MAP_WRITE staging pool. mappedAtCreation allocates + zero-inits host-visible
// memory every call (~3.9 GB/s, ~1.7s for a 6.6 GB dataset); a persistent pool pays that
// once, then map/unmap-reuses, so the per-load upload becomes just the (fast, wide) CPU copy
// + a GPU-internal copyBufferToBuffer. Cleared on device loss.
let STAGING: GPUBuffer[] = [];
let STAGING_BYTES = 0;
let STAGING_READY: Promise<void> = Promise.resolve();
function ensureStaging(device: GPUDevice, count: number, bytes: number): void {
  if (STAGING.length >= count && STAGING_BYTES >= bytes) return;
  STAGING.forEach((b) => b.destroy());
  STAGING_BYTES = Math.max(bytes, STAGING_BYTES);
  STAGING = Array.from({ length: count }, () => device.createBuffer({ size: STAGING_BYTES, usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC }));
}
onGPULost(() => { STAGING = []; STAGING_BYTES = 0; STAGING_READY = Promise.resolve(); });

// Upload each spec's compressed bytes into a plain STORAGE buffer via the staging pool:
// map a pooled staging buffer (no per-load alloc), wide-copy the bytes in, then a GPU copy
// to the STORAGE buffer the decoder reads. Returns the raw buffers (caller destroys them).
async function uploadViaStaging(device: GPUDevice, specs: Bslz4Spec[]): Promise<GPUBuffer[]> {
  const align4 = (n: number) => Math.ceil(n / 4) * 4;
  const rawBufs = await fillStaging(device, specs);
  const enc = device.createCommandEncoder();
  specs.forEach((s, i) => enc.copyBufferToBuffer(STAGING[i], 0, rawBufs[i], 0, align4(s.compressed.byteLength)));
  device.queue.submit([enc.finish()]);   // ordered before the decode submit; next group's mapAsync waits on it
  STAGING_READY = device.queue.onSubmittedWorkDone().then(() => undefined);
  return rawBufs;
}

// Wait for the previous group's staging copy, then wide-copy each spec into its pooled
// staging buffer and allocate the STORAGE buffer the decoder will read.
async function fillStaging(device: GPUDevice, specs: Bslz4Spec[]): Promise<GPUBuffer[]> {
  const align4 = (n: number) => Math.ceil(n / 4) * 4;
  const maxBytes = specs.reduce((m, s) => Math.max(m, align4(s.compressed.byteLength)), 0);
  await STAGING_READY.catch(() => undefined);
  ensureStaging(device, specs.length, maxBytes);
  const rawBufs = specs.map((s) => device.createBuffer({ size: align4(s.compressed.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }));
  for (let i = 0; i < specs.length; i++) {
    const s = specs[i];
    const sz = align4(s.compressed.byteLength);
    await STAGING[i].mapAsync(GPUMapMode.WRITE, 0, sz);   // reused buffer: maps without re-alloc
    copyWide(STAGING[i].getMappedRange(0, sz), s.compressed);
    STAGING[i].unmap();
  }
  return rawBufs;
}

// Decode N bslz4 specs into N packed GPU stack buffers, batching the GPU work in groups so
// at most `groupSize` chunks' transient buffers are live at once - one submit + one await
// per group. The pipelined default prepares (host copy + upload) the next group while the
// current one decodes; __BSLZ4_PIPELINE_STAGING = false runs the groups strictly in turn.
export async function decodeBslz4Batch(specs: Bslz4Spec[], dtype: DecodeDtype = "uint8", srcDtype: SourceDtype = "uint16", groupSize = 14): Promise<{ device: GPUDevice; buffers: GPUBuffer[]; mode: number; profile: Bslz4BatchProfile } | null> {
  validateDecodeDtypes(dtype, srcDtype);
  const device = await getGPUDevice();
  if (!device) return null;
  const { variant, build } = decodeJobBuilder(dtype, srcDtype);
  const pipelineStaging = bslz4PipelineStaging();
  const buffers: GPUBuffer[] = []; let mode = 0;
  const profile: Bslz4BatchProfile = {
    variant: `${variant}/${pipelineStaging ? "stagingPipeline" : "staging"}`,
    groups: 0,
    specs: specs.length,
    compressedMB: +(specs.reduce((n, s) => n + s.compressed.byteLength, 0) / 1e6).toFixed(1),
    uploadMs: 0,
    buildMs: 0,
    gpuWaitMs: 0,
    totalMs: 0,
  };
  const tAll = performance.now();
  type PreparedDecodeGroup = { jobs: DecodeJob[]; uploadMs: number; buildMs: number };
  const prepare = async (groupSpecs: Bslz4Spec[]): Promise<PreparedDecodeGroup> => {
    const tUpload = performance.now();
    const raws = await uploadViaStaging(device, groupSpecs);
    const uploadMs = performance.now() - tUpload;
    const tBuild = performance.now();
    const jobs = groupSpecs.map((s, i) => build(device, s, raws[i]));
    return { jobs, uploadMs, buildMs: performance.now() - tBuild };
  };
  const submit = (prepared: PreparedDecodeGroup): Promise<number> => {
    const enc = device.createCommandEncoder();
    for (const j of prepared.jobs) j.record(enc);
    device.queue.submit([enc.finish()]);
    const tGpu = performance.now();
    return device.queue.onSubmittedWorkDone().then(() => performance.now() - tGpu);
  };
  const accept = (prepared: PreparedDecodeGroup, gpuMs: number): void => {
    profile.gpuWaitMs += gpuMs;
    profile.uploadMs += prepared.uploadMs;
    profile.buildMs += prepared.buildMs;
    for (const j of prepared.jobs) { j.releaseTemps(); buffers.push(j.stack); mode = j.mode; }
    profile.groups++;
  };
  if (pipelineStaging) {
    let next: Promise<PreparedDecodeGroup> | null = specs.length ? prepare(specs.slice(0, groupSize)) : null;
    for (let g = 0; g < specs.length; g += groupSize) {
      const prepared = await next!;
      const done = submit(prepared);
      const nextStart = g + groupSize;
      next = nextStart < specs.length ? prepare(specs.slice(nextStart, nextStart + groupSize)) : null;
      accept(prepared, await done);
    }
  } else {
    for (let g = 0; g < specs.length; g += groupSize) {
      const prepared = await prepare(specs.slice(g, g + groupSize));
      accept(prepared, await submit(prepared));
    }
  }
  profile.totalMs = performance.now() - tAll;
  return { device, buffers, mode, profile };
}

// Decode a bslz4 stack to a packed GPU buffer ([scanPos][detPixel]). dtype "uint8"
// (clip 0-255, 4 px/u32, offline default - half the memory), "uint16"
// (lossless, 2 px/u32), or "uint32" (lossless, 1 px/u32). Layout matches
// DetectorCompute's sample() for that mode exactly.
// Returns null if WebGPU is unavailable. Throws (validation) only on misuse.
export async function decodeBslz4ToStack(spec: Bslz4Spec, dtype: DecodeDtype = "uint8", srcDtype: SourceDtype = "uint16"): Promise<{ device: GPUDevice; buffer: GPUBuffer; mode: number } | null> {
  validateDecodeDtypes(dtype, srcDtype);
  const device = await getGPUDevice();
  if (!device) return null;
  const job = decodeJobBuilder(dtype, srcDtype).build(device, spec);
  const enc = device.createCommandEncoder();
  job.record(enc);
  device.queue.submit([enc.finish()]);
  await device.queue.onSubmittedWorkDone();
  job.releaseTemps();
  return { device, buffer: job.stack, mode: job.mode };
}

function uniform(device: GPUDevice, values: number[]): GPUBuffer {
  const buffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(buffer, 0, new Uint32Array(values).buffer);
  return buffer;
}
