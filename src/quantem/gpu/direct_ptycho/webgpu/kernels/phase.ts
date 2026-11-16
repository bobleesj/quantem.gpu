// Phase-only weak-object inversion. Each measured frequency owns a small
// Hermitian normal system coupling the output frequencies that alias onto it.
export function phaseLayout(n: number, factor: number, rowStep: number, colStep: number, cutoff: number) {
  const size = n * factor;
  const rows: number[][] = [];
  let aliases = 0;
  for (let r = 0; r < n; r++) for (let c = 0; c < n / 2; c++) {
    if (r === n / 2 || (c === 0 && (r === 0 || r >= n / 2))) continue;
    const row = [r * (n / 2 + 1) + c];
    for (let ar = r; ar < size; ar += n) for (let ac = c; ac < size; ac += n) {
      const x = Math.fround((ar < size / 2 ? ar : ar - size) * rowStep);
      const y = Math.fround((ac < size / 2 ? ac : ac - size) * colStep);
      if (Math.fround(Math.fround(x*x) + Math.fround(y*y)) < Math.fround(cutoff*cutoff)) row.push(ar*size + ac);
    }
    if (row.length === 1) continue;
    aliases = Math.max(aliases, row.length - 1);
    rows.push(row);
  }
  if (!aliases) throw new Error("The calibrated aperture supports no output frequencies.");
  const map = new Uint32Array(rows.length * (aliases + 1)).fill(0xffffffff);
  rows.forEach((row,i)=>map.set(row,i*(aliases+1)));
  return {map, aliases, count: rows.length};
}

export function phaseNormalShader(n: number, size: number, aliases: number, rowStep: number, colStep: number) {
  const entries = aliases*(aliases+1)/2+aliases;
  return `
@compute @workgroup_size(64)
fn phaseDetectorCounts(@builtin(global_invocation_id) id:vec3<u32>) {
  if(id.x<params.chunk_bf){phaseStack[params.bf_offset+id.x]=gqk[id.x*${n*(n/2+1)}u].x;}
}
@group(0) @binding(17) var<storage,read> aliasMap:array<u32>;
@compute @workgroup_size(64)
fn accumulatePhase(@builtin(global_invocation_id) id:vec3<u32>) {
  let f=id.x;
  let base=f*${aliases+1}u;
  if(base>=arrayLength(&aliasMap)){return;}
  var qs:array<vec2<f32>,${aliases}>;
  var count=0u;
  for(var j=0u;j<${aliases}u;j++){
    let p=aliasMap[base+1u+j];
    if(p==0xffffffffu){break;}
    let x=i32(p/${size}u); let y=i32(p%${size}u);
    qs[j]=vec2<f32>(f32(select(x,x-${size},x>=${size/2}))*${rowStep},
                    f32(select(y,y-${size},y>=${size/2}))*${colStep});
    count++;
  }
  var sums:array<vec2<f32>,${entries}>;
  for(var k=0u;k<params.chunk_bf;k++){
    let bf=k+params.bf_offset; let bg=bfGeom[bf]; if(bg.w==0.0){continue;}
    let g=gqk[k*${n*(n/2+1)}u+aliasMap[base]];
    var v:array<vec2<f32>,${aliases}>;
    for(var a=0u;a<count;a++){
      let t=transfer(qs[a].x,qs[a].y,bg.x,bg.y,bg.z,cachedProbe[bf]); v[a]=vec2<f32>(-t.y,t.x);
    }
    for(var a=0u;a<count;a++){
      let conjA=vec2<f32>(v[a].x,-v[a].y);
      for(var b=a;b<count;b++){
        let e=a*(2u*${aliases}u+1u-a)/2u+b-a;
        sums[e]+=cmul(conjA,v[b]);
      }
      sums[${entries-aliases}u+a]+=cmul(conjA,g);
    }
  }
  for(var j=0u;j<${entries}u;j++){stage[f*${entries}u+j]+=sums[j];}
}`;
}

// Float64 Cholesky, matching the native Metal host solve. Scatter phi and
// its Hermitian mirror before the GPU inverse FFT.
export function solvePhase(layout: ReturnType<typeof phaseLayout>, sums:Float32Array, size:number,
    factor:number, rowStep:number, colStep:number, dose:number, strength:number) {
  const {map,aliases:m,count}=layout, gram=m*(m+1)/2;
  const field=new Float32Array(size*size*2);
  const l=new Float64Array(2*m*m), y=new Float64Array(2*m);
  for(let f=0;f<count;f++){
    const base=f*(m+1), start=2*f*(gram+m);
    let dim=0;
    while(dim<m && map[base+1+dim]!==0xffffffff) dim++;
    for(let i=0;i<dim;i++){
      const p=map[base+1+i], r=Math.floor(p/size), c=p%size;
      const qr=Math.fround((r<size/2?r:r-size)*rowStep), qc=Math.fround((c<size/2?c:c-size)*colStep);
      const prior=Math.fround(1+Math.fround(Math.fround(qr*qr)+Math.fround(qc*qc)));
      for(let j=0;j<=i;j++){
        const e=start+2*(j*(2*m+1-j)/2+i-j);
        let re=sums[e], im=-sums[e+1];
        for(let k=0;k<j;k++){
          const a=2*(i*m+k), b=2*(j*m+k);
          re-=l[a]*l[b]+l[a+1]*l[b+1];im-=l[a+1]*l[b]-l[a]*l[b+1];
        }
        const a=2*(i*m+j);
        if(i===j){
          const d=re+strength*Math.fround(prior*prior);
          if(!(d>0)) throw new Error("Phase normal system is not positive definite.");
          l[a]=Math.sqrt(d);l[a+1]=0;
        }else{const d=l[2*(j*m+j)];l[a]=re/d;l[a+1]=im/d;}
      }
      let re=sums[start+2*(gram+i)]/dose, im=sums[start+2*(gram+i)+1]/dose;
      for(let k=0;k<i;k++){
        const a=2*(i*m+k);re-=l[a]*y[2*k]-l[a+1]*y[2*k+1];im-=l[a]*y[2*k+1]+l[a+1]*y[2*k];
      }
      y[2*i]=re/l[2*(i*m+i)];y[2*i+1]=im/l[2*(i*m+i)];
    }
    for(let i=dim-1;i>=0;i--){
      let re=y[2*i],im=y[2*i+1];
      for(let k=i+1;k<dim;k++){
        const a=2*(k*m+i);re-=l[a]*y[2*k]+l[a+1]*y[2*k+1];im-=l[a]*y[2*k+1]-l[a+1]*y[2*k];
      }
      y[2*i]=re/l[2*(i*m+i)];y[2*i+1]=im/l[2*(i*m+i)];
      const p=map[base+1+i],mirror=((size-Math.floor(p/size))%size)*size+(size-p%size)%size;
      field[2*p]=field[2*mirror]=Math.fround(y[2*i])*factor*factor;
      field[2*p+1]=Math.fround(y[2*i+1])*factor*factor;field[2*mirror+1]=-field[2*p+1];
    }
  }
  return field;
}
