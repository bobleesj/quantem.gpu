// Each BF/frequency pair supplies all 28 derivatives. Generate scalar WGSL
// expressions so the compiler can keep the complex powers and sums in registers.
const orders = [[2,1],[2,3],[3,0],[3,2],[3,4],[4,1],[4,3],[4,5],[5,0],[5,2],[5,4],[5,6]];
const coordinates = ["params.factor*100.0*a2", "params.factor*100.0*z2.x",
  "params.factor*100.0*z2.y", "", "", ""];
for (const [n,m] of orders) {
  const radial = (n+1-m)/2;
  for (const part of (m ? ["x","y"] : ["x"])) {
    coordinates.push([m ? `z${m}.${part}` : "1.0", ...Array(radial).fill("r2")].join("*"));
  }
}
function basis(name: string, point: string) {
  return `let ${name}_a = ${point} * params.wavelength;
    let ${name}_a2 = dot(${name}_a,${name}_a);
    let ${name}_low2 = cmul(${name}_a,${name}_a);
    let ${name}_z1 = ${name}_a / params.semiangle_rad;
    let ${name}_r2 = dot(${name}_z1,${name}_z1);
    ${Array.from({length:5},(_,i)=>`let ${name}_z${i+2}=cmul(${name}_z${i+1},${name}_z1);`).join("\n")}`;
}
function term(j: number, name: string) {
  return coordinates[j].replace(/a2|r2|z[1-6]/g, t =>
    name + "_" + (j < 3 && t === "z2" ? "low2" : t));
}
export const agreementGradient = (stride: number) => `
@group(0) @binding(16) var<storage, read_write> gradientTerms: array<f32>;
fn sincPrime(x: f32) -> f32 {
  if (abs(x)<0.01) { return x*(-1.0/3.0 + x*x/30.0); }
  return (x*cos(x)-sin(x))/(x*x);
}
@compute @workgroup_size(64)
fn modelGradient(@builtin(global_invocation_id) id: vec3<u32>) {

  let i = id.x;
  let size = arrayLength(&fitFrequencies);
  if (i>=size) { return; }
  let sums = agreementTerms[i].xyz;
  if (sums.z<=0.0) { return; }
  let width=params.n/2u+1u;
  let q=vec2<f32>(qx1d[fitFrequencies[i].x/width],qy1d[fitFrequencies[i].x%width]);
  let projection=sums.xy/sums.z;
  let shift=6.283185307*dot(q,vec2<f32>(params.tilt_row_rad,params.tilt_col_rad));
  ${coordinates.map((_,j)=>`var grad${j}=0.0;`).join("\n")}
  for (var local=0u; local<params.chunk_bf; local++) {
    let bf=params.bf_offset+local;
    let bg=bfGeom[bf];
    let km=bg.xy-q; let kp=bg.xy+q;
    let gm=compute_geometry(km.x,km.y); let gp=compute_geometry(kp.x,kp.y);
    if (gm.w==0.0 && gp.w==0.0) { continue; }
    let base=chi_full(bg.x,bg.y);
    let m=chi_full(km.x,km.y)-base; let p=chi_full(kp.x,kp.y)-base;
    let a=bg.w*gm.w*vec2<f32>(cos(m),-sin(m));
    let b=bg.w*gp.w*vec2<f32>(cos(p),sin(p));
    let rm=-params.factor*(gm.x-bg.z)-shift;
    let rp=params.factor*(gp.x-bg.z)-shift;
    let xm=0.5*params.thickness*rm; let xp=0.5*params.thickness*rp;
    let t1=a*thick_sinc(xm); let t2=b*thick_sinc(xp);
    let g=gqk[local*${stride}u+fitFrequencies[i].y];
    let residual=g-cmul(projection,t1-t2);
    // Re(conj(residual) * projection * dGamma) is linear in dGamma.
    let adjoint=2.0*cmul(residual,vec2<f32>(projection.x,-projection.y));
    let wm=dot(adjoint,vec2<f32>(t1.y,-t1.x));
    let wp=dot(adjoint,vec2<f32>(t2.y,-t2.x));
    let sm=dot(adjoint,a)*sincPrime(xm);
    let sp=dot(adjoint,b)*sincPrime(xp);
    let tiltWeight=(sm-sp)*(-0.5*params.thickness*6.283185307*0.005);
    grad3+=tiltWeight*q.x; grad4+=tiltWeight*q.y;
    grad5+=50.0*(sm*rm-sp*rp);
    ${basis("baseK","bg.xy")}
    ${basis("minusK","km")}
    ${basis("plusK","kp")}
    ${coordinates.map((expr,j)=>expr ?
      `grad${j}+=wm*((${term(j,"minusK")})-(${term(j,"baseK")}))+wp*((${term(j,"plusK")})-(${term(j,"baseK")}));` : "").join("\n")}
  }
  ${coordinates.map((_,j)=>`gradientTerms[${j}u*size+i]+=grad${j};`).join("\n")}
}
`;
