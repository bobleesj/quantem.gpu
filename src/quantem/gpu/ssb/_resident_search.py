"""Device-resident multivariate TPE and trust-region Newton joint fitting.

Only shape/scheduling metadata crosses to the host during optimization. Trial
points, model scores, density ranking, finite differences, eigensolves and
trust-region decisions remain on the selected GPU. Results are downloaded
once for the public fit record. The search retains groups of eight, 24 TPE
candidates, the startup budget, and three distinct Newton starts.

Objective per backend: ``fit_resident`` (tilt=True) maximizes the
least-squares agreement of the thick-sample SSB model with G(q, k) on CUDA and
MPS. ``fit_standard_resident`` (tilt=False) is the same search and objective
with the sample tilt and depth spread held at 0; MPS uses it, while CUDA's
tilt=False still minimizes the phase variance (``cuda/optimizer.py``).
"""

import math
import re
import time
from functools import lru_cache

import numpy as np

from quantem.gpu.ssb.thick_sample_fit import DEPTH_SPREAD_RANGE_A
from quantem.gpu.ssb._execution import checkpoint
from quantem.gpu.ssb._resident_precision import HEADER


class _Device:
    """Compile the same decision kernels for Metal and CUDA."""

    def __init__(self, backend: str):
        self.backend = backend
        if backend == "mps":
            from quantem.gpu.ssb.mps.hardware import require_mlx

            self.xp = require_mlx()
        elif backend == "cuda":
            import cupy

            self.xp = cupy
        else:
            raise ValueError(
                f"Resident joint fitting requires mps or cuda, got {backend!r}."
            )

    def pack(self, values):
        # Public scalar configuration crosses once, losslessly, before fitting.
        values = np.asarray(values, dtype=np.float64)
        high = values.astype(np.float32)
        middle = (values - high).astype(np.float32)
        low = (values - high - middle).astype(np.float32)
        return self.xp.array(np.stack((high, middle, low), axis=-1))

    def host(self, values):
        self.sync(values)
        raw = np.asarray(values) if self.backend == "mps" else values.get()
        return raw.astype(np.float64).sum(axis=-1)

    def sync(self, *values):
        if self.backend == "mps":
            self.xp.eval(*values)
        else:
            self.xp.cuda.get_current_stream().synchronize()

    def run(self, name, inputs, shapes, body, *, count=1, constants=""):
        # Counts and dispatch offsets must not compile a new shader for each
        # adaptive batch or Newton iteration. They are launch metadata only.
        launch = []

        def parameterize(match):
            key, value = match.groups()
            if key not in {
                "COUNT",
                "DONE",
                "BATCH",
                "OFFSET",
                "ACTIVE",
                "ITERATION",
                "TOTAL",
                "BAND",
            }:
                return match.group(0)
            slot = len(launch)
            launch.append(int(value))
            return f"#define {key} int(launch_constants[{slot}])"

        constants = re.sub(
            r"#define ([A-Z_]+) ([0-9]+)(?![0-9a-z])", parameterize, constants
        )
        if launch:
            inputs = {
                **inputs,
                "launch_constants": self.xp.array(launch, dtype=self.xp.float32),
            }
        names = tuple(inputs)
        result = _kernel(self.backend, name, names, len(shapes), body, constants)
        if self.backend == "mps":
            return result(
                inputs=list(inputs.values()),
                grid=(count, 1, 1),
                threadgroup=(min(128, count), 1, 1),
                output_shapes=shapes,
                output_dtypes=[self.xp.float32] * len(shapes),
            )
        outputs = [self.xp.empty(shape, dtype=self.xp.float32) for shape in shapes]
        result(((count + 127) // 128,), (128,), (*inputs.values(), *outputs))
        return outputs


@lru_cache(maxsize=512)
def _kernel(backend, name, names, n_outputs, body, constants):
    outputs = [f"out{i}" for i in range(n_outputs)]
    if backend == "mps":
        from quantem.gpu.ssb.mps.hardware import require_mlx

        return require_mlx().fast.metal_kernel(
            name=name,
            input_names=list(names),
            output_names=outputs,
            header=HEADER + constants,
            source="uint index=thread_position_in_grid.x;\n" + body,
            compile_options={"math_mode": "safe"},
        )
    import cupy as cp

    signature = ",".join(
        [f"const float *{n}" for n in names] + [f"float *{n}" for n in outputs]
    )
    source = (
        HEADER + constants + f'\nextern "C" __global__ void {name}({signature}){{\n'
    )
    source += "unsigned int index=blockIdx.x*blockDim.x+threadIdx.x;\n" + body + "\n}"
    # A bounded six-dimensional solve contains nested fixed-size loops. Letting
    # NVRTC inline and unroll every trust-region solve makes cold compilation
    # grow by minutes without providing useful parallelism in this one thread.
    for function in ("trust_step", "bounded_step"):
        source = source.replace(
            f"FN void {function}", f"__device__ __noinline__ void {function}"
        )
    source = re.sub(r"\bfor\s*\(", "\n#pragma unroll 1\nfor(", source)
    return cp.RawKernel(source, name, options=("--std=c++17", "--fmad=false"))


# MT19937 and the 53-bit uniform construction used by the existing sampler.
# Startup and TPE use separate streams, both initialized with the fit seed.
RANDOM_HEADER = r"""
FN unsigned int next_mt(LOCAL unsigned int *state, LOCAL int &position) {
    if(position==624){
        for(int i=0;i<624;i++){
            unsigned int y=(state[i]&0x80000000u)|(state[(i+1)%624]&0x7fffffffu);
            state[i]=state[(i+397)%624]^(y>>1)^((y&1u)?0x9908b0dfu:0u);
        }
        position=0;
    }
    unsigned int y=state[position++];y^=y>>11;y^=(y<<7)&0x9d2c5680u;y^=(y<<15)&0xefc60000u;y^=y>>18;return y;
}
"""

RANDOM_BODY = r"""
if(index>0)return;
unsigned int state[624];state[0]=SEED;
for(int i=1;i<624;i++)state[i]=1812433253u*(state[i-1]^(state[i-1]>>30))+i;
int position=624;
for(int i=0;i<TOTAL;i++){
    unsigned int a=next_mt(state,position)>>5,b=next_mt(state,position)>>6;
    out0[4*i]=float(a>>16);out0[4*i+1]=float(a&65535);
    out0[4*i+2]=float(b>>16);out0[4*i+3]=float(b&65535);
}
"""

UNIFORM_BODY = r"""
if(index>=TOTAL)return;
int i=int(index);
Real a=Real(words[4*i])*Real(65536)+Real(words[4*i+1]);
Real b=Real(words[4*i+2])*Real(65536)+Real(words[4*i+3]);
store_real(out0,i,(a*Real(67108864.0f)+b)/Real(9007199254740992.0f));
"""

DENSITY_BODY = r"""
if(index>=DONE+1)return;
int i=int(index), good_count=min(25,(DONE+9)/10), rank=0, group_order=0;
if(i<DONE)rank=int(ranks[i]);
bool good=rank<good_count;
if(i==DONE)good=false;
if(i<DONE)for(int j=0;j<i;j++)if((int(ranks[j])<good_count)==good)group_order++;
int n=good?good_count:DONE-good_count;
Real sigma=max_r((Real(1)/Real(5))*exp_r(-log_r(Real(max(n,1)))/Real(DIMENSIONS+4)),
                 Real(1)/Real(min(100,n+2)));
if(i==DONE)sigma=Real(1);
Real norm=Real(0);
for(int d=0;d<6;d++){
    if((FIXED && d<3)||(THIN && d>=3))continue;
    Real mu=i<DONE?load_real(history,i*6+d):Real(0.5f);
    Real mass=cdf_r((Real(1)-mu)/sigma)-cdf_r(-mu/sigma);
    norm=norm+log_r(mass)+log_r(sigma);
}
Real weight(1);
if(n>=25 && group_order<n-25){
    weight=n==26?Real(1)/Real(n):Real(1)/Real(n)+(Real(1)-Real(1)/Real(n))*Real(group_order)/Real(n-26);
}
store_real(out0,i*5,good?Real(1):Real(0));
store_real(out0,i*5+1,sigma);store_real(out0,i*5+2,norm);
store_real(out0,i*5+3,weight);store_real(out0,i*5+4,Real(group_order));
"""

SAMPLE_BODY = r"""
if(index>=BATCH*24*6)return;
int d=int(index)%6,candidate=int(index)/6,b=candidate/24,c=candidate%24;
if((FIXED && d<3)||(THIN && d>=3)){store_real(out0,int(index),Real(0));return;}
int below=min(25,(DONE+9)/10);
Real choose=load_real(random,(OFFSET+b)*(DIMENSIONS+1)*24+c)*Real(below+1);
int chosen=int(scalar(choose)),selected=DONE;
for(int i=0;i<DONE;i++)if(load_real(density,i*5)>Real(0.5f) && int(scalar(load_real(density,i*5+4)))==chosen)selected=i;
Real sigma=selected==DONE?Real(1):load_real(density,selected*5+1);
Real mu=selected==DONE?Real(0.5f):load_real(history,selected*6+d);
// Optuna's multivariate dimensions are alphabetical after startup.
int order=d<3?d:(d==3?5:(d==4?4:3));
if(FIXED)order-=3;
Real q=load_real(random,(OFFSET+b)*(DIMENSIONS+1)*24+24+order*24+c);
Real lo=cdf_r(-mu/sigma),hi=cdf_r((Real(1)-mu)/sigma);
Real p=clip_r(lo+q*(hi-lo),Real(1e-18f),Real(1)-Real(1e-18f));
store_real(out0,int(index),clip_r(mu+sigma*inverse_cdf_r(p),Real(0),Real(1)));
"""

SELECT_BODY = r"""
if(index>=BATCH)return;
int b=int(index),best=0;
for(int c=1;c<24;c++)if(load_real(scores,b*24+c)>load_real(scores,b*24+best))best=c;
for(int d=0;d<6;d++)store_real(out0,b*6+d,load_real(candidates,(b*24+best)*6+d));
"""

DECODE_BODY = r"""
if(index>=COUNT*6)return;
int d=int(index)%6;
Real lo=load_real(bounds,d),hi=load_real(bounds,6+d);
store_real(out0,int(index),lo+load_real(points,int(index))*(hi-lo));
"""

TRIAL_BODY = r"""
if(index>=COUNT)return;
int b=int(index);Real angle=load_real(points,b*6+2)*Real(2);
out0[b*7]=scalar(load_real(points,b*6));out0[b*7+1]=scalar(load_real(points,b*6+1));
out0[b*7+2]=scalar(cos_r(angle));out0[b*7+3]=scalar(sin_r(angle));
out0[b*7+4]=scalar(load_real(points,b*6+5));
out0[b*7+5]=scalar(load_real(points,b*6+3)/Real(1000));
out0[b*7+6]=scalar(load_real(points,b*6+4)/Real(1000));
"""


def _random(device, seed, trials, width):
    constants = (
        RANDOM_HEADER
        + f"\n#define SEED {seed & 0xFFFFFFFF}u\n#define TOTAL {trials * width}\n"
    )
    # MT state transitions are serial. The expensive extended-precision
    # conversion is independent for every number, so give it a GPU thread.
    # Four exactly represented 16-bit words retain the original 53-bit stream.
    words = device.run(
        "ssb_joint_random_words",
        {},
        [(trials * width, 4)],
        RANDOM_BODY,
        constants=constants,
    )[0]
    return device.run(
        "ssb_joint_uniform",
        {"words": words},
        [(trials, width, 3)],
        UNIFORM_BODY,
        count=trials * width,
        constants=f"\n#define TOTAL {trials * width}\n",
    )[0]


def _decode(device, points, bounds):
    return device.run(
        "ssb_joint_decode",
        {"points": points, "bounds": bounds},
        [points.shape],
        DECODE_BODY,
        count=points.shape[0] * 6,
        constants=f"\n#define COUNT {points.shape[0]}\n",
    )[0]


def _trial(device, points):
    return device.run(
        "ssb_joint_trial",
        {"points": points},
        [(points.shape[0], 7)],
        TRIAL_BODY,
        count=points.shape[0],
        constants=f"\n#define COUNT {points.shape[0]}\n",
    )[0]


RANK_BODY = r"""
if(index>=COUNT)return;
int i=int(index),rank=0;Real value=load_real(scores,i);
for(int j=0;j<COUNT;j++){
    Real other=load_real(scores,j);
    if(other>value || (!(other<value)&&j<i))rank++;
}
out0[i]=float(rank);
"""

NEWTON_HEADER = r"""
FN void encode_point(LOCAL Real *p,LOCAL Real *z){
    for(int d=0;d<6;d++)z[d]=p[d];
    z[1]=p[1]*cos_r(Real(2)*p[2]);z[2]=p[1]*sin_r(Real(2)*p[2]);
}
FN void decode_point(LOCAL Real *z,LOCAL Real *p){
    for(int d=0;d<6;d++)p[d]=z[d];
    p[1]=sqrt_r(z[1]*z[1]+z[2]*z[2]);p[2]=atan2_r(z[2],z[1])/Real(2);p[5]=abs_r(z[5]);
}
FN void trust_step(LOCAL Real *g,LOCAL Real *h,LOCAL Real *steps,int n,Real radius,LOCAL Real *result){
    Real a[36],v[36],along[6],scaled_g[6];
    for(int i=0;i<n;i++){
        scaled_g[i]=g[i]*steps[i];
        for(int j=0;j<n;j++){a[i*6+j]=h[i*6+j]*steps[i]*steps[j];v[i*6+j]=Real(i==j?1:0);}
    }
    // Cyclic Jacobi for a small symmetric matrix; no host LAPACK fallback.
    for(int sweep=0;sweep<32;sweep++){
        Real maximum(0);
        for(int p=0;p<n;p++)for(int q=p+1;q<n;q++){
            Real off=a[p*6+q];maximum=max_r(maximum,abs_r(off));
            if(abs_r(off)<Real(1e-25f))continue;
            Real tau=(a[q*6+q]-a[p*6+p])/(Real(2)*off);
            Real t=(tau<Real(0)?Real(-1):Real(1))/(abs_r(tau)+sqrt_r(Real(1)+tau*tau));
            Real c=Real(1)/sqrt_r(Real(1)+t*t),s=t*c;
            Real pp=a[p*6+p],qq=a[q*6+q];
            a[p*6+p]=pp-t*off;a[q*6+q]=qq+t*off;a[p*6+q]=a[q*6+p]=Real(0);
            for(int k=0;k<n;k++){
                if(k!=p&&k!=q){Real x=a[k*6+p],y=a[k*6+q];a[k*6+p]=a[p*6+k]=c*x-s*y;a[k*6+q]=a[q*6+k]=s*x+c*y;}
                Real x=v[k*6+p],y=v[k*6+q];v[k*6+p]=c*x-s*y;v[k*6+q]=s*x+c*y;
            }
        }
        if(maximum<Real(1e-24f))break;
    }
    Real largest(-1e30f),gnorm(0),norm(0);
    for(int i=0;i<n;i++){
        largest=max_r(largest,a[i*6+i]);along[i]=Real(0);gnorm=gnorm+scaled_g[i]*scaled_g[i];
        for(int j=0;j<n;j++)along[i]=along[i]+v[j*6+i]*scaled_g[j];
        Real denom=-a[i*6+i];if(abs_r(denom)>Real(1e-30f)){Real y=along[i]/denom;norm=norm+y*y;}else norm=Real(1e30f);
    }
    Real mu(0);
    if(!(largest<Real(0)&&sqrt_r(norm)<=radius)){
        Real low=max_r(Real(0),largest),high=low+sqrt_r(gnorm)/radius+Real(1e-30f);
        low=low+Real(1e-12f)*(abs_r(low)+Real(1));
        for(int k=0;k<100;k++){
            Real middle=(low+high)/Real(2);norm=Real(0);
            for(int i=0;i<n;i++){Real y=along[i]/(middle-a[i*6+i]);norm=norm+y*y;}
            if(sqrt_r(norm)>radius)low=middle;else high=middle;
        }
        mu=high;
    }
    for(int i=0;i<n;i++){
        Real y(0);for(int j=0;j<n;j++)y=y+v[i*6+j]*(along[j]/(mu-a[j*6+j]));result[i]=y*steps[i];
    }
}
FN void bounded_step(LOCAL Real *center,LOCAL Real *g,LOCAL Real *h,LOCAL Real *step,
                     Real radius,LOCAL Real *lower,LOCAL Real *upper,LOCAL Real *move){
    bool free[6];Real held[6];for(int d=0;d<6;d++){free[d]=!(FIXED&&d<3);held[d]=Real(0);}
    for(int pass=0;pass<6;pass++){
        int ids[6],n=0;for(int d=0;d<6;d++){move[d]=held[d];if(free[d])ids[n++]=d;}
        if(n){
            Real fg[6],fh[36],fs[6],fm[6];
            for(int i=0;i<n;i++){
                int d=ids[i];fg[i]=g[d];fs[i]=step[d];
                for(int j=0;j<6;j++)if(!free[j])fg[i]=fg[i]+h[d*6+j]*held[j];
                for(int j=0;j<n;j++)fh[i*6+j]=h[d*6+ids[j]];
            }
            trust_step(fg,fh,fs,n,radius,fm);for(int i=0;i<n;i++)move[ids[i]]=fm[i];
        }
        bool any=false;
        for(int d=0;d<6;d++)if(free[d]&&(center[d]+move[d]<lower[d]||center[d]+move[d]>upper[d])){
            held[d]=clip_r(center[d]+move[d],lower[d],upper[d])-center[d];free[d]=false;any=true;
        }
        if(!any)break;
    }
}
"""

INIT_NEWTON_BODY = r"""
if(index>0)return;
Real chosen[18];int kept=0;
Real distinct[6]={Real(50),Real(20),Real(0.3f),Real(2),Real(2),Real(50)};
if(HAS_START){for(int d=0;d<6;d++)chosen[d]=load_real(start,d);kept=1;}
for(int rank=0;rank<COUNT && kept<STARTS;rank++){
    int idx=0;for(int i=0;i<COUNT;i++)if(int(ranks[i])==rank)idx=i;
    Real p[6];for(int d=0;d<6;d++)p[d]=load_real(points,idx*6+d);
    bool different=true;
    for(int k=0;k<kept;k++){
        Real dist(0);for(int d=0;d<6;d++)dist=max_r(dist,abs_r(p[d]-chosen[k*6+d])/distinct[d]);
        if(dist<=Real(1))different=false;
    }
    if(different){for(int d=0;d<6;d++)chosen[kept*6+d]=p[d];kept++;}
}
for(int k=0;k<STARTS;k++){
    for(int j=0;j<68;j++)store_real(out0,k*68+j,Real(0));
    if(k<kept){
        Real z[6],p[6];for(int d=0;d<6;d++)p[d]=chosen[k*6+d];encode_point(p,z);
        for(int d=0;d<6;d++){store_real(out0,k*68+d,z[d]);store_real(out0,k*68+62+d,z[d]);}
        store_real(out0,k*68+55,Real(8));store_real(out0,k*68+56,Real(1));store_real(out0,k*68+57,Real(1));
    }
    out1[k]=k<kept?1:0;out1[STARTS+k]=1;
}
"""

DESIGN_BODY = r"""
if(index>=COUNT)return;
int point=int(index),which=0,local=point;
for(int k=0;k<ACTIVE;k++){int size=int(plan[ACTIVE+k]);if(local<size){which=int(plan[k]);break;}local-=size;}
Real z[6],p[6];for(int d=0;d<6;d++)z[d]=load_real(state,which*68+d);
Real steps[6]={Real(2),Real(2),Real(2),Real(0.25f),Real(0.25f),Real(5)};
int first=FIXED?3:0,n=6-first;
if(local>0 && local<=2*n){int d=first+(local-1)/2;z[d]=z[d]+steps[d]*Real(local%2?1:-1);}
if(local>2*n){int pair=2*n+1;for(int i=first;i<6;i++)for(int j=i+1;j<6;j++){if(pair==local){z[i]=z[i]+steps[i];z[j]=z[j]+steps[j];}pair++;}}
decode_point(z,p);for(int d=0;d<6;d++)store_real(out0,point*6+d,p[d]);
"""

UPDATE_NEWTON_BODY = r"""
if(index>0)return;
Real all[204];for(int k=0;k<STARTS*68;k++)all[k]=load_real(state,k);
Real steps[6]={Real(2),Real(2),Real(2),Real(0.25f),Real(0.25f),Real(5)};
Real tol[6]={Real(0.05f),Real(0.05f),Real(0.05f),Real(0.005f),Real(0.005f),Real(0.5f)};
Real lower[6],upper[6];for(int d=0;d<6;d++){lower[d]=load_real(bounds,d);upper[d]=load_real(bounds,6+d);}
lower[1]=-upper[1];lower[2]=-upper[1];upper[2]=upper[1];
int offset=0,first=FIXED?3:0,n=6-first;
for(int active=0;active<ACTIVE;active++){
    int k=int(plan[active]),base=k*68,size=int(plan[ACTIVE+active]);
    Real value=load_real(values,offset)/load_real(scale,0),center[6],g[6],h[36];
    bool previous=all[base+58]>Real(0.5f);Real moved(0);
    for(int d=0;d<6;d++){
        center[d]=all[base+d];Real delta=(center[d]-all[base+6+d])/steps[d];moved=moved+delta*delta;
        g[d]=Real(0);for(int j=0;j<6;j++)h[d*6+j]=all[base+19+d*6+j];
    }
    moved=sqrt_r(moved);
    if(previous && value<all[base+12]){
        for(int d=0;d<6;d++){center[d]=all[base+6+d];g[d]=all[base+13+d];}
        all[base+55]=min_r(all[base+55],moved)/Real(4);
    }else{
        Real plus[6],minus[6];
        for(int d=first;d<6;d++){
            plus[d]=load_real(values,offset+1+2*(d-first))/load_real(scale,0);
            minus[d]=load_real(values,offset+2+2*(d-first))/load_real(scale,0);
            g[d]=(plus[d]-minus[d])/(Real(2)*steps[d]);
            h[d*6+d]=(plus[d]-Real(2)*value+minus[d])/(steps[d]*steps[d]);
        }
        if(size>2*n+1){int pair=offset+2*n+1;for(int i=first;i<6;i++)for(int j=i+1;j<6;j++){
            Real mixed=(load_real(values,pair++)/load_real(scale,0)-plus[i]-plus[j]+value)/(steps[i]*steps[j]);h[i*6+j]=h[j*6+i]=mixed;
        }}
        if(previous && moved>=Real(0.99f)*all[base+55])all[base+55]=Real(2)*all[base+55];
        all[base+56]=Real(previous && moved>Real(2)?1:0);
        for(int d=0;d<6;d++){all[base+6+d]=center[d];all[base+13+d]=g[d];for(int j=0;j<6;j++)all[base+19+d*6+j]=h[d*6+j];}
        all[base+12]=value;all[base+58]=Real(1);
    }
    Real move[6];bounded_step(center,g,h,steps,all[base+55],lower,upper,move);
    Real proposal[6];for(int d=0;d<6;d++)proposal[d]=FIXED&&d<3?center[d]:clip_r(center[d]+move[d],lower[d],upper[d]);
    Real norm=sqrt_r(proposal[1]*proposal[1]+proposal[2]*proposal[2]);
    if(!FIXED&&norm>upper[1]){proposal[1]=proposal[1]*upper[1]/norm;proposal[2]=proposal[2]*upper[1]/norm;}
    bool converged=true;for(int d=first;d<6;d++)if(abs_r(proposal[d]-center[d])>=tol[d])converged=false;
    if(converged){all[base+57]=Real(0);all[base+61]=Real(1);}
    for(int d=0;d<6;d++){all[base+d]=converged?center[d]:proposal[d];all[base+62+d]=all[base+d];}
    all[base+59]=all[base+59]+Real(1);all[base+60]=all[base+60]+Real(size);offset+=size;
}
for(int k=0;k<STARTS;k++)if(all[k*68+57]>Real(0.5f))for(int j=0;j<STARTS;j++){
    if(j==k || all[j*68+58]<Real(0.5f))continue;
    bool near=true,heading=true;
    for(int d=first;d<6;d++){
        if(abs_r(all[k*68+6+d]-all[j*68+6+d])>=Real(4)*steps[d])near=false;
        if(abs_r(all[k*68+62+d]-all[j*68+62+d])>=Real(2)*steps[d])heading=false;
    }
    bool worse=all[k*68+12]<all[j*68+12] || (!(all[k*68+12]>all[j*68+12])&&k>j);
    if((near||heading)&&worse){all[k*68+57]=Real(0);break;}
}
if(ITERATION==1){
    int best=-1;for(int k=0;k<STARTS;k++)if(all[k*68+57]>Real(0.5f))if(best<0||all[k*68+12]>all[best*68+12])best=k;
    for(int k=0;k<STARTS;k++)if(k!=best)all[k*68+57]=Real(0);
}
for(int k=0;k<STARTS;k++){
    for(int j=0;j<68;j++)store_real(out0,k*68+j,all[k*68+j]);
    out1[k]=scalar(all[k*68+57]);out1[STARTS+k]=scalar(all[k*68+56]);
}
"""

FINAL_NEWTON_BODY = r"""
if(index>0)return;
int best=0;for(int k=1;k<STARTS;k++)if(load_real(state,k*68+58)>Real(0.5f)&&load_real(state,k*68+12)>load_real(state,best*68+12))best=k;
Real z[6],p[6];for(int d=0;d<6;d++)z[d]=load_real(state,best*68+6+d);decode_point(z,p);
for(int d=0;d<6;d++)store_real(out0,d,p[d]);
store_real(out1,0,load_real(state,best*68+12)*load_real(scale,0));
"""


# The standard fit (tilt=False) holds coordinates 3-5 (sample tilt and depth spread) at 0, as FIXED holds 0-2 in a
# fixed-optics fit. Its kernels are derived from the joint fit's by these exact edits, not parameterized by another
# flag, so the joint fit's generated source (MPS and CUDA) is unchanged; every edit must match exactly once.
_STANDARD_EDITS = {
    "header": [("free[d]=!(FIXED&&d<3);", "free[d]=d<3;")],
    "design": [("int first=FIXED?3:0,n=6-first;", "int first=0,n=3;"),
               ("for(int i=first;i<6;i++)for(int j=i+1;j<6;j++)", "for(int i=0;i<3;i++)for(int j=i+1;j<3;j++)")],
    "update": [("int offset=0,first=FIXED?3:0,n=6-first;", "int offset=0,first=0,n=3;"),
               ("for(int d=first;d<6;d++){\n            plus[d]", "for(int d=0;d<3;d++){\n            plus[d]"),
               ("for(int i=first;i<6;i++)for(int j=i+1;j<6;j++){", "for(int i=0;i<3;i++)for(int j=i+1;j<3;j++){"),
               ("proposal[d]=FIXED&&d<3?center[d]:", "proposal[d]=d>=3?center[d]:"),
               ("bool converged=true;for(int d=first;d<6;d++)", "bool converged=true;for(int d=0;d<3;d++)"),
               ("for(int d=first;d<6;d++){\n        if(abs_r(all[k*68+6+d]", "for(int d=0;d<3;d++){\n        if(abs_r(all[k*68+6+d]")],
}


def _standard_source(source: str, part: str) -> str:
    for old, new in _STANDARD_EDITS[part]:
        if source.count(old) != 1:
            raise RuntimeError(f"The standard-fit Newton edit {old!r} no longer matches the joint kernel once.")
        source = source.replace(old, new)
    return source


STANDARD_NEWTON_HEADER = _standard_source(NEWTON_HEADER, "header")
STANDARD_DESIGN_BODY = _standard_source(DESIGN_BODY, "design")
STANDARD_UPDATE_NEWTON_BODY = _standard_source(UPDATE_NEWTON_BODY, "update")


def _rank(device, scores):
    return device.run(
        "ssb_joint_rank",
        {"scores": scores},
        [(len(scores),)],
        RANK_BODY,
        count=len(scores),
        constants=f"\n#define COUNT {len(scores)}\n",
    )[0]


def _search(device, evaluate, bounds, trials, seed, *, fixed=False, thin=False):
    xp = device.xp
    startup = max(30, trials // 6) if not thin else 10
    dimensions = 3 if (fixed or thin) else 6
    flags = f"\n#define FIXED {int(fixed)}\n#define THIN {int(thin)}\n#define DIMENSIONS {dimensions}\n"
    uniform = _random(device, seed, trials, dimensions)
    tpe_random = _random(device, seed, trials, (dimensions + 1) * 24)
    history = xp.zeros((0, 6, 3), dtype=xp.float32)
    scores = xp.zeros((0, 3), dtype=xp.float32)
    tpe_offset = 0
    for done in range(0, trials, 8):
        checkpoint()
        batch = min(8, trials - done)
        constants = (
            flags
            + f"\n#define DONE {done}\n#define BATCH {batch}\n#define OFFSET {tpe_offset}\n"
        )
        if done < startup:
            points = uniform[done : done + batch]
            if dimensions == 3:
                zero = xp.zeros((batch, 3, 3), dtype=xp.float32)
                points = xp.concatenate(
                    (zero, points) if fixed else (points, zero), axis=1
                )
        else:
            ranks = _rank(device, scores)
            density = device.run(
                "ssb_joint_density",
                {"history": history, "scores": scores, "ranks": ranks},
                [(done + 1, 5, 3)],
                DENSITY_BODY,
                count=done + 1,
                constants=constants,
            )[0]
            candidates = device.run(
                "ssb_joint_sample",
                {"history": history, "density": density, "random": tpe_random},
                [(batch, 24, 6, 3)],
                SAMPLE_BODY,
                count=batch * 24 * 6,
                constants=constants,
            )[0]
            components = device.run(
                "ssb_joint_components",
                {"history": history, "density": density, "candidates": candidates},
                [(batch, 24, done + 1, 3)],
                COMPONENT_BODY,
                count=batch * 24 * (done + 1),
                constants=constants,
            )[0]
            maxima = device.run(
                "ssb_joint_mixture_max",
                {"density": density, "components": components},
                [(batch, 24, 4, 3)],
                MAX_MIXTURE_BODY,
                count=batch * 24,
                constants=constants,
            )[0]
            powers = device.run(
                "ssb_joint_mixture_exp",
                {"density": density, "components": components, "maxima": maxima},
                [(batch, 24, done + 1, 2, 3)],
                EXP_MIXTURE_BODY,
                count=batch * 24 * (done + 1),
                constants=constants,
            )[0]
            ratios = device.run(
                "ssb_joint_mixture_sum",
                {"components": powers, "maxima": maxima},
                [(batch, 24, 3)],
                SUM_MIXTURE_BODY,
                count=batch * 24,
                constants=constants,
            )[0]
            points = device.run(
                "ssb_joint_select",
                {"candidates": candidates, "scores": ratios},
                [(batch, 6, 3)],
                SELECT_BODY,
                count=batch,
                constants=constants,
            )[0]
            tpe_offset += batch
        real_points = _decode(device, points, bounds)
        values = evaluate(_trial(device, real_points))
        history = xp.concatenate((history, points), axis=0)
        scores = xp.concatenate((scores, values), axis=0)
        device.sync(history, scores)
    return _decode(device, history, bounds), scores


def _newton(device, evaluate, points, scores, bounds, *, start, starts, fixed=False, standard=False):
    starts = min(3, int(starts))
    header = STANDARD_NEWTON_HEADER if standard else NEWTON_HEADER
    design_body = STANDARD_DESIGN_BODY if standard else DESIGN_BODY
    update_body = STANDARD_UPDATE_NEWTON_BODY if standard else UPDATE_NEWTON_BODY
    prefix = "ssb_standard" if standard else "ssb_joint"
    ranks = _rank(device, scores)
    count = len(points)
    flags = f"\n#define STARTS {starts}\n#define COUNT {count}\n#define FIXED {int(fixed)}\n#define HAS_START {int(start is not None)}\n"
    initial = device.pack([0.0] * 6 if start is None else start)
    state, plan = device.run(
        f"{prefix}_newton_init",
        {"points": points, "scores": scores, "ranks": ranks, "start": initial},
        [(starts, 68, 3), (starts * 2,)],
        INIT_NEWTON_BODY,
        constants=flags + header,
    )
    scale = device.run(
        "ssb_joint_scale",
        {"scores": scores},
        [(1, 3)],
        "if(index>0)return;Real s(1e-30f);for(int i=0;i<COUNT;i++)s=max_r(s,abs_r(load_real(scores,i)));store_real(out0,0,s);",
        constants=flags,
    )[0]
    for iteration in range(30):
        checkpoint()
        device.sync(state, plan)
        # Active/full-design flags are dispatch metadata. Numerical points,
        # scores, gradients and matrices are never read back here.
        metadata = np.asarray(plan) if device.backend == "mps" else plan.get()
        active = [i for i in range(starts) if metadata[i] > 0]
        if not active:
            break
        n = 3 if fixed or standard else 6
        sizes = [
            1 + 2 * n + (n * (n - 1) // 2 if metadata[starts + i] > 0 else 0)
            for i in active
        ]
        dispatch = device.xp.array(active + sizes, dtype=device.xp.float32)
        total = sum(sizes)
        constants = (
            f"\n#define FIXED {int(fixed)}\n#define STARTS {starts}\n#define ACTIVE {len(active)}\n#define COUNT {total}\n#define ITERATION {iteration}\n"
            + header
        )
        design = device.run(
            f"{prefix}_newton_design",
            {"state": state, "plan": dispatch},
            [(total, 6, 3)],
            design_body,
            count=total,
            constants=constants,
        )[0]
        values = evaluate(_trial(device, design))
        state, plan = device.run(
            f"{prefix}_newton_update",
            {
                "state": state,
                "plan": dispatch,
                "values": values,
                "scale": scale,
                "bounds": bounds,
            },
            [(starts, 68, 3), (starts * 2,)],
            update_body,
            constants=constants,
        )
    point, value = device.run(
        f"{prefix}_newton_result",
        {"state": state, "scale": scale},
        [(1, 6, 3), (1, 3)],
        FINAL_NEWTON_BODY,
        constants=flags + header,
    )
    device.sync(point, value, state)
    return point, value, state


def fit_resident(
    evaluate,
    *,
    backend: str,
    trials: int = 300,
    tilt_limit_mrad: float = 25.0,
    depth_spread_range_A: tuple[float, float] = DEPTH_SPREAD_RANGE_A,
    c10_range: tuple[float, float] = (-300.0, 300.0),
    c12_max: float = 200.0,
    seed: int = 0,
    verbose: bool = True,
    band_inv_A: tuple[float, float] = (0.2, 0.9),
    polish_starts: int = 3,
    polish: str = "newton",
    compare_standard: bool = True,
    start=None,
    fixed_aberrations: dict[str, float] | None = None,
) -> dict[str, object]:
    """Fit the joint model with resident adaptive search and Newton refinement.

    Parameters use the existing engine units. ``evaluate`` accepts resident
    encoded model trials and returns resident extended scores. Public callers
    use ``SSB.find_aberrations(tilt=True)``; no solver setting is needed.
    """
    from quantem.gpu.ssb.thick_sample_fit import PARAMETERS

    if polish not in ("newton", None):
        raise ValueError("Resident joint fitting supports refinement='newton' or None.")
    if trials < 0:
        raise ValueError(
            "trials must be non-negative; use 200 for a complete joint search."
        )
    if trials == 0 and start is None:
        raise ValueError("Zero-trial refinement needs a previous joint fit as start.")
    if not 0 <= polish_starts <= 3:
        raise ValueError("Resident Newton supports zero to three distinct starts.")
    device = _Device(backend)
    lower = [
        c10_range[0],
        0.0,
        -math.pi / 2,
        -tilt_limit_mrad,
        -tilt_limit_mrad,
        depth_spread_range_A[0],
    ]
    upper = [
        c10_range[1],
        c12_max,
        math.pi / 2,
        tilt_limit_mrad,
        tilt_limit_mrad,
        depth_spread_range_A[1],
    ]
    if fixed_aberrations is not None:
        for d, name in enumerate(PARAMETERS[:3]):
            lower[d] = upper[d] = fixed_aberrations[name]
    bounds = device.pack(lower + upper)
    timings = {}
    begin = time.perf_counter()
    standard = None
    standard_fit = None
    if compare_standard:
        thin_bounds = device.pack(
            lower[:3] + [0.0, 0.0, 0.0] + upper[:3] + [0.0, 0.0, 0.0]
        )
        thin_points, thin_scores = _search(
            device, evaluate, thin_bounds, 200, seed, thin=True
        )
        thin_ranks = _rank(device, thin_scores)
        thin_point, thin_value = device.run(
            "ssb_joint_best",
            {"points": thin_points, "scores": thin_scores, "ranks": thin_ranks},
            [(1, 6, 3), (1, 3)],
            "if(index>0)return;for(int i=0;i<COUNT;i++)if(int(ranks[i])==0){for(int d=0;d<6;d++)store_real(out0,d,load_real(points,i*6+d));store_real(out1,0,load_real(scores,i));}",
            constants="\n#define COUNT 200\n",
        )
        standard = dict(
            zip(PARAMETERS[:3], device.host(thin_point)[0, :3].tolist(), strict=True)
        )
        standard_fit = float(device.host(thin_value)[0])
    timings["standard_comparison_s"] = time.perf_counter() - begin
    begin = time.perf_counter()
    if trials:
        points, scores = _search(
            device,
            evaluate,
            bounds,
            int(trials),
            int(seed),
            fixed=fixed_aberrations is not None,
        )
    else:
        points = device.pack([start])
        scores = evaluate(_trial(device, points))
        device.sync(scores)
    timings["coarse_search_s"] = time.perf_counter() - begin
    begin = time.perf_counter()
    if polish is None:
        polish_starts = 0
    if polish_starts:
        point, value, state = _newton(
            device,
            evaluate,
            points,
            scores,
            bounds,
            start=start,
            starts=polish_starts,
            fixed=fixed_aberrations is not None,
        )
    else:
        ranks = _rank(device, scores)
        point, value = device.run(
            "ssb_joint_best",
            {"points": points, "scores": scores, "ranks": ranks},
            [(1, 6, 3), (1, 3)],
            "if(index>0)return;for(int i=0;i<COUNT;i++)if(int(ranks[i])==0){for(int d=0;d<6;d++)store_real(out0,d,load_real(points,i*6+d));store_real(out1,0,load_real(scores,i));}",
            constants=f"\n#define COUNT {len(points)}\n",
        )
        state = None
    device.sync(point, value)
    timings["refinement_s"] = time.perf_counter() - begin
    best = dict(zip(PARAMETERS, device.host(point)[0].tolist(), strict=True))
    fit = float(device.host(value)[0])
    if not math.isfinite(fit) or fit <= 0:
        raise ValueError(
            "Joint tilt fitting found no positive finite agreement. Check scan sampling and signal in the fitting band, or use tilt=False."
        )
    record_points = device.host(points)
    record_scores = device.host(scores)
    records = (
        [
            {
                "number": i,
                "params": dict(zip(PARAMETERS, row.tolist(), strict=True)),
                "loss": -float(score),
            }
            for i, (row, score) in enumerate(
                zip(record_points, record_scores, strict=True)
            )
        ]
        if trials
        else []
    )
    best["phi12"] = (best["phi12"] + math.pi / 2) % math.pi - math.pi / 2
    newton = [] if state is None else device.host(state)
    diagnostics = [
        {
            "iterations": int(row[59]),
            "evaluations": int(row[60]),
            "converged": bool(row[61]),
        }
        for row in newton
        if row[58] > 0
    ]
    return {
        **best,
        "fit": fit,
        "standard_fit": standard_fit,
        "gain": None
        if standard_fit is None
        else fit / standard_fit
        if standard_fit > 0
        else float("nan"),
        "standard": standard,
        "band_inv_A": list(band_inv_A),
        "trials": int(trials),
        "trial_records": records,
        "timings": timings,
        "optimizer": {
            "search": "resident multivariate TPE",
            "refinement": polish if polish_starts else None,
            "device": backend,
            "host_numerical_search": False,
            "precision": "three-float expansion" if backend == "mps" else "float64",
            "newton_starts": diagnostics,
        },
    }


def fit_standard_resident(
    evaluate,
    *,
    backend: str,
    lower: tuple[float, float, float],
    upper: tuple[float, float, float],
    trials: int = 200,
    seed: int = 42,
    polish: str | None = "newton",
    lock: tuple[int, ...] = (),
    start=None,
    polish_starts: int = 3,
) -> dict[str, object]:
    """Standard SSB fit (tilt=False): the joint fit's search and objective with sample tilt and depth spread at 0.

    ``lower`` and ``upper`` bound (C10, C12, phi12) in engine units (Angstrom, Angstrom, rad); a coefficient with
    ``lower == upper`` is held at that value in every trial. The search is the resident multivariate TPE of the
    joint fit's standard comparison (``_search(thin=True)``, 10 startup trials). ``polish``:

    - ``"newton"``: the resident trust-region Newton from the best ``polish_starts`` distinct trials (and ``start``),
      on (C10, C12 cos 2 phi12, C12 sin 2 phi12). It keeps C10 in its range and C12 below its maximum, so it needs
      a C12 range from 0 and the full phi12 period, and no ``lock``;
    - ``"nelder-mead"``: bounded SciPy Nelder-Mead from the same starts, the starts' simplices evaluated together.
      ``lock`` (indices into (C10, C12, phi12)) holds those coefficients at the best trial's values; phi12 wraps
      when its range is the whole period;
    - ``None``: the best trial (or ``start``) as is.

    ``trials=0`` polishes from ``start`` alone. Returns the fitted point, its agreement and the trial records.
    """
    from quantem.gpu.ssb.batch_refinement import _minimize_batch

    if trials < 0:
        raise ValueError("trials must be non-negative; use 200 for a complete standard search.")
    if trials == 0 and start is None:
        raise ValueError("Zero-trial refinement needs a start point.")
    if polish not in ("newton", "nelder-mead", None):
        raise ValueError("A standard fit polishes with 'newton', 'nelder-mead' or None.")
    full_period = lower[2] <= -math.pi / 2 + 1e-9 and upper[2] >= math.pi / 2 - 1e-9
    if polish == "newton" and (lock or lower[1] != 0.0 or not full_period):
        raise ValueError("Newton refinement bounds C10 and the C12 maximum only; use 'nelder-mead' to lock a "
                         "coefficient, bound phi12 or hold a nonzero C12.")
    device = _Device(backend)
    bounds = device.pack(list(lower) + [0.0, 0.0, 0.0] + list(upper) + [0.0, 0.0, 0.0])
    timings = {}
    begin = time.perf_counter()
    if trials:
        points, scores = _search(device, evaluate, bounds, int(trials), int(seed), thin=True)
    else:
        points = device.pack([[*start[:3], 0.0, 0.0, 0.0]])
        scores = evaluate(_trial(device, points))
        device.sync(scores)
    timings["coarse_search_s"] = time.perf_counter() - begin
    begin = time.perf_counter()
    host_points = device.host(points)[:, :3]
    host_scores = device.host(scores)
    order = np.argsort(-host_scores, kind="stable")
    nfev = 0
    if polish == "newton" and polish_starts:
        point, value, state = _newton(device, evaluate, points, scores, bounds,
                                      start=None if start is None or not trials else [*start[:3], 0.0, 0.0, 0.0],
                                      starts=polish_starts, standard=True)
        best = device.host(point)[0, :3]
        fit = float(device.host(value)[0])
        nfev = int(sum(row[60] for row in device.host(state) if row[58] > 0))
    elif polish == "nelder-mead" and polish_starts:
        candidates = ([np.asarray(start[:3], dtype=np.float64)] if start is not None and trials else []) + [
            host_points[index] for index in order]
        starts: list[np.ndarray] = []
        for candidate in candidates:
            if len(starts) >= polish_starts:
                break
            if all(np.max(np.abs(candidate - kept) / np.array([50.0, 20.0, 0.3])) > 1.0 for kept in starts):
                starts.append(np.array(candidate, dtype=np.float64))
        held = host_points[order[0]] if trials else np.asarray(start[:3], dtype=np.float64)
        free = [index for index in range(3) if index not in lock]
        for kept in starts:
            kept[list(lock)] = held[list(lock)]
        scale = max(abs(float(host_scores[order[0]])), 1e-30)
        if not free:
            best, fit = starts[0], float(host_scores[order[0]])
        else:
            def rows(free_points):
                full = np.tile(starts[0], (len(free_points), 1))
                full[:, free] = free_points
                full[:, 1] = np.abs(full[:, 1])
                if full_period:
                    full[:, 2] = (full[:, 2] + math.pi / 2) % math.pi - math.pi / 2
                return full

            def objective(free_points):
                full = rows(np.asarray(free_points, dtype=np.float64))
                packed = device.pack(np.concatenate((full, np.zeros((len(full), 3))), axis=1))
                return -device.host(evaluate(_trial(device, packed))) / scale

            box = [(lower[0], upper[0]), (lower[1], upper[1]), (None, None) if full_period else (lower[2], upper[2])]
            polished = _minimize_batch(objective, [kept[free] for kept in starts], bounds=[box[i] for i in free],
                                       options={"xatol": 0.05, "fatol": 1e-6, "maxiter": 400})
            outcome = min(polished, key=lambda result: result.fun)
            best, fit = rows(outcome.x[None])[0], float(-outcome.fun) * scale
            nfev = int(sum(result.nfev for result in polished))
    else:
        best, fit = host_points[order[0]], float(host_scores[order[0]])
    timings["refinement_s"] = time.perf_counter() - begin
    if not math.isfinite(fit) or fit <= 0:
        raise ValueError("The standard SSB fit found no positive finite model agreement. Check the scan sampling and "
                         "the signal in the fitting band.")
    best = np.asarray(best, dtype=np.float64)
    best[1] = abs(best[1])
    best[2] = (best[2] + math.pi / 2) % math.pi - math.pi / 2
    records = [{"number": i, "params": {"C10": float(row[0]), "C12": float(row[1]), "phi12": float(row[2])},
                "loss": -float(score)} for i, (row, score) in enumerate(zip(host_points, host_scores, strict=True))
               ] if trials else []
    return {"C10": float(best[0]), "C12": float(best[1]), "phi12": float(best[2]), "fit": fit, "trials": int(trials),
            "trial_records": records, "timings": timings, "nfev": nfev,
            "optimizer": {"search": "resident multivariate TPE", "refinement": polish if polish_starts else None,
                          "device": backend, "objective": "agreement, depth spread 0"}}

# Probability components are independent work. Expose them as GPU threads
# instead of serially evaluating every mixture component within a candidate.
COMPONENT_BODY = r"""
if(index>=BATCH*24*(DONE+1))return;
int component=int(index)%(DONE+1),candidate=int(index)/(DONE+1);
Real width=load_real(density,component*5+1),sum(0);
for(int d=0;d<6;d++){
    if((FIXED&&d<3)||(THIN&&d>=3))continue;
    Real mu=component==DONE?Real(0.5f):load_real(history,component*6+d);
    Real delta=load_real(candidates,candidate*6+d)-mu;sum=sum+delta*delta;
}
Real weight=component==DONE?Real(1):load_real(density,component*5+3);
store_real(out0,int(index),-sum/(Real(2)*width*width)-load_real(density,component*5+2)+log_r(weight));
"""
MAX_MIXTURE_BODY = r"""
if(index>=BATCH*24)return;
int candidate=int(index);Real good(-1e30f),bad(-1e30f),good_w(0),bad_w(0);
for(int i=0;i<=DONE;i++){
    bool prior=i==DONE,below=load_real(density,i*5)>Real(0.5f);
    Real value=load_real(components,candidate*(DONE+1)+i),w=prior?Real(1):load_real(density,i*5+3);
    if(prior||below){good=max_r(good,value);good_w=good_w+w;}
    if(prior||!below){bad=max_r(bad,value);bad_w=bad_w+w;}
}
store_real(out0,candidate*4,good);store_real(out0,candidate*4+1,bad);
store_real(out0,candidate*4+2,good_w);store_real(out0,candidate*4+3,bad_w);
"""
EXP_MIXTURE_BODY = r"""
if(index>=BATCH*24*(DONE+1))return;
int component=int(index)%(DONE+1),candidate=int(index)/(DONE+1);
bool prior=component==DONE,below=load_real(density,component*5)>Real(0.5f);
Real value=load_real(components,int(index));
store_real(out0,int(index)*2,(prior||below)?exp_r(value-load_real(maxima,candidate*4)):Real(0));
store_real(out0,int(index)*2+1,(prior||!below)?exp_r(value-load_real(maxima,candidate*4+1)):Real(0));
"""
SUM_MIXTURE_BODY = r"""
if(index>=BATCH*24)return;
int candidate=int(index);Real good(0),bad(0);
for(int i=0;i<=DONE;i++){
    good=good+load_real(components,(candidate*(DONE+1)+i)*2);
    bad=bad+load_real(components,(candidate*(DONE+1)+i)*2+1);
}
store_real(out0,candidate,log_r(good)+load_real(maxima,candidate*4)-log_r(load_real(maxima,candidate*4+2))
                        -log_r(bad)-load_real(maxima,candidate*4+1)+log_r(load_real(maxima,candidate*4+3)));
"""
