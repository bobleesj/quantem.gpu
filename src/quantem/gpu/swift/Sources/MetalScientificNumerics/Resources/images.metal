#include <metal_stdlib>
using namespace metal;
constant float PI = 3.14159265358979323846f;
kernel void complex_image(device const float *a [[buffer(0)]],device float2 *out [[buffer(1)]],
    constant uint &n [[buffer(2)]],uint i [[thread_position_in_grid]]) {if(i<n)out[i]=float2(a[i],0);}
kernel void normalize_pdf(device float *a [[buffer(0)]],device const float *total [[buffer(1)]],
    constant uint &n [[buffer(2)]],uint i [[thread_position_in_grid]]) {if(i<n)a[i]/=total[0];}
kernel void gaussian_pdf(device float *out [[buffer(0)]],constant uint &n [[buffer(1)]],
    constant float &sigma [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    #pragma clang fp contract(off)
    if(i>=n)return;float value=(float(i)-float(n-1)*.5f)/sigma;
    value=value*value;out[i]=exp(-.5f*value);
}
kernel void sobel_weights(device float *out [[buffer(0)]],constant uint &axis [[buffer(1)]],
    uint i [[thread_position_in_grid]]) {
    if(i>=9)return;int row=int(i/3)-1,col=int(i%3)-1;
    out[i]=axis==0?float(row*(col==0?2:1)):float(col*(row==0?2:1));
}
inline float normalized_translation(uint index, uint length, float shift) {
    // Match the separately rounded linspace, multiply and subtract operations
    // used to build the reference normalized grid. Contracting the first pair
    // moves detector coordinates by an extra float32 ULP near the scan edge.
    #pragma clang fp contract(off)
    float step = 2.0f / float(length - 1);
    float base = float(index) * step - 1.0f;
    return base - 2.0f * shift / float(length);
}
inline float2 cmul(float2 a, float2 b) { return float2(a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x); }
inline float at(device const float *a,int r,int c,int h,int w) { return r>=0&&r<h&&c>=0&&c<w?a[r*w+c]:0; }
inline float bilinear(device const float *a,float r,float c,int h,int w) {
    #pragma clang fp contract(off)
    int rr=int(floor(r)),cc=int(floor(c)); float fr=r-rr,fc=c-cc;
    float result=at(a,rr,cc,h,w)*((1-fr)*(1-fc));
    result=fma(at(a,rr,cc+1,h,w),(1-fr)*fc,result);
    result=fma(at(a,rr+1,cc,h,w),fr*(1-fc),result);
    return fma(at(a,rr+1,cc+1,h,w),fr*fc,result);
}
kernel void mean_images(device const float *a [[buffer(0)]],device float *out [[buffer(1)]],
    constant uint3 &p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    uint lane=i%p.z,pixel=i/p.z;if(pixel>=p.y)return;float value=0;
    for(uint j=lane;j<p.x;j+=p.z)value+=a[j*p.y+pixel];
    for(uint stride=p.z/2;stride;stride/=2)value+=simd_shuffle_down(value,stride);
    if(lane==0)out[pixel]=value/float(p.x);
}
kernel void magnitude(device const float *a [[buffer(0)]],device const float *b [[buffer(1)]],device float *out [[buffer(2)]],
    constant uint &n [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    #pragma clang fp contract(off)
    if(i<n)out[i]=sqrt(a[i]*a[i]+b[i]*b[i]);
}
inline float hann(uint index,float factor) {
    #pragma clang fp contract(off)
    return cos(float(index)*factor)*(-.5f)+.5f;
}
inline float tukey(uint i,uint n,float4 v) {
    #pragma clang fp contract(off)
    if(v.x==0)return 1;if(v.x==2)return hann(i,v.w);
    float edge=v.y*.5f;
    if(i<edge)return .5f*(1+cos(PI*(2*float(i)/v.y-1)));
    if(i>=float(n-1)-edge)return .5f*(1+cos(PI*(2*float(i)/v.y-v.z+1)));
    return 1;
}
kernel void image_window(device const float *a [[buffer(0)]],device float *out [[buffer(1)]],
    constant uint4 &p [[buffer(2)]],constant float4 *coefficients [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    #pragma clang fp contract(off)
    uint h=p.x,w=p.y,pad=p.z,ww=w+2*pad;if(i>=(h+2*pad)*ww)return;
    int r=int(i/ww)-int(pad),c=int(i%ww)-int(pad);float value=0;
    if(r>=0&&r<int(h)&&c>=0&&c<int(w)){
        float wr=1,wc=1;
        if(p.w==1){wr=tukey(r,h,coefficients[0]);wc=tukey(c,w,coefficients[1]);}
        if(p.w==2){wr=hann(r,coefficients[0].w);wc=hann(c,coefficients[1].w);}
        value=a[r*w+c]*(wr*wc);}out[i]=value;
}
kernel void shift_image(device const float *a [[buffer(0)]],device const float2 *shifts [[buffer(1)]],
    device float *out [[buffer(2)]],constant uint4 &p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.x*p.y)return;float2 s=shifts[p.z];
    // Retain the align_corners=true normalized-grid displacement convention.
    float nr=normalized_translation(i/p.y,p.x,s.x);
    float nc=normalized_translation(i%p.y,p.y,s.y);
    out[i]=bilinear(a,(nr+1)*.5f*float(p.x-1),(nc+1)*.5f*float(p.y-1),p.x,p.y);
}
// Eight independent accumulation chains and two SIMD reductions preserve
// float32 rounding for batched image sums while exposing parallel work.
inline float chain_sum(device const float *a,device const float *w,uint n,
    uint lane,uint width,bool weighted) {
    #pragma clang fp contract(off)
    float accumulators[8]={0,0,0,0,0,0,0,0};
    for(uint base=lane*8;base<n;base+=width*8)
        for(uint j=0;j<8&&base+j<n;j++)
            accumulators[j]+=weighted?a[base+j]*w[base+j]:a[base+j];
    float result=accumulators[0];
    for(uint j=1;j<8;j++)result+=accumulators[j];
    return result;
}
inline float group_sum(float value,uint lane,uint width,threadgroup float *partial) {
    value=simd_sum(value);
    if(lane%32==0)partial[lane/32]=value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    value=lane<width/32?partial[lane]:0;
    return simd_sum(value);
}
kernel void sum_pixels(device const float *a [[buffer(0)]],device float *out [[buffer(1)]],
    constant uint &n [[buffer(2)]],uint lane [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]],uint width [[threads_per_threadgroup]]) {
    threadgroup float partial[32];
    float value=chain_sum(a+group*n,a,n,lane,width,false);
    value=group_sum(value,lane,width,partial);
    if(lane==0)out[group]=value;
}
kernel void window_mean(device const float *a [[buffer(0)]],device const float *w [[buffer(1)]],
    device const float *total [[buffer(2)]],device float *mean [[buffer(3)]],
    constant uint &n [[buffer(4)]],uint lane [[thread_index_in_threadgroup]],
    uint width [[threads_per_threadgroup]]) {
    threadgroup float partial[32];
    float value=chain_sum(a,w,n,lane,width,true);
    value=group_sum(value,lane,width,partial);
    if(lane==0)mean[0]=value/total[0];
}
kernel void window_center(device const float *a [[buffer(0)]],device const float *w [[buffer(1)]],
    device const float *mean [[buffer(2)]],device float *out [[buffer(3)]],constant uint &n [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {if(i<n)out[i]=(a[i]-mean[0])*w[i];}
kernel void fill_value(device float *a [[buffer(0)]],constant uint &n [[buffer(1)]],constant float &value [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {if(i<n)a[i]=value;}
kernel void spectrum_product(device const float2 *a [[buffer(0)]],device const float2 *b [[buffer(1)]],
    device float2 *out [[buffer(2)]],constant uint &n [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(i<n)out[i]=cmul(a[i],float2(b[i].x,-b[i].y));
}
inline float frequency(uint i,uint n){return float(i<(n+1)/2?int(i):int(i)-int(n));}
kernel void spectrum_blend(device const float2 *a [[buffer(0)]],device const float2 *b [[buffer(1)]],
    device const float2 *shift [[buffer(2)]],device float2 *out [[buffer(3)]],constant uint4 &p [[buffer(4)]],
    constant float4 &coefficients [[buffer(5)]],uint i [[thread_position_in_grid]]) {
    #pragma clang fp contract(off)
    if(i>=p.x*p.y)return;float2 s=p.w?shift[0]:float2(0);
    float row=frequency(i/p.y,p.x)*coefficients.x;
    float col=frequency(i%p.y,p.y)*coefficients.y;
    float angle=(-2*PI)*(row*s.x+col*s.y);
    float2 translated=cmul(b[i],float2(cos(angle),sin(angle)));
    out[i]=a[i]*coefficients.z+translated/float(p.z+1);
}
kernel void peak_fit(device const float *a [[buffer(0)]],device float2 *shift [[buffer(1)]],
    constant uint4 &p [[buffer(2)]],uint lane [[thread_index_in_threadgroup]]) {
    #pragma clang fp contract(off)
    threadgroup float values[256];threadgroup uint positions[256];float best=-INFINITY;uint at=0;
    for(uint i=lane;i<p.x*p.y;i+=256){if(a[i]>best||(a[i]==best&&i<at)){best=a[i];at=i;}}
    values[lane]=best;positions[lane]=at;threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint d=128;d;d/=2){if(lane<d){float b=values[lane+d];uint j=positions[lane+d];
        if(b>values[lane]||(b==values[lane]&&j<positions[lane])){values[lane]=b;positions[lane]=j;}}
        threadgroup_barrier(mem_flags::mem_threadgroup);}
    if(lane)return;uint r=positions[0]/p.y,c=positions[0]%p.y;
    float top=a[((r+p.x-1)%p.x)*p.y+c],bottom=a[((r+1)%p.x)*p.y+c];
    float left=a[r*p.y+(c+p.y-1)%p.y],right=a[r*p.y+(c+1)%p.y];
    float2 den=float2(4*values[0]-2*bottom-2*top,4*values[0]-2*right-2*left);
    float2 delta=float2(den.x!=0?(bottom-top)/den.x:0,den.y!=0?(right-left)/den.y:0);
    if(p.z==0)shift[0]=rint((float2(r,c)+delta)*2)*.5f;
    else {float center=floor(ceil(float(p.z)*1.5f)/2);
        shift[0]=rint(shift[0]*float(p.z))/float(p.z)+(float2(r,c)-center+delta)/float(p.z);}
}
kernel void dft_kernels(device const float2 *shift [[buffer(0)]],device float2 *rows [[buffer(1)]],
    device float2 *cols [[buffer(2)]],constant uint4 &p [[buffer(3)]],
    constant float2 &factors [[buffer(4)]],uint i [[thread_position_in_grid]]) {
    #pragma clang fp contract(off)
    float2 rounded=rint(shift[0]*float(p.z))/float(p.z);
    float2 center=floor(float(p.w)*.5f)-float(p.z)*rounded;
    // Preserve the reference's integer frequency times coordinate, followed
    // by one multiplication by the rounded complex exponential coefficient.
    if(i<p.w*p.x){uint r=i/p.x,k=i%p.x;int frequency=k<(p.x+1)/2?int(k):int(k)-int(p.x);
        float angle=(float(frequency)*(float(r)-center.x))*factors.x;
        rows[i]=float2(cos(angle),sin(angle));}
    if(i<p.y*p.w){uint k=i/p.w,c=i%p.w;int frequency=k<(p.y+1)/2?int(k):int(k)-int(p.y);
        float angle=(float(frequency)*(float(c)-center.y))*factors.y;
        cols[i]=float2(cos(angle),sin(angle));}
}
kernel void shift_record(device float2 *shifts [[buffer(0)]],device const float2 *value [[buffer(1)]],
    constant uint4 &p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i)return;float2 v=value[0];
    if(p.w){v.x=fmod(v.x+float(p.x)*.5f,float(p.x))-float(p.x)*.5f;
        v.y=fmod(v.y+float(p.y)*.5f,float(p.y))-float(p.y)*.5f;}
    shifts[p.z]+=v;
}
kernel void shift_center(device float2 *shifts [[buffer(0)]],constant uint &n [[buffer(1)]],uint i [[thread_position_in_grid]]) {
    if(i)return;float2 lanes[32];uint per=(n+31)/32;
    for(uint lane=0;lane<32;lane++){
        float2 chain[8]={};uint end=min((lane+1)*per,n);
        for(uint j=lane*per;j<end;j++)chain[j%8]+=shifts[j];
        lanes[lane]=chain[0];for(uint j=1;j<8;j++)lanes[lane]+=chain[j];
    }
    for(uint step=16;step;step/=2)for(uint lane=0;lane<step;lane++)lanes[lane]+=lanes[lane+step];
    float2 mean=lanes[0]/float(n);for(uint j=0;j<n;j++)shifts[j]-=mean;
}
kernel void origin(device const float *a [[buffer(0)]],device uint2 *out [[buffer(1)]],constant uint2 &p [[buffer(2)]],uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float vals[256];threadgroup uint ids[256];float v=-INFINITY;uint id=0;
    for(uint i=lane;i<p.x*p.y;i+=256){if(a[i]>v||(a[i]==v&&i<id)){v=a[i];id=i;}}
    vals[lane]=v;ids[lane]=id;threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint d=128;d;d/=2){if(lane<d&&(vals[lane+d]>vals[lane]||(vals[lane+d]==vals[lane]&&ids[lane+d]<ids[lane]))){vals[lane]=vals[lane+d];ids[lane]=ids[lane+d];}threadgroup_barrier(mem_flags::mem_threadgroup);}
    if(!lane)out[0]=uint2(ids[0]/p.y,ids[0]%p.y);
}
kernel void interior_window(device float *out [[buffer(0)]],constant uint2 &p [[buffer(1)]],uint i [[thread_position_in_grid]]) {
    if(i<p.x*p.y){uint r=i/p.y,c=i%p.y;out[i]=(r>0&&r+1<p.x&&c>0&&c+1<p.y)?1:0;}
}
kernel void shift_scan_mask(device const float *a [[buffer(0)]],device const float2 *shifts [[buffer(1)]],
    device float *out [[buffer(2)]],constant uint4 &p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.x*p.y)return;float2 s=shifts[p.z];
    float nr=2*(float(i/p.y)-s.x)/float(p.x-1)-1,nc=2*(float(i%p.y)-s.y)/float(p.y-1)-1;
    out[i]=bilinear(a,(nr+1)*.5f*float(p.x-1),(nc+1)*.5f*float(p.y-1),p.x,p.y);
}
kernel void add_image(device const float *a [[buffer(0)]],device float *out [[buffer(1)]],constant uint &n [[buffer(2)]],uint i [[thread_position_in_grid]]) {if(i<n)out[i]+=a[i];}
kernel void complement_clamp(device float *out [[buffer(0)]],constant uint &n [[buffer(1)]],uint i [[thread_position_in_grid]]) {if(i<n)out[i]=1-clamp(out[i],0.0f,1.0f);}
inline float raw_at(device const uchar *a,int r,int c,int dr,int dc,constant int *p) {
    if(r<p[5]||r>=p[6]||c<0||c>=p[1]||dr<0||dr>=p[2]||dc<0||dc>=p[3])return 0;
    ulong at=((ulong(r-p[5])*p[1]+c)*p[2]+dr)*p[3]+dc;
    return p[7]==1?float(a[at]):(p[7]==2?float(reinterpret_cast<device const ushort *>(a)[at]):float(reinterpret_cast<device const uint *>(a)[at]));
}
inline float scan_sample(device const uchar *a,int r,int c,int dr,int dc,float2 fraction,int2 delta,constant int *p) {
    float value=0;
    value+=raw_at(a,r+delta.x,c+delta.y,dr,dc,p)*((1-fraction.x)*(1-fraction.y));
    value+=raw_at(a,r+delta.x,c+delta.y+1,dr,dc,p)*((1-fraction.x)*fraction.y);
    value+=raw_at(a,r+delta.x+1,c+delta.y,dr,dc,p)*(fraction.x*(1-fraction.y));
    value+=raw_at(a,r+delta.x+1,c+delta.y+1,dr,dc,p)*(fraction.x*fraction.y);
    return value;
}
kernel void sample_accumulate(device const uchar *raw [[buffer(0)]],device const float *scan_weight [[buffer(1)]],
    device const float *det_weight [[buffer(2)]],device const float2 *scan_shifts [[buffer(3)]],
    device const float2 *det_shifts [[buffer(4)]],device float *numerator [[buffer(5)]],
    device float *denominator [[buffer(6)]],constant int *p [[buffer(7)]],uint i [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],uint width [[threads_per_simdgroup]]) {
    // p: scan rows/columns, detector rows/columns, first output row,
    // decoded first/stop row, native bytes, output rows, source index.
    uint pixels=p[2]*p[3];if(i>=uint(p[8]*p[1])*pixels)return;
    int scan=i/pixels,r=scan/p[1]+p[4],c=scan%p[1],pixel=i%pixels;
    float2 offset=-scan_shifts[p[9]],fraction=offset-floor(offset);int2 delta=int2(floor(offset));
    float2 ds=det_shifts[p[9]];
    float nr=normalized_translation(pixel/p[3],p[2],ds.x);
    float nc=normalized_translation(pixel%p[3],p[3],ds.y);
    float rr=(nr+1)*.5f*float(p[2]-1),cc=(nc+1)*.5f*float(p[3]-1);
    int dr=int(floor(rr)),dc=int(floor(cc));float fr=rr-dr,fc=cc-dc;
    // Neighboring lanes usually need the same interpolated detector column.
    // Reuse only after checking the exact integer coordinates and scan frame;
    // subgroup/row boundaries retain the original four-gather calculation.
    float upper=scan_sample(raw,r,c,dr,dc,fraction,delta,p);
    float lower=scan_sample(raw,r,c,dr+1,dc,fraction,delta,p);
    float upperNext=simd_shuffle_down(upper,1);
    float lowerNext=simd_shuffle_down(lower,1);
    int scanNext=simd_shuffle_down(scan,1),drNext=simd_shuffle_down(dr,1),dcNext=simd_shuffle_down(dc,1);
    bool reuse=lane+1<width && i+1<uint(p[8]*p[1])*pixels
        && scanNext==scan && drNext==dr && dcNext==dc+1;
    if(!reuse) {
        upperNext=scan_sample(raw,r,c,dr,dc+1,fraction,delta,p);
        lowerNext=scan_sample(raw,r,c,dr+1,dc+1,fraction,delta,p);
    }
    float value=upper*((1-fr)*(1-fc)) +upperNext*((1-fr)*fc)
        +lower*(fr*(1-fc)) +lowerNext*(fr*fc);
    float weight=scan_weight[r*p[1]+c];
    numerator[i]+=value*weight;denominator[i]+=weight*det_weight[pixel];
}
kernel void weighted_finish(device float *num [[buffer(0)]],device const float *den [[buffer(1)]],
    device const float *edge [[buffer(2)]],constant uint2 &p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(i<p.x){float d=den[i]+edge[i%p.y];num[i]=d==0?0:num[i]/d;}
}

// Invariant sampling geometry is prepared once per shape and displacement.
// Coordinates and arithmetic match sample_accumulate, including its normalized grid.
kernel void translated_detector_plan(device const float2 *shifts [[buffer(0)]],
    device int4 *indices [[buffer(1)]], device float4 *weights [[buffer(2)]],
    constant uint4 &p [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if(i>=p.x*p.y)return;
    float2 s=shifts[p.z];
    float nr=normalized_translation(i/p.y,p.x,s.x),nc=normalized_translation(i%p.y,p.y,s.y);
    float rr=(nr+1)*.5f*float(p.x-1),cc=(nc+1)*.5f*float(p.y-1);
    int r=int(floor(rr)),c=int(floor(cc));float fr=rr-r,fc=cc-c;
    int4 rows=int4(r,r,r+1,r+1),cols=int4(c,c+1,c,c+1);
    indices[i]=select(rows*int(p.y)+cols,int4(-1),rows<0||rows>=int(p.x)||cols<0||cols>=int(p.y));
    weights[i]=float4((1-fr)*(1-fc),(1-fr)*fc,fr*(1-fc),fr*fc);
}
kernel void translated_scan_plan(device const float2 *shifts [[buffer(0)]],
    device int4 *indices [[buffer(1)]],device float4 *weights [[buffer(2)]],
    constant uint4 &p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.x*p.y)return;
    float2 off=-shifts[p.z],f=off-floor(off);int2 d=int2(floor(off));
    int r=int(i/p.y)+d.x,c=int(i%p.y)+d.y;
    int4 rows=int4(r,r,r+1,r+1),cols=int4(c,c+1,c,c+1);
    indices[i]=select(rows*int(p.y)+cols,int4(-1),rows<0||rows>=int(p.x)||cols<0||cols>=int(p.y));
    if(i==0)weights[0]=float4((1-f.x)*(1-f.y),(1-f.x)*f.y,f.x*(1-f.y),f.x*f.y);
}
template<typename T>
inline float prepared_raw(device const T *raw,int scan,int pixel,constant uint4 &p) {
    return scan<0||pixel<0?0:float(raw[ulong(uint(scan-int(p.y)))*p.x+uint(pixel)]);
}
template<typename T>
inline float prepared_scan(device const T *raw,int4 scan,int pixel,float4 w,constant uint4 &p) {
    float value=0;
    value+=prepared_raw(raw,scan.x,pixel,p)*w.x;
    value+=prepared_raw(raw,scan.y,pixel,p)*w.y;
    value+=prepared_raw(raw,scan.z,pixel,p)*w.z;
    value+=prepared_raw(raw,scan.w,pixel,p)*w.w;
    return value;
}
template<typename T>
inline void prepared_accumulate(device const T *raw,device const int4 *scans,
    device const int4 *pixels,device const float4 *scanCoefficients,device const float4 *detectorCoefficients,
    device const float *scanWeight,device const float *detectorWeight,device float *numerator,
    device float *denominator,constant uint4 &p,uint2 at,uint lane,uint width) {
    if(at.x>=p.x||at.y>=p.w)return;
    uint scan=at.y+p.z,i=at.y*p.x+at.x;
    int4 s=scans[scan],d=pixels[at.x];float4 sw=scanCoefficients[0],dw=detectorCoefficients[at.x];
    float upper=prepared_scan(raw,s,d.x,sw,p),lower=prepared_scan(raw,s,d.z,sw,p);
    float upperNext=simd_shuffle_down(upper,1),lowerNext=simd_shuffle_down(lower,1);
    int nextUpper=simd_shuffle_down(d.x,1),nextLower=simd_shuffle_down(d.z,1);
    bool reuse=lane+1<width&&at.x+1<p.x&&nextUpper==d.y&&nextLower==d.w;
    if(!reuse){upperNext=prepared_scan(raw,s,d.y,sw,p);lowerNext=prepared_scan(raw,s,d.w,sw,p);}
    float value=upper*dw.x+upperNext*dw.y+lower*dw.z+lowerNext*dw.w;
    float weight=scanWeight[scan];
    numerator[i]+=value*weight;denominator[i]+=weight*detectorWeight[at.x];
}
#define PREPARED_KERNEL(NAME,TYPE) \
kernel void NAME(device const TYPE *raw [[buffer(0)]],device const int4 *scans [[buffer(1)]], \
    device const int4 *pixels [[buffer(2)]],device const float4 *sw [[buffer(3)]], \
    device const float4 *dw [[buffer(4)]],device const float *scanWeight [[buffer(5)]], \
    device const float *detectorWeight [[buffer(6)]],device float *numerator [[buffer(7)]], \
    device float *denominator [[buffer(8)]],constant uint4 &p [[buffer(9)]], \
    uint2 at [[thread_position_in_grid]],uint lane [[thread_index_in_simdgroup]],uint width [[threads_per_simdgroup]]) { \
    prepared_accumulate(raw,scans,pixels,sw,dw,scanWeight,detectorWeight,numerator,denominator,p,at,lane,width); \
}
PREPARED_KERNEL(sample_prepared_u8,uchar)
PREPARED_KERNEL(sample_prepared_u16,ushort)
PREPARED_KERNEL(sample_prepared_u32,uint)
