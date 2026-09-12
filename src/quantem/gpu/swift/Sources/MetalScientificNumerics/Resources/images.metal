#include <metal_stdlib>
using namespace metal;
constant float PI = 3.14159265358979323846f;
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
inline int reflected(int i, int n) { while (i < 0 || i >= n) i = i < 0 ? -i : 2*n-2-i; return i; }
inline float at(device const float *a,int r,int c,int h,int w) { return r>=0&&r<h&&c>=0&&c<w?a[r*w+c]:0; }
inline float bilinear(device const float *a,float r,float c,int h,int w) {
    int rr=int(floor(r)),cc=int(floor(c)); float fr=r-rr,fc=c-cc;
    return at(a,rr,cc,h,w)*((1-fr)*(1-fc))+at(a,rr,cc+1,h,w)*((1-fr)*fc)
        +at(a,rr+1,cc,h,w)*(fr*(1-fc))+at(a,rr+1,cc+1,h,w)*(fr*fc);
}
kernel void gaussian(device const float *a [[buffer(0)]],device float *b [[buffer(1)]],
    constant uint4 &p [[buffer(2)]],constant float &sigma [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.x*p.y)return; int r=i/p.y,c=i%p.y,rad=int(p.z); float sum=0,weight=0;
    for(int d=-rad;d<=rad;d++){float v=exp(-float(d*d)/(2*sigma*sigma));
        int rr=reflected(r+(p.w==0?d:0),p.x),cc=reflected(c+(p.w==1?d:0),p.y);
        sum+=a[rr*p.y+cc]*v;weight+=v;} b[i]=sum/weight;
}
kernel void sobel(device const float *a [[buffer(0)]],device float *rout [[buffer(1)]],device float *cout [[buffer(2)]],
    constant uint2 &p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.x*p.y)return; int r=i/p.y,c=i%p.y;float row=0,col=0;
    for(int dr=-1;dr<=1;dr++)for(int dc=-1;dc<=1;dc++){
        float v=a[reflected(r+dr,p.x)*p.y+reflected(c+dc,p.y)];
        row+=v*float(dr*(dc==0?2:1));col+=v*float(dc*(dr==0?2:1));}rout[i]=row;cout[i]=col;
}
kernel void magnitude(device const float *a [[buffer(0)]],device const float *b [[buffer(1)]],device float *out [[buffer(2)]],
    constant uint &n [[buffer(3)]],uint i [[thread_position_in_grid]]) {if(i<n)out[i]=sqrt(a[i]*a[i]+b[i]*b[i]);}
inline float tukey(uint i,uint n,float alpha) {
    if(alpha<=0)return 1; if(alpha>=1)return .5f*(1-cos(2*PI*float(i)/float(n)));
    float edge=alpha*float(n-1)*.5f;
    if(i<edge)return .5f*(1+cos(PI*(2*float(i)/(alpha*float(n-1))-1)));
    if(i>=float(n-1)-edge)return .5f*(1+cos(PI*(2*float(i)/(alpha*float(n-1))-2/alpha+1)));
    return 1;
}
kernel void image_window(device const float *a [[buffer(0)]],device float *out [[buffer(1)]],
    constant uint4 &p [[buffer(2)]],constant float &edge [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    uint h=p.x,w=p.y,pad=p.z,ww=w+2*pad; if(i>=(h+2*pad)*ww)return;
    int r=int(i/ww)-int(pad),c=int(i%ww)-int(pad);float value=0;
    if(r>=0&&r<int(h)&&c>=0&&c<int(w)){
        float wr=1,wc=1;
        if(p.w==1){wr=tukey(r,h,2*edge/h);wc=tukey(c,w,2*edge/w);}
        if(p.w==2){wr=.5f*(1-cos(2*PI*float(r)/h));wc=.5f*(1-cos(2*PI*float(c)/w));}
        value=a[r*w+c]*wr*wc;}out[i]=value;
}
kernel void shift_image(device const float *a [[buffer(0)]],device const float2 *shifts [[buffer(1)]],
    device float *out [[buffer(2)]],constant uint4 &p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.x*p.y)return;float2 s=shifts[p.z];
    // Retain the align_corners=true normalized-grid displacement convention.
    float nr=normalized_translation(i/p.y,p.x,s.x);
    float nc=normalized_translation(i%p.y,p.y,s.y);
    out[i]=bilinear(a,(nr+1)*.5f*float(p.x-1),(nc+1)*.5f*float(p.y-1),p.x,p.y);
}
kernel void window_mean(device const float *a [[buffer(0)]],device const float *w [[buffer(1)]],
    device float2 *mean [[buffer(2)]],constant uint &n [[buffer(3)]],uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float2 sums[256];float2 sum=0;
    for(uint i=lane;i<n;i+=256)sum+=float2(a[i]*w[i],w[i]);sums[lane]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint d=128;d;d/=2){if(lane<d)sums[lane]+=sums[lane+d];threadgroup_barrier(mem_flags::mem_threadgroup);}
    if(!lane)mean[0]=float2(sums[0].x/sums[0].y,0);
}
kernel void window_center(device const float *a [[buffer(0)]],device const float *w [[buffer(1)]],
    device const float2 *mean [[buffer(2)]],device float *out [[buffer(3)]],constant uint &n [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {if(i<n)out[i]=(a[i]-mean[0].x)*w[i];}
kernel void fill_value(device float *a [[buffer(0)]],constant uint &n [[buffer(1)]],constant float &value [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {if(i<n)a[i]=value;}
kernel void spectrum_product(device const float2 *a [[buffer(0)]],device const float2 *b [[buffer(1)]],
    device float2 *out [[buffer(2)]],constant uint &n [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(i<n)out[i]=cmul(a[i],float2(b[i].x,-b[i].y));
}
inline float frequency(uint i,uint n){return float(i<(n+1)/2?int(i):int(i)-int(n))/float(n);}
kernel void spectrum_blend(device const float2 *a [[buffer(0)]],device const float2 *b [[buffer(1)]],
    device const float2 *shift [[buffer(2)]],device float2 *out [[buffer(3)]],constant uint4 &p [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {
    if(i>=p.x*p.y)return;float2 s=p.w?shift[0]:float2(0);
    float angle=-2*PI*(frequency(i/p.y,p.x)*s.x+frequency(i%p.y,p.y)*s.y);
    float2 translated=cmul(b[i],float2(cos(angle),sin(angle)));
    out[i]=a[i]*(float(p.z)/float(p.z+1))+translated/float(p.z+1);
}
kernel void peak_fit(device const float *a [[buffer(0)]],device float2 *shift [[buffer(1)]],
    constant uint4 &p [[buffer(2)]],uint lane [[thread_index_in_threadgroup]]) {
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
    device float2 *cols [[buffer(2)]],constant uint4 &p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    float2 rounded=rint(shift[0]*float(p.z))/float(p.z);
    float2 center=floor(float(p.w)*.5f)-float(p.z)*rounded;
    if(i<p.w*p.x){uint r=i/p.x,k=i%p.x;float angle=-2*PI*frequency(k,p.x)*(float(r)-center.x)/float(p.z);
        rows[i]=float2(cos(angle),sin(angle));}
    if(i<p.y*p.w){uint k=i/p.w,c=i%p.w;float angle=-2*PI*frequency(k,p.y)*(float(c)-center.y)/float(p.z);
        cols[i]=float2(cos(angle),sin(angle));}
}
kernel void dft_rows(device const float2 *a [[buffer(0)]],device const float2 *row [[buffer(1)]],
    device float2 *out [[buffer(2)]],constant uint4 &p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.w*p.y)return;uint r=i/p.y,c=i%p.y;float2 sum=0;
    for(uint k=0;k<p.x;k++){float2 v=a[k*p.y+c];sum+=cmul(row[r*p.x+k],float2(v.x,-v.y));}out[i]=sum;
}
kernel void dft_cols(device const float2 *a [[buffer(0)]],device const float2 *col [[buffer(1)]],
    device float *out [[buffer(2)]],constant uint4 &p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.w*p.w)return;uint r=i/p.w,c=i%p.w;float sum=0;
    for(uint k=0;k<p.y;k++){float2 a0=a[r*p.y+k],b=col[k*p.w+c];sum+=a0.x*b.x-a0.y*b.y;}out[i]=sum;
}
kernel void shift_record(device float2 *shifts [[buffer(0)]],device const float2 *value [[buffer(1)]],
    constant uint4 &p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i)return;float2 v=value[0];
    if(p.w){v.x=fmod(v.x+float(p.x)*.5f,float(p.x))-float(p.x)*.5f;
        v.y=fmod(v.y+float(p.y)*.5f,float(p.y))-float(p.y)*.5f;}
    shifts[p.z]+=v;
}
kernel void shift_center(device float2 *shifts [[buffer(0)]],constant uint &n [[buffer(1)]],uint i [[thread_position_in_grid]]) {
    if(i)return;float2 sum=0;for(uint j=0;j<n;j++)sum+=shifts[j];sum/=float(n);for(uint j=0;j<n;j++)shifts[j]-=sum;
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
    return p[7]==1?float(a[at]):float(reinterpret_cast<device const ushort *>(a)[at]);
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
    device float *denominator [[buffer(6)]],constant int *p [[buffer(7)]],uint i [[thread_position_in_grid]]) {
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
    float value=scan_sample(raw,r,c,dr,dc,fraction,delta,p)*((1-fr)*(1-fc))
        +scan_sample(raw,r,c,dr,dc+1,fraction,delta,p)*((1-fr)*fc)
        +scan_sample(raw,r,c,dr+1,dc,fraction,delta,p)*(fr*(1-fc))
        +scan_sample(raw,r,c,dr+1,dc+1,fraction,delta,p)*(fr*fc);
    float weight=scan_weight[r*p[1]+c];
    numerator[i]+=value*weight;denominator[i]+=weight*det_weight[pixel];
}
kernel void weighted_finish(device float *num [[buffer(0)]],device const float *den [[buffer(1)]],
    device const float *edge [[buffer(2)]],constant uint2 &p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(i<p.x){float d=den[i]+edge[i%p.y];num[i]=d==0?0:num[i]/d;}
}
