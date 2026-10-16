#include <metal_stdlib>
using namespace metal;
// Scalars are explicitly packed; k and q use (row, column), inverse Angstrom.
struct JointGrid {
    uint n, outputN, count, bfOffset, totalBF, bandCount, sourceHalf, chunks;
    float wl, semiangle, angularRow, angularColumn, dqRow, dqColumn, dcReal, dcImag, factor;
};
inline float4 joint_geometry(float2 k, constant JointGrid& p) {
    float r2=dot(k,k), r=sqrt(r2), a=r*p.wl;
    float inv=r2>1e-30f?1/r2:0;
    float d=r>1e-15f?length(k*float2(p.angularRow,p.angularColumn))/r:0;
    float aperture=d>1e-15f?clamp((p.semiangle-a)/d+0.5f,0.0f,1.0f):1;
    return float4(a*a,(k.x*k.x-k.y*k.y)*inv,2*k.x*k.y*inv,aperture);
}
inline float joint_sinc(float x) { return abs(x)<1e-6f?1.0f:precise::sin(x)/x; }
inline float2 joint_gamma(float2 q, float2 k, constant JointGrid& p, constant float* t) {
    float4 gk=joint_geometry(k,p), gm=joint_geometry(q-k,p), gp=joint_geometry(q+k,p);
    float f=p.factor;
    float ck=f*gk.x*(t[0]+t[1]*(gk.y*t[2]+gk.z*t[3]));
    float cm=f*gm.x*(t[0]+t[1]*(gm.y*t[2]+gm.z*t[3]));
    float cp=f*gp.x*(t[0]+t[1]*(gp.y*t[2]+gp.z*t[3]));
    float shift=2*M_PI_F*(q.x*t[5]+q.y*t[6]);
    float b1=joint_sinc(0.5f*t[4]*(-f*(gm.x-gk.x)-shift))*gm.w*gk.w;
    float b2=joint_sinc(0.5f*t[4]*( f*(gp.x-gk.x)-shift))*gp.w*gk.w;
    return float2(b1*precise::cos(cm-ck)-b2*precise::cos(cp-ck),
                 -b1*precise::sin(cm-ck)-b2*precise::sin(cp-ck));
}
inline float2 joint_read(device const float2* g,uint row,uint col,uint k,constant JointGrid& p) {
    if (!p.sourceHalf) return g[(size_t)k*p.n*p.n+row*p.n+col];
    if (col<=p.n/2) return g[(size_t)k*p.n*(p.n/2+1)+row*(p.n/2+1)+col];
    float2 v=g[(size_t)k*p.n*(p.n/2+1)+((p.n-row)%p.n)*(p.n/2+1)+p.n-col];
    return float2(v.x,-v.y);
}
kernel void joint_gather(device const float2* g [[buffer(0)]], device float2* band [[buffer(1)]],
    device const uint2* indices [[buffer(2)]],constant JointGrid& p [[buffer(3)]],uint2 tid [[thread_position_in_grid]]) {
    if(tid.x>=p.bandCount||tid.y>=p.count)return;
    uint2 q=indices[tid.x];
    band[(size_t)(p.bfOffset+tid.y)*p.bandCount+tid.x]=joint_read(g,q.x,q.y,tid.y,p);
}
kernel void joint_objective(device const float2* g [[buffer(0)]],device const float2* k [[buffer(1)]],
    device const float2* q [[buffer(2)]],device float4* partial [[buffer(3)]],
    constant JointGrid& p [[buffer(4)]],constant float* t [[buffer(5)]],uint2 tid [[thread_position_in_grid]]) {
    if(tid.x>=p.bandCount||tid.y>=p.chunks)return;
    float2 numerator=0;float denominator=0;
    for(uint i=tid.y*256;i<min(p.totalBF,(tid.y+1)*256);i++) {
        float2 gamma=joint_gamma(q[tid.x],k[i],p,t);
        float2 value=g[(size_t)i*p.bandCount+tid.x];
        numerator+=float2(dot(value,gamma),value.y*gamma.x-value.x*gamma.y);
        denominator+=dot(gamma,gamma);
    }
    partial[(size_t)tid.y*p.bandCount+tid.x]=float4(numerator,denominator,0);
}
kernel void joint_power(device const float4* partial [[buffer(0)]],device const float* weights [[buffer(1)]],
    device float* power [[buffer(2)]],constant JointGrid& p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.bandCount)return;
    float3 sum=0;for(uint c=0;c<p.chunks;c++)sum+=partial[(size_t)c*p.bandCount+i].xyz;
    power[i]=sum.z>0?weights[i]*dot(sum.xy,sum.xy)/sum.z:0;
}
kernel void joint_correct(device const float2* g [[buffer(0)]],device const float2* k [[buffer(1)]],
    device float2* corrected [[buffer(2)]],constant JointGrid& p [[buffer(3)]],
    constant float* t [[buffer(4)]],uint2 tid [[thread_position_in_grid]]) {
    uint plane=p.outputN*p.outputN;
    if(tid.x>=plane||tid.y>=p.count)return;
    uint row=tid.x/p.outputN,col=tid.x%p.outputN;
    int qr=row<p.outputN/2?int(row):int(row)-int(p.outputN);
    int qc=col<p.outputN/2?int(col):int(col)-int(p.outputN);
    float2 q=float2(qr*p.dqRow,qc*p.dqColumn);
    uint r=uint((qr%int(p.n)+int(p.n))%int(p.n));
    uint c=uint((qc%int(p.n)+int(p.n))%int(p.n));
    float2 value=joint_read(g,r,c,tid.y,p);
    float2 gamma=joint_gamma(q,k[p.bfOffset+tid.y],p,t);
    float2 result=float2(dot(value,gamma),value.y*gamma.x-value.x*gamma.y)/max(length(gamma),1e-8f);
    if(tid.x==0)result=float2(p.dcReal,p.dcImag);
    corrected[(size_t)tid.y*plane+tid.x]=result;
}
kernel void joint_phase_add(device const float2* object [[buffer(0)]],device float* phase [[buffer(1)]],
    constant JointGrid& p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    uint plane=p.outputN*p.outputN;if(i>=plane)return;
    float sum=0;for(uint k=0;k<p.count;k++){float2 v=object[(size_t)k*plane+i];sum+=atan2(v.y,v.x);}
    phase[i]+=sum;
}
kernel void joint_phase_normalize(device float* phase [[buffer(0)]],constant uint& count [[buffer(1)]],
    constant float& divisor [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i<count)phase[i]/=divisor;
}
