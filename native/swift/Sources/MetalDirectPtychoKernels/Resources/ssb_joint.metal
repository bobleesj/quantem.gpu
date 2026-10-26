#include <metal_stdlib>
using namespace metal;
// Scalars are explicitly packed; k and q use (row, column), inverse Angstrom.
struct JointGrid {
    uint n, outputN, count, bfOffset, totalBF, bandCount, sourceHalf, chunks;
    float wl, semiangle, angularRow, angularColumn, dqRow, dqColumn, dcReal, dcImag, factor;
};
// The float32 geometry of Python MPS `tf_geometry` (thick_sample.py), expression for expression.
inline float4 joint_geometry_xy(float dx, float dy, float wl, float semiangle, float ang_y, float ang_x) {
    float r2 = dx * dx + dy * dy;
    float r = metal::sqrt(r2);
    float alpha = r * wl;
    float inv_r2 = (r2 > 1e-30f) ? (1.0f / r2) : 0.0f;
    float cos2 = (dx * dx - dy * dy) * inv_r2;
    float sin2 = 2.0f * dx * dy * inv_r2;
    float inv_r = (r > 1e-15f) ? (1.0f / r) : 0.0f;
    float denom = metal::sqrt(dx * ang_y * dx * ang_y + dy * ang_x * dy * ang_x) * inv_r;
    float edge = (denom > 1e-15f) ? ((semiangle - alpha) / denom + 0.5f) : 1.0f;
    return float4(alpha * alpha, cos2, sin2, metal::clamp(edge, 0.0f, 1.0f));
}
inline float4 joint_geometry(float2 k, constant JointGrid& p) {
    return joint_geometry_xy(k.x, k.y, p.wl, p.semiangle, p.angularRow, p.angularColumn);
}
inline float joint_sinc(float x) { return (metal::abs(x) < 1e-6f) ? 1.0f : metal::precise::sin(x) / x; }
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
kernel void joint_probe_geometry(device const float2* k [[buffer(0)]],device float4* geometry [[buffer(1)]],
    constant JointGrid& p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i<p.totalBF)geometry[i]=joint_geometry(k[i],p);
}
// Sum the chunk partials of each band and candidate in chunk order; the host forms the power.
kernel void joint_power_batch(device const float4* partial [[buffer(0)]],device float4* sums [[buffer(1)]],
    constant JointGrid& p [[buffer(2)]],constant uint& batch [[buffer(3)]],uint2 tid [[thread_position_in_grid]]) {
    if(tid.x>=p.bandCount||tid.y>=batch)return;
    float nr=0.0f, ni=0.0f, dd=0.0f;
    for(uint c=0;c<p.chunks;c++){
        float4 v=partial[((size_t)c*batch+tid.y)*p.bandCount+tid.x];
        nr+=v.x; ni+=v.y; dd+=v.z;
    }
    sums[(size_t)tid.y*p.bandCount+tid.x]=float4(nr,ni,dd,0);
}
