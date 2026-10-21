// For real counts, Gamma(-q,k) = -conj(Gamma(q,k)), so the corrected
// spectrum with DC removed is anti-Hermitian. Rotate it by -i and transform
// only its Hermitian half; the object is DC / N^2 + i*h. Each BF phase is
// still evaluated before averaging (mean_phase, never phase_of_mean).
// The host proves both Nyquist boundaries are outside aperture support.
// Unsupported geometries retain the general complex transform.
//
// Three 512-point transforms plus a radix-3 combination give 1536. The shared
// 512-point routine is also used by ordinary SSB. Phase accumulation retains
// the existing groups of eight BF pixels, including partial cache/tail groups.
inline float2 joint_value_at(device const float2* g,device const float2* k,
    uint row,uint col,uint bf,float4 gk,float chi,constant JointGrid& p,constant float* t) {
    if(row==0 && col==0)return float2(p.dcReal,p.dcImag);
    int qr=row<768?int(row):int(row)-1536;
    int qc=col<768?int(col):int(col)-1536;
    float2 q=float2(qr*p.dqRow,qc*p.dqColumn),pixel=k[p.bfOffset+bf];
    float4 gm=joint_geometry(q-pixel,p),gp=joint_geometry(q+pixel,p);
    if(gm.w*gk.w==0 && gp.w*gk.w==0)return 0;
    uint r=uint((qr%int(p.n)+int(p.n))%int(p.n));
    uint c=uint((qc%int(p.n)+int(p.n))%int(p.n));
    if(r==0u && c==0u)return 0;  // replicas of the scan mean stay zero
    float2 value=joint_read(g,r,c,bf,p),gamma=joint_gamma_geometry_chi(q,gk,gm,gp,p,t,chi);
    return float2(dot(value,gamma),value.y*gamma.x-value.x*gamma.y)/max(length(gamma),1e-8f);
}
inline void joint_ifft1536_finish(threadgroup float2* scratch,device const float2* twiddle,
    uint index,thread float2& a,thread float2& b,thread float2& c) {
    float2 x=scratch[index],y=complex_mul(scratch[512+index],twiddle[512+index]);
    float2 z=complex_mul(scratch[1024+index],twiddle[512+2*index]);
    float2 s=y+z,d=y-z;
    constexpr float root=0.86602540378443864676f;
    float2 skew=float2(-root*d.y,root*d.x);
    a=(x+s)*(1.0f/1536.0f);
    b=(x-0.5f*s+skew)*(1.0f/1536.0f);
    c=(x-0.5f*s-skew)*(1.0f/1536.0f);
}

inline bool joint_active_column(uint col,float2 pixel,constant JointGrid& p) {
    float radius=(p.semiangle+0.5f*max(p.angularRow,p.angularColumn))/p.wl;
    return float(col)*abs(p.dqColumn) <= (abs(pixel.y)+radius)*1.00001f;
}
[[max_total_threads_per_threadgroup(192)]]
kernel void joint_ifft1536_half_columns(device const float2* source [[buffer(0)]],
    device const float2* k [[buffer(1)]],device float2* intermediate [[buffer(2)]],
    constant JointGrid& p [[buffer(3)]],constant float* t [[buffer(4)]],
    device const float2* twiddle [[buffer(5)]],device const float4* geometry [[buffer(6)]],uint tid [[thread_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]) {
    threadgroup float2 scratch[1536];
    uint sub=tid/64,local=tid%64,col=group.x,bf=group.y;
    if(!joint_active_column(col,k[p.bfOffset+bf],p))return;
    float4 gk=geometry[p.bfOffset+bf];
    float chi=p.factor*gk.x*(t[0]+t[1]*(gk.y*t[2]+gk.z*t[3]));
    float2 r[8];
    for(uint i=0;i<8;i++) {
        uint row=3*octal_reverse_512(local*8+i)+sub;
        float2 v=joint_value_at(source,k,row,col,bf,gk,chi,p,t);
        r[i]=(row==0 && col==0)?float2(0):float2(v.y,-v.x);
    }
    ifft512_radix8_registers(r[0],r[1],r[2],r[3],r[4],r[5],r[6],r[7],local,twiddle,scratch+512*sub);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=0;i<8;i++)scratch[sub*512+local+64*i]=r[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint index=tid;index<512;index+=192) {
        float2 a,b,c;joint_ifft1536_finish(scratch,twiddle,index,a,b,c);
        intermediate[((size_t)bf*1536+index)*769+col]=a;
        intermediate[((size_t)bf*1536+index+512)*769+col]=b;
        intermediate[((size_t)bf*1536+index+1024)*769+col]=c;
    }
}
[[max_total_threads_per_threadgroup(192)]]
kernel void joint_ifft1536_half_rows_phase(device const float2* intermediate [[buffer(0)]],
    device float* phase [[buffer(1)]],device const float2* twiddle [[buffer(2)]],
    constant JointGrid& p [[buffer(3)]],device const float2* k [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],uint row [[threadgroup_position_in_grid]]) {
    threadgroup float2 scratch[1536];
    uint sub=tid/64,local=tid%64;
    float sums[9]={0,0,0,0,0,0,0,0,0};
    float2 dc=float2(p.dcReal,p.dcImag)*(1.0f/(1536.0f*1536.0f));
    for(uint bf=0;bf<p.count;bf++) {
        float2 pixel=k[p.bfOffset+bf],r[8];
        for(uint i=0;i<8;i++) {
            uint col=3*octal_reverse_512(local*8+i)+sub;
            uint src=col<=768?col:1536-col;
            float2 v=joint_active_column(src,pixel,p)?intermediate[((size_t)bf*1536+row)*769+src]:float2(0);
            r[i]=col<=768?v:float2(v.x,-v.y);
        }
        ifft512_radix8_registers(r[0],r[1],r[2],r[3],r[4],r[5],r[6],r[7],local,twiddle,scratch+512*sub);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint i=0;i<8;i++)scratch[sub*512+local+64*i]=r[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint slot=0;slot<3;slot++) {
            uint index=tid+slot*192;if(index>=512)continue;
            float2 a,b,c;joint_ifft1536_finish(scratch,twiddle,index,a,b,c);
            sums[3*slot]+=atan2(dc.y+a.x,dc.x);
            sums[3*slot+1]+=atan2(dc.y+b.x,dc.x);
            sums[3*slot+2]+=atan2(dc.y+c.x,dc.x);
            if((bf&7)==7 || bf+1==p.count) {
                phase[(size_t)row*1536+index]+=sums[3*slot];
                phase[(size_t)row*1536+index+512]+=sums[3*slot+1];
                phase[(size_t)row*1536+index+1024]+=sums[3*slot+2];
                sums[3*slot]=0;sums[3*slot+1]=0;sums[3*slot+2]=0;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
