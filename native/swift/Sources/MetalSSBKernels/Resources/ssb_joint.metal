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
inline float2 joint_gamma_geometry_chi(float2 q, float4 gk, float4 gm, float4 gp,
                                      constant JointGrid& p, constant float* t, float ck) {
    float f=p.factor;
    float cm=f*gm.x*(t[0]+t[1]*(gm.y*t[2]+gm.z*t[3]));
    float cp=f*gp.x*(t[0]+t[1]*(gp.y*t[2]+gp.z*t[3]));
    float shift=2*M_PI_F*(q.x*t[5]+q.y*t[6]);
    float b1=joint_sinc(0.5f*t[4]*(-f*(gm.x-gk.x)-shift))*gm.w*gk.w;
    float b2=joint_sinc(0.5f*t[4]*( f*(gp.x-gk.x)-shift))*gp.w*gk.w;
    return float2(b1*precise::cos(cm-ck)-b2*precise::cos(cp-ck),
                 -b1*precise::sin(cm-ck)-b2*precise::sin(cp-ck));
}
inline float2 joint_gamma_geometry(float2 q, float4 gk, float4 gm, float4 gp,
                                  constant JointGrid& p, constant float* t) {
    float ck=p.factor*gk.x*(t[0]+t[1]*(gk.y*t[2]+gk.z*t[3]));
    return joint_gamma_geometry_chi(q,gk,gm,gp,p,t,ck);
}
inline float2 joint_gamma(float2 q, float2 k, constant JointGrid& p, constant float* t) {
    return joint_gamma_geometry(q,joint_geometry(k,p),joint_geometry(q-k,p),
                                joint_geometry(q+k,p),p,t);
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
kernel void joint_probe_geometry(device const float2* k [[buffer(0)]],device float4* geometry [[buffer(1)]],
    constant JointGrid& p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i<p.totalBF)geometry[i]=joint_geometry(k[i],p);
}
kernel void joint_trial_probe(device const float4* geometry [[buffer(0)]],device float* chi [[buffer(1)]],
    constant float* parameters [[buffer(2)]],constant JointGrid& p [[buffer(3)]],
    constant uint& batch [[buffer(4)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.totalBF)return;
    float4 gk=geometry[i];
    for(uint b=0;b<batch;b++) {
        constant float* t=parameters+b*7;
        chi[(size_t)i*batch+b]=p.factor*gk.x*(t[1]*(gk.y*t[2]+gk.z*t[3])+t[0]);
    }
}
// The agreement objective of Python MPS `_thick_fit_batch_kernel`: per band frequency (one thread) and chunk of
// 256 BF pixels, the partial sums of G conj(gamma) and |gamma|^2 for up to eight candidates, in the same
// float32 expressions and BF order. The chunk partials are reduced afterwards, the band sum on the host in
// double precision.
kernel void joint_objective_batch(device const float2* g [[buffer(0)]],device const float2* k [[buffer(1)]],
    device const float2* q [[buffer(2)]],device float4* partial [[buffer(3)]],
    constant JointGrid& p [[buffer(4)]],constant float* parameters [[buffer(5)]],
    constant uint& batch [[buffer(6)]],device const float4* geometry [[buffer(7)]],
    device const float* chi [[buffer(8)]],uint2 tid [[thread_position_in_grid]]) {
    if(tid.x>=p.bandCount||tid.y>=p.chunks)return;
    float wl=p.wl, semiangle=p.semiangle, ang_y=p.angularRow, ang_x=p.angularColumn, factor=p.factor;
    float qx=q[tid.x].x, qy=q[tid.x].y;
    float nr[8], ni[8], dd[8];
    for(uint b=0;b<batch;b++){nr[b]=0.0f;ni[b]=0.0f;dd[b]=0.0f;}
    uint k1=metal::min(p.totalBF,(tid.y+1)*256);
    for(uint i=tid.y*256;i<k1;i++) {
        float kxv=k[i].x, kyv=k[i].y;
        float4 gk=geometry[i];
        float4 gm=joint_geometry_xy(qx-kxv,qy-kyv,wl,semiangle,ang_y,ang_x);
        float4 gp=joint_geometry_xy(qx+kxv,qy+kyv,wl,semiangle,ang_y,ang_x);
        float a1=gm.w*gk.w, a2=gp.w*gk.w;
        if(a1==0.0f && a2==0.0f)continue;
        float2 value=g[(size_t)i*p.bandCount+tid.x];
        float gre=value.x, gim=value.y;
        for(uint b=0;b<batch;b++) {
            constant float* t=parameters+b*7;
            float C10=t[0], C12=t[1], c2=t[2], s2=t[3], depth=t[4], thr=t[5], thc=t[6];
            float chi_k=chi[(size_t)i*batch+b];
            float b1=0.0f, b2=0.0f, c1=0.0f, cp=0.0f, s1=0.0f, sp=0.0f;
            bool thick=depth!=0.0f;
            if(a1!=0.0f) {
                float chi_m=factor*gm.x*(C12*(gm.y*c2+gm.z*s2)+C10);
                s1=metal::precise::sincos(chi_m-chi_k,c1);
                b1=a1;
                if(thick) {
                    float shift=6.283185307179586f*(qx*thr+qy*thc);
                    b1=joint_sinc(0.5f*depth*(-factor*(gm.x-gk.x)-shift))*a1;
                }
            }
            if(a2!=0.0f) {
                float chi_p=factor*gp.x*(C12*(gp.y*c2+gp.z*s2)+C10);
                sp=metal::precise::sincos(chi_p-chi_k,cp);
                b2=a2;
                if(thick) {
                    float shift=6.283185307179586f*(qx*thr+qy*thc);
                    b2=joint_sinc(0.5f*depth*(factor*(gp.x-gk.x)-shift))*a2;
                }
            }
            float gr=b1*c1-b2*cp;
            float gi=-b1*s1-b2*sp;
            nr[b]+=gre*gr+gim*gi;
            ni[b]+=gim*gr-gre*gi;
            dd[b]+=gr*gr+gi*gi;
        }
    }
    for(uint b=0;b<batch;b++)
        partial[((size_t)tid.y*batch+b)*p.bandCount+tid.x]=float4(nr[b],ni[b],dd[b],0);
}
// The same objective with angle-sum phases (Python MPS `resident_fit._ANGLE_SUM_SOURCE`). The two-term chi is a
// quadratic form, so chi(q -/+ k) - chi(k) = Psi -/+ u.k, and the depth rates are y -/+ P with y = C + t f q.k:
// one phasor of u.k and one of y give both sidebands, Psi and P once per (q, trial). Stored BF pixels follow
// detector rows, so both phasors advance by one row step per pixel and are evaluated exactly at a chunk's first
// pixel, after a row break and every eight pixels. Four candidates per thread; grid z tiles the candidates.
inline float joint_depth_sinc(float x, float s) {
    float x2 = x * x;
    float series = 1.0f - x2 * (1.0f / 6.0f) * (1.0f - x2 * (1.0f / 20.0f) * (1.0f - x2 * (1.0f / 42.0f)
                 * (1.0f - x2 * (1.0f / 72.0f) * (1.0f - x2 * (1.0f / 110.0f)))));
    return metal::abs(x) < 1.0f ? series : s / x;
}
kernel void joint_objective_angle_sums(device const float2* g [[buffer(0)]],device const float2* k [[buffer(1)]],
    device const float2* q [[buffer(2)]],device float4* partial [[buffer(3)]],
    constant JointGrid& p [[buffer(4)]],device const float* trial [[buffer(5)]],
    constant uint& TOTAL [[buffer(6)]],device const float4* probe_geometry [[buffer(7)]],
    device const float* neighbour [[buffer(8)]],constant float2& step_k [[buffer(9)]],
    uint3 tid [[thread_position_in_grid]]) {
    constexpr int B = 4;
    constexpr uint ANCHOR = 8;
    uint qi = tid.x, chunk = tid.y, base = tid.z * B;
    if (qi >= p.bandCount || chunk >= p.chunks || base >= TOTAL) return;
    float wl = p.wl, semi = p.semiangle, ang_y = p.angularRow, ang_x = p.angularColumn, factor = p.factor;
    float f = factor * wl * wl;
    float qx = q[qi].x, qy = q[qi].y;
    float ux[B], uy[B], cos_psi[B], sin_psi[B], c_q[B], cos_p[B], sin_p[B], p_q[B], bx[B], by[B];
    float2 n_minus[B], n_plus[B];
    float s_minus[B], s_plus[B];
    for (int b = 0; b < B; ++b) {
        uint i = metal::min(base + b, TOTAL - 1u);
        float c10 = trial[i * 7], c12 = trial[i * 7 + 1], c2 = trial[i * 7 + 2], s2 = trial[i * 7 + 3];
        float t = trial[i * 7 + 4], thr = trial[i * 7 + 5], thc = trial[i * 7 + 6];
        float m00 = f * (c10 + c12 * c2), m01 = f * (c12 * s2), m11 = f * (c10 - c12 * c2);
        ux[b] = 2.0f * (m00 * qx + m01 * qy);
        uy[b] = 2.0f * (m01 * qx + m11 * qy);
        float psi = qx * (m00 * qx + m01 * qy) + qy * (m01 * qx + m11 * qy);
        sin_psi[b] = metal::precise::sincos(psi, cos_psi[b]);
        c_q[b] = -3.141592653589793f * t * (qx * thr + qy * thc);
        p_q[b] = 0.5f * t * f * (qx * qx + qy * qy);
        sin_p[b] = metal::precise::sincos(p_q[b], cos_p[b]);
        bx[b] = t * f * qx;
        by[b] = t * f * qy;
        n_minus[b] = n_plus[b] = float2(0.0f);
        s_minus[b] = s_plus[b] = 0.0f;
    }
    uint k0 = chunk * 256;
    uint k1 = metal::min(p.totalBF, k0 + 256u);
    float2 eu[B], ey[B], su_step[B], sy_step[B];
    for (int b = 0; b < B; ++b) {
        float c, sn = metal::precise::sincos(ux[b] * step_k[0] + uy[b] * step_k[1], c);
        su_step[b] = float2(c, sn);
        sn = metal::precise::sincos(bx[b] * step_k[0] + by[b] * step_k[1], c);
        sy_step[b] = float2(c, sn);
        eu[b] = ey[b] = float2(1.0f, 0.0f);
    }
    for (uint kk = k0; kk < k1; ++kk) {
        float kxv = k[kk].x, kyv = k[kk].y;
        bool anchor = kk == k0 || neighbour[kk] == 0.0f || (kk - k0) % ANCHOR == 0;
        for (int b = 0; b < B; ++b) {
            if (anchor) {
                float c, sn = metal::precise::sincos(ux[b] * kxv + uy[b] * kyv, c);
                eu[b] = float2(c, sn);
                sn = metal::precise::sincos(c_q[b] + (bx[b] * kxv + by[b] * kyv), c);
                ey[b] = float2(c, sn);
            } else {
                eu[b] = float2(eu[b].x * su_step[b].x - eu[b].y * su_step[b].y,
                               eu[b].x * su_step[b].y + eu[b].y * su_step[b].x);
                ey[b] = float2(ey[b].x * sy_step[b].x - ey[b].y * sy_step[b].y,
                               ey[b].x * sy_step[b].y + ey[b].y * sy_step[b].x);
            }
        }
        float4 gk = probe_geometry[kk];
        float4 gm = joint_geometry_xy(qx - kxv, qy - kyv, wl, semi, ang_y, ang_x);
        float4 gp = joint_geometry_xy(qx + kxv, qy + kyv, wl, semi, ang_y, ang_x);
        float a1 = gm.w * gk.w, a2 = gp.w * gk.w;
        if (a1 == 0.0f && a2 == 0.0f) continue;
        float2 value = g[(size_t)kk * p.bandCount + qi];
        float gre = value.x, gim = value.y;
        for (int b = 0; b < B; ++b) {
            float cu = eu[b].x, su = eu[b].y;
            float hr = gre * cu + gim * su, hi = gim * cu - gre * su;
            float wa = a1, wb = a2;
            if (p_q[b] != 0.0f || c_q[b] != 0.0f || bx[b] != 0.0f || by[b] != 0.0f) {
                float y = c_q[b] + (bx[b] * kxv + by[b] * kyv);
                float cy = ey[b].x, sy = ey[b].y;
                if (a1 != 0.0f) wa = joint_depth_sinc(y - p_q[b], sy * cos_p[b] - cy * sin_p[b]) * a1;
                if (a2 != 0.0f) wb = joint_depth_sinc(y + p_q[b], sy * cos_p[b] + cy * sin_p[b]) * a2;
            }
            float d = wa - wb, sm = wa + wb;
            n_minus[b] += d * float2(hr, hi);
            n_plus[b] += sm * float2(hr, hi);
            s_minus[b] += d * d;
            s_plus[b] += sm * sm;
        }
    }
    for (int b = 0; b < B; ++b) {
        if (base + b >= TOTAL) break;
        partial[((size_t)chunk * TOTAL + base + b) * p.bandCount + qi] = float4(
            cos_psi[b] * n_minus[b].x - sin_psi[b] * n_plus[b].y,
            cos_psi[b] * n_minus[b].y + sin_psi[b] * n_plus[b].x,
            cos_psi[b] * cos_psi[b] * s_minus[b] + sin_psi[b] * sin_psi[b] * s_plus[b], 0);
    }
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
    float2 pixel=k[p.bfOffset+tid.y];
    float4 gk=joint_geometry(pixel,p),gm=joint_geometry(q-pixel,p),gp=joint_geometry(q+pixel,p);
    float2 result=0;
    // The scan mean is background: its replicas at nonzero output frequencies stay zero.
    bool replica=r==0u && c==0u;
    if(!replica && (gm.w*gk.w!=0 || gp.w*gk.w!=0)) {
        float2 value=joint_read(g,r,c,tid.y,p);
        float2 gamma=joint_gamma_geometry(q,gk,gm,gp,p,t);
        result=float2(dot(value,gamma),value.y*gamma.x-value.x*gamma.y)/max(length(gamma),1e-8f);
    }
    if(tid.x==0)result=float2(p.dcReal,p.dcImag);
    corrected[(size_t)tid.y*plane+tid.x]=result;
}
kernel void joint_phase_add(device const float2* object [[buffer(0)]],device float* phase [[buffer(1)]],
    constant JointGrid& p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    uint plane=p.outputN*p.outputN;if(i>=plane)return;
    for(uint start=0;start<p.count;start+=8) {
        float sum=0;for(uint k=start;k<min(start+8,p.count);k++){float2 v=object[(size_t)k*plane+i];sum+=atan2(v.y,v.x);}
        phase[i]+=sum;
    }
}
kernel void joint_phase_normalize(device float* phase [[buffer(0)]],constant uint& count [[buffer(1)]],
    constant float& divisor [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i<count)phase[i]/=divisor;
}
