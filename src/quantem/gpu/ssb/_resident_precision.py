"""Extended arithmetic for resident joint-fit decisions.

Metal stores a nonoverlapping three-float expansion (about 72 significand
bits), preserving the accuracy of the former host float64 decision path.
CUDA uses native float64 with the same three-float interchange format. This
is optimizer/reduction arithmetic; the scientific model kernels keep their
existing float32 arithmetic and measured detector counts.
"""

HEADER = r"""
#ifdef __METAL_VERSION__
#define FN inline
#define PTR device
#define LOCAL thread
struct Real {
    float a, b, c;
    Real(): a(0), b(0), c(0) {}
    Real(float v): a(v), b(0), c(0) {}
    Real(int v): a(float(v)), b(0), c(0) {}
};
FN void add_term(LOCAL Real &r, float v) {
    float s=r.a+v, z=s-r.a, e=(r.a-(s-z))+(v-z); r.a=s;
    s=r.b+e; z=s-r.b; e=(r.b-(s-z))+(e-z); r.b=s;
    r.c+=e;
    s=r.b+r.c; e=r.c-(s-r.b); r.b=s; r.c=e;
    s=r.a+r.b; e=r.b-(s-r.a); r.a=s;
    s=e+r.c; r.c=r.c-(s-e); r.b=s;
}
FN Real operator+(Real x, Real y) { add_term(x,y.a); add_term(x,y.b); add_term(x,y.c); return x; }
FN Real operator-(Real x) { x.a=-x.a;x.b=-x.b;x.c=-x.c;return x; }
FN Real operator-(Real x, Real y) {return x+(-y);}
FN Real operator*(Real x, Real y) {
    Real r; float xx[3]={x.a,x.b,x.c}, yy[3]={y.a,y.b,y.c};
    for(int i=0;i<3;i++) for(int j=0;j<3-i;j++) {
        float p=xx[i]*yy[j]; add_term(r,p); add_term(r,fma(xx[i],yy[j],-p));
    }
    return r;
}
FN Real operator/(Real x, Real y) {
    Real q;
    for(int i=0;i<4;i++) { float t=x.a/y.a; add_term(q,t); x=x-y*Real(t); }
    return q;
}
FN bool operator<(Real x,Real y) {Real d=x-y;return d.a<0;}
FN bool operator>(Real x,Real y) {return y<x;}
FN bool operator<=(Real x,Real y) {return !(y<x);}
FN bool operator>=(Real x,Real y) {return !(x<y);}
FN float scalar(Real x) {return x.a;}
#else
#define FN __device__ inline
#define PTR
#define LOCAL
struct Real {
    double v;
    FN Real():v(0) {}
    FN Real(double x):v(x) {}
};
FN Real operator+(Real x,Real y){return Real(x.v+y.v);}
FN Real operator-(Real x,Real y){return Real(x.v-y.v);}
FN Real operator-(Real x){return Real(-x.v);}
FN Real operator*(Real x,Real y){return Real(x.v*y.v);}
FN Real operator/(Real x,Real y){return Real(x.v/y.v);}
FN bool operator<(Real x,Real y){return x.v<y.v;}
FN bool operator>(Real x,Real y){return x.v>y.v;}
FN bool operator<=(Real x,Real y){return x.v<=y.v;}
FN bool operator>=(Real x,Real y){return x.v>=y.v;}
FN float scalar(Real x){return float(x.v);}
#endif
FN Real load_real(const PTR float *p,int i) {
#ifdef __METAL_VERSION__
    Real r; r.a=p[3*i];r.b=p[3*i+1];r.c=p[3*i+2];return r;
#else
    return Real(double(p[3*i])+double(p[3*i+1])+double(p[3*i+2]));
#endif
}
#ifdef __METAL_VERSION__
FN Real load_real(const constant float *p,int i) {
    Real r;r.a=p[3*i];r.b=p[3*i+1];r.c=p[3*i+2];return r;
}
#endif
FN void store_real(PTR float *p,int i,Real r) {
#ifdef __METAL_VERSION__
    p[3*i]=r.a;p[3*i+1]=r.b;p[3*i+2]=r.c;
#else
    float a=float(r.v),b=float(r.v-double(a));
    p[3*i]=a;p[3*i+1]=b;p[3*i+2]=float(r.v-double(a)-double(b));
#endif
}
FN Real abs_r(Real x){return x<Real(0)?-x:x;}
FN Real min_r(Real x,Real y){return x<y?x:y;}
FN Real max_r(Real x,Real y){return x>y?x:y;}
FN Real clip_r(Real x,Real lo,Real hi){return min_r(max_r(x,lo),hi);}
FN Real sqrt_r(Real x) {
    if(x<=Real(0))return Real(0);
#ifdef __METAL_VERSION__
    Real y(sqrt(x.a)); for(int i=0;i<3;i++)y=(y+x/y)*Real(0.5f); return y;
#else
    return Real(sqrt(x.v));
#endif
}
FN Real pi_r(){return Real(3.1415927410125732f)+Real(-8.742277657347586e-8f)+Real(-3.4302490200117637e-15f);}
FN Real ln2_r(){return Real(0.6931471824645996f)+Real(-1.9046542121259336e-9f)+Real(-8.783183432405266e-17f);}
FN Real exp_r(Real x) {
#ifndef __METAL_VERSION__
    return Real(exp(x.v));
#else
    int n=int(rint(scalar(x/ln2_r())));
    if(n < -125)return Real(0);
    Real r=x-Real(n)*ln2_r();
Real total=(Real(1.56192068145415131e-16f)+Real(1.54044714659419928e-24f)+Real(-1.27412229414971146e-32f));
total=total*r+(Real(2.81145735896637036e-15f)+Real(-1.04620847397636578e-22f)+Real(-2.20149422016349591e-30f));
total=total*r+(Real(4.77947725613294544e-14f)+Real(7.62544404448642983e-22f)+Real(-5.8709668504926312e-30f));
total=total*r+(Real(7.6471636098127127e-13f)+Real(1.22007104711782877e-20f)+Real(-9.39354696078820992e-29f));
total=total*r+(Real(1.14707453605089604e-11f)+Real(2.37220768923123805e-19f)+Real(-4.64020618551434324e-27f));
total=total*r+(Real(1.60590443720742826e-10f)+Real(-5.35252651156272563e-18f)+Real(-1.68360469329305087e-25f));
total=total*r+(Real(2.08767558795841524e-09f)+Real(1.10828391474598523e-16f)+Real(3.18798788031371376e-24f));
total=total*r+(Real(2.50521079436794025e-08f)+Real(4.41762304464836646e-16f)+Real(1.17860757509285847e-23f));
total=total*r+(Real(2.75573199814971304e-07f)+Real(-7.57511220905119492e-15f)+Real(-1.87990526693313626e-22f));
total=total*r+(Real(2.75573188446287531e-06f)+Real(3.79357122429722915e-14f)+Real(1.50822652208406509e-21f));
total=total*r+(Real(2.48015876422869042e-05f)+Real(-3.40699609366681977e-13f)+Real(6.79777592851575006e-21f));
total=total*r+(Real(0.000198412701138295233f)+Real(-2.72559687493345582e-12f)+Real(5.43822074281260005e-20f));
total=total*r+(Real(0.00138888892251998186f)+Real(-3.3631094437103215e-11f)+Real(1.4648776180200379e-18f));
total=total*r+(Real(0.00833333376795053482f)+Real(-4.34617203337595015e-10f)+Real(1.85037180421299903e-18f));
total=total*r+(Real(0.0416666679084300995f)+Real(-1.24176346982807218e-09f)+Real(3.70074352570793681e-17f));
total=total*r+(Real(0.166666671633720398f)+Real(-4.9670538793122887e-09f)+Real(1.48029741028317472e-16f));
total=total*r+(Real(0.5f)+Real(0)+Real(0));
total=total*r+(Real(1.0f)+Real(0)+Real(0));
total=total*r+(Real(1.0f)+Real(0)+Real(0));
    return total*Real(ldexp(1.0f,n));
#endif
}
FN Real log_r(Real x) {
#ifndef __METAL_VERSION__
    return Real(log(x.v));
#else
    int e=0; frexp(x.a,e); Real m=x*Real(ldexp(1.0f,-e));
    if(m<Real(0.70710677f)){m=m*Real(2);e--;}
    Real z=(m-Real(1))/(m+Real(1)), z2=z*z;
Real total=(Real(0.0322580635547637939f)+Real(9.61365209661835252e-10f)+Real(2.86509159106563107e-17f));
total=total*z2+(Real(0.034482758492231369f)+Real(1.28458285675314698e-10f)+Real(4.78544405383298913e-19f));
total=total*z2+(Real(0.0370370373129844666f)+Real(-2.75947431571665902e-10f)+Real(2.05596857943881975e-18f));
total=total*z2+(Real(0.0399999991059303284f)+Real(8.94069651646844932e-10f)+Real(1.99840152373462058e-17f));
total=total*z2+(Real(0.0434782616794109344f)+Real(-8.09845690508126381e-10f)+Real(-2.65488117461426521e-17f));
total=total*z2+(Real(0.0476190485060214996f)+Real(-8.86973883229558169e-10f)+Real(2.64338820310636156e-18f));
total=total*z2+(Real(0.0526315793395042419f)+Real(-3.92135823812367335e-10f)+Real(2.92163956025516491e-18f));
total=total*z2+(Real(0.0588235296308994293f)+Real(-2.19134724255276581e-10f)+Real(8.16340462324308532e-19f));
total=total*z2+(Real(0.0666666701436042786f)+Real(-3.47693762670076012e-09f)+Real(1.48029744337039923e-17f));
total=total*z2+(Real(0.076923079788684845f)+Real(-2.86560797313484272e-09f)+Real(5.12410629295243365e-17f));
total=total*z2+(Real(0.0909090936183929443f)+Real(-2.70930211598852111e-09f)+Real(8.07434951063549849e-17f));
total=total*z2+(Real(0.1111111119389534f)+Real(-8.27842294714997706e-10f)+Real(6.16790573831645926e-18f));
total=total*z2+(Real(0.142857149243354797f)+Real(-6.3862120036617398e-09f)+Real(6.34413185289139025e-17f));
total=total*z2+(Real(0.200000002980232239f)+Real(-2.98023228317845224e-09f)+Real(4.44089216467507517e-17f));
total=total*z2+(Real(0.333333343267440796f)+Real(-9.9341077586245774e-09f)+Real(2.96059482056634945e-16f));
total=total*z2+(Real(1.0f)+Real(0)+Real(0));
    return Real(2)*z*total+Real(e)*ln2_r();
#endif
}
FN Real sin_r(Real x){
#ifndef __METAL_VERSION__
    return Real(sin(x.v));
#else
    Real term=x,total=x,xx=-x*x;
    for(int i=1;i<=17;i++){term=term*xx/Real((2*i)*(2*i+1));total=total+term;}
    return total;
#endif
}
FN Real cos_r(Real x){
#ifndef __METAL_VERSION__
    return Real(cos(x.v));
#else
    Real term(1),total(1),xx=-x*x;
    for(int i=1;i<=18;i++){term=term*xx/Real((2*i-1)*(2*i));total=total+term;}
    return total;
#endif
}
FN Real atan_r(Real x){
#ifndef __METAL_VERSION__
    return Real(atan(x.v));
#else
    bool neg=x<Real(0);x=abs_r(x); bool inv=x>Real(1);if(inv)x=Real(1)/x;
    bool shift=x>Real(0.41421356f);if(shift)x=(x-Real(1))/(x+Real(1));
    Real term=x,total=x,xx=-x*x;
    for(int i=3;i<=61;i+=2){term=term*xx;total=total+term/Real(i);}
    if(shift)total=total+pi_r()/Real(4);if(inv)total=pi_r()/Real(2)-total;
    return neg?-total:total;
#endif
}
FN Real atan2_r(Real y,Real x){
    if(abs_r(x)<Real(1e-30f))return y<Real(0)?-pi_r()/Real(2):pi_r()/Real(2);
    Real a=atan_r(y/x);if(x<Real(0))a=a+(y<Real(0)?-pi_r():pi_r());return a;
}
FN Real cdf_r(Real x){
#ifndef __METAL_VERSION__
    return Real(0.5*erfc(-x.v/sqrt(2.0)));
#else
    // Cephes rational normal CDF (SciPy/xsf, BSD-3-Clause).
    Real z=abs_r(x)*(Real(0.70710676908493042f)+Real(1.21016174858823433e-08f)+Real(-3.81403371918205645e-16f));
    if(z<Real(1)){
        Real z2=z*z;
Real p=(Real(9.60497379302978516f)+Real(-5.31592689867466106e-08f)+Real(2.17986615658523764e-16f));
p=p*z2+(Real(90.0260162353515625f)+Real(3.4850327210733667e-06f)+Real(-1.46516659075461922e-14f));
p=p*z2+(Real(2232.00537109375f)+Real(-2.51469064096454531e-05f)+Real(-3.98094534767579056e-13f));
p=p*z2+(Real(7003.3251953125f)+Real(-5.41844492545351386e-05f)+Real(9.26513860343433984e-15f));
p=p*z2+(Real(55592.30078125f)+Real(0.000519789522513747215f)+Real(-2.62369466430545373e-11f));
Real q=(Real(1.0f)+Real(0)+Real(0));
q=q*z2+(Real(33.5617141723632812f)+Real(-7.61297158646812022e-09f)+Real(3.01168123049982842e-16f));
q=q*z2+(Real(521.35797119140625f)+Real(-2.14112533285515383e-05f)+Real(-2.41653458311549829e-13f));
q=q*z2+(Real(4594.32373046875f)+Real(9.92410496110096574e-05f)+Real(1.66886036240265767e-12f));
q=q*z2+(Real(22629.0f)+Real(6.13890952081419528e-05f)+Real(-1.78354195561752915e-12f));
q=q*z2+(Real(49267.39453125f)+Real(-0.00027038640109822154f)+Real(-6.79317843296778356e-12f));
        Real erf=z*p/q;return Real(0.5f)+(x<Real(0)?-erf:erf)/Real(2);
    }
    if(z>Real(14))return x<Real(0)?Real(0):Real(1);
    Real ratio;
    if(z<Real(8)){
Real p=(Real(2.46196979469814892e-10f)+Real(2.00371558014384499e-18f)+Real(4.0707869707519775e-26f));
p=p*z+(Real(0.564189553260803223f)+Real(1.15702656344751631e-08f)+Real(-3.51544117186176772e-17f));
p=p*z+(Real(7.4632105827331543f)+Real(-1.83104553741486598e-08f)+Real(2.04143662080665141e-16f));
p=p*z+(Real(48.6371955871582031f)+Real(1.51140989146369975e-06f)+Real(4.20726990843934417e-14f));
p=p*z+(Real(196.520828247070312f)+Real(4.70900658910977654e-06f)+Real(1.96632220597105223e-13f));
p=p*z+(Real(526.4451904296875f)+Real(4.56578982266364619e-06f)+Real(3.59673533632202252e-14f));
p=p*z+(Real(934.52850341796875f)+Real(2.37539879890391603e-05f)+Real(8.68500855000925442e-13f));
p=p*z+(Real(1027.5518798828125f)+Real(7.01234466760070063e-06f)+Real(-6.48806976568695737e-14f));
p=p*z+(Real(557.53533935546875f)+Real(-3.98606925955391489e-06f)+Real(-1.62920078903422416e-13f));
Real q=(Real(1.0f)+Real(0)+Real(0));
q=q*z+(Real(13.2281951904296875f)+Real(-7.49551887224697566e-08f)+Real(4.7326975958076928e-16f));
q=q*z+(Real(86.70721435546875f)+Real(-2.66869761844645836e-07f)+Real(-1.39224543962220178e-14f));
q=q*z+(Real(354.937774658203125f)+Real(4.22961693402612582e-06f)+Real(-1.67964126223003885e-13f));
q=q*z+(Real(975.70849609375f)+Real(5.64945548831019551e-06f)+Real(1.44280447625480838e-15f));
q=q*z+(Real(1823.9091796875f)+Real(-1.28084029711317271e-05f)+Real(3.34021737135800634e-13f));
q=q*z+(Real(2246.337646484375f)+Real(-3.82972648367285728e-05f)+Real(-3.45351432997839658e-13f));
q=q*z+(Real(1656.6630859375f)+Real(6.00411340201389976e-06f)+Real(9.9806102239145128e-14f));
q=q*z+(Real(557.53533935546875f)+Real(1.46225897879048716e-06f)+Real(-5.32444876762735303e-14f));
        ratio=p/q;
    }else{
Real p=(Real(0.564189612865447998f)+Real(-2.93176931620564574e-08f)+Real(2.37993573577339741e-16f));
p=p*z+(Real(1.27536666393280029f)+Real(4.36669793657529226e-08f)+Real(1.38543832286906364e-15f));
p=p*z+(Real(5.01905059814453125f)+Real(-1.75632720811336185e-07f)+Real(-5.66452394091718735e-15f));
p=p*z+(Real(6.16021108627319336f)+Real(-1.06342660899372277e-07f)+Real(3.39194724892810835e-15f));
p=p*z+(Real(7.40974283218383789f)+Real(-1.32679346620534488e-07f)+Real(-1.8784905898725163e-15f));
p=p*z+(Real(2.9788665771484375f)+Real(7.65725616247436847e-08f)+Real(3.28195642083711162e-15f));
Real q=(Real(1.0f)+Real(0)+Real(0));
q=q*z+(Real(2.260528564453125f)+Real(6.77480471722446964e-08f)+Real(5.9365532306091158e-16f));
q=q*z+(Real(9.39603519439697266f)+Real(5.49830403429041326e-08f)+Real(1.347575844247678e-15f));
q=q*z+(Real(12.0489540100097656f)+Real(-2.92001001156450002e-08f)+Real(1.51144995859420781e-16f));
q=q*z+(Real(17.0814456939697266f)+Real(-6.19213153640885139e-07f)+Real(1.68005851065985631e-14f));
q=q*z+(Real(9.60896778106689453f)+Real(3.09565962197666522e-07f)+Real(2.05306344502691716e-15f));
q=q*z+(Real(3.36907649040222168f)+Real(-3.94014065818737436e-08f)+Real(6.26862415959958971e-17f));
        ratio=p/q;
    }
    Real tail=exp_r(-z*z)*ratio/Real(2);return x<Real(0)?tail:Real(1)-tail;
#endif
}
FN Real inverse_cdf_r(Real p){
    bool flip=p>Real(0.5f);if(flip)p=Real(1)-p;
    // Log-probability Newton is stable in the small tail.
    Real x=p<Real(0.01f)?-sqrt_r(-Real(2)*log_r(p)):
        log_r(p/(Real(1)-p))*Real(0.62665707f);
    Real target=log_r(p);
    Real invsqrt=Real(0.3989422917366028f)+Real(-1.133517013585105e-8f)+Real(3.058787842895743e-17f);
    for(int i=0;i<12;i++){
        Real cdf=max_r(cdf_r(x),Real(1e-30f));
        Real delta=(log_r(cdf)-target)*cdf/(exp_r(-x*x/Real(2))*invsqrt);
        x=x-delta;if(abs_r(delta)<Real(1e-16f))break;
    }
    return flip?-x:x;
}
"""
