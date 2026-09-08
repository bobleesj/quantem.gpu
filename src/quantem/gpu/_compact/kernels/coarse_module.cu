
using u32=unsigned int;using u64=unsigned long long;using u8=unsigned char;
using i8=signed char;constexpr int fields=36;
extern "C" __global__ void add_coarse(const u64* addresses,const u8* widths,
    const int* selected,const i8* coefficients,int count,u32* output) {
    const int chunk=blockIdx.y,scan=(blockIdx.x*blockDim.x+threadIdx.x)*4;
    __shared__ u64 data[fields];__shared__ int bits[fields],signs[fields];
    for(int i=threadIdx.x;i<count;i+=blockDim.x) {
        const int field=selected[i];data[i]=addresses[chunk*fields+field];
        bits[i]=widths[chunk*fields+field];signs[i]=coefficients[i];
    }
    __syncthreads();
    const u64 position=(u64(chunk)*16384+scan)/4;
    uint4 sum=reinterpret_cast<uint4*>(output)[position];
    for(int i=0;i<count;++i) {
        const int width=bits[i];
        if(width) {
            const auto pointer=reinterpret_cast<const u32*>(data[i]);
            const int bit=scan*width,word=bit>>5,shift=bit&31;
            const u32 a=pointer[word],b=pointer[min(word+1,width*512-1)];
            const u32 c=shift+width*4>64?pointer[word+2]:0;
            const u64 low=u64(a)|(u64(b)<<32),high=u64(b)|(u64(c)<<32);
            const u32 mask=(1u<<width)-1;const int sign=signs[i];
            #pragma unroll
            for(int j=0;j<4;++j) {
                const int offset=shift+j*width;
                const u32 v=u32(offset<32?(low>>offset):(high>>(offset-32)))&mask;
                (&sum.x)[j]+=v*sign;
            }
        }
    }
    reinterpret_cast<uint4*>(output)[position]=sum;
}
