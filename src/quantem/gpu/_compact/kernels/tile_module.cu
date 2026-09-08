// Same 1032 exact fields, locally packed in 1024-scan packets.
using u32=unsigned int;
using u64=unsigned long long;
using i8=signed char;
constexpr int fields=1032;
extern "C" __global__ void seed_tiles(
    const u64* addresses,const u32* descriptors,const int* selected,const i8* coefficients,
    int count,int seed,const u32* total,u32* output) {
    const int chunk=blockIdx.y,packet=blockIdx.x;
    const int local=threadIdx.x*4,scan=packet*1024+local;
    __shared__ u64 data[fields];
    __shared__ u32 packed_meta[fields];
    __shared__ int signs[fields];
    for(int i=threadIdx.x;i<count;i+=blockDim.x) {
        const int field=selected[i];
        const u32 descriptor=descriptors[(chunk*16+packet)*fields+field];
        data[i]=addresses[chunk*fields+field]+u64(descriptor&65535u)*4;
        packed_meta[i]=descriptor>>16;
        signs[i]=coefficients[i];
    }
    __syncthreads();
    const u64 position=(u64(chunk)*16384+scan)/4;
    uint4 sum=seed==0?make_uint4(0,0,0,0):(seed==1?
        reinterpret_cast<const uint4*>(total)[position]:reinterpret_cast<const uint4*>(output)[position]);
    u32 base_sum=0;
    for(int i=0;i<count;++i) {
        const u32 meta=packed_meta[i],width=meta&15u,base=meta>>4;
        const int sign=signs[i];
        base_sum+=base*sign;
        if(width) {
            const auto pointer=reinterpret_cast<const u32*>(data[i]);
            const u32 bit=local*width,word=bit>>5,shift=bit&31;
            // Host-validated width <= 10 and aligned local scan fit in 64 bits.
            // Clamp the second load within this exact packet, including its last word.
            const u64 packed=(u64(pointer[word])|(u64(pointer[min(word+1,width*32-1)])<<32))>>shift;
            const u32 mask=(1u<<width)-1;
            sum.x+=u32(packed&mask)*sign;
            sum.y+=u32((packed>>width)&mask)*sign;
            sum.z+=u32((packed>>(width*2))&mask)*sign;
            sum.w+=u32((packed>>(width*3))&mask)*sign;
        }
    }
    sum.x+=base_sum;sum.y+=base_sum;sum.z+=base_sum;sum.w+=base_sum;
    reinterpret_cast<uint4*>(output)[position]=sum;
}
