__device__ unsigned int joint_sparse_lanes2_threads256=1;

__device__ __forceinline__ unsigned int read_sparse_event(const unsigned int* packed,unsigned int i) {
    const unsigned int pwords=packed[1],fwords=packed[2],rwords=packed[3];
    const auto positions=packed+4,flags=positions+pwords,ranks=flags+fwords;
    const auto values=reinterpret_cast<const unsigned char*>(ranks+rwords);
    const unsigned int bit=i*9,word=bit>>5,shift=bit&31;
    const unsigned int position=__funnelshift_r(positions[word],word+1<pwords?positions[word+1]:0,shift)&511u;
    const unsigned int flag=flags[i>>5];
    unsigned int count=1;
    if((flag>>(i&31))&1u) {
        unsigned int rank=ranks[i>>8];
        for(unsigned int j=(i>>8)*8;j<(i>>5);++j)rank+=__popc(flags[j]);
        rank+=__popc(flag&((1u<<(i&31))-1));
        count=values[rank];
    }
    return position|(count<<9);
}

__device__ __forceinline__ void sparse_bounds(const unsigned int* packed,int stream,
    int cached,unsigned int& begin,unsigned int& end) {
    const int lane=stream&31,whole=lane>>2,tail=lane&3;
    const auto words=packed+cached+1+(stream>>5)*8;
    unsigned int sum=0;
    for(int i=0;i<whole;++i)sum=__dp4a(words[i],0x01010101u,sum);
    if(tail)sum=__dp4a(words[whole]&((1u<<(tail*8))-1),0x01010101u,sum);
    begin=packed[stream>>5]+sum;
    end=begin+((words[whole]>>((lane&3)*8))&255u);
}
// One CTA owns one native512-scan region across all selected sparse columns.
// Exact modular addition preserves both positive and negative detector changes.
extern "C" __global__ void integrate_sparse(
    const unsigned long long* event_addresses,const unsigned long long* offset_addresses,
    const int* selected,const signed char* coefficients,int selected_count,
    int cached_columns,int scan_blocks,unsigned int* output) {
    const int chunk=blockIdx.y,scan_block=blockIdx.x;
    const auto events=reinterpret_cast<const unsigned int*>(event_addresses[chunk]);
    const auto offsets=reinterpret_cast<const unsigned int*>(offset_addresses[chunk]);
    __shared__ unsigned int sums[512];
    for(int scan=threadIdx.x;scan<512;scan+=blockDim.x)sums[scan]=0;
    __syncthreads();
    const int lane=threadIdx.x&(2-1),first=threadIdx.x/2;
    for(int entry=first;entry<selected_count;entry+=blockDim.x/2) {
        const int stream=scan_block*cached_columns+selected[entry];
        const int coefficient=coefficients[entry];
        unsigned int begin,end;sparse_bounds(offsets,stream,cached_columns,begin,end);
        for(unsigned int at=begin+lane;at<end;at+=2) {
            const unsigned int event=read_sparse_event(events,at);
            atomicAdd(sums+(event&511),(event>>9)*coefficient);
        }
    }
    __syncthreads();
    output+=(unsigned long long)chunk*16384+scan_block*512;
    for(int scan=threadIdx.x;scan<512;scan+=blockDim.x) {
        const unsigned int sum=sums[scan];
        if(sum)output[scan]+=sum;
    }
}
