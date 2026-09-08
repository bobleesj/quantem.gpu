// Exact mixed-model compaction across all264 contexts and both signs.
using i8 = signed char;
using u8 = unsigned char;
constexpr int palette_capacity = 5;

template<bool scatter> __device__ void pack_palette(
    const int* counts, const int* old_offsets, const int* old_selected,
    const int* offsets, int* group_counts, int* selected, u8* slots,
    int* contexts, i8* signs, u8* lengths, u8* model_counts, int* models) {
    if (threadIdx.x) return;
    const int partition=blockIdx.x, context=partition/2, sign=partition&1;
    int group=0, used=0, columns=0;
    const int first=scatter?offsets[partition]:0;
    for(int model=0;model<81;++model) {
        const int segment=(context*81+model)*2+sign;
        const int count=counts[segment];
        int consumed=0;
        while(consumed<count) {
            if(used==palette_capacity) {
                if(scatter) {
                    contexts[first+group]=context;signs[first+group]=sign?-1:1;
                    lengths[first+group]=columns;model_counts[first+group]=used;
                }
                ++group;used=columns=0;
            }
            const int take=min(32-columns,count-consumed);
            if(scatter) {
                models[(first+group)*palette_capacity+used]=model;
                for(int i=0;i<take;++i) {
                    selected[(first+group)*32+columns+i]=old_selected[old_offsets[segment]*32+consumed+i];
                    slots[(first+group)*32+columns+i]=used;
                }
            }
            ++used;columns+=take;consumed+=take;
            if(columns==32) {
                if(scatter) {
                    contexts[first+group]=context;signs[first+group]=sign?-1:1;
                    lengths[first+group]=columns;model_counts[first+group]=used;
                }
                ++group;used=columns=0;
            }
        }
    }
    if(columns) {
        if(scatter) {
            contexts[first+group]=context;signs[first+group]=sign?-1:1;
            lengths[first+group]=columns;model_counts[first+group]=used;
        }
        ++group;
    }
    if(!scatter)group_counts[partition]=group;
}

extern "C" __global__ void count_palette(const int* counts,int* group_counts) {
    pack_palette<false>(counts,nullptr,nullptr,nullptr,group_counts,nullptr,nullptr,
                        nullptr,nullptr,nullptr,nullptr,nullptr);
}
extern "C" __global__ void scatter_palette(
    const int* counts,const int* old_offsets,const int* old_selected,
    const int* offsets,int* selected,u8* slots,int* contexts,i8* signs,
    u8* lengths,u8* model_counts,int* models) {
    pack_palette<true>(counts,old_offsets,old_selected,offsets,nullptr,selected,slots,
                       contexts,signs,lengths,model_counts,models);
}
