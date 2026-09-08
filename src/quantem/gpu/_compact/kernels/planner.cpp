// CPU detector planning only. The full66 scientific reduction remains in CUDA.
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

extern "C" int plan167(
    const int8_t* current, const int8_t* previous, const int8_t* valid,
    const int64_t* leaf_of, int leaf_count, const double* column_cost,
    double tile_cost, const int32_t* parents, const int32_t* omitted,
    const int32_t* stored, int parent_count, int8_t* output_difference,
    int8_t* output_signs, int8_t* output_coarse, double* output_cost) {
    constexpr int pixels=36864, fields=1032;
    if (leaf_count!=1184 || parent_count!=152) return -1;
    double best=std::numeric_limits<double>::infinity(); int winner=-1;
    for (int seed=0;seed<(previous?3:2);++seed) {
        std::array<double,1184> positive{},negative{},zero{};
        std::array<uint8_t,1184> has_positive{},has_negative{};
        std::array<int8_t,pixels> difference{},residual{};
        for (int q=0;q<pixels;++q) {
            int v=int(current[q])-(seed==1?valid[q]:(seed==2?previous[q]:0));
            difference[q]=int8_t(v); int leaf=int(leaf_of[q]);
            if (leaf<0 || leaf>=leaf_count || v < -1 || v > 1) return -2;
            if(v>0){positive[leaf]+=column_cost[q];has_positive[leaf]=1;}
            else if(v<0){negative[leaf]+=column_cost[q];has_negative[leaf]=1;}
            else zero[leaf]+=column_cost[q];
        }
        std::array<int8_t,1184> leaf{};
        for(int l=0;l<leaf_count;++l) {
            if(positive[l]>zero[l]+tile_cost && negative[l]==0) leaf[l]=1;
            if(negative[l]>zero[l]+tile_cost && positive[l]==0) leaf[l]=-1;
            if(has_positive[l]&&has_negative[l])leaf[l]=0;
        }
        std::array<int8_t,fields> signs{};
        std::copy_n(leaf.data(),576,signs.data());
        for(int j=0;j<parent_count;++j) {
            int base=leaf[576+j*4+omitted[j]];signs[parents[j]]=int8_t(base);
            for(int k=0;k<3;++k)signs[576+j*3+k]=int8_t(leaf[576+j*4+stored[j*3+k]]-base);
        }
        std::array<int8_t,36> coarse{};
        const int options[3]={0,-1,1};
        for(int cr=0;cr<6;++cr)for(int cc=0;cc<6;++cc) {
            int cost_min=100,choice=0;
            for(int option:options) {
                int cost=option!=0;
                for(int y=0;y<4;++y)for(int x=0;x<4;++x)cost+=signs[(cr*4+y)*24+cc*4+x]!=option;
                if(cost<cost_min){cost_min=cost;choice=option;}
            }
            coarse[cr*6+cc]=int8_t(choice);
            for(int y=0;y<4;++y)for(int x=0;x<4;++x)signs[(cr*4+y)*24+cc*4+x]-=int8_t(choice);
        }
        double cost=0;int nonzero=0;
        for(int q=0;q<pixels;++q) {
            int v=valid[q]?int(difference[q])-leaf[leaf_of[q]]:0;
            if(v < -1 || v > 1)return -3;
            residual[q]=int8_t(v);if(v)cost+=column_cost[q];
        }
        for(auto v:signs)nonzero+=v!=0;
        for(auto v:coarse)nonzero+=v!=0;
        cost+=tile_cost*nonzero;
        if(cost<best) {
            best=cost;winner=seed;
            std::copy(residual.begin(),residual.end(),output_difference);
            std::copy(signs.begin(),signs.end(),output_signs);
            std::copy(coarse.begin(),coarse.end(),output_coarse);
        }
    }
    *output_cost=best;return winner;
}
