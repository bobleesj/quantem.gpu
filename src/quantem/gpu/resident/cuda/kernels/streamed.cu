// Runtime-shaped count rANS. The probability model affects bytes, never counts.
typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;
static const u32 SC_LOWER = 1u << 23;
static const int SC_SCALE = 10;
static const int SC_MODELS = 64;

__device__ u32 sc_raw(const void* data, u64 at, int itemsize) {
    return itemsize == 1 ? ((const u8*)data)[at] : ((const u16*)data)[at];
}

struct StreamReader {
    __device__ StreamReader() = default;
    const u8* payload;
    const u32* table;
    u32 cursor, end, state, model, constant;
    bool valid;
    __device__ StreamReader(const u8* bytes, const u32* offsets,
                           const u8* models, const u32* decoding, u32 stream)
        : payload(bytes), cursor(offsets[stream]), end(offsets[stream + 1]),
          state(SC_LOWER), model(models[stream]), constant(0), valid(true) {
        table = decoding + (model < SC_MODELS ? model * 1024 : 0);
        if (model == 252) { state=0; valid=(end-cursor)%2==0; return; }
        if (model == 253) return;
        if (model == 255) {
            valid = end - cursor == 2;
            if (valid) { constant = payload[cursor] | (u32(payload[cursor+1]) << 8); cursor += 2; }
        } else if (model != 254) {
            valid = model < SC_MODELS && end - cursor >= 4;
            if (valid) {
                state = 0;
                for (int i=0;i<4;++i) state |= u32(payload[cursor++]) << (8*i);
                valid = state >= SC_LOWER && state < (1u<<31);
            }
        }
    }
    __device__ u32 next() {
        if (!valid) return 0;
        if (model == 252) {
            u32 position=state++;
            if (cursor==end) return 0;
            u32 event=payload[cursor] | (u32(payload[cursor+1])<<8);
            if ((event>>7)<position) { valid=false; return 0; }
            if ((event>>7)!=position) return 0;
            cursor+=2; return (event&127u)+1;
        }
        if (model == 253 || model == 255) return constant;
        if (model == 254) {
            if (end-cursor < 2) { valid=false; return 0; }
            u32 value = payload[cursor] | (u32(payload[cursor+1])<<8); cursor+=2;
            return value;
        }
        u32 slot = state & 1023u, code = table[slot];
        state = (code >> 16) * (state >> SC_SCALE) + slot - ((code >> 6) & 1023u);
        while (state < SC_LOWER) {
            if (cursor >= end) { valid=false; return 0; }
            state = (state << 8) | payload[cursor++];
        }
        u32 symbol=code&63u;
        if (symbol==32) {
            if (end-cursor<2) { valid=false; return 0; }
            symbol=payload[cursor] | (u32(payload[cursor+1])<<8); cursor+=2;
        }
        return symbol;
    }
    __device__ bool finished() const {
        return valid && cursor == end && (model >= 252 || state == SC_LOWER);
    }
};

// `total` (one u64 per detector pixel, or null) receives the exact counts of every stream: the
// statistics pass reads each count anyway, so the detector sum costs one addition per count.
extern "C" __global__ void sc_encode(
    const void* raw, int itemsize, u32 scans, u32 pixels, u32 interval,
    const u32* encoding, u8* scratch, u32* sizes, u32* states, u8* models,
    u32 streams, u64* total
) {
    u32 stream = blockIdx.x * blockDim.x + threadIdx.x;
    if (stream >= streams) return;
    u32 pixel=stream%pixels, first=(stream/pixels)*interval;
    u32 count=min(interval,scans-first), maximum=0, minimum=65535, sum=0, nonzero=0;
    u64 counts=0;
    for (u32 i=0;i<count;++i) {
        u32 v=sc_raw(raw,u64(first+i)*pixels+pixel,itemsize);
        maximum=max(maximum,v); minimum=min(minimum,v); sum+=min(v,32u); nonzero+=v!=0; counts+=v;
    }
    if (total && counts) atomicAdd(total+pixel,counts);
    if (minimum == maximum) {
        models[stream]=maximum ? 255 : 253;
        sizes[stream]=maximum ? 2 : 0; states[stream]=maximum; return;
    }
    if (maximum<=128 && nonzero<=2) {
        models[stream]=252; sizes[stream]=2*nonzero; return;
    }
    float mean=float(sum)/count;
    int model=max(0,min(63,__float2int_rn((logf(fmaxf(mean,0.002f))-logf(0.002f))
                                         * (63.0f/logf(16000.0f)))));
    u32 state=SC_LOWER, emitted=0;
    bool literal=false;
    for (u32 i=count;i>0;--i) {
        // Once coding cannot beat literal storage, avoid further work and
        // bound scratch even when a symbol requires two literal bytes.
        if (emitted+4>=2*count) { literal=true; break; }
        u32 value=sc_raw(raw,u64(first+i-1)*pixels+pixel,itemsize);
        if (value>=32) {
            scratch[u64(emitted++)*streams+stream]=u8(value>>8);
            scratch[u64(emitted++)*streams+stream]=u8(value);
        }
        u32 code=encoding[model*33+min(value,32u)], frequency=code>>16, start=code&65535u;
        u32 limit=((SC_LOWER >> SC_SCALE) << 8)*frequency;
        while (state >= limit) {
            scratch[u64(emitted++)*streams+stream]=u8(state); state >>= 8;
        }
        u32 quotient=state/frequency;
        state=(quotient<<SC_SCALE)+(state-quotient*frequency)+start;
    }
    u32 bytes=literal ? 2*count : min(emitted+4,2*count);
    models[stream]=bytes<2*count ? model : 254;
    if (maximum<=128 && 2*nonzero<bytes) {
        models[stream]=252; bytes=2*nonzero;
    }
    sizes[stream]=bytes; states[stream]=state;
}

extern "C" __global__ void sc_compact(
    const void* raw, int itemsize, u32 scans, u32 pixels, u32 interval,
    const u8* scratch, const u32* offsets, const u32* states,
    const u8* models, u8* payload, u32 streams
) {
    u32 stream=blockIdx.x*blockDim.x+threadIdx.x;
    if (stream>=streams) return;
    u32 begin=offsets[stream], size=offsets[stream+1]-begin, model=models[stream];
    if (model==253) return;
    if (model==252) {
        u32 first=(stream/pixels)*interval,pixel=stream%pixels,at=begin;
        for (u32 i=0;i<min(interval,scans-first);++i) {
            u32 value=sc_raw(raw,u64(first+i)*pixels+pixel,itemsize);
            if (value) {
                u32 event=(i<<7)|(value-1);
                payload[at++]=u8(event); payload[at++]=u8(event>>8);
            }
        }
        return;
    }
    if (model==255) {
        payload[begin]=u8(states[stream]); payload[begin+1]=u8(states[stream]>>8); return;
    }
    if (model==254) {
        u32 first=(stream/pixels)*interval, pixel=stream%pixels;
        for (u32 i=0;i<size/2;++i) {
            u32 v=sc_raw(raw,u64(first+i)*pixels+pixel,itemsize);
            payload[begin+2*i]=u8(v); payload[begin+2*i+1]=u8(v>>8);
        }
        return;
    }
    for (int i=0;i<4;++i) payload[begin+i]=u8(states[stream]>>(8*i));
    for (u32 i=4;i<size;++i) payload[begin+i]=scratch[u64(size-1-i)*streams+stream];
}

template<typename Output>
__device__ void sc_decode_range(
    const u8* payload, const u32* offsets, const u8* models,
    const u32* decoding, Output* raw, u32* errors, u32 scans, u32 pixels,
    u32 interval, u32 first, u32 count, u32 first_stream, u32 stop_stream,
    u32 detector_cols, u32 row0, u32 col0, u32 crop_rows, u32 crop_cols
) {
    u32 stream=first_stream+blockIdx.x*blockDim.x+threadIdx.x;
    if (stream>=stop_stream) return;
    u32 selected_pixels=crop_rows*crop_cols;
    u32 pixel=stream%selected_pixels;
    u32 block_first=(stream/selected_pixels)*interval;
    u32 source_pixel=(row0+pixel/crop_cols)*detector_cols+col0+pixel%crop_cols;
    stream=(stream/selected_pixels)*pixels+source_pixel;
    u32 block_count=min(interval,scans-block_first),stop=first+count;
    StreamReader reader(payload,offsets,models,decoding,stream);
    for (u32 i=0;i<block_count;++i) {
        u32 value=reader.next(),scan=block_first+i;
        if (scan>=first && scan<stop)
            raw[u64(scan-first)*selected_pixels+pixel]=Output(value);
    }
    if (!reader.finished()) atomicOr(errors,1u);
}

extern "C" __global__ void sc_decode_range_u8(
    const u8* payload, const u32* offsets, const u8* models,
    const u32* decoding, u8* raw, u32* errors, u32 scans, u32 pixels,
    u32 interval, u32 first, u32 count, u32 first_stream, u32 stop_stream,
    u32 detector_cols, u32 row0, u32 col0, u32 crop_rows, u32 crop_cols
) {
    sc_decode_range(payload,offsets,models,decoding,raw,errors,scans,pixels,
                    interval,first,count,first_stream,stop_stream,
                    detector_cols,row0,col0,crop_rows,crop_cols);
}

extern "C" __global__ void sc_decode_range_u16(
    const u8* payload, const u32* offsets, const u8* models,
    const u32* decoding, u16* raw, u32* errors, u32 scans, u32 pixels,
    u32 interval, u32 first, u32 count, u32 first_stream, u32 stop_stream,
    u32 detector_cols, u32 row0, u32 col0, u32 crop_rows, u32 crop_cols
) {
    sc_decode_range(payload,offsets,models,decoding,raw,errors,scans,pixels,
                    interval,first,count,first_stream,stop_stream,
                    detector_cols,row0,col0,crop_rows,crop_cols);
}

// Exact detector sum of one chunk without expanding it. A stream holds one detector pixel over
// `interval` scans, so each thread adds its own stream in a register and contributes one total:
// the dense route wrote every count to memory, masked it and read it back to add it.
extern "C" __global__ void sc_total(
    const u8* payload, const u32* offsets, const u8* models, const u32* decoding,
    u64* total, u32* errors, u32 scans, u32 pixels, u32 interval
) {
    u32 stream=blockIdx.x*blockDim.x+threadIdx.x;
    if (stream>=((scans+interval-1)/interval)*pixels) return;
    u32 count=min(interval,scans-(stream/pixels)*interval);
    StreamReader reader(payload,offsets,models,decoding,stream);
    u64 sum=0;
    for (u32 i=0;i<count;++i) sum+=reader.next();
    if (!reader.finished()) atomicOr(errors,1u);
    if (sum) atomicAdd(total+stream%pixels,sum);
}

// The spatial index of one scan: the sum of every 8x8 detector tile (the leaves), then of every
// 32x32 tile (the roots). One warp sums one leaf from the counts.
extern "C" __global__ void sc_fields(
    const void* raw, int itemsize, const u8* valid, u32* fields,
    u32 scans, u32 rows, u32 cols, u32 field_count
) {
    u64 item=(u64(blockIdx.x)*blockDim.x+threadIdx.x)/32;
    u32 lane=threadIdx.x&31;
    u32 leaf_cols=(cols+7)/8, leaves=((rows+7)/8)*leaf_cols;
    if (item>=u64(scans)*leaves) return;
    u32 scan=item/leaves, leaf=item%leaves;
    u32 row0=(leaf/leaf_cols)*8, col0=(leaf%leaf_cols)*8, sum=0;
    for (u32 offset=lane;offset<64;offset+=32) {
        u32 row=row0+offset/8,col=col0+offset%8;
        if (row<rows && col<cols && valid[row*cols+col])
            sum+=sc_raw(raw,(u64(scan)*rows+row)*cols+col,itemsize);
    }
    sum=__reduce_add_sync(0xffffffffu,sum);
    if (!lane) fields[u64(scan)*field_count+leaf]=sum;
}

// A root is the sum of its sixteen leaves, which sc_fields has written: summing its 1024 counts
// instead read the whole block of counts a second time.
extern "C" __global__ void sc_field_roots(u32* fields, u32 scans, u32 rows, u32 cols, u32 field_count) {
    u64 item=u64(blockIdx.x)*blockDim.x+threadIdx.x;
    u32 leaf_rows=(rows+7)/8, leaf_cols=(cols+7)/8, leaves=leaf_rows*leaf_cols, roots=field_count-leaves;
    if (item>=u64(scans)*roots) return;
    u32 scan=item/roots, root=item%roots, root_cols=(cols+31)/32;
    u32 leaf_row0=(root/root_cols)*4, leaf_col0=(root%root_cols)*4, sum=0;
    u32* scan_fields=fields+u64(scan)*field_count;
    for (u32 row=leaf_row0;row<min(leaf_row0+4,leaf_rows);++row)
        for (u32 col=leaf_col0;col<min(leaf_col0+4,leaf_cols);++col) sum+=scan_fields[row*leaf_cols+col];
    scan_fields[leaves+root]=sum;
}

extern "C" __global__ void sc_field_sizes(
    const u32* values, u8* widths, u64* sizes, u32 scans, u32 fields, u32 interval
) {
    u32 stream=blockIdx.x*blockDim.x+threadIdx.x;
    if (stream>=((scans+interval-1)/interval)*fields) return;
    u32 first=(stream/fields)*interval, field=stream%fields, bits=0;
    u32 count=min(interval,scans-first);
    for (u32 i=0;i<count;++i) bits|=values[u64(first+i)*fields+field];
    u32 width=bits ? 32-__clz(bits) : 0;
    widths[stream]=width; sizes[stream]=(u64(count)*width+31)/32;
}

extern "C" __global__ void sc_pack_fields(
    const u32* values, const u8* widths, const u64* offsets, u32* payload,
    u32 scans, u32 fields, u32 interval
) {
    u32 stream=blockIdx.x*blockDim.x+threadIdx.x;
    if (stream>=((scans+interval-1)/interval)*fields) return;
    u32 width=widths[stream]; if (!width) return;
    u32 first=(stream/fields)*interval,field=stream%fields,available=0;
    u64 reservoir=0,at=offsets[stream];
    for (u32 i=0;i<min(interval,scans-first);++i) {
        reservoir|=u64(values[u64(first+i)*fields+field])<<available;
        available+=width;
        if (available>=32) { payload[at++]=u32(reservoir); reservoir>>=32;available-=32; }
    }
    if (available) payload[at]=u32(reservoir);
}

// One 12-word row per encoded chunk (StreamedSeriesCompute builds them): 0 payload,
// 1 stream offsets, 2 models, 3 decoding table, 4 field words, 5 field starts,
// 6 field widths, 7 first scan, 8 scans, 9 acquisition, 10 valid pixels, 11 interval.
static const u32 SC_DESCRIPTOR = 12;
__device__ StreamReader chunk_reader(const u64* d, u32 local_stream) {
    return StreamReader((const u8*)d[0],(const u32*)d[1],(const u8*)d[2],(const u32*)d[3],local_stream);
}

__device__ u32 sc_field(const u64* d, u32 scan, u32 field, u32 fields, u32 interval) {
    interval=d[11];
    u32 stream=(scan/interval)*fields+field;
    u32 width=((const u8*)d[6])[stream];
    if (!width) return 0;
    u64 bit=u64(scan%interval)*width;
    u64 at=((const u64*)d[5])[stream]+bit/32;
    u64 value=((const u32*)d[4])[at];
    if ((bit%32)+width>32) value|=u64(((const u32*)d[4])[at+1])<<32;
    return u32(value>>(bit%32)) & (width==32 ? 0xffffffffu : (1u<<width)-1);
}

template<typename Output>
__device__ void sc_index_sum(const u64* descriptors, const u32* selected,
                            const int* coefficients, u32 count, const Output* previous,
                            Output* output, u64 scans, u32 fields, u32 interval, int delta) {
    const u64* d=descriptors+u64(blockIdx.y)*SC_DESCRIPTOR;
    u32 scan=blockIdx.x*blockDim.x+threadIdx.x;
    if (scan>=d[8]) return;
    u64 at=d[9]*scans+d[7]+scan;
    Output value=delta ? previous[at] : 0;
    for (u32 i=0;i<count;++i) value+=Output((long long)coefficients[i]*sc_field(d,scan,selected[i],fields,interval));
    output[at]=value;
}

template<typename Output>
__device__ void sc_residual(const u64* descriptors, const u32* selected,
                           const int* coefficients, u32 count, Output* output,
                           u32* errors, u64 scans, u32 pixels, u32 interval) {
    const u64* d=descriptors+u64(blockIdx.y)*SC_DESCRIPTOR;
    interval=d[11];
    u32 lane=threadIdx.x&31, group=blockIdx.x, warp=threadIdx.x/32;
    u32 groups=(count+31)/32, stream_block=(group/groups)*(blockDim.x/32)+warp;
    u32 selected_at=(group%groups)*32+lane;
    u32 first=stream_block*interval;
    if (first>=d[8]) return;
    u32 pixel=selected_at<count ? selected[selected_at] : 0;
    int coefficient=selected_at<count && ((const u8*)d[10])[pixel] ? coefficients[selected_at] : 0;
    u32 length=min(interval,u32(d[8])-first);
    u32 stream=stream_block*pixels+pixel;
    u32 model=((const u8*)d[2])[stream];
    const u8* payload=(const u8*)d[0];
    const u32* offsets=(const u32*)d[1];
    int constant=0;
    if (coefficient && model==252) {
        for (u32 at=offsets[stream];at<offsets[stream+1];at+=2) {
            u32 event=payload[at]|(u32(payload[at+1])<<8),position=event>>7;
            if (position>=length) { atomicOr(errors,1u); break; }
            atomicAdd(output+d[9]*scans+d[7]+first+position,Output((long long)coefficient*((event&127u)+1)));
        }
        coefficient=0;
    }
    if (coefficient && model==255) {
        u32 at=offsets[stream];
        constant=coefficient*int(payload[at]|(u32(payload[at+1])<<8));
        coefficient=0;
    }
    if (model==253) coefficient=0;
    constant=__reduce_add_sync(0xffffffffu,constant);
    if (constant) for (u32 i=lane;i<length;i+=32)
        atomicAdd(output+d[9]*scans+d[7]+first+i,Output((long long)constant));
    if (__all_sync(0xffffffffu,coefficient==0)) return;
    StreamReader reader=chunk_reader(d,stream);
    for (u32 batch=0;batch<length;batch+=32) {
        int result=0;
        for (u32 j=0;j<min(32u,length-batch);++j) {
            int value=coefficient ? int(reader.next())*coefficient : 0;
            value=__reduce_add_sync(0xffffffffu,value);
            if (lane==j) result=value;
        }
        if (batch+lane<length && result)
            atomicAdd(output+d[9]*scans+d[7]+first+batch+lane,Output((long long)result));
    }
    if (coefficient && !reader.finished()) atomicOr(errors,1u);
}

#define SC_SUM(BITS, TYPE) \
extern "C" __global__ void sc_index_u##BITS(const u64* d,const u32* selected,const int* coefficients,u32 count,const TYPE* previous,TYPE* output,u64 scans,u32 fields,u32 interval,int delta) { sc_index_sum(d,selected,coefficients,count,previous,output,scans,fields,interval,delta); } \
extern "C" __global__ void sc_residual_u##BITS(const u64* d,const u32* selected,const int* coefficients,u32 count,TYPE* output,u32* errors,u64 scans,u32 pixels,u32 interval) { sc_residual(d,selected,coefficients,count,output,errors,scans,pixels,interval); }
SC_SUM(32,u32)
SC_SUM(64,u64)

// Two weightings of the same pixels from one decode of their streams: the row and the column
// moment of a centre of mass. Decoding is nearly all of a residual pass, so the second sum costs
// one multiply and one warp reduction per count instead of a second pass. Sums as sc_residual.
extern "C" __global__ void sc_moment_pair(
    const u64* descriptors, const u32* selected, const int* first_weights, const int* second_weights,
    u32 count, u64* first_output, u64* second_output, u32* errors, u64 scans, u32 pixels, u32 interval
) {
    const u64* d=descriptors+u64(blockIdx.y)*SC_DESCRIPTOR;
    interval=d[11];
    u32 lane=threadIdx.x&31, group=blockIdx.x, warp=threadIdx.x/32;
    u32 groups=(count+31)/32, stream_block=(group/groups)*(blockDim.x/32)+warp;
    u32 selected_at=(group%groups)*32+lane;
    u32 first=stream_block*interval;
    if (first>=d[8]) return;
    u32 pixel=selected_at<count ? selected[selected_at] : 0;
    bool counted=selected_at<count && ((const u8*)d[10])[pixel];
    int first_weight=counted ? first_weights[selected_at] : 0;
    int second_weight=counted ? second_weights[selected_at] : 0;
    u32 length=min(interval,u32(d[8])-first);
    u32 stream=stream_block*pixels+pixel;
    u32 model=((const u8*)d[2])[stream];
    const u8* payload=(const u8*)d[0];
    const u32* offsets=(const u32*)d[1];
    u64 at_scan=d[9]*scans+d[7]+first;
    bool decodes=counted && model!=253;
    int first_constant=0, second_constant=0;
    if (decodes && model==252) {
        for (u32 at=offsets[stream];at<offsets[stream+1];at+=2) {
            u32 event=payload[at]|(u32(payload[at+1])<<8),position=event>>7;
            if (position>=length) { atomicOr(errors,1u); break; }
            long long value=(event&127u)+1;
            atomicAdd(first_output+at_scan+position,u64(first_weight*value));
            atomicAdd(second_output+at_scan+position,u64(second_weight*value));
        }
        decodes=false;
    }
    if (decodes && model==255) {
        int constant=int(payload[offsets[stream]]|(u32(payload[offsets[stream]+1])<<8));
        first_constant=first_weight*constant; second_constant=second_weight*constant;
        decodes=false;
    }
    first_constant=__reduce_add_sync(0xffffffffu,first_constant);
    second_constant=__reduce_add_sync(0xffffffffu,second_constant);
    if (first_constant || second_constant) for (u32 i=lane;i<length;i+=32) {
        atomicAdd(first_output+at_scan+i,u64((long long)first_constant));
        atomicAdd(second_output+at_scan+i,u64((long long)second_constant));
    }
    if (__all_sync(0xffffffffu,!decodes)) return;
    StreamReader reader=chunk_reader(d,stream);
    for (u32 batch=0;batch<length;batch+=32) {
        int first_result=0, second_result=0;
        for (u32 j=0;j<min(32u,length-batch);++j) {
            int value=decodes ? int(reader.next()) : 0;
            int first_sum=__reduce_add_sync(0xffffffffu,value*first_weight);
            int second_sum=__reduce_add_sync(0xffffffffu,value*second_weight);
            if (lane==j) { first_result=first_sum; second_result=second_sum; }
        }
        if (batch+lane<length) {
            if (first_result) atomicAdd(first_output+at_scan+batch+lane,u64((long long)first_result));
            if (second_result) atomicAdd(second_output+at_scan+batch+lane,u64((long long)second_result));
        }
    }
    if (decodes && !reader.finished()) atomicOr(errors,1u);
}

// Sum, or maximum, of selected scan positions per detector pixel without expanding a block. Each
// thread decodes one stream as far as its block's last selected scan (`stops`, per block of every
// descriptor row from `bases`; 0 skips the block) and adds the selected counts, times how often
// each scan was selected. Decoding whole blocks to gather a few frames wrote 512 patterns per block.
extern "C" __global__ void sc_selected(
    const u64* descriptors, const u32* multiplicity, const u32* stops, const u32* bases,
    u64* output, u32* errors, u32 pixels, int maximum
) {
    const u64* d=descriptors+u64(blockIdx.y)*SC_DESCRIPTOR;
    u32 interval=d[11], stream=blockIdx.x*blockDim.x+threadIdx.x;
    if (stream>=((u32(d[8])+interval-1)/interval)*pixels) return;
    u32 block=stream/pixels, stop=stops[bases[blockIdx.y]+block];
    if (!stop) return;
    const u32* times=multiplicity+d[7]+u64(block)*interval;
    StreamReader reader=chunk_reader(d,stream);
    u64 value=0;
    for (u32 i=0;i<stop;++i) {
        u64 count=reader.next();
        if (maximum) value=times[i] ? max(value,count) : value;
        else value+=count*times[i];
    }
    if (!reader.valid) atomicOr(errors,1u);
    u64* target=output+d[9]*pixels+stream%pixels;
    if (maximum) atomicMax(target,value);
    else if (value) atomicAdd(target,value);
}

template<typename Output>
__device__ void sc_frame(const u64* descriptors, Output* output, u32* errors,
                        u32 index, u32 pixels, u32 interval) {
    const u64* d=descriptors+u64(blockIdx.y)*SC_DESCRIPTOR;
    u32 pixel=blockIdx.x*blockDim.x+threadIdx.x;
    if (pixel>=pixels || index<d[7] || index>=d[7]+d[8]) return;
    interval=d[11];
    u32 local=index-d[7], offset=local%interval, stream=(local/interval)*pixels+pixel;
    StreamReader reader=chunk_reader(d,stream);
    u32 value=0;
    for (u32 i=0;i<=offset;++i) value=reader.next();
    output[d[9]*pixels+pixel]=value;
    if (!reader.valid) atomicOr(errors,1u);
}
extern "C" __global__ void sc_frame_u8(const u64* d,u8* output,u32* errors,u32 index,u32 pixels,u32 interval) { sc_frame(d,output,errors,index,pixels,interval); }
extern "C" __global__ void sc_frame_u16(const u64* d,u16* output,u32* errors,u32 index,u32 pixels,u32 interval) { sc_frame(d,output,errors,index,pixels,interval); }
