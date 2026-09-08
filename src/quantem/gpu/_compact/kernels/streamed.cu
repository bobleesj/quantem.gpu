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

extern "C" __global__ void sc_encode(
    const void* raw, int itemsize, u32 scans, u32 pixels, u32 interval,
    const u32* encoding, u8* scratch, u32* sizes, u32* states, u8* models,
    u32 streams
) {
    u32 stream = blockIdx.x * blockDim.x + threadIdx.x;
    if (stream >= streams) return;
    u32 pixel=stream%pixels, first=(stream/pixels)*interval;
    u32 count=min(interval,scans-first), maximum=0, minimum=65535, sum=0, nonzero=0;
    for (u32 i=0;i<count;++i) {
        u32 v=sc_raw(raw,u64(first+i)*pixels+pixel,itemsize);
        maximum=max(maximum,v); minimum=min(minimum,v); sum+=min(v,32u); nonzero+=v!=0;
    }
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

extern "C" __global__ void sc_decode(
    const u8* payload, const u32* offsets, const u8* models,
    const u32* decoding, u16* raw, u32* errors, u32 scans, u32 pixels,
    u32 interval, u32 streams
) {
    u32 stream=blockIdx.x*blockDim.x+threadIdx.x;
    if (stream>=streams) return;
    u32 first=(stream/pixels)*interval, pixel=stream%pixels;
    StreamReader reader(payload,offsets,models,decoding,stream);
    for (u32 i=0;i<min(interval,scans-first);++i)
        raw[u64(first+i)*pixels+pixel]=reader.next();
    if (!reader.finished()) atomicOr(errors,1u);
}

extern "C" __global__ void sc_fields(
    const void* raw, int itemsize, const u8* valid, u32* fields,
    u32 scans, u32 rows, u32 cols, u32 field_count
) {
    u64 item=(u64(blockIdx.x)*blockDim.x+threadIdx.x)/32;
    u32 lane=threadIdx.x&31;
    if (item>=u64(scans)*field_count) return;
    u32 scan=item/field_count, field=item%field_count;
    u32 leaves=((rows+7)/8)*((cols+7)/8);
    u32 side=field<leaves ? 8 : 32;
    if (field>=leaves) field-=leaves;
    u32 field_cols=(cols+side-1)/side;
    u32 row0=(field/field_cols)*side, col0=(field%field_cols)*side, sum=0;
    for (u32 offset=lane;offset<side*side;offset+=32) {
        u32 row=row0+offset/side,col=col0+offset%side;
        if (row<rows && col<cols && valid[row*cols+col])
            sum+=sc_raw(raw,(u64(scan)*rows+row)*cols+col,itemsize);
    }
    sum=__reduce_add_sync(0xffffffffu,sum);
    if (!lane) fields[item]=sum;
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

static const u32 SC_DESCRIPTOR = 23;
struct CountReader {
    StreamReader encoded;
    Reader portable;
    const u64* descriptor;
    u64 stream;
    u32 index, kind;
    bool valid;
    __device__ CountReader(const u64* d, u32 local_stream, u32 pixels)
        : descriptor(d), index(0), kind(d[11]), valid(true) {
        stream=(d[7]/d[12])*pixels+local_stream;
        if (!kind) encoded=StreamReader((const u8*)d[0],(const u32*)d[1],(const u8*)d[2],(const u32*)d[3],local_stream);
        else if (kind==1) portable=Reader((const u8*)d[14],(const u64*)d[15],(const u32*)d[16],
            (const u32*)d[17],(const u16*)d[18],(const u16*)d[19],(const u16*)d[20],(const u8*)d[21],stream,d[13]);
    }
    __device__ u32 next() {
        if (!kind) { u32 value=encoded.next(); valid=encoded.valid; return value; }
        if (kind==1) { u32 value=portable.next(); valid=portable.valid; return value; }
        return packed_count((const u32*)descriptor[14],(const u64*)descriptor[15],(const u8*)descriptor[16],stream,index++);
    }
    __device__ bool finished() const {
        return !kind ? encoded.finished() : kind==1 ? portable.finished() : true;
    }
};

__device__ u32 sc_field(const u64* d, u32 scan, u32 field, u32 fields, u32 interval) {
    interval=d[12];
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
    interval=d[12];
    u32 lane=threadIdx.x&31, group=blockIdx.x, warp=threadIdx.x/32;
    u32 groups=(count+31)/32, stream_block=(group/groups)*(blockDim.x/32)+warp;
    u32 selected_at=(group%groups)*32+lane;
    u32 first=stream_block*interval;
    if (first>=d[8]) return;
    u32 pixel=selected_at<count ? selected[selected_at] : 0;
    int coefficient=selected_at<count && ((const u8*)d[10])[pixel] ? coefficients[selected_at] : 0;
    u32 length=min(interval,u32(d[8])-first);
    if (!d[11]) {
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
    }
    CountReader reader(d,stream_block*pixels+pixel,pixels);
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

template<typename Output>
__device__ void sc_frame(const u64* descriptors, Output* output, u32* errors,
                        u32 index, u32 pixels, u32 interval) {
    const u64* d=descriptors+u64(blockIdx.y)*SC_DESCRIPTOR;
    u32 pixel=blockIdx.x*blockDim.x+threadIdx.x;
    if (pixel>=pixels || index<d[7] || index>=d[7]+d[8]) return;
    interval=d[12];
    u32 local=index-d[7], offset=local%interval, stream=(local/interval)*pixels+pixel;
    CountReader reader(d,stream,pixels);
    u32 value=0;
    for (u32 i=0;i<=offset;++i) value=reader.next();
    output[d[9]*pixels+pixel]=value;
    if (!reader.valid) atomicOr(errors,1u);
}
extern "C" __global__ void sc_frame_u8(const u64* d,u8* output,u32* errors,u32 index,u32 pixels,u32 interval) { sc_frame(d,output,errors,index,pixels,interval); }
extern "C" __global__ void sc_frame_u16(const u64* d,u16* output,u32* errors,u32 index,u32 pixels,u32 interval) { sc_frame(d,output,errors,index,pixels,interval); }
