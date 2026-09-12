typedef unsigned long long u64;
__device__ float count_at(const unsigned* words,const u64* offsets,const unsigned char* widths,const unsigned char* valid,int r,int c,int d,int rows,int cols,int pixels,int block){
 if(r<0||r>=rows||c<0||c>=cols||!valid[d])return 0;
 u64 scan=(u64)r*cols+c,stream=(scan/block)*pixels+d;
 unsigned width=widths[stream];if(!width)return 0;
 u64 bit=(scan%block)*width,word=offsets[stream]+bit/32;
 unsigned shift=bit%32;u64 value=words[word];
 if(shift+width>32)value|=(u64)words[word+1]<<32;
 return float((value>>shift)&((1u<<width)-1));
}
extern "C" __global__ void sample(const unsigned* words,const u64* offsets,const unsigned char* widths,const unsigned char* valid,const float* shifts,float* out,int first,int scans,int rows,int cols,int pixels,int block){
 u64 item=(u64)blockIdx.x*blockDim.x+threadIdx.x,vectors=(pixels+3)/4;
 if(item>=(u64)scans*vectors)return;
 int local_scan=item/vectors,first_d=(item%vectors)*4,scan=first+local_scan;
 double rr=-double(shifts[0]),cc=-double(shifts[1]);
 int r0=int(floor(rr)),c0=int(floor(cc));double rf=rr-r0,cf=cc-c0;
 for(int lane=0;lane<4;lane++){
  int d=first_d+lane;if(d>=pixels)continue;float acc=0;
  for(int dr=0;dr<2;dr++)for(int dc=0;dc<2;dc++){
   float weight=float((dr?rf:1-rf)*(dc?cf:1-cf));
   float value=count_at(words,offsets,widths,valid,scan/cols+r0+dr,scan%cols+c0+dc,d,rows,cols,pixels,block);
   acc=__fadd_rn(acc,__fmul_rn(value,weight));
  }
  out[(u64)local_scan*pixels+d]=acc;
 }
}
__device__ float dense_count_at(const void* values,int item_bytes,const unsigned char* valid,int r,int c,int d,int rows,int cols,int pixels,int first_row,int decoded_rows){
 if(r<0||r>=rows||c<0||c>=cols||r<first_row||r>=first_row+decoded_rows||!valid[d])return 0;
 u64 index=(u64(r-first_row)*cols+c)*pixels+d;
 return item_bytes==1 ? float(((const unsigned char*)values)[index]) : float(((const unsigned short*)values)[index]);
}
extern "C" __global__ void sample_dense(const void* values,int item_bytes,const unsigned char* valid,const float* shifts,float* out,int first,int scans,int rows,int cols,int pixels,int first_row,int decoded_rows){
 u64 item=(u64)blockIdx.x*blockDim.x+threadIdx.x,vectors=(pixels+3)/4;
 if(item>=(u64)scans*vectors)return;
 int local_scan=item/vectors,first_d=(item%vectors)*4,scan=first+local_scan;
 double rr=-double(shifts[0]),cc=-double(shifts[1]);
 int r0=int(floor(rr)),c0=int(floor(cc));double rf=rr-r0,cf=cc-c0;
 for(int lane=0;lane<4;lane++){
  int d=first_d+lane;if(d>=pixels)continue;float acc=0;
  for(int dr=0;dr<2;dr++)for(int dc=0;dc<2;dc++){
   float weight=float((dr?rf:1-rf)*(dc?cf:1-cf));
   float value=dense_count_at(values,item_bytes,valid,scan/cols+r0+dr,scan%cols+c0+dc,d,rows,cols,pixels,first_row,decoded_rows);
   acc=__fadd_rn(acc,__fmul_rn(value,weight));
  }
  out[(u64)local_scan*pixels+d]=acc;
 }
}
extern "C" __global__ void accumulate(const float* sample,const float* shift,const float* wi,float* num,int scans,int height,int width,int initialize){
 int pixels=height*width;u64 item=(u64)blockIdx.x*blockDim.x+threadIdx.x,vectors=(pixels+3)/4;
 if(item>=(u64)scans*vectors)return;
 int scan=item/vectors,first_d=(item%vectors)*4;
 double rr=-double(shift[0])*(height-1)/height,cc=-double(shift[1])*(width-1)/width;
 int r0=int(floor(rr)),c0=int(floor(cc));double rf=rr-r0,cf=cc-c0;
 for(int lane=0;lane<4;lane++){
  int d=first_d+lane;if(d>=pixels)continue;int r=d/width,c=d%width;float acc=0;
  for(int dr=0;dr<2;dr++)for(int dc=0;dc<2;dc++){
   int sr=r+r0+dr,sc=c+c0+dc;if(sr<0||sr>=height||sc<0||sc>=width)continue;
   float weight=float((dr?rf:1-rf)*(dc?cf:1-cf));
   acc=__fadd_rn(acc,__fmul_rn(sample[(u64)scan*pixels+sr*width+sc],weight));
  }
  u64 output=(u64)scan*pixels+d;float value=__fmul_rn(wi[scan],acc);
  num[output]=initialize?value:__fadd_rn(num[output],value);
 }
}
extern "C" __global__ void normalize(float* values,const float* real_weights,const float* detector_weights,const float* edge,int first,int scans,int total_scans,int pixels,int sources){
 u64 item=(u64)blockIdx.x*blockDim.x+threadIdx.x,vectors=(pixels+3)/4;
 if(item>=(u64)scans*vectors)return;
 int scan=item/vectors,first_d=(item%vectors)*4;
 for(int lane=0;lane<4;lane++){
  int d=first_d+lane;if(d>=pixels)continue;float denominator=edge[d];
  for(int source=0;source<sources;source++){
   float wi=real_weights[(u64)source*total_scans+first+scan];
   float wd=detector_weights[(u64)source*pixels+d];
   denominator=__fadd_rn(denominator,__fmul_rn(wi,wd));
  }
  u64 output=(u64)scan*pixels+d;
  values[output]=denominator==0?0:values[output]/denominator;
 }
}
