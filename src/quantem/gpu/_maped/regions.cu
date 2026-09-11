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
 u64 item=(u64)blockIdx.x*blockDim.x+threadIdx.x;if(item>=(u64)scans*pixels)return;
 int scan=first+item/pixels,d=item%pixels;
 double rr=-double(shifts[0]),cc=-double(shifts[1]);
 int r0=int(floor(rr)),c0=int(floor(cc));double rf=rr-r0,cf=cc-c0;
 float acc=0;
 for(int dr=0;dr<2;dr++)for(int dc=0;dc<2;dc++){
  float weight=float((dr?rf:1-rf)*(dc?cf:1-cf));
  float value=count_at(words,offsets,widths,valid,scan/cols+r0+dr,scan%cols+c0+dc,d,rows,cols,pixels,block);
  acc=__fadd_rn(acc,__fmul_rn(value,weight));
 }
 out[item]=acc;
}
extern "C" __global__ void accumulate(const float* sample,const float* shift,const float* wi,float* num,int scans,int height,int width){
 int pixels=height*width;u64 item=(u64)blockIdx.x*blockDim.x+threadIdx.x;
 if(item>=(u64)scans*pixels)return;
 int scan=item/pixels,d=item%pixels,r=d/width,c=d%width;
 double rr=-double(shift[0])*(height-1)/height,cc=-double(shift[1])*(width-1)/width;
 int r0=int(floor(rr)),c0=int(floor(cc));double rf=rr-r0,cf=cc-c0;
 float acc=0;
 for(int dr=0;dr<2;dr++)for(int dc=0;dc<2;dc++){
  int sr=r+r0+dr,sc=c+c0+dc;if(sr<0||sr>=height||sc<0||sc>=width)continue;
  float weight=float((dr?rf:1-rf)*(dc?cf:1-cf));
  acc=__fadd_rn(acc,__fmul_rn(sample[(u64)scan*pixels+sr*width+sc],weight));
 }
 num[item]=__fadd_rn(num[item],__fmul_rn(wi[scan],acc));
}
