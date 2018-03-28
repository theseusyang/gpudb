#define TEST 1

#define utype  unsigned long long
#define type long long
#define type_len (sizeof(type) * 8)

double ave_time  = 0;

const int know_stop_size=10000010;

type nlz(utype x){
   type n;
   if(sizeof(n)==4){
     if (x == 0) return(type_len);
     n = 1;
     if ((x >> 16) == 0) {n = n +16; x = x <<16;}
     if ((x >> 24) == 0) {n = n + 8; x = x << 8;}
     if ((x >> 28) == 0) {n = n + 4; x = x << 4;}
     if ((x >> 30) == 0) {n = n + 2; x = x << 2;}
     n = n - (x >> 31);
  }else{
     if (x == 0) return(64);
     n = 1;
     if ((x >> 32) == 0) {n = n+ 32; x = x << 32;}
     if ((x >> 48) == 0) {n = n +16; x = x << 16;}
     if ((x >> 56) == 0) {n = n + 8; x = x << 8;}
     if ((x >> 60) == 0) {n = n + 4; x = x << 4;}
     if ((x >> 62) == 0) {n = n + 2; x = x << 2;}
     n = n - (x >> 63);
  }
   return n;
}
__global__ void static equal(type * a, int n, utype constC){
    int offset = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i=offset; i<n; i+=stride){    
        a[i] = constC;
    }
}


__global__ void static genScanFilter_int_lth_bit(type * col,int n, type *constC,int group_size, type * lt, type * eq){
    int offset = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;  
    for(int i=offset; i < n; i+=stride){
        lt[i] = lt[i] | (eq[i] & ~constC[i/group_size] & col[i]);
        eq[i] = eq[i] & ~(col[i] ^ constC[i/group_size]);
        //printf(" %d %u %u %u\n",i,lt[i],eq[i],col[i]);
    }
}
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", 
            cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}
inline type bit_constC(type where,int j){

        type constC = ((((utype)1 << (type_len - 1  - j) )) & where)>>(type_len - 1 - j);
        if(constC != 0) {
          constC =  - 1;
        }
        return constC;
}
inline type ran(){
  type x = rand();
  if(sizeof(type) == 8) return (x << 32) + rand();
  return x;
}