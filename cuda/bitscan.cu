/*
   Copyright (c) 2012-2013 The Ohio State University.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include "scanImpl.cu"
#include "../include/common.h"
#include "../include/gpuCudaLib.h"
#include <iostream>
using namespace std;
#define TEST  1

#define utype  unsigned long long
#define type long long
#define type_len (sizeof(type) * 8)
double ave_time ;
__global__ void static equal(type * a, int n, utype constC){
    int offset = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i=offset; i<n; i+=stride){    
        a[i] = constC;
    }
}

__global__ void static genScanFilter_int_lth_bit(type * col,int n, utype constC,type * lt, type * eq){
    int offset = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;  

    for(int i=offset; i < n; i+=stride){
        lt[i] = lt[i] | (eq[i] & ~constC & col[i]);
        eq[i] = eq[i] & ~(col[i] ^ constC);
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
        if(constC != 0) 
          constC = - 1;
        return constC;
}
inline type ran(){
  type x = rand();
  if(sizeof(type) == 8) return (x << 32) + rand();
  return x;
}
void profilebitscan(type        *h_a, 
                   type       *h_b, 
                   type       *d, 
                   type *lt,
                   type *eq,
                   int  n,
                   utype where,
                   char         *desc,
                   unsigned int loopTotal)
{

  dim3 block(256);
  dim3 grid(1024);
  float time,stime;
  // events for timing
  cudaEvent_t startEvent, stopEvent; 
  checkCuda( cudaEventCreate(&startEvent) );
  checkCuda( cudaEventCreate(&stopEvent) ); 
  stime=0;
  utype bytes=n * sizeof(type);
  for(int loop = 1; loop <= loopTotal; loop++){  
      checkCuda( cudaEventRecord(startEvent, 0) );
      checkCuda( cudaMemcpy(d, h_a, bytes, cudaMemcpyHostToDevice) );
      checkCuda(cudaThreadSynchronize());
      utype c = 0;
      for(int i = 0;i < type_len;i++) c += ((utype)1 << i);
      equal<<<grid,block>>>(lt, n/type_len, 0) ;
      equal<<<grid,block>>>(eq, n/type_len, c) ;
      checkCuda(cudaThreadSynchronize());

      for(int j = 0; j < type_len; ++j){
            int constC =bit_constC(where,j);            
            genScanFilter_int_lth_bit<<<grid,block>>>(d + j * (n / type_len), n / type_len,  constC, lt, eq);
            checkCuda(cudaThreadSynchronize());
      }

      checkCuda( cudaMemcpy(h_b, lt, n / type_len * sizeof(type), cudaMemcpyDeviceToHost) );
      checkCuda( cudaMemcpy(h_b + n / type_len , eq, n / type_len * sizeof(type), cudaMemcpyDeviceToHost) );
      checkCuda( cudaEventRecord(stopEvent, 0) );
      checkCuda( cudaEventSynchronize(stopEvent) );

      checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );

      ave_time += time;
      //printf("time=%f\n",stime);
  }
  //cerr<<stime<<endl;
  //printf("%f\n" ,bytes * 1e-6/(stime / loopTotal));
  checkCuda( cudaEventDestroy(startEvent) );
  checkCuda( cudaEventDestroy(stopEvent) );
}



int main(int argc, char ** argv)
{
  #ifdef TEST
      freopen("scan.in","r",stdin);
      freopen("scan_bit.out","w",stdout);
  #endif
  int inputN;
  sscanf(argv[1],"%d",&inputN);
  utype nElements = inputN;
  #ifdef TEST
      scanf("%d",&nElements);
  #endif
  utype bytes = nElements * sizeof(type);

  // host arrays
  type *h_aPageable, *h_bPageable,*know_stop_constC_cpu;   
  type *h_bitPageable;
  type *know_stop_len_cpu;


  // device array
  type *d_a;
  type *lt,*eq;

  // allocate and initialize
  h_aPageable = (type*)malloc(bytes );          
  h_bPageable = (type*)malloc(bytes );
  h_bitPageable =(type *)malloc(bytes );
  know_stop_len_cpu = (type *)malloc(bytes );     
  know_stop_constC_cpu = (type *)malloc(bytes );           // host pageable
  //checkCuda( cudaMallocHost((void**)&h_aPinned, bytes  ) ); // host pinned
  //checkCuda( cudaMallocHost((void**)&h_bPinned, bytes  ) );  
  checkCuda( cudaMalloc((void**)&d_a, bytes  ) );           // device
  checkCuda( cudaMalloc((void**)&lt, bytes ) ); // device return
  checkCuda( cudaMalloc((void**)&eq, bytes  ) );  

  int early_size = 1024*1024; 
  sscanf(argv[2],"%d",&early_size);  
  srand(0);
  for (int i = 0; i < nElements; ++i) h_aPageable[i] = ran()%((utype)1<<(type_len - 1));  
  #ifdef TEST
      for (int i = 0; i < nElements; ++i){ 
      if(sizeof(type)==4) scanf("%d",h_aPageable + i);
      else scanf("%lld",h_aPageable + i);
     // cerr<<h_aPageable[i]<<endl;
      }
  #endif

  for (int i = 0; i < nElements; ++i) 
    for(int j = type_len - 1; j >= 0; --j){
        h_bitPageable[i / type_len + (type_len - 1 - j)*(nElements/type_len)] += (((h_aPageable[i] &((utype)1<<j))>>j)<<(type_len - 1 - i % type_len));
        //h_bitPageable[i / 32 + (31-j)*(nElements/32)] += 0;
    }



  // output device info and transfer size
  cudaDeviceProp prop;

  checkCuda( cudaGetDeviceProperties(&prop, 0) );

  // printf("\nDevice: %s\n", prop.name);
  // if(bytes< 1024){
  //     printf("scan size (B): %d\n", bytes);
  // }else if (bytes < 1024 * 1024)
  // {
  //     printf("scan size (KB): %d\n", bytes / (1024));              
  // }else{
  //     printf("scan size (MB): %d\n", bytes / (1024 * 1024));
  // }



  type constC = ran()%((utype)1<<(type_len - 1));
  // perform  scan eq
 // profilescan(h_aPageable, h_bPageable, d_a, filter, nElements, constC,"Pageable",20);
  //profilescan(h_aPinned, h_bPinned, d_a, filter,nElements, constC,"Pinned",20);
  #ifdef TEST
      int test_num = 0;
      scanf("%d",&test_num);
      for(int i  = 0; i < test_num; i++){
        if(sizeof(type)==4)
        scanf("%d",&constC);
        else scanf("%lld",&constC);
        profilebitscan(h_bitPageable, h_bPageable, d_a, lt, eq, nElements, constC,"Pageable",1);
      }
  #else 
    constC = ran()%((utype)1<<(type_len - 1));
      profilebitscan(h_bitPageable, h_bPageable, d_a, lt, eq, nElements, constC,"Pageable",1);
  #endif

  // for(int i = 0; i < nElements; i++) printf("%3u ",h_aPageable[i]);printf("\n");
  // for(int i = 0; i < nElements; i++) printf("%3u ",((h_bPageable[i/32] & (1u << (31 - i % 32)))>> (31 - i % 32)));printf("\n");
  // for(int i = 0; i < nElements; i++) printf("%3u ",((h_bPageable[i/32 + nElements/32] & (1u << (31 - i % 32)))>> (31 - i % 32)));printf("\n");
  // for(int i = 0; i < nElements; i++) printf("%3u ",h_bitPageable[i]);printf("\n");
  #ifdef TEST
      for(int i = 0; i < nElements; i++) {
          int x =(h_bPageable[i/type_len] & ((utype)1 << (type_len - 1 - i % type_len)))>> (type_len - 1 - i % type_len);
          int y =(h_bPageable[i/type_len + nElements/type_len] & ((utype)1 << (type_len - 1 - i % type_len)))>> (type_len - 1 - i % type_len);
          if(x ==0 && y == 0) printf("%d\n",1);
          else printf("%d\n", 0);

      }
          printf("%.6f\n",bytes* 1e-6 /(ave_time / test_num));
      cerr<<bytes* 1e-6 /(ave_time / test_num)<<endl;

  #endif
  // cleanup

  cudaFree(lt);
  cudaFree(eq);
  free(h_aPageable); 
}
