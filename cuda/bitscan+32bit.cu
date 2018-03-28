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

//#define TEST  1


__global__ void static equal(int * a, int n, unsigned int constC){
    int offset = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i=offset; i<n; i+=stride){    
        a[i] = constC;
    }
}
__global__ void static genScanFilter_int_lth_bit(int * col,int n, unsigned int constC,int * lt, int * eq){
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
inline int bit_constC(int where,int j){

        int constC = (((1U << (31 - j )) & where)>>(31 - j));
        if(constC != 0) 
          constC = (1LL << 32) - 1;
        return constC;
}

void profilebitscan(int        *h_a, 
                   int        *h_b, 
                   int        *d, 
                   int *lt,
                   int *eq,
                   int  n,
                   unsigned int where,
                   char         *desc,
                   unsigned int loopTotal)
{

  dim3 block(256);
  dim3 grid(2048);
  float time,stime;
  // events for timing
  cudaEvent_t startEvent, stopEvent; 
  checkCuda( cudaEventCreate(&startEvent) );
  checkCuda( cudaEventCreate(&stopEvent) ); 
  stime=0;
  int bytes=n * sizeof(int);
  for(int loop = 1; loop <= loopTotal; loop++){  
      checkCuda( cudaEventRecord(startEvent, 0) );
      checkCuda( cudaMemcpy(d, h_a, bytes, cudaMemcpyHostToDevice) );
      checkCuda(cudaThreadSynchronize());
      unsigned int c = 0;
      for(int i = 0;i < 32;i++) c += (1u << i);
      equal<<<grid,block>>>(lt, n/32, 0) ;
      equal<<<grid,block>>>(eq, n/32, c) ;
      checkCuda(cudaThreadSynchronize());

      for(int j = 0; j < 32; ++j){
            int constC =bit_constC(where,j);
               // printf("%u %u\n",j,((((1U << (31 - j )) & where)>>(31-j))<< k));
            
            genScanFilter_int_lth_bit<<<grid,block>>>(d + j * (n / 32), n / 32,  constC, lt, eq);
            checkCuda(cudaThreadSynchronize());
      }

      checkCuda( cudaMemcpy(h_b, lt, n / 32 * 4, cudaMemcpyDeviceToHost) );
      checkCuda( cudaMemcpy(h_b + n / 32 , eq, n / 32 * 4, cudaMemcpyDeviceToHost) );
      checkCuda( cudaEventRecord(stopEvent, 0) );
      checkCuda( cudaEventSynchronize(stopEvent) );

      checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
      stime += time;
      printf("time=%f\n",stime);
  }
  printf("%f\n" ,bytes * 1e-6/(stime / loopTotal));
  checkCuda( cudaEventDestroy(startEvent) );
  checkCuda( cudaEventDestroy(stopEvent) );
}



int main(int argc, char ** argv)
{
  #ifdef TEST
      freopen("scan.in","r",stdin);
      freopen("scan_bit.out","w",stdout);
  #endif
  dim3 block(256);
  dim3 grid(2048);
  int inputN;
  sscanf(argv[1],"%d",&inputN);
  unsigned int nElements = inputN;
  const unsigned int bytes = nElements * sizeof(int);
  #ifdef TEST
      scanf("%d",&nElements);
  #endif
  // host arrays
  int *h_aPageable, *h_bPageable,*h_bitPageable;   
  int *h_aPinned, *h_bPinned;

  // device array
  int *d_a,*lt,*eq;

  // allocate and initialize
  h_aPageable = (int*)malloc(bytes );          
  h_bPageable = (int*)malloc(bytes );
  h_bitPageable =(int *)malloc(bytes );             // host pageable
  checkCuda( cudaMallocHost((void**)&h_aPinned, bytes  ) ); // host pinned
  checkCuda( cudaMallocHost((void**)&h_bPinned, bytes  ) );  
  checkCuda( cudaMalloc((void**)&d_a, bytes  ) );           // device
  checkCuda( cudaMalloc((void**)&lt, bytes ) ); // device return
  checkCuda( cudaMalloc((void**)&eq, bytes  ) );  
  srand(time(0));
  for (int i = 0; i < nElements; ++i) h_aPageable[i] = rand()%(1U<<31);     
  #ifdef TEST
      for (int i = 0; i < nElements; ++i) scanf("%d",h_aPageable + i);
  #endif
  for (int i = 0; i < nElements; ++i) 
    for(int j = 31; j >= 0; --j){
        h_bitPageable[i / 32 + (31-j)*(nElements/32)] += (((h_aPageable[i] &(1<<j))>>j)<<(31 - i % 32));

        //h_bitPageable[i / 32 + (31-j)*(nElements/32)] += 0;
    }
  //for(int i = 0;i < nElements; i++) h_bitPageable[i] = rand()%(1<<31);
  memcpy(h_aPinned, h_aPageable, bytes  );

  memset(h_bPageable, 0, bytes);
  memset(h_bPinned, 0, bytes);

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


  int constC = rand()%(1U<<31);
  #ifdef TEST
      scanf("%d",&constC);
  #endif

  // perform  scan eq
 // profilescan(h_aPageable, h_bPageable, d_a, filter, nElements, constC,"Pageable",20);
  //profilescan(h_aPinned, h_bPinned, d_a, filter,nElements, constC,"Pinned",20);
  profilebitscan(h_bitPageable, h_bPageable, d_a, lt, eq, nElements, constC,"Pageable",1);
  // for(int i = 0; i < nElements; i++) printf("%3u ",h_aPageable[i]);printf("\n");
  // for(int i = 0; i < nElements; i++) printf("%3u ",((h_bPageable[i/32] & (1u << (31 - i % 32)))>> (31 - i % 32)));printf("\n");
  // for(int i = 0; i < nElements; i++) printf("%3u ",((h_bPageable[i/32 + nElements/32] & (1u << (31 - i % 32)))>> (31 - i % 32)));printf("\n");
  // for(int i = 0; i < nElements; i++) printf("%3u ",h_bitPageable[i]);printf("\n");
    #ifdef TEST
      for(int i = 0; i < nElements; i++) {
          int x =(h_bPageable[i/32] & (1u << (31 - i % 32)))>> (31 - i % 32);
          int y =(h_bPageable[i/32 + nElements/32] & (1u << (31 - i % 32)))>> (31 - i % 32);
          if(x ==0 && y == 0) printf("%d\n",1);
          else printf("%d\n", 0);
        //  printf("%d|%d\n",x,y);
      }
  #endif
  // cleanup

  cudaFree(lt);
  cudaFree(eq);
  cudaFreeHost(h_aPinned);
  cudaFreeHost(h_bPinned);
  free(h_aPageable); 
}
