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
#include <algorithm>
#include <vector>
#include "scanImpl.cu"
#include "../include/common.h"
#include "../include/gpuCudaLib.h"

using namespace std;
//#define TEST 1


#define CHECK_POINTER(p)   do {                     \
    if(p == NULL){                                  \
        perror("Failed to allocate host memory");   \
        exit(-1);                                   \
    }} while(0)
const int know_stop_size=10000010;
vector<int > know_stop_num[know_stop_size];
type nlz(utype x){
   type n;
     if (x == 0) return(32);
     n = 1;
     if ((x >> 16) == 0) {n = n +16; x = x <<16;}
     if ((x >> 24) == 0) {n = n + 8; x = x << 8;}
     if ((x >> 28) == 0) {n = n + 4; x = x << 4;}
     if ((x >> 30) == 0) {n = n + 2; x = x << 2;}
     n = n - (x >> 31);

}
__global__ void static equal(int * a, int n, unsigned int constC){
    int offset = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i=offset; i<n; i+=stride){    
        a[i] = constC;
    }
}


__global__ void static genScanFilter_int_lth_bit(int * col,int n, int *constC,int group_size, int * lt, int * eq){
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
inline int bit_constC(int where,int j){

        int constC = (((1U << (31 - j )) & where)>>(31 - j));
        if(constC != 0) 
          constC = (1L << 32) - 1;
        return constC;
}
void profilebitweavscan(int        *h_a, 
                   int        *h_b, 
                   int        *d, 
                   int *lt,
                   int *eq,
                   int *know_stop_len_cpu,
                   int *know_stop_constC_cpu,
                   int early_size,
                   int group_size,
                   int  n,
                   int *gpu_constC,
                   int *constC,
                   int queryC,
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
      #ifndef TEST
        queryC = rand()% (1U<<31);
      #endif
      for(int i = 0 ; i < n/group_size; ++i){
            assert(queryC > 0);
            vector<int>::iterator ii = lower_bound(know_stop_num[i]. begin(),know_stop_num[i].end(),queryC);
            know_stop_constC_cpu[i] = *ii;
            int last_constC = *(--ii);
            know_stop_len_cpu[i] =nlz(last_constC ^ know_stop_constC_cpu[i]) + 1; 
              //      printf(" %d %d ",last_constC,know_stop_constC_cpu[i], know_stop_len_cpu[i]);
      }

      unsigned int c = 0;
      for(int i = 0;i < 32;i++) c += (1u << i);
      equal<<<grid,block>>>(lt, n/32, 0) ;
      equal<<<grid,block>>>(eq, n/32, c) ;
      checkCuda(cudaThreadSynchronize());

      
      double len_max=0.0, len_sum = 0.0;
        for(int k  = 0; k < n ; k += early_size ){
            //printf("%d \n", know_stop_len_cpu[k / (early_size / 32)]);    

            int len = 0;
            for(int i = 0; i < early_size / group_size; i++)
                len = max (len , know_stop_len_cpu[k / group_size + i]);
            checkCuda( cudaMemcpy(d + k, h_a + k, early_size / 32 *4 * len,  cudaMemcpyHostToDevice) );

            len_max=max(len_max,(double)len);
            len_sum=len_sum + len;

            for(int j = 0; j < len; ++j){     
                for(int i = 0; i < early_size / group_size; i ++ ) {
                    constC[i] = bit_constC(know_stop_constC_cpu[k / group_size + i], j);
                    //printf("constC=%d\n",constC[i]);
                }
                checkCuda( cudaMemcpy(gpu_constC, constC, early_size / group_size *4 ,  cudaMemcpyHostToDevice) );            
                int place = k + early_size / 32 * j;
                genScanFilter_int_lth_bit<<<grid,block>>>(d + place, early_size / 32,  gpu_constC, group_size/32, lt + k/32, eq + k/32);
                checkCuda(cudaThreadSynchronize());
            }

        }
      
      printf("%.2f %.2f\n",len_max, len_sum*1.0/(n/early_size));
      checkCuda( cudaMemcpy(h_b, lt, n / 32 * 4, cudaMemcpyDeviceToHost) );
      checkCuda( cudaMemcpy(h_b + n / 32 , eq, n / 32 * 4, cudaMemcpyDeviceToHost) );
      checkCuda(cudaThreadSynchronize());
      checkCuda( cudaEventRecord(stopEvent, 0) );
      checkCuda( cudaEventSynchronize(stopEvent) );

      checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
      stime += time;

      //printf("%f\n" ,bytes * 1e-6/(stime / loop));
      //printf("%f\n",stime);
  }

  printf("%d %f\n" ,group_size,bytes * 1e-6/(stime / loopTotal));
  checkCuda( cudaEventDestroy(startEvent) );
  checkCuda( cudaEventDestroy(stopEvent) );
}
int main(int argc, char ** argv)
{
  #ifdef TEST
      freopen("scan.in","r",stdin);
      freopen("scan_smallstore.out","w",stdout);
  #endif
  dim3 block(256);
  dim3 grid(2048);
  int inputN;
  sscanf(argv[1],"%d",&inputN);
  unsigned int nElements = inputN;
  #ifdef TEST
      scanf("%d",&nElements);
  #endif
  const unsigned int bytes = nElements * sizeof(int);

  // host arrays
  int *h_aPageable, *h_bPageable,*h_bitPageable,*know_stop_len_cpu,*know_stop_constC_cpu;   
  int *h_aPinned, *h_bPinned;

  // device array
  int *d_a,*lt,*eq;

  // allocate and initialize
  h_aPageable = (int*)malloc(bytes );          
  h_bPageable = (int*)malloc(bytes );
  h_bitPageable =(int *)malloc(bytes );
  know_stop_len_cpu = (int *)malloc(bytes );     
  know_stop_constC_cpu = (int *)malloc(bytes );           // host pageable
  checkCuda( cudaMallocHost((void**)&h_aPinned, bytes  ) ); // host pinned
  checkCuda( cudaMallocHost((void**)&h_bPinned, bytes  ) );  
  checkCuda( cudaMalloc((void**)&d_a, bytes  ) );           // device
  checkCuda( cudaMalloc((void**)&lt, bytes ) ); // device return
  checkCuda( cudaMalloc((void**)&eq, bytes  ) );  

  int early_size = 1024*1024; 
  int group_size = 1024*64;
  sscanf(argv[2],"%d",&early_size);  
  sscanf(argv[3],"%d",&group_size); 
  srand(time(0));
  for (int i = 0; i < nElements; ++i) h_aPageable[i] = rand()%(1U<<31);  
  #ifdef TEST
      for (int i = 0; i < nElements; ++i) scanf("%d",h_aPageable + i);
  #endif
  for   (int i = 0;i < nElements/ group_size; i++){
        know_stop_num[i].push_back(0);
        know_stop_num[i].push_back((1U<<31)-1);
        //0----2^31-1
        for(int j=0;j < group_size; j++)
          know_stop_num[i].push_back(h_aPageable[i * group_size + j]);
        sort(know_stop_num[i].begin(),know_stop_num[i].end());
        know_stop_num[i].erase(unique(know_stop_num[i].begin(), know_stop_num[i].end()), know_stop_num[i].end());
  }   
   
  for (int i = 0; i < nElements; i += early_size) 
          for(int j = 31; j >= 0; --j)
               for(int k = 0; k < early_size; ++k)
                      {
                          h_bitPageable[i + (31 -j) * early_size / 32 + k / 32] += (((h_aPageable[i + k] &(1<<j))>>j)<<(31 - k % 32));

                      }
  // for(int i = 0;i < nElements; i++) 
  //   for(int j = 0 ;j < 32;j++)h_bitPageable[i] = rand()%(1<<31);
  memcpy(h_aPinned, h_aPageable, bytes  );
  memset(h_bPageable, 0, bytes);
  memset(h_bPinned, 0, bytes);
  memset(know_stop_len_cpu, 0, bytes);

  // output device info and transfer size
  cudaDeviceProp prop;

  checkCuda( cudaGetDeviceProperties(&prop, 0) );



  int constC = rand()%(1U<<31);
  #ifdef TEST
      scanf("%d",&constC);
  #endif
  for(int i = 0 ; i < nElements/group_size; ++i){
        assert(constC > 0);
        know_stop_constC_cpu[i] = *lower_bound(know_stop_num[i]. begin(),know_stop_num[i].end(),constC);
        int last_constC = *(--lower_bound(know_stop_num[i]. begin(),know_stop_num[i].end(),constC));

        know_stop_len_cpu[i] =nlz(last_constC ^ know_stop_constC_cpu[i]) + 1; 
          //      printf(" %d %d ",last_constC,know_stop_constC_cpu[i], know_stop_len_cpu[i]);
  }


  // perform  scan eq
 // profilescan(h_aPageable, h_bPageable, d_a, filter, nElements, constC,"Pageable",20);
  //profilescan(h_aPinned, h_bPinned, d_a, filter,nElements, constC,"Pinned",20);
  int *gpu_constC,*cpu_constC;
  checkCuda( cudaMalloc((void**)&gpu_constC, bytes) ); 
  cpu_constC=(int*)malloc(bytes);   
  profilebitweavscan(h_bitPageable, h_bPageable, d_a, lt, eq, know_stop_len_cpu, know_stop_constC_cpu,early_size, group_size, nElements,gpu_constC,cpu_constC,constC,"Pageable",1);
  // printf("constC=%d\n",constC);
  // for(int i = 0; i < nElements; i++) printf("%u ",h_aPageable[i]);printf("\n");
  // for(int i = 0; i < nElements; i++) printf("%u ",((h_bPageable[i/32] & (1u << (31 - i % 32)))>> (31 - i % 32)));printf("\n");
  // for(int i = 0; i < nElements; i++) printf("%u ",((h_bPageable[i/32 + nElements/32] & (1u << (31 - i % 32)))>> (31 - i % 32)));printf("\n");
  // //for(int i = 0; i < nElements; i++) printf("%3u ",h_bitPageable[i]);printf("\n");
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
