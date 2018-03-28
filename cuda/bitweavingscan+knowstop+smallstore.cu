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
#include <iostream>
#include "scanImpl.cu"
#include "../include/common.h"
#include "../include/gpuCudaLib.h"

using namespace std;
#define TEST 1
#define utype  unsigned long long
#define type long long
#define type_len (sizeof(type) * 8)

double ave_time  = 0;

const int know_stop_size=10000010;
vector<type > know_stop_num[know_stop_size];
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
void profilebitweavscan(type        *h_a, 
                   type       *h_b, 
                   type        *d, 
                   type *lt,
                   type *eq,
                   type *know_stop_len_cpu,
                   type *know_stop_constC_cpu,
                   int early_size,
                   int group_size,
                   int  n,
                   type *gpu_constC,
                   type *constC,
                   type queryC,
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
  int bytes=n * sizeof(type);

  for(int loop = 1; loop <= loopTotal; loop++){  

      checkCuda( cudaEventRecord(startEvent, 0) );
      for(int i = 0 ; i < n/group_size; ++i){
            assert(queryC > 0);
            vector<type>::iterator ii = lower_bound(know_stop_num[i]. begin(),know_stop_num[i].end(),queryC);
            know_stop_constC_cpu[i] = *ii;
            type last_constC = *(--ii);
            know_stop_len_cpu[i] =nlz(last_constC ^ know_stop_constC_cpu[i]) + 1; 
              //      printf(" %d %d ",last_constC,know_stop_constC_cpu[i], know_stop_len_cpu[i]);
      }

      utype c = 0;
      for(int i = 0;i < type_len;i++) c += ((utype)1 << i);
      equal<<<grid,block>>>(lt, n/type_len, 0) ;
      equal<<<grid,block>>>(eq, n/type_len, c) ;
      checkCuda(cudaThreadSynchronize());

      
      double len_max=0.0, len_sum = 0.0;
        for(int k  = 0; k < n ; k += early_size ){
            //printf("%d \n", know_stop_len_cpu[k / (early_size / type_len)]);    

            type len = 0;
            for(int i = 0; i < early_size / group_size; i++){
                len = max (len , know_stop_len_cpu[k / group_size + i]);
               // printf("len=%d\n",know_stop_len_cpu[k / group_size + i]);
            }
            checkCuda( cudaMemcpy(d + k, h_a + k, early_size / type_len * sizeof(type) * len,  cudaMemcpyHostToDevice) );

            len_max=max(len_max,(double)len);
            len_sum=len_sum + len;
        //    printf("xxx=%d\n",len);
            for(int j = 0; j < len; ++j){     
                for(int i = 0; i < early_size / group_size; i ++ ) {
                    constC[i] = bit_constC(know_stop_constC_cpu[k / group_size + i], j);

                }
                checkCuda( cudaMemcpy(gpu_constC, constC, early_size / group_size * sizeof(type) ,  cudaMemcpyHostToDevice) );            
                int place = k + early_size / type_len * j;
                genScanFilter_int_lth_bit<<<grid,block>>>(d + place, early_size / type_len,  gpu_constC, group_size/type_len, lt + k/type_len, eq + k/type_len);
                checkCuda(cudaThreadSynchronize());
            }

        }
      #ifndef TEST
        printf("%.2f %.2f\n",len_max, len_sum*1.0/(n/early_size));
      #endif
      checkCuda( cudaMemcpy(h_b, lt, n / type_len * sizeof(type), cudaMemcpyDeviceToHost) );
      checkCuda( cudaMemcpy(h_b + n / type_len , eq, n / type_len * sizeof(type), cudaMemcpyDeviceToHost) );
      checkCuda(cudaThreadSynchronize());
      checkCuda( cudaEventRecord(stopEvent, 0) );
      checkCuda( cudaEventSynchronize(stopEvent) );

      checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
      stime += time;
      ave_time += stime;

      //printf("%f\n" ,bytes * 1e-6/(stime / loop));
      //printf("%f\n",stime);
  }
    //  cerr<<bytes<<" "<<stime<<" "<<bytes* 1e-6 /(stime)<<endl;
  #ifndef TEST
    printf("%d %f\n" ,group_size,bytes * 1e-6/(stime / loopTotal));
  #endif
  checkCuda( cudaEventDestroy(startEvent) );
  checkCuda( cudaEventDestroy(stopEvent) );
}
int main(int argc, char ** argv)
{
  #ifdef TEST
      freopen("scan.in","r",stdin);
      freopen("scan_smallstore.out","w",stdout);
  #endif

  int inputN;
  sscanf(argv[1],"%d",&inputN);
  unsigned int nElements = inputN;
  #ifdef TEST
      scanf("%d",&nElements);
  #endif
  const unsigned int bytes = nElements * sizeof(type);

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
  int group_size = 1024*64;
  sscanf(argv[2],"%d",&early_size);  
  sscanf(argv[3],"%d",&group_size); 
  srand(time(0));

  #ifdef TEST
      for (int i = 0; i < nElements; ++i){ 
      if(sizeof(type)==4) scanf("%d",h_aPageable + i);
      else scanf("%lld",h_aPageable + i);
     // cerr<<h_aPageable[i]<<endl;
      }
  #else
    for (int i = 0; i < nElements; ++i) h_aPageable[i] = ran()%((utype)1<<(type_len - 1));  
  #endif

  for   (int i = 0;i < nElements/ group_size; i++){
        know_stop_num[i].push_back(0);
        know_stop_num[i].push_back(((utype)1<<(type_len - 1))-1);
        //0----2^31-1
        for(int j=0;j < group_size; j++)
          know_stop_num[i].push_back(h_aPageable[i * group_size + j]);
        sort(know_stop_num[i].begin(),know_stop_num[i].end());
        know_stop_num[i].erase(unique(know_stop_num[i].begin(), know_stop_num[i].end()), know_stop_num[i].end());
  }   

  for (int i = 0; i < nElements; i += early_size) 
          for(int j = type_len -1; j >= 0; --j)
               for(int k = 0; k < early_size; ++k)
                      {
                          int idx = i + (type_len - 1 -j) * early_size / type_len + k / type_len;

                          h_bitPageable[idx] += (((h_aPageable[i + k] &((utype)1<<j))>>j)<<(type_len - 1 - k % type_len));
                          //if(j<=12)cerr<<h_aPageable[i+k]<<" "<<(h_aPageable[i + k] &((utype)1<<j))<<endl;
                      }

  //memcpy(h_aPinned, h_aPageable, bytes  );
  memset(h_bPageable, 0, bytes);
  //memset(h_bPinned, 0, bytes);

  memset(know_stop_len_cpu, 0, bytes);

  // output device info and transfer size
  cudaDeviceProp prop;

  checkCuda( cudaGetDeviceProperties(&prop, 0) );



  type constC = 0;



  // perform  scan eq
  type *gpu_constC,*cpu_constC;
  checkCuda( cudaMalloc((void**)&gpu_constC, bytes) ); 
  cpu_constC=(type*)malloc(bytes);   
  #ifdef TEST
      int test_num = 0;
      scanf("%d",&test_num);
      for(int i  = 0; i < test_num; i++){
        if(sizeof(type)==4)
        scanf("%d",&constC);
        else scanf("%lld",&constC);
        profilebitweavscan(h_bitPageable, h_bPageable, d_a, lt, eq, know_stop_len_cpu, know_stop_constC_cpu,early_size, group_size, nElements,gpu_constC,cpu_constC,constC,"Pageable",1);
      }
  #else 
    constC = ran()%((utype)1<<(type_len - 1));
    profilebitweavscan(h_bitPageable, h_bPageable, d_a, lt, eq, know_stop_len_cpu, know_stop_constC_cpu,early_size, group_size, nElements,gpu_constC,cpu_constC,constC,"Pageable",1);
  #endif
  // printf("constC=%d\n",constC);
  // for(int i = 0; i < nElements; i++) printf("%u ",h_aPageable[i]);printf("\n");
  // for(int i = 0; i < nElements; i++) printf("%u ",((h_bPageable[i/type_len] & (1u << (31 - i % type_len)))>> (31 - i % type_len)));printf("\n");
  // for(int i = 0; i < nElements; i++) printf("%u ",((h_bPageable[i/type_len + nElements/type_len] & (1u << (31 - i % type_len)))>> (31 - i % type_len)));printf("\n");
  // //for(int i = 0; i < nElements; i++) printf("%3u ",h_bitPageable[i]);printf("\n");
  #ifdef TEST
      for(int i = 0; i < nElements; i++) {
          int x =(h_bPageable[i/type_len] & ((utype)1 << (type_len - 1 - i % type_len)))>> (type_len - 1 - i % type_len);
          int y =(h_bPageable[i/type_len + nElements/type_len] & ((utype)1 << (type_len - 1 - i % type_len)))>> (type_len - 1 - i % type_len);
          if(x ==0 && y == 0) printf("%d\n",1);
          else printf("%d\n", 0);
        //  printf("%d|%d\n",x,y);
      }
    printf("%.6f\n",bytes* 1e-6 /(ave_time / test_num));
      cerr<<bytes* 1e-6 /(ave_time / test_num)<<endl;
  #endif
  // cleanup
  cudaFree(lt);
  cudaFree(eq);
  free(h_aPageable); 
}
