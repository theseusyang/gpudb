#include <stdio.h>
#include <assert.h>

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
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

void profileCopies(float        *h_a, 
                   float        *h_b, 
                   float        *d, 
                   unsigned int  n,
                   char         *desc,
                   unsigned int loopTotal)
{
  //printf("\n%s transfers\n", desc);

  unsigned int bytes = n * sizeof(float);

  float time,stime;
  // events for timing
  cudaEvent_t startEvent, stopEvent; 
  checkCuda( cudaEventCreate(&startEvent) );
  checkCuda( cudaEventCreate(&stopEvent) );	
  stime=0;
  for(int i = 1; i <= loopTotal; i++){	
	  checkCuda( cudaEventRecord(startEvent, 0) );
	  checkCuda( cudaMemcpy(d, h_a, bytes, cudaMemcpyHostToDevice) );
	  checkCuda( cudaEventRecord(stopEvent, 0) );
	  checkCuda( cudaEventSynchronize(stopEvent) );

	  checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
	  stime += time;
  }
  printf("%f\n", bytes * 1e-6 / stime *loopTotal);
  stime=0;
  for(int i = 1; i <= loopTotal; i++){	
	  checkCuda( cudaEventRecord(startEvent, 0) );
	  checkCuda( cudaMemcpy(h_b, d, bytes, cudaMemcpyDeviceToHost) );
	  checkCuda( cudaEventRecord(stopEvent, 0) );
	  checkCuda( cudaEventSynchronize(stopEvent) );

	  checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
	  stime += time;
  }
 //printf("%f\n", bytes * 1e-6 / stime * loopTotal);

  for (int i = 0; i < n; ++i) {
    if (h_a[i] != h_b[i]) {
      printf("*** %s transfers failed ***\n", desc);
      break;
    }
  }

  // clean up events
  checkCuda( cudaEventDestroy(startEvent) );
  checkCuda( cudaEventDestroy(stopEvent) );
}

int main(int argc, char ** argv)
{
  int inputN;
  sscanf(argv[1],"%d",&inputN);
  unsigned int nElements = inputN/4;
  const unsigned int bytes = nElements * sizeof(float);

  // host arrays
  float *h_aPageable, *h_bPageable;   
  float *h_aPinned, *h_bPinned;

  // device array
  float *d_a;

  // allocate and initialize
  h_aPageable = (float*)malloc(bytes);                    // host pageable
  h_bPageable = (float*)malloc(bytes);                    // host pageable
  checkCuda( cudaMallocHost((void**)&h_aPinned, bytes) ); // host pinned
  checkCuda( cudaMallocHost((void**)&h_bPinned, bytes) ); // host pinned
  checkCuda( cudaMalloc((void**)&d_a, bytes) );           // device

  char busid[16];
  int dev;
  checkCuda(cudaGetDevice(&dev));
  checkCuda(cudaDeviceGetPCIBusId(busid, 16, dev));
  printf("%s\n", busid);

  for (int i = 0; i < nElements; ++i) h_aPageable[i] = i;      
  memcpy(h_aPinned, h_aPageable, bytes);
  memset(h_bPageable, 0, bytes);
  memset(h_bPinned, 0, bytes);

  // output device info and transfer size
  cudaDeviceProp prop;
  checkCuda( cudaGetDeviceProperties(&prop, 0) );

  // printf("\nDevice: %s\n", prop.name);
  // if(bytes< 1024){
  //     printf("Transfer size (B): %d\n", bytes);
  // }else if (bytes < 1024 * 1024)
  // {
  //     printf("Transfer size (KB): %d\n", bytes / (1024));              
  // }else{
  //     printf("Transfer size (MB): %d\n", bytes / (1024 * 1024));
  // }


  // perform copies and report bandwidth
  profileCopies(h_aPageable, h_bPageable, d_a, nElements, "Pageable",20);
  //profileCopies(h_aPinned, h_bPinned, d_a, nElements, "Pinned",20);

  //printf("n");

  // cleanup
  cudaFree(d_a);
  cudaFreeHost(h_aPinned);
  cudaFreeHost(h_bPinned);
  free(h_aPageable);
  free(h_bPageable);

  return 0;
}
