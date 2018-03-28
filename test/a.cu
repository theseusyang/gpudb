#include <cuda.h>
#include <stdio.h>

int main(){
	int devCount;
	cudaGetDeviceCount(&devCount);
	printf("%d\n", devCount);
	return 0;
}
