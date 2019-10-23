/*
 * Copyright 2011-2015 NVIDIA Corporation. All rights reserved
 *
 * Sample app to demonstrate use of CUPTI library to obtain metric values
 * using callbacks for CUDA runtime APIs
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
// #include <cupti.h>
#include <math_constants.h>
#include "lcutil.h"
#include <cuda_profiler_api.h>

#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                            \
  (((uintptr_t) (buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t) (buffer) & ((align)-1))) : (buffer))


#define COMP_ITERATIONS (512) //512
#define THREADS (1024)
#define BLOCKS (32760)
#define REGBLOCK_SIZE (4)
#define UNROLL_ITERATIONS (32)
#define deviceNum (0)

template <class T> __global__ void benchmark(){

	__shared__ T shared[THREADS];

	T r0 = shared[threadIdx.x],
	  r1 = r0,
	  r2 = r0,
	  r3 = r0;

	#pragma unroll 512
	for(int i=0; i<UNROLL_ITERATIONS; i++){
		r0 = r0 * r0 + r1;//r0;
		r1 = r1 * r1 + r2;//r1;
		r2 = r2 * r2 + r3;//r2;
		r3 = r3 * r3 + r0;//r3;
		r0 = r0 * r0 + r1;//r0;
		r1 = r1 * r1 + r2;//r1;
		r2 = r2 * r2 + r3;//r2;
		r3 = r3 * r3 + r0;//r3;
		r0 = r0 * r0 + r1;//r0;
		r1 = r1 * r1 + r2;//r1;
		r2 = r2 * r2 + r3;//r2;
		r3 = r3 * r3 + r0;//r3;
		r0 = r0 * r0 + r1;//r0;
		r1 = r1 * r1 + r2;//r1;
		r2 = r2 * r2 + r3;//r2;
		r3 = r3 * r3 + r0;//r3;
		r0 = r0 * r0 + r1;//r0;
		r1 = r1 * r1 + r2;//r1;
		r2 = r2 * r2 + r3;//r2;
		r3 = r3 * r3 + r0;//r3;
		r0 = r0 * r0 + r1;//r0;
		r1 = r1 * r1 + r2;//r1;
		r2 = r2 * r2 + r3;//r2;
		r3 = r3 * r3 + r0;//r3;
		r0 = r0 * r0 + r1;//r0;
		r1 = r1 * r1 + r2;//r1;
		r2 = r2 * r2 + r3;//r2;
		r3 = r3 * r3 + r0;//r3;
		r0 = r0 * r0 + r1;//r0;
		r1 = r1 * r1 + r2;//r1;
		r2 = r2 * r2 + r3;//r2;
		r3 = r3 * r3 + r0;//r3;
		r0 = r0 * r0 + r1;//r0;
		r1 = r1 * r1 + r2;//r1;
		r2 = r2 * r2 + r3;//r2;
		r3 = r3 * r3 + r0;//r3;
		r0 = r0 * r0 + r1;//r0;
		r1 = r1 * r1 + r2;//r1;
		r2 = r2 * r2 + r3;//r2;
		r3 = r3 * r3 + r0;//r3;
		r0 = r0 * r0 + r1;//r0;
		r1 = r1 * r1 + r2;//r1;
		r2 = r2 * r2 + r3;//r2;
		r3 = r3 * r3 + r0;//r3;
		r0 = r0 * r0 + r1;//r0;
		r1 = r1 * r1 + r2;//r1;
		r2 = r2 * r2 + r3;//r2;
		r3 = r3 * r3 + r0;//r3;
		r0 = r0 * r0 + r1;//r0;
		r1 = r1 * r1 + r2;//r1;
		r2 = r2 * r2 + r3;//r2;
		r3 = r3 * r3 + r0;//r3;
		r0 = r0 * r0 + r1;//r0;
		r1 = r1 * r1 + r2;//r1;
		r2 = r2 * r2 + r3;//r2;
		r3 = r3 * r3 + r0;//r3;
		r0 = r0 * r0 + r1;//r0;
		r1 = r1 * r1 + r2;//r1;
		r2 = r2 * r2 + r3;//r2;
		r3 = r3 * r3 + r0;//r3;
		r0 = r0 * r0 + r1;//r0;
		r1 = r1 * r1 + r2;//r1;
		r2 = r2 * r2 + r3;//r2;
		r3 = r3 * r3 + r0;//r3;
		r0 = r0 * r0 + r1;//r0;
		r1 = r1 * r1 + r2;//r1;
		r2 = r2 * r2 + r3;//r2;
		r3 = r3 * r3 + r0;//r3;
		r0 = r0 * r0 + r1;//r0;
		r1 = r1 * r1 + r2;//r1;
		r2 = r2 * r2 + r3;//r2;
		r3 = r3 * r3 + r0;//r3;
		r0 = r0 * r0 + r1;//r0;
		r1 = r1 * r1 + r2;//r1;
		r2 = r2 * r2 + r3;//r2;
		r3 = r3 * r3 + r0;//r3;
		r0 = r0 * r0 + r1;//r0;
		r1 = r1 * r1 + r2;//r1;
		r2 = r2 * r2 + r3;//r2;
		r3 = r3 * r3 + r0;//r3;
		r0 = r0 * r0 + r1;//r0;
		r1 = r1 * r1 + r2;//r1;
		r2 = r2 * r2 + r3;//r2;
		r3 = r3 * r3 + r0;//r3;
		r0 = r0 * r0 + r1;//r0;
		r1 = r1 * r1 + r2;//r1;
		r2 = r2 * r2 + r3;//r2;
		r3 = r3 * r3 + r0;//r3;
		r0 = r0 * r0 + r1;//r0;
		r1 = r1 * r1 + r2;//r1;
		r2 = r2 * r2 + r3;//r2;
		r3 = r3 * r3 + r0;//r3;
		r0 = r0 * r0 + r1;//r0;
		r1 = r1 * r1 + r2;//r1;
		r2 = r2 * r2 + r3;//r2;
		r3 = r3 * r3 + r0;//r3;
		r0 = r0 * r0 + r1;//r0;
		r1 = r1 * r1 + r2;//r1;
		r2 = r2 * r2 + r3;//r2;
		r3 = r3 * r3 + r0;//r3;
		r0 = r0 * r0 + r1;//r0;
		r1 = r1 * r1 + r2;//r1;
		r2 = r2 * r2 + r3;//r2;
		r3 = r3 * r3 + r0;//r3;
		r0 = r0 * r0 + r1;//r0;
		r1 = r1 * r1 + r2;//r1;
		r2 = r2 * r2 + r3;//r2;
		r3 = r3 * r3 + r0;//r3;
		r0 = r0 * r0 + r1;//r0;
		r1 = r1 * r1 + r2;//r1;
		r2 = r2 * r2 + r3;//r2;
		r3 = r3 * r3 + r0;//r3;
		r0 = r0 * r0 + r1;//r0;
		r1 = r1 * r1 + r2;//r1;
		r2 = r2 * r2 + r3;//r2;
		r3 = r3 * r3 + r0;//r3;
		r0 = r0 * r0 + r1;//r0;
		r1 = r1 * r1 + r2;//r1;
		r2 = r2 * r2 + r3;//r2;
		r3 = r3 * r3 + r0;//r3;
		r0 = r0 * r0 + r1;//r0;
		r1 = r1 * r1 + r2;//r1;
		r2 = r2 * r2 + r3;//r2;
		r3 = r3 * r3 + r0;//r3;
		r0 = r0 * r0 + r1;//r0;
		r1 = r1 * r1 + r2;//r1;
		r2 = r2 * r2 + r3;//r2;
		r3 = r3 * r3 + r0;//r3;
	}

	shared[threadIdx.x] = r0;

}

double median(int n, double x[][4],int col) {
    double temp;
    int i, j;
    // the following two loops sort the array x in ascending order
    for(i=0; i<n-1; i++) {
        for(j=i+1; j<n; j++) {
            if(x[j][col] < x[i][col]) {
                // swap elements
                temp = x[i][col];
                x[i][col] = x[j][col];
                x[j][col] = temp;
            }
        }
    }
    if(n%2==0) {
        // if there is an even number of elements, return mean of the two elements in the middle
        return((x[n/2][col] + x[n/2 - 1][col]) / 2.0);
    } else {
        // else return the element in the middle
        return x[n/2][col];
    }
}

void initializeEvents(cudaEvent_t *start, cudaEvent_t *stop){
	CUDA_SAFE_CALL( cudaEventCreate(start) );
	CUDA_SAFE_CALL( cudaEventCreate(stop) );
	CUDA_SAFE_CALL( cudaEventRecord(*start, 0) );
}

float finalizeEvents(cudaEvent_t start, cudaEvent_t stop){
	CUDA_SAFE_CALL( cudaGetLastError() );
	CUDA_SAFE_CALL( cudaEventRecord(stop, 0) );
	CUDA_SAFE_CALL( cudaEventSynchronize(stop) );
	float kernel_time;
	CUDA_SAFE_CALL( cudaEventElapsedTime(&kernel_time, start, stop) );
	CUDA_SAFE_CALL( cudaEventDestroy(start) );
	CUDA_SAFE_CALL( cudaEventDestroy(stop) );
	return kernel_time;
}

void runbench(int type, double* kernel_time, double* flops){

	const long long computations = 2*(long long)(COMP_ITERATIONS)*REGBLOCK_SIZE*THREADS*BLOCKS;

	dim3 dimBlock(THREADS, 1, 1);
    dim3 dimGrid(BLOCKS, 1, 1);
	cudaEvent_t start, stop;

	initializeEvents(&start, &stop);

    benchmark<int><<< dimGrid, dimBlock >>>();

	cudaDeviceSynchronize();

	double time = finalizeEvents(start, stop);
	double result = ((double)computations)/(double)time*1000./(double)(1000*1000*1000);

	*kernel_time = time;
	*flops=result;

}

int main(int argc, char *argv[]){
	// CUpti_SubscriberHandle subscriber;
	// CUcontext context = 0;
	CUdevice device = 0;
	int deviceCount;
	// cupti_eventData cuptiEvent;
	// RuntimeApiTrace_t trace;


	printf("Usage: %s [device_num] [metric_name]\n", argv[0]);
	int ntries;
	if (argc>1){
		ntries = atoi(argv[1]);
	}else{
		ntries = 1;
	}

	cudaSetDevice(deviceNum);
	double time[ntries][2],value[ntries][4];

	(cuDeviceGetCount(&deviceCount));
	if (deviceCount == 0) {
		printf("There is no device supporting CUDA.\n");
	return -2;
	}

	printf("CUDA Device Number: %d\n", deviceNum);

	(cuDeviceGet(&device, deviceNum));
	// CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, device));
	int measure;
	(cuDeviceGetAttribute(&measure,CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,device));
	printf("Max Threads per block = %d\n", measure);


	// DRIVER_API_CALL(cuCtxCreate(&context, 0, device));
	int i;

	for (i=0;i<ntries;i++){

		runbench(0,&time[0][0],&value[0][0]);


		printf("Registered time: %f ms\n",time[0][0]);

	}




	CUDA_SAFE_CALL( cudaDeviceReset());

	return 0;
}
