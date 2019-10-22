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
// #include "../../lcutil.h"
#include <cuda_profiler_api.h>

#define CUDA_SAFE_CALL( call) {                                    \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    } }


#define DRIVER_API_CALL(apiFuncCall)                                           \
do {                                                                           \
    CUresult _status = apiFuncCall;                                            \
    if (_status != CUDA_SUCCESS) {                                             \
        fprintf(stderr, "%s:%d: error: function %s failed with error %d.\n",   \
                __FILE__, __LINE__, #apiFuncCall, _status);                    \
        exit(-1);                                                              \
    }                                                                          \
} while (0)

#define RUNTIME_API_CALL(apiFuncCall)                                          \
do {                                                                           \
    cudaError_t _status = apiFuncCall;                                         \
    if (_status != cudaSuccess) {                                              \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",   \
                __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status));\
        exit(-1);                                                              \
    }                                                                          \
} while (0)


#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                            \
  (((uintptr_t) (buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t) (buffer) & ((align)-1))) : (buffer))

// #define COMP_ITERATIONS (512)
#define THREADS (1024)
#define BLOCKS (3276)
#define N (10)

#define REGBLOCK_SIZE (4)
// #define UNROLL_ITERATIONS (32)
#define deviceNum (0)

// #define OFFSET

#define INNER_REPS 512
#define UNROLLS 8

// __constant__ __device__ int off [16] = {0,4,8,12,9,13,1,5,2,6,10,14,11,15,3,7}; //512 threads
// __constant__ __device__ int off [16] = {0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3}; //512 threads
// __constant__ __device__ int off [16] = {0,2,4,6,8,10,12,14,11,9,15,13,3,1,7,5}; //256 threads

template <class T> __global__ void benchmark (T* cdin, T* cdout){

	// const int total = THREADS*BLOCKS+THREADS;

	const int ite = blockIdx.x * THREADS + threadIdx.x;
	T r0;
	// printf("%d - %d\n", blockIdx.x,off[blockIdx.x]);
	// T r1,r2,r3;

	// r0=cdin[ite];
	for (int k=0; k<N;k++){
        #pragma unroll 512
		for(int j=0; j<INNER_REPS; j+=UNROLLS){
			r0 = cdin[ite];
			cdout[ite]=r0;
			r0 = cdin[ite];
			cdout[ite]=r0;
            r0 = cdin[ite];
			cdout[ite]=r0;
            r0 = cdin[ite];
			cdout[ite]=r0;
            r0 = cdin[ite];
			cdout[ite]=r0;
            r0 = cdin[ite];
			cdout[ite]=r0;
            r0 = cdin[ite];
			cdout[ite]=r0;
            r0 = cdin[ite];
			cdout[ite]=r0;
		}
	}
	cdout[ite]=r0;
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


void runbench(int type, double* kernel_time, double* bandw,double* cdin,double* cdout){

	cudaEvent_t start, stop;
	initializeEvents(&start, &stop);
	dim3 dimBlock(THREADS, 1, 1);
	dim3 dimGrid(BLOCKS, 1, 1);
	// if (type==0){
	benchmark<float><<< dimGrid, dimBlock >>>((float*)cdin,(float*)cdout);
	// }else{
	// 	benchmark<double><<< dimGrid, dimBlock >>>(cdin,cdout, inner_reps, unrolls);
	// }

	long long shared_access = 2*(long long)(INNER_REPS)*N*THREADS*BLOCKS;

	cudaDeviceSynchronize();

	double time = finalizeEvents(start, stop);
	double result;
	if (type==0)
		result = ((double)shared_access)*4/(double)time*1000./(double)(1024*1024*1024);
	else
		result = ((double)shared_access)*8/(double)time*1000./(double)(1024*1024*1024);

	*kernel_time = time;
	*bandw=result;

}

int main(int argc, char *argv[]){
	// CUpti_SubscriberHandle subscriber;
	CUcontext context = 0;
	CUdevice device = 0;
	int deviceCount;
	char deviceName[32];

    int outer_reps;
    // , vector_size, tile_dim;

    if (argc>1){
        outer_reps = atoi(argv[1]);
    }else{
        outer_reps = 1;
    }



	// cupti_eventData cuptiEvent;
	// RuntimeApiTrace_t trace;
	cudaDeviceProp deviceProp;

	printf("Usage: %s [device_num] [metric_name]\n", argv[0]);

	cudaSetDevice(deviceNum);
	double mean[4];
	double time[outer_reps][2],value[outer_reps][4],sum_dev_median[4],sum_dev_mean[4],medianv[4],std_dev_mean[4],std_dev_median[4];
	long SPresult[outer_reps],DPresult[outer_reps],timeresult[outer_reps][2];
	int L2size;
	int counters;
	// StoreDeviceInfo_DRAM(stdout,&L2size);
	int size = THREADS*BLOCKS*sizeof(double);
	size_t freeCUDAMem, totalCUDAMem;
	cudaMemGetInfo(&freeCUDAMem, &totalCUDAMem);
	printf("Total GPU memory %lu, free %lu\n", totalCUDAMem, freeCUDAMem);
	printf("Buffer size: %dMB\n", size*sizeof(double)/(1024*1024));
	SPresult[0]=0;
	DPresult[0]=0;

	//Initialize Global Memory
	double *cdin,L2=32;
	double *cdout;
	CUDA_SAFE_CALL(cudaMalloc((void**)&cdin, size));
	CUDA_SAFE_CALL(cudaMalloc((void**)&cdout, size));

	// Copy data to device memory
	CUDA_SAFE_CALL(cudaMemset(cdin, 1, size));  // initialize to zeros
	CUDA_SAFE_CALL(cudaMemset(cdout, 0, size));  // initialize to zeros
	// Synchronize in order to wait for memory operations to finish
	CUDA_SAFE_CALL(cudaThreadSynchronize());
	// make sure activity is enabled before any CUDA API

	DRIVER_API_CALL(cuDeviceGetCount(&deviceCount));
	if (deviceCount == 0) {
		printf("There is no device supporting CUDA.\n");
	return -2;
	}

	printf("CUDA Device Number: %d\n", deviceNum);

	DRIVER_API_CALL(cuDeviceGet(&device, deviceNum));
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, device));
	DRIVER_API_CALL(cuDeviceGetName(deviceName, 32, device));

	int i;
	class type;

	uint64_t L2units;
	size_t sizet=sizeof(L2units);

    for (i=0;i<outer_reps;i++){

        uint32_t all = 1;

		runbench(0,&time[0][0],&value[0][0],cdin,cdout);


        printf("Registered time: %f ms\n",time[0][0]);

    }

    CUDA_SAFE_CALL( cudaDeviceReset());
	return 0;
}
