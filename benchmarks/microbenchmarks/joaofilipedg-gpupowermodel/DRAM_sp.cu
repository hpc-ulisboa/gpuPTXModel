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
#include <math_constants.h>
#include "lcutil.h"
#include <cuda_profiler_api.h>
// #include <gpuCUPTISampler.cuh>

#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                            \
  (((uintptr_t) (buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t) (buffer) & ((align)-1))) : (buffer))

#define COMP_ITERATIONS (1024)
#define THREADS (1024)
#define BLOCKS (32768)
#define STRIDE (64*1024)


#define REGBLOCK_SIZE (4)
#define UNROLL_ITERATIONS (32)

#define deviceNum (0)

template <class T>
__global__ void benchmark( T* cdin,  T* cdout){

	const long ite=blockIdx.x * THREADS + threadIdx.x;
	T r0;

    // for(int j=0; j<COMP_ITERATIONS; j+=UNROLL_ITERATIONS){
    //     #pragma unroll
    //     for(int i=0; i<UNROLL_ITERATIONS; i++){
    //         r0=cdin[ite+STRIDE*i];
    //         cdout[ite+STRIDE*i]=r0;
    //     }
    // }


    // #pragma unroll 1024
    // for(int j=0; j<COMP_ITERATIONS; j+=UNROLL_ITERATIONS){
    //     #pragma unroll 32
    //     for(int i=0; i<UNROLL_ITERATIONS; i++){
    //         r0=cdin[ite+STRIDE*i];
    //         cdout[ite+STRIDE*i]=r0;
    //     }
    // }

    int i;
    #pragma unroll 1024
    for(int j=0; j<COMP_ITERATIONS; j+=UNROLL_ITERATIONS) {
        i=0;
        r0=cdin[ite+STRIDE*i];
    	cdout[ite+STRIDE*i]=r0;
        i++;
        r0=cdin[ite+STRIDE*i];
    	cdout[ite+STRIDE*i]=r0;
        i++;
        r0=cdin[ite+STRIDE*i];
    	cdout[ite+STRIDE*i]=r0;
        i++;
        r0=cdin[ite+STRIDE*i];
    	cdout[ite+STRIDE*i]=r0;
        i++;
        r0=cdin[ite+STRIDE*i];
    	cdout[ite+STRIDE*i]=r0;
        i++;
        r0=cdin[ite+STRIDE*i];
    	cdout[ite+STRIDE*i]=r0;
        i++;
        r0=cdin[ite+STRIDE*i];
    	cdout[ite+STRIDE*i]=r0;
        i++;
        r0=cdin[ite+STRIDE*i];
    	cdout[ite+STRIDE*i]=r0;
        i++;
        r0=cdin[ite+STRIDE*i];
    	cdout[ite+STRIDE*i]=r0;
        i++;
        r0=cdin[ite+STRIDE*i];
    	cdout[ite+STRIDE*i]=r0;
        i++;
        r0=cdin[ite+STRIDE*i];
    	cdout[ite+STRIDE*i]=r0;
        i++;
        r0=cdin[ite+STRIDE*i];
    	cdout[ite+STRIDE*i]=r0;
        i++;
        r0=cdin[ite+STRIDE*i];
    	cdout[ite+STRIDE*i]=r0;
        i++;
        r0=cdin[ite+STRIDE*i];
    	cdout[ite+STRIDE*i]=r0;
        i++;
        r0=cdin[ite+STRIDE*i];
    	cdout[ite+STRIDE*i]=r0;
        i++;
        r0=cdin[ite+STRIDE*i];
    	cdout[ite+STRIDE*i]=r0;
        i++;
        r0=cdin[ite+STRIDE*i];
    	cdout[ite+STRIDE*i]=r0;
        i++;
        r0=cdin[ite+STRIDE*i];
    	cdout[ite+STRIDE*i]=r0;
        i++;
        r0=cdin[ite+STRIDE*i];
    	cdout[ite+STRIDE*i]=r0;
        i++;
        r0=cdin[ite+STRIDE*i];
    	cdout[ite+STRIDE*i]=r0;
        i++;
        r0=cdin[ite+STRIDE*i];
    	cdout[ite+STRIDE*i]=r0;
        i++;
        r0=cdin[ite+STRIDE*i];
    	cdout[ite+STRIDE*i]=r0;
        i++;
        r0=cdin[ite+STRIDE*i];
    	cdout[ite+STRIDE*i]=r0;
        i++;
        r0=cdin[ite+STRIDE*i];
    	cdout[ite+STRIDE*i]=r0;
        i++;
        r0=cdin[ite+STRIDE*i];
    	cdout[ite+STRIDE*i]=r0;
        i++;
        r0=cdin[ite+STRIDE*i];
    	cdout[ite+STRIDE*i]=r0;
        i++;
        r0=cdin[ite+STRIDE*i];
    	cdout[ite+STRIDE*i]=r0;
        i++;
        r0=cdin[ite+STRIDE*i];
    	cdout[ite+STRIDE*i]=r0;
        i++;
        r0=cdin[ite+STRIDE*i];
    	cdout[ite+STRIDE*i]=r0;
        i++;
        r0=cdin[ite+STRIDE*i];
    	cdout[ite+STRIDE*i]=r0;
        i++;
        r0=cdin[ite+STRIDE*i];
    	cdout[ite+STRIDE*i]=r0;
        i++;
        r0=cdin[ite+STRIDE*i];
    	cdout[ite+STRIDE*i]=r0;
        i++;
    }
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

void runbench(int type, double* kernel_time, double* bandw,double* cdin,double* cdout,int L2size){

	cudaEvent_t start, stop;
	initializeEvents(&start, &stop);
	dim3 dimBlock(THREADS, 1, 1);
	dim3 dimGrid(BLOCKS, 1, 1);
	benchmark<float><<< dimGrid, dimBlock >>>((float*)cdin,(float*)cdout);


	long long shared_access = 2*(long long)(COMP_ITERATIONS)*THREADS*BLOCKS;

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
	CUdevice device = 0;
	int deviceCount;
	char deviceName[32];
    int L2size;

	cudaDeviceProp deviceProp;
    int dodouble=0;


	printf("Usage: %s [device_num] [metric_name]\n", argv[0]);
	int ntries;
	if (argc>1){
		ntries = atoi(argv[1]);
	}else{
		ntries = 1;
	}

	cudaSetDevice(deviceNum);
	double time[ntries][2],value[ntries][4];
	// StoreDeviceInfo_DRAM(stdout,&L2size);
	int size = (THREADS*BLOCKS+32*STRIDE)*sizeof(double);
	size_t freeCUDAMem, totalCUDAMem;
	cudaMemGetInfo(&freeCUDAMem, &totalCUDAMem);
	printf("Total GPU memory %lu, free %lu\n", totalCUDAMem, freeCUDAMem);
	printf("Buffer size: %dMB\n", size*sizeof(double)/(1024*1024));

	//Initialize Global Memory
	double *cdin;
	double *cdout;
	CUDA_SAFE_CALL(cudaMalloc((void**)&cdin, size));
	CUDA_SAFE_CALL(cudaMalloc((void**)&cdout, size));


	// Copy data to device memory
	CUDA_SAFE_CALL(cudaMemset(cdin, 0, size));  // initialize to zeros
	CUDA_SAFE_CALL(cudaMemset(cdout, 0, size));  // initialize to zeros
	// Synchronize in order to wait for memory operations to finish
	CUDA_SAFE_CALL(cudaThreadSynchronize());
	// make sure activity is enabled before any CUDA API

	cuDeviceGetCount(&deviceCount);
	if (deviceCount == 0) {
		printf("There is no device supporting CUDA.\n");
	return -2;
	}

	printf("CUDA Device Number: %d\n", deviceNum);

	cuDeviceGet(&device, deviceNum);
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, device));
	cuDeviceGetName(deviceName, 32, device);

	int i;
	class type;

    for (i=0;i<ntries;i++){
        runbench(dodouble,&time[0][0],&value[0][0],cdin,cdout,L2size);
    	printf("Registered time: %f ms\n",time[0][0]);
	}

    CUDA_SAFE_CALL( cudaDeviceReset());

    printf("-----------------------------------------------------------------------\n");
	return 0;
}
