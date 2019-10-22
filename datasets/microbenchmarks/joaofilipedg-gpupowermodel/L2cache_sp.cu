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
// #include <gpuCUPTISampler.cuh>

#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                            \
  (((uintptr_t) (buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t) (buffer) & ((align)-1))) : (buffer))

#define COMP_ITERATIONS (512)
#define THREADS (1024)
#define BLOCKS (32760)
#define N (10)

#define REGBLOCK_SIZE (4)
#define UNROLL_ITERATIONS (32)
#define deviceNum (0)

// #define OFFSET

// __constant__ __device__ int off [16] = {0,4,8,12,9,13,1,5,2,6,10,14,11,15,3,7}; //512 threads
// __constant__ __device__ int off [16] = {0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3}; //512 threads
// __constant__ __device__ int off [16] = {0,2,4,6,8,10,12,14,11,9,15,13,3,1,7,5}; //256 threads

template <class T> __global__ void benchmark (T* cdin, T* cdout){


	const int ite = blockIdx.x * THREADS + threadIdx.x;
	T r0;

	// for (int k=0; k<N;k++){
    #pragma unroll 512
	for(int j=0; j<COMP_ITERATIONS*N; j+=UNROLL_ITERATIONS){
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
	benchmark<float><<< dimGrid, dimBlock >>>((float*)cdin,(float*)cdout);

	long long shared_access = 2*(long long)(COMP_ITERATIONS)*N*THREADS*BLOCKS;

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
	CUdevice device = 0;
	int deviceCount;
	char deviceName[32];

	cudaDeviceProp deviceProp;

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
	int size = THREADS*BLOCKS*sizeof(double);
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
	CUDA_SAFE_CALL(cudaMemset(cdin, 1, size));  // initialize to zeros
	CUDA_SAFE_CALL(cudaMemset(cdout, 0, size));  // initialize to zeros
	// Synchronize in order to wait for memory operations to finish
	CUDA_SAFE_CALL(cudaThreadSynchronize());
	// make sure activity is enabled before any CUDA API
	// CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));

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
		runbench(0,&time[0][0],&value[0][0],cdin,cdout);


        printf("Registered time: %f ms\n",time[0][0]);
    }




    CUDA_SAFE_CALL( cudaDeviceReset());

	// switch(deviceProp.major){
	// 	// case 5: //Maxwell
	// 		// counters=2;
	// 		// break;
	// 	case 3: //Kepler
	// 		counters=8;
	// 		break;
	// 	default: //Fermi
	// 		counters=4;
	// 		break;
	// 	}
    //
	// /* SP/DP ACCESS*/{
	// for(int j=0;j<counters;j++){
	// 	CUPTI_CALL(cuptiEventGroupCreate(context, &cuptiEvent.eventGroup, 0));
	// 	CUPTI_CALL(cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)getEventValueCallback , &trace));
	// 	cuptiEvent.numEvents=1;
	// 	switch(deviceProp.major){
	// 		// case 5://Maxwell
	// 			// cuptiEvent.numEvents=2;
	// 			// switch (j){
	// 				// case(1):
	// 					// cuptiEvent.eventName[0] = "l2_subp0_total_read_sector_queries";
	// 					// CUPTI_CALL(cuptiEventGetIdFromName(device, cuptiEvent.eventName[0], &cuptiEvent.eventId[0]));
	// 					// CUPTI_CALL(cuptiEventGroupAddEvent(cuptiEvent.eventGroup,cuptiEvent.eventId[0]));
	// 					// cuptiEvent.eventName[1] = "l2_subp1_total_read_sector_queries";
	// 					// CUPTI_CALL(cuptiEventGetIdFromName(device, cuptiEvent.eventName[1], &cuptiEvent.eventId[1]));
	// 					// CUPTI_CALL(cuptiEventGroupAddEvent(cuptiEvent.eventGroup,cuptiEvent.eventId[1]));
	// 					// break;
	// 				// default:
	// 					// cuptiEvent.eventName[0] = "l2_subp0_total_write_sector_queries";
	// 					// CUPTI_CALL(cuptiEventGetIdFromName(device, cuptiEvent.eventName[0], &cuptiEvent.eventId[0]));
	// 					// CUPTI_CALL(cuptiEventGroupAddEvent(cuptiEvent.eventGroup,cuptiEvent.eventId[0]));
	// 					// cuptiEvent.eventName[1] = "l2_subp1_total_write_sector_queries";
	// 					// CUPTI_CALL(cuptiEventGetIdFromName(device, cuptiEvent.eventName[1], &cuptiEvent.eventId[1]));
	// 					// CUPTI_CALL(cuptiEventGroupAddEvent(cuptiEvent.eventGroup,cuptiEvent.eventId[1]));
	// 					// break;
	// 			// }
	// 			// break;
	// 		case 3: //Kepler
	// 			switch (j){
	// 				case(1):
	// 					cuptiEvent.eventName[0] = "l2_subp1_total_write_sector_queries";
	// 					break;
	// 				case(2):
	// 					cuptiEvent.eventName[0] = "l2_subp2_total_write_sector_queries";
	// 					break;
	// 				case(3):
	// 					cuptiEvent.eventName[0] = "l2_subp3_total_write_sector_queries";
	// 					break;
	// 				case(4):
	// 					cuptiEvent.eventName[0] = "l2_subp0_total_read_sector_queries";
	// 					break;
	// 				case(5):
	// 					cuptiEvent.eventName[0] = "l2_subp1_total_read_sector_queries";
	// 					break;
	// 				case(6):
	// 					cuptiEvent.eventName[0] = "l2_subp2_total_read_sector_queries";
	// 					break;
	// 				case(7):
	// 					cuptiEvent.eventName[0] = "l2_subp3_total_read_sector_queries";
	// 					break;
	// 				default:
	// 					cuptiEvent.eventName[0] = "l2_subp0_total_write_sector_queries";
	// 					break;
	// 			}
	// 			CUPTI_CALL(cuptiEventGetIdFromName(device, cuptiEvent.eventName[0], &cuptiEvent.eventId[0]));
	// 			CUPTI_CALL(cuptiEventGroupAddEvent(cuptiEvent.eventGroup,cuptiEvent.eventId[0]));
	// 			break;
	// 		default: //Fermi
	// 			switch (j){
	// 				case(1):
	// 					cuptiEvent.eventName[0] = "l2_subp1_total_write_sector_queries";
	// 					break;
	// 				case(2):
	// 					cuptiEvent.eventName[0] = "l2_subp0_total_read_sector_queries";
	// 					break;
	// 				case(3):
	// 					cuptiEvent.eventName[0] = "l2_subp1_total_read_sector_queries";
	// 					break;
	// 				default:
	// 					cuptiEvent.eventName[0] = "l2_subp0_total_write_sector_queries";
	// 					break;
	// 			}
	// 			CUPTI_CALL(cuptiEventGetIdFromName(device, cuptiEvent.eventName[0], &cuptiEvent.eventId[0]));
	// 			CUPTI_CALL(cuptiEventGroupAddEvent(cuptiEvent.eventGroup,cuptiEvent.eventId[0]));
	// 			break;
	// 	}
    //
	// 	for (i=0;i<1;i++){
	// 		if(i==0)
	// 			printf("--------------------- Test SP %d - %s ---------------------\n",i,cuptiEvent.eventName[0]);
	// 		else
	// 			printf("--------------------- Test DP %d - %s ---------------------\n",i,cuptiEvent.eventName[0]);
    //
    //
	// 		printf("Event ID: %ld\n", cuptiEvent.eventId);
    //
	// 		trace.eventData = &cuptiEvent;
	// 		uint32_t all = 1;
	// 		CUPTI_CALL(cuptiEventGroupSetAttribute(cuptiEvent.eventGroup,CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES,sizeof(all),&all));
    //
	// 		CUPTI_CALL(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020));
    //
	// 		runbench(i,&time[0][0],&value[0][0],cdin,cdout);
	// 		if (i==0)
	// 			SPresult[0]+=displayEventVal(&trace, &cuptiEvent,device);
	// 		else
	// 			DPresult[0]+=displayEventVal(&trace, &cuptiEvent,device);
    //
	// 		printf("Registered time: %f ms\n",time[0][0]);
    //
	// 		trace.eventData = NULL;
	// 	}
	// 	CUPTI_CALL(cuptiEventGroupRemoveEvent(cuptiEvent.eventGroup, cuptiEvent.eventId[0]));
	// 	CUPTI_CALL(cuptiEventGroupDestroy(cuptiEvent.eventGroup));
	// 	CUPTI_CALL(cuptiUnsubscribe(subscriber));
	// 	CUDA_SAFE_CALL(cudaDeviceReset());
	// 	}
	// }
    //
	// /* CLOCKS CYCLES QUERY*/{
	// CUPTI_CALL(cuptiEventGroupCreate(context, &cuptiEvent.eventGroup, 0));
	// CUPTI_CALL(cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)getEventValueCallback , &trace));
	// cuptiEvent.eventName[0] = "elapsed_cycles_sm";
	// CUPTI_CALL(cuptiEventGetIdFromName(device, cuptiEvent.eventName[0], &cuptiEvent.eventId[0]));
	// CUPTI_CALL(cuptiEventGroupAddEvent(cuptiEvent.eventGroup,cuptiEvent.eventId[0]));
	// cuptiEvent.numEvents=1;
    //
	// int dodouble=0;
	// for (i=0;i<ntries;i++){
    //
	// 	if(i<ntries){
	// 		printf("--------------------- Test SP %d - %s ---------------------\n",i,cuptiEvent.eventName[0]);
	// 	}else{
	// 		dodouble=1;
	// 		printf("--------------------- Test DP %d - %s ---------------------\n",i,cuptiEvent.eventName[0]);
	// 	}
    //
	// 	printf("Event ID: %d\n", cuptiEvent.eventId[0]);
    //
	// 	trace.eventData = &cuptiEvent;
	// 	uint32_t all = 1;
	// 	CUPTI_CALL(cuptiEventGroupSetAttribute(cuptiEvent.eventGroup,CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES,sizeof(all),&all));
    //
	// 	CUPTI_CALL(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020));
    //
	// 	runbench(dodouble,&time[0][0],&value[0][0],cdin,cdout);
    //
	// 	timeresult[i%ntries][dodouble]=displayEventVal(&trace, &cuptiEvent,device);
    //
	// 	printf("Registered time: %f ms\n",time[0][0]);
    //
	// 	trace.eventData = NULL;
	// 	CUDA_SAFE_CALL(cudaDeviceReset());
    //
	// }
	// CUPTI_CALL(cuptiEventGroupRemoveEvent(cuptiEvent.eventGroup, cuptiEvent.eventId[0]));
	// CUPTI_CALL(cuptiEventGroupDestroy(cuptiEvent.eventGroup));
	// CUPTI_CALL(cuptiUnsubscribe(subscriber));
	// }
    //
	// // // BENCHMARK
	// // for(i=0;i<ntries;i++){
	// 	// runbench(0,&time[i][0],&value[i][0],cdin,cdout);
	// // }
	// // for(i=0;i<ntries;i++){
	// 	// runbench(1,&time[i][1],&value[i][1],cdin,cdout);
	// // }
    //
	// CUDA_SAFE_CALL( cudaDeviceReset());
    //
	// //Print results
	// printf("-----------------------------------------------------------------------\n");
	// //Check if flops results are equal
	// for(i=0;i<0;i++){
	// 	if(SPresult[i]!=SPresult[i+1]){
	// 		printf("SPresult= %ld\n",SPresult[i]);
	// 		printf("SP Result %d is wrong\n", i);
	// 		break;
	// 	}
	// 	if(DPresult[i]!=DPresult[i+1]){
	// 		printf("DPresult= %ld\n",DPresult[i]);
	// 		printf("DP Result %d is wrong\n", i);
	// 		break;
	// 	}
	// }
	// const long long access = (long long) BLOCKS *N* THREADS * COMP_ITERATIONS/8;
	// double result = ((double)SPresult[0]) /(double)access;
	// printf("Run size fit L2 size = %s\n",THREADS*BLOCKS*4*2>L2size?"False":"True");
	// printf("SP Experimental / Theoretical (total=%ld) = %f \n",SPresult[0], result);
	// result = ((double)DPresult[0]) /(double)access;
	// printf("DP Experimental / Theoretical = %f \n", result);
    //
	// printf("-----------------------------------------------------------------------\n");
	// printf("  Single-Precision Floating Point   | Double-Precision Floating Point\n");
	// printf("Test    Ex. Time(ms)    GB/sec  |  Ex. Time(ms)    GB/sec\n");
    //
	// mean[0]=mean[1]=mean[2]=mean[3]=0;
	// sum_dev_median[0]=sum_dev_median[1]=sum_dev_median[2]=sum_dev_median[3]=0;
	// sum_dev_mean[0]=sum_dev_mean[1]=sum_dev_mean[2]=sum_dev_mean[3]=0;
    //
	// for (i=0; i < ntries; i++){
	// 	value[i][2]=(double)SPresult[0]/timeresult[i][0];
	// 	value[i][3]=(double)DPresult[0]/timeresult[i][1];
	// 	printf("%2d %11.2f %16.2f     |  %5.2f %14.2f\n",i,time[i][0],value[i][0],time[i][1],value[i][1]);
	// 	mean[0]+=value[i][0];
	// 	mean[1]+=value[i][1];
	// 	mean[2]+=value[i][2];
	// 	mean[3]+=value[i][3];
	// }
    //
	// mean[0]=mean[0]/ntries;
	// mean[1]=mean[1]/ntries;
	// mean[2]=mean[2]/ntries;
	// mean[3]=mean[3]/ntries;
    //
    //
	// medianv[0] = median(ntries,value,0);
	// medianv[1] = median(ntries,value,1);
	// medianv[2] = median(ntries,value,2);
	// medianv[3] = median(ntries,value,3);
    //
 // 	for(i=0; i<ntries; i++){
	// 	for(int j=0; j<4;j++){
	// 		sum_dev_mean[j]+=(value[i][j]-mean[j])*(value[i][j]-mean[j]);
	// 		sum_dev_median[j]+=(value[i][j]-medianv[j])*(value[i][j]-medianv[j]);
	// 	}
	// }
    //
	// for(i=0; i<4;i++){
	// 	std_dev_mean[i]= sqrt(sum_dev_mean[i]/ntries);
	// 	std_dev_median[i]= sqrt(sum_dev_median[i]/ntries);
	// }
    //
	// printf("-----------------------------------------------------RESULTS-----------------------------------------------------\n");
	// printf("	             |\t\t Benchmark (GB/s) \t\t|\t\t CUPTI Query (Accesses/clk) \t|\n");
	// printf("                     |\t  Single \t|\t Double \t|\t Single \t|\t Double \t|\n");
	// printf("Median               |\t  %.2f \t|\t %.2f \t|\t %.3f \t|\t %.3f \t|\n", medianv[0],medianv[1],medianv[2],medianv[3]);
	// printf("Mean                 |\t  %.2f \t|\t %.2f \t|\t %.3f \t|\t %.3f \t|\n",mean[0], mean[1],mean[2], mean[3]);
	// printf("Std. Dev. Median     |\t  %.5f \t|\t %.5f \t|\t %.3f \t\t|\t %.3f \t\t|\n", std_dev_median[0],std_dev_median[1],std_dev_median[2],std_dev_median[3]);
	// printf("Std. Dev. Mean       |\t  %.5f \t|\t %.5f \t|\t %.3f \t\t|\t %.3f \t\t|\n", std_dev_mean[0],std_dev_mean[1],std_dev_mean[2],std_dev_mean[3]);
	// printf("Percentage (Median)  |\t  %.2f% \t|\t %.2f% \t|\t %.2f% \t|\t %.2f% \t|\n",(medianv[0]/L2*100),(medianv[1]/L2*100),(medianv[2]/L2*100),(medianv[3]/L2*100));
	// printf("Percentage (Mean)    |\t  %.2f% \t|\t %.2f% \t|\t %.2f% \t|\t %.2f% \t|\n",(mean[0]/L2*100),(mean[1]/L2*100),(mean[2]/L2*100),(mean[3]/L2*100));
	// printf("Std. Dev Median (%)  |\t  %.2f% \t|\t %.2f% \t\t|\t %.2f% \t\t|\t %.2f% \t\t|\n",(std_dev_median[0]/medianv[0]*100),(std_dev_median[1]/medianv[1]*100),(std_dev_median[2]/medianv[2]*100),(std_dev_median[3]/medianv[3]*100));
	// printf("Std. Dev Mean (%)    |\t  %.2f% \t|\t %.2f% \t\t|\t %.2f% \t\t|\t %.2f% \t\t|\n\n",(std_dev_mean[0]/mean[0]*100),(std_dev_mean[1]/mean[1]*100),(std_dev_mean[2]/mean[2]*100),(std_dev_mean[3]/mean[3]*100));


	return 0;
}
