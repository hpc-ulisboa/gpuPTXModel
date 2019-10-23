#include <stdio.h>
#include <iostream>
#include <cuda_profiler_api.h>
//#include <cutil.h>
#include <cuda_runtime.h>
#include <string>

#define SHARED_MEM_ELEMENTS 1024
#define GLOBAL_MEM_ELEMENTS 4096

int num_blocks;
int num_threads_per_block;
int num_iterations;
int divergence;

float* h_A;
float* h_B;
float* h_C;
float* h_res;
float* d_A;
float* d_B;
float* d_C;
float* d_res;

__global__ void init_memory (unsigned long long ** my_ptr_array, unsigned long long * my_array, int stride, int num_blocks_k, int num_threads_per_block_k) {

    int block_id;
    int warp_id;
    int i;

    int index;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    void **ptr_array = (void **)my_ptr_array;
    unsigned long long *array = (unsigned long long *)my_array;

    if (tid == 0) {
//        int elements_per_block = GLOBAL_MEM_ELEMENTS / num_blocks_k;
        int num_warps_per_block = num_threads_per_block_k / 32;
        //int elements_per_warp = elements_per_block / num_warps_per_block;
        int elements_per_warp = GLOBAL_MEM_ELEMENTS / num_warps_per_block;

//        for (block_id = 0; block_id < num_blocks_k; block_id++) {
            for (warp_id = 0; warp_id < num_warps_per_block; warp_id++) {
                for (i = 0; i < elements_per_warp; i++) {
                    //index = (block_id * elements_per_block) + (warp_id * elements_per_warp);
                    index = (warp_id * elements_per_warp);
                    ptr_array[index + i] = (void*)&array[(index + ((i + 16) % elements_per_warp))];
                }
            }

/*        for (i = 0; i < GLOBAL_MEM_ELEMENTS; i++) {
            ptr_array[i] = (void*)&array[(i + 32)%GLOBAL_MEM_ELEMENTS];
        }
*/
        for (i = 0; i < GLOBAL_MEM_ELEMENTS; i++) {
            //array[i] = (unsigned long long)ptr_array[(i+stride)%GLOBAL_MEM_ELEMENTS];
            array[i] = (unsigned long long)ptr_array[i];
        }
    }

    __syncthreads();

}

__global__ void shared_latency (unsigned long long ** my_ptr_array, unsigned long long * my_array, int array_length, int iterations, unsigned long long * duration, int stride, int divergence, int num_blocks_k, int num_threads_per_block_k) {

//    unsigned long long int start_time, end_time;
    unsigned long long int sum_time = 0;
    int i, k;

    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / 32;
    int warp_thread_id = threadIdx.x % 32;

//    int elements_per_block = GLOBAL_MEM_ELEMENTS / num_blocks_k;
    int num_warps_per_block = num_threads_per_block_k / 32;
//    int elements_per_warp = elements_per_block / num_warps_per_block;
    int elements_per_warp = GLOBAL_MEM_ELEMENTS / num_warps_per_block;

    //int index1 = (block_id * elements_per_block) + (warp_id * elements_per_warp) + warp_thread_id;
    int index1 = (warp_id * elements_per_warp) + warp_thread_id;

    void **ptr_array = (void **)my_ptr_array;
    unsigned long long int *array = (unsigned long long int *)my_array;

    void **tmp_ptr;

    //tmp_ptr = (void *)sdata;
    //tmp_ptr = (void **)(&(ptr_array[(threadIdx.x * stride)%GLOBAL_MEM_ELEMENTS]));
    //tmp_ptr = (void **)(&(ptr_array[(tid * stride)%GLOBAL_MEM_ELEMENTS]));
    //tmp_ptr = (void **)(&(ptr_array[index1]));
    tmp_ptr = (void **)(&(array[index1]));

        double f1, f2, f3;
        f1 = 1.1;
        f2 = 2.5;
    if (warp_thread_id < divergence) {
/*        __asm volatile (
                ".reg .f32 %r14;\n\t"
                "mov.f32 %r14, 2.2;\n\t"
                );
*/

        for (int l = 0; l < iterations; l++) {
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            tmp_ptr = (void**)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
            f1 = f1 + (unsigned long long)(*tmp_ptr);
        }
    }
//    __syncthreads();

    //    if ((blockDim.x * blockIdx.x + threadIdx.x) == 0)
    duration[tid] = (unsigned long long)(*tmp_ptr) + (f1 * tid);

//    __syncthreads();
}

void usage() {
    std::cout << "Usage ./binary <num_blocks> <num_threads_per_block> <iterations>" "threads active per warp" << std::endl;
}

void parametric_measure_shared(int N, int iterations, int stride) {

    cudaProfilerStop();
    int i;
    unsigned long long int * h_a;
    unsigned long long int * d_a;

    unsigned long long ** h_ptr_a;
    unsigned long long ** d_ptr_a;

    unsigned long long * duration;
    unsigned long long * latency;

    cudaError_t error_id;

    /* allocate array on CPU */
    h_a = (unsigned long long *)malloc(sizeof(unsigned long long int) * N);

    h_ptr_a = (unsigned long long **)malloc(sizeof(unsigned long long int*)*N);

    latency = (unsigned long long *)malloc(sizeof(unsigned long long) * num_threads_per_block * num_blocks);

    /* initialize array elements on CPU */


    for (i = 0; i < N; i++) {
        h_ptr_a[i] = (unsigned long long *)&h_a[i];
    }
    for (i = 0; i < N; i++) {
        h_a[i] = (unsigned long long)h_ptr_a[(i + 1 + stride) % N];
    }

    /* allocate arrays on GPU */
    cudaMalloc ((void **) &d_a, sizeof(unsigned long long int) * N );
    cudaMalloc ((void **) &d_ptr_a, sizeof(unsigned long long int*) * N );
    cudaMalloc ((void **) &duration, sizeof(unsigned long long) * num_threads_per_block * num_blocks);

    cudaThreadSynchronize ();
    error_id = cudaGetLastError();
    if (error_id != cudaSuccess) {
        printf("Error 1 is %s\n", cudaGetErrorString(error_id));
    }

    /* copy array elements from CPU to GPU */
    cudaMemcpy((void *)d_a, (void *)h_a, sizeof(unsigned long long int) * N, cudaMemcpyHostToDevice);
    cudaMemcpy((void *)d_ptr_a, (void *)h_ptr_a, sizeof(unsigned long long int *) * N, cudaMemcpyHostToDevice);
    cudaMemcpy((void *)duration, (void *)latency, sizeof(unsigned long long) * num_threads_per_block * num_blocks, cudaMemcpyHostToDevice);

    cudaThreadSynchronize ();

    error_id = cudaGetLastError();
    if (error_id != cudaSuccess) {
        printf("Error 2 is %s\n", cudaGetErrorString(error_id));
    }

    init_memory <<<1, 1>>>(d_ptr_a, d_a, stride, num_blocks, num_threads_per_block);
    cudaDeviceSynchronize();

    /* launch kernel*/
    //dim3 Db = dim3(13);
    //dim3 Dg = dim3(768,1,1);

    //printf("Launch kernel with parameters: %d, N: %d, stride: %d\n", iterations, N, stride);
    //	int sharedMemSize =  sizeof(unsigned long long int) * N ;

    cudaEvent_t start, stop;
    float time;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    cudaProfilerStart();

    cudaFuncSetCacheConfig(shared_latency, cudaFuncCachePreferL1);
    //shared_latency <<<Dg, Db, sharedMemSize>>>(d_a, N, iterations, duration);
    //shared_latency <<<num_blocks, num_threads_per_block, sharedMemSize>>>(d_a, N, num_iterations, duration, stride, divergence);
    shared_latency <<<num_blocks, num_threads_per_block>>>(d_ptr_a, d_a, N, num_iterations, duration, stride, divergence, num_blocks, num_threads_per_block);

    cudaDeviceSynchronize();
    ///cudaThreadSynchronize ();

    cudaProfilerStop();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);

    error_id = cudaGetLastError();
    if (error_id != cudaSuccess) {
        printf("Error 3 is %s\n", cudaGetErrorString(error_id));
    }

    /* copy results from GPU to CPU */

    cudaMemcpy((void *)h_a, (void *)d_a, sizeof(unsigned long long int) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy((void *)latency, (void *)duration, sizeof(unsigned long long) * num_threads_per_block * num_blocks, cudaMemcpyDeviceToHost);

    cudaThreadSynchronize ();

    /* print results*/


    unsigned long long max_dur = latency[0];
    unsigned long long min_dur = latency[0];
    unsigned long long avg_lat = latency[0];
    for (int i = 1; i < num_threads_per_block * num_blocks; i++) {
        avg_lat += latency[i];
        if (latency[i] > max_dur) {
            max_dur = latency[i];
        } else if (latency[i] < min_dur) {
            min_dur = latency[i];
        }
    }


    //	printf("  %d, %f, %f, %f, %f\n",stride,(double)(avg_lat/(num_threads_per_block * num_blocks * 256.0 *num_iterations)), (double)(min_dur/(256.0 * num_iterations)), (double)(max_dur/(256.0 * num_iterations)), time);

    printf("%f\n", time);


    /* free memory on GPU */
    cudaFree(d_a);
    cudaFree(d_ptr_a);
    cudaFree(duration);
    cudaThreadSynchronize ();

    /*free memory on CPU */
    free(h_a);
    free(h_ptr_a);
    free(latency);


}


int main(int argc, char **argv)
{
    int N;

    if (argc != 6) {
        usage();
        exit(1);
    }

    num_blocks = atoi(argv[1]);
    num_threads_per_block = atoi(argv[2]);
    num_iterations = atoi(argv[3]);
    divergence = atoi(argv[4]);
    int stride = atoi(argv[5]);

    N = GLOBAL_MEM_ELEMENTS;
    parametric_measure_shared(N, 10, stride);

    return 0;
}
