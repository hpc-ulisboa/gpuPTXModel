// Utilities and system includes
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
// #include <cupti.h>
#include <cuda_profiler_api.h>

#define DATA_TYPE 0 // 0-SP, 1-INT, 2-DP
#define THREADS 1024

#define TILE_DIM 1024
#define SIZE 60000000

#define INNER_REPS 1

template <class T> __global__ void simpleKernel2()
{
    __shared__ T shared[THREADS];
    T r0;
    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;


    if (xIndex < SIZE) {
        #pragma unroll 1
        for (int i=0;i<INNER_REPS;i++) {
            r0 = shared[threadIdx.x];
            shared[THREADS - threadIdx.x - 1] = r0;
        }
    }
}


int main(int argc, char **argv) {
    int inner_reps, outer_reps, vector_size, tile_dim;
    inner_reps = INNER_REPS;
    vector_size = SIZE;
    tile_dim = TILE_DIM;

    if (argc>1){
        outer_reps = atoi(argv[1]);
    }else{
        outer_reps = 1;
    }

    // execution configuration parameters
    dim3 grid(vector_size/tile_dim, 1), threads(tile_dim, 1);

    // CUDA events
    cudaEvent_t start, stop;

    // print out common data for all kernels
    printf("\nVector size: %d  TotalBlocks: %d blockSize: %d\n\n", vector_size, grid.x, threads.x);

    // initialize events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // take measurements for loop over kernel launches
    cudaEventRecord(start, 0);

    for (int i=0; i < outer_reps; i++)
    {
        simpleKernel2<float><<<grid, threads>>>();
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float kernelTime;
    cudaEventElapsedTime(&kernelTime, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaDeviceReset();

    printf("Test passed\n");

    exit(EXIT_SUCCESS);
}
