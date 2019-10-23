// Utilities and system includes
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_profiler_api.h>

#define DATA_TYPE 1 // 0-SP, 1-INT, 2-DP
#define SIZE 60000000
#define TILE_DIM 1024

#define INNER_REPS 4

template <class T> __global__ void simpleKernel(T *A, T *C1, T *C2, T *C3, T *C4)
{
    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    T ra, rb, rc, rd;

    if (xIndex < SIZE) {
        ra=A[xIndex];
        rb=A[SIZE-xIndex];
        rc=A[xIndex];
        rd=A[SIZE-xIndex];

        // rb=A[xIndex];
        #pragma unroll 4
        for (int i=0;i<INNER_REPS;i++) {
          ra=ra+rb;
          rb=rb+rc;
          rc=rc+rd;
          rd=rd+ra;
        }
        C1[xIndex]=ra;
        C2[xIndex]=rb;
        C3[xIndex]=rc;
        C4[xIndex]=rd;

    }
}


int main(int argc, char **argv) {
    int outer_reps, vector_size, tile_dim;
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

    size_t mem_size = static_cast<size_t>(sizeof(int) * vector_size);
    // allocate host memory
    int *h_iA = (int *) malloc(mem_size);
    int *h_oC1 = (int *) malloc(mem_size);
    int *h_oC2 = (int *) malloc(mem_size);
    int *h_oC3 = (int *) malloc(mem_size);
    int *h_oC4 = (int *) malloc(mem_size);
    // initalize host data
    for (int i = 0; i < vector_size; ++i)
    {
        h_iA[i] = (int) i+3;
        // h_iB[i] = (float) i+3;
    }
    // allocate device memory
    int *d_iA, *d_iB, *d_oC1, *d_oC2, *d_oC3, *d_oC4;

    cudaMalloc((void **) &d_iA, mem_size);
    // cudaMalloc((void **) &d_iB, mem_size);
    cudaMalloc((void **) &d_oC1, mem_size);
    cudaMalloc((void **) &d_oC2, mem_size);
    cudaMalloc((void **) &d_oC3, mem_size);
    cudaMalloc((void **) &d_oC4, mem_size);

    // copy host data to device
    cudaMemcpy(d_iA, h_iA, mem_size, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_iB, h_iB, mem_size, cudaMemcpyHostToDevice);

    // print out common data for all kernels
    printf("\nVector size: %d  TotalBlocks: %d blockSize: %d\n\n", vector_size, grid.x, threads.x);

    // initialize events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // take measurements for loop over kernel launches
    cudaEventRecord(start, 0);

    for (int i=0; i < outer_reps; i++)
    {
        simpleKernel<int><<<grid, threads>>>(d_iA, d_oC1, d_oC2, d_oC3, d_oC4);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float kernelTime;
    cudaEventElapsedTime(&kernelTime, start, stop);

    // take measurements for loop inside kernel
    cudaMemcpy(h_oC1, d_oC1, mem_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_oC2, d_oC2, mem_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_oC3, d_oC3, mem_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_oC4, d_oC4, mem_size, cudaMemcpyDeviceToHost);

    printf("teste: %f\n", h_oC1[0]);

    // report effective bandwidths
    float kernelBandwidth = 2.0f * 1000.0f * mem_size/(1024*1024*1024)/(kernelTime/outer_reps);
    printf("simpleKernel, Throughput = %.4f GB/s, Time = %.5f ms, Size = %u fp32 elements, NumDevsUsed = %u, Workgroup = %u\n",
           kernelBandwidth,
           kernelTime/outer_reps,
           vector_size, 1, tile_dim * 1);

    free(h_iA);
    // free(h_iB);
    free(h_oC1);
    free(h_oC2);
    free(h_oC3);
    free(h_oC4);

    cudaFree(d_iA);
    // cudaFree(d_iB);
    cudaFree(d_oC1);
    cudaFree(d_oC2);
    cudaFree(d_oC3);
    cudaFree(d_oC4);


    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaDeviceReset();

    printf("Test passed\n");

    exit(EXIT_SUCCESS);
}
