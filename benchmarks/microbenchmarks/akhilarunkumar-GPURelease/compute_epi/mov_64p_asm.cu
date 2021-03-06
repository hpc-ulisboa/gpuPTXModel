#include <stdio.h>
#include <iostream>
#include <cuda_profiler_api.h>
//#include <cutil.h>
#include <cuda_runtime.h>

float* h_A;
float* h_B;
float* h_C;
float* h_res;
float* d_A;
float* d_B;
float* d_C;
float* d_res;

__global__
//void compute(const float* A, const float* B, const float* C, float* D, int n) {
void compute(float* D, int n, int div) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    float I1 = tid * 2.0;

    int thread_id = threadIdx.x % 32;

    if (thread_id < div) {
        __asm volatile (
                " .reg .f64 %r112;\n\t"
                " .reg .f64 %r113;\n\t"
                " .reg .f64 %r114;\n\t"
                " .reg .f64 %r115;\n\t"
                " .reg .f64 %r116;\n\t"
                " .reg .f64 %r117;\n\t"
                " .reg .f64 %r118;\n\t"
                " .reg .f64 %r119;\n\t"
                " .reg .f64 %r120;\n\t"
                " .reg .f64 %r121;\n\t"
                " .reg .f64 %r122;\n\t"
                " .reg .f64 %r123;\n\t"
                " .reg .f64 %r124;\n\t"
                " .reg .f64 %r125;\n\t"
                " .reg .f64 %r126;\n\t"
                " .reg .f64 %r127;\n\t"
                " .reg .f64 %r128;\n\t"
                " .reg .f64 %r129;\n\t"
                " .reg .f64 %r30;\n\t"
                "mov.f64 %r112, 4.4;\n\t"
                "mov.f64 %r113, 4.4;\n\t"
                "mov.f64 %r114, 2.2;\n\t"
                "mov.f64 %r115, 3.3;\n\t"
                "mov.f64 %r116, 1.23;\n\t"
                "mov.f64 %r117, 2.42;\n\t"
                "mov.f64 %r118, 3.34;\n\t"
                "mov.f64 %r119, 5.62;\n\t"
                "mov.f64 %r120, 2.56;\n\t"
                "mov.f64 %r121, 1.56;\n\t"
                "mov.f64 %r122, 2.56;\n\t"
                "mov.f64 %r123, 5.56;\n\t"
                "mov.f64 %r124, 8.56;\n\t"
                "mov.f64 %r125, 3.56;\n\t"
                "mov.f64 %r126, 5.56;\n\t"
                "mov.f64 %r127, 6.56;\n\t"
                "mov.f64 %r128, 0.56;\n\t"
                "mov.f64 %r129, 0.56;\n\t"

                );
        for (int k = 0; k < n; k++) {
            __asm volatile (
                    "mov.f64 %r122, %r121;\n\t" 
                    "mov.f64 %r121, %r120;\n\t" 
                    "mov.f64 %r120, %r119;\n\t" 
                    "mov.f64 %r119, %r118;\n\t" 
                    "mov.f64 %r118, %r117;\n\t" 
                    "mov.f64 %r117, %r116;\n\t" 
                    "mov.f64 %r116, %r115;\n\t" 
                    "mov.f64 %r115, %r30;\n\t" 
                    "mov.f64 %r30, %r129;\n\t" 
                    "mov.f64 %r129, %r128;\n\t" 
                    "mov.f64 %r128, %r127;\n\t" 
                    "mov.f64 %r127, %r126;\n\t" 
                    "mov.f64 %r126, %r125;\n\t" 
                    "mov.f64 %r125, %r124;\n\t" 
                    "mov.f64 %r124, %r123;\n\t" 
                    "mov.f64 %r123, %r122;\n\t" 
                    "mov.f64 %r122, %r121;\n\t" 
                    "mov.f64 %r121, %r120;\n\t" 
                    "mov.f64 %r120, %r119;\n\t" 
                    "mov.f64 %r119, %r118;\n\t" 
                    "mov.f64 %r118, %r117;\n\t" 
                    "mov.f64 %r117, %r116;\n\t" 
                    "mov.f64 %r116, %r115;\n\t" 
                    "mov.f64 %r115, %r30;\n\t" 
                    "mov.f64 %r30, %r129;\n\t" 
                    "mov.f64 %r129, %r128;\n\t" 
                    "mov.f64 %r128, %r127;\n\t" 
                    "mov.f64 %r127, %r126;\n\t" 
                    "mov.f64 %r126, %r125;\n\t" 
                    "mov.f64 %r125, %r124;\n\t" 
                    "mov.f64 %r124, %r123;\n\t" 
                    "mov.f64 %r123, %r122;\n\t" 
                    "mov.f64 %r122, %r121;\n\t" 
                    "mov.f64 %r121, %r120;\n\t" 
                    "mov.f64 %r120, %r119;\n\t" 
                    "mov.f64 %r119, %r118;\n\t" 
                    "mov.f64 %r118, %r117;\n\t" 
                    "mov.f64 %r117, %r116;\n\t" 
                    "mov.f64 %r116, %r115;\n\t" 
                    "mov.f64 %r115, %r30;\n\t" 
                    "mov.f64 %r30, %r129;\n\t" 
                    "mov.f64 %r129, %r128;\n\t" 
                    "mov.f64 %r128, %r127;\n\t" 
                    "mov.f64 %r127, %r126;\n\t" 
                    "mov.f64 %r126, %r125;\n\t" 
                    "mov.f64 %r125, %r124;\n\t" 
                    "mov.f64 %r124, %r123;\n\t" 
                    "mov.f64 %r123, %r122;\n\t" 
                    "mov.f64 %r122, %r121;\n\t" 
                    "mov.f64 %r121, %r120;\n\t" 
                    "mov.f64 %r120, %r119;\n\t" 
                    "mov.f64 %r119, %r118;\n\t" 
                    "mov.f64 %r118, %r117;\n\t" 
                    "mov.f64 %r117, %r116;\n\t" 
                    "mov.f64 %r116, %r115;\n\t" 
                    "mov.f64 %r115, %r30;\n\t" 
                    "mov.f64 %r122, %r121;\n\t" 
                    "mov.f64 %r121, %r120;\n\t" 
                    "mov.f64 %r120, %r119;\n\t" 
                    "mov.f64 %r119, %r118;\n\t" 
                    "mov.f64 %r118, %r117;\n\t" 
                    "mov.f64 %r117, %r116;\n\t" 
                    "mov.f64 %r116, %r115;\n\t" 
                    "mov.f64 %r115, %r30;\n\t" 
                    "mov.f64 %r30, %r129;\n\t" 
                    "mov.f64 %r129, %r128;\n\t" 
                    "mov.f64 %r128, %r127;\n\t" 
                    "mov.f64 %r127, %r126;\n\t" 
                    "mov.f64 %r126, %r125;\n\t" 
                    "mov.f64 %r125, %r124;\n\t" 
                    "mov.f64 %r124, %r123;\n\t" 
                    "mov.f64 %r123, %r122;\n\t" 
                    "mov.f64 %r122, %r121;\n\t" 
                    "mov.f64 %r121, %r120;\n\t" 
                    "mov.f64 %r120, %r119;\n\t" 
                    "mov.f64 %r119, %r118;\n\t" 
                    "mov.f64 %r118, %r117;\n\t" 
                    "mov.f64 %r117, %r116;\n\t" 
                    "mov.f64 %r116, %r115;\n\t" 
                    "mov.f64 %r115, %r30;\n\t" 
                    "mov.f64 %r30, %r129;\n\t" 
                    "mov.f64 %r129, %r128;\n\t" 
                    "mov.f64 %r128, %r127;\n\t" 
                    "mov.f64 %r127, %r126;\n\t" 
                    "mov.f64 %r126, %r125;\n\t" 
                    "mov.f64 %r125, %r124;\n\t" 
                    "mov.f64 %r124, %r123;\n\t" 
                    "mov.f64 %r123, %r122;\n\t" 
                    "mov.f64 %r122, %r121;\n\t" 
                    "mov.f64 %r121, %r120;\n\t" 
                    "mov.f64 %r120, %r119;\n\t" 
                    "mov.f64 %r119, %r118;\n\t" 
                    "mov.f64 %r118, %r117;\n\t" 
                    "mov.f64 %r117, %r116;\n\t" 
                    "mov.f64 %r116, %r115;\n\t" 
                    "mov.f64 %r115, %r30;\n\t" 
                    "mov.f64 %r30, %r129;\n\t" 
                    "mov.f64 %r129, %r128;\n\t" 
                    "mov.f64 %r128, %r127;\n\t" 
                    "mov.f64 %r127, %r126;\n\t" 
                    "mov.f64 %r126, %r125;\n\t" 
                    "mov.f64 %r125, %r124;\n\t" 
                    "mov.f64 %r124, %r123;\n\t" 
                    "mov.f64 %r123, %r122;\n\t" 
                    "mov.f64 %r122, %r121;\n\t" 
                    "mov.f64 %r121, %r120;\n\t" 
                    "mov.f64 %r120, %r119;\n\t" 
                    "mov.f64 %r119, %r118;\n\t" 
                    "mov.f64 %r118, %r117;\n\t" 
                    "mov.f64 %r117, %r116;\n\t" 
                    "mov.f64 %r116, %r115;\n\t" 
                    "mov.f64 %r115, %r30;\n\t" 
                    "mov.f64 %r30, %r129;\n\t" 
                    "mov.f64 %r129, %r128;\n\t" 
                    "mov.f64 %r128, %r127;\n\t" 
                    "mov.f64 %r127, %r126;\n\t" 
                    "mov.f64 %r126, %r125;\n\t" 
                    "mov.f64 %r125, %r124;\n\t" 
                    "mov.f64 %r124, %r123;\n\t" 
                    "mov.f64 %r123, %r122;\n\t" 
                    "mov.f64 %r122, %r121;\n\t" 
                    "mov.f64 %r121, %r120;\n\t" 
                    "mov.f64 %r120, %r119;\n\t" 
                    "mov.f64 %r119, %r118;\n\t" 
                    "mov.f64 %r118, %r117;\n\t" 
                    "mov.f64 %r117, %r116;\n\t" 
                    "mov.f64 %r116, %r115;\n\t" 
                    "mov.f64 %r115, %r30;\n\t" 
                    "mov.f64 %r30, %r129;\n\t" 
                    "mov.f64 %r129, %r128;\n\t" 
                    "mov.f64 %r128, %r127;\n\t" 
                    "mov.f64 %r127, %r126;\n\t" 
                    "mov.f64 %r126, %r125;\n\t" 
                    "mov.f64 %r125, %r124;\n\t" 
                    "mov.f64 %r124, %r123;\n\t" 
                    "mov.f64 %r123, %r122;\n\t" 
                    "mov.f64 %r122, %r121;\n\t" 
                    "mov.f64 %r121, %r120;\n\t" 
                    "mov.f64 %r120, %r119;\n\t" 
                    "mov.f64 %r119, %r118;\n\t" 
                    "mov.f64 %r118, %r117;\n\t" 
                    "mov.f64 %r117, %r116;\n\t" 
                    "mov.f64 %r116, %r115;\n\t" 
                    "mov.f64 %r115, %r30;\n\t" 
                    "mov.f64 %r30, %r129;\n\t" 
                    "mov.f64 %r129, %r128;\n\t" 
                    "mov.f64 %r128, %r127;\n\t" 
                    "mov.f64 %r127, %r126;\n\t" 
                    "mov.f64 %r126, %r125;\n\t" 
                    "mov.f64 %r125, %r124;\n\t" 
                    "mov.f64 %r124, %r123;\n\t" 
                    "mov.f64 %r123, %r122;\n\t" 
                    "mov.f64 %r122, %r121;\n\t" 
                    "mov.f64 %r121, %r120;\n\t" 
                    "mov.f64 %r120, %r119;\n\t" 
                    "mov.f64 %r119, %r118;\n\t" 
                    "mov.f64 %r118, %r117;\n\t" 
                    "mov.f64 %r117, %r116;\n\t" 
                    "mov.f64 %r116, %r115;\n\t" 
                    "mov.f64 %r115, %r30;\n\t" 
                    "mov.f64 %r30, %r129;\n\t" 
                    "mov.f64 %r129, %r128;\n\t" 
                    "mov.f64 %r128, %r127;\n\t" 
                    "mov.f64 %r127, %r126;\n\t" 
                    "mov.f64 %r126, %r125;\n\t" 
                    "mov.f64 %r125, %r124;\n\t" 
                    "mov.f64 %r124, %r123;\n\t" 
                    "mov.f64 %r123, %r122;\n\t" 
                    "mov.f64 %r122, %r121;\n\t" 
                    "mov.f64 %r121, %r120;\n\t" 
                    "mov.f64 %r120, %r119;\n\t" 
                    "mov.f64 %r119, %r118;\n\t" 
                    "mov.f64 %r118, %r117;\n\t" 
                    "mov.f64 %r117, %r116;\n\t" 
                    "mov.f64 %r116, %r115;\n\t" 
                    "mov.f64 %r115, %r30;\n\t" 
                    "mov.f64 %r30, %r129;\n\t" 
                    "mov.f64 %r129, %r128;\n\t" 
                    "mov.f64 %r128, %r127;\n\t" 
                    "mov.f64 %r127, %r126;\n\t" 
                    "mov.f64 %r126, %r125;\n\t" 
                    "mov.f64 %r125, %r124;\n\t" 
                    "mov.f64 %r124, %r123;\n\t" 
                    "mov.f64 %r123, %r122;\n\t" 
                    "mov.f64 %r122, %r121;\n\t" 
                    "mov.f64 %r121, %r120;\n\t" 
                    "mov.f64 %r120, %r119;\n\t" 
                    "mov.f64 %r119, %r118;\n\t" 
                    "mov.f64 %r118, %r117;\n\t" 
                    "mov.f64 %r117, %r116;\n\t" 
                    "mov.f64 %r116, %r115;\n\t" 
                    "mov.f64 %r115, %r30;\n\t" 
                    "mov.f64 %r30, %r129;\n\t" 
                    "mov.f64 %r129, %r128;\n\t" 
                    "mov.f64 %r128, %r127;\n\t" 
                    "mov.f64 %r127, %r126;\n\t" 
                    "mov.f64 %r126, %r125;\n\t" 
                    "mov.f64 %r125, %r124;\n\t" 
                    "mov.f64 %r124, %r123;\n\t" 
                    "mov.f64 %r123, %r122;\n\t" 
                    "mov.f64 %r122, %r121;\n\t" 
                    "mov.f64 %r121, %r120;\n\t" 
                    "mov.f64 %r120, %r119;\n\t" 
                    "mov.f64 %r119, %r118;\n\t" 
                    "mov.f64 %r118, %r117;\n\t" 
                    "mov.f64 %r117, %r116;\n\t" 
                    "mov.f64 %r116, %r115;\n\t" 
                    "mov.f64 %r115, %r30;\n\t" 
                    "mov.f64 %r30, %r129;\n\t" 
                    "mov.f64 %r129, %r128;\n\t" 
                    "mov.f64 %r128, %r127;\n\t" 
                    "mov.f64 %r127, %r126;\n\t" 
                    "mov.f64 %r126, %r125;\n\t" 
                    "mov.f64 %r125, %r124;\n\t" 
                    "mov.f64 %r124, %r123;\n\t" 
                    "mov.f64 %r123, %r122;\n\t" 
                    "mov.f64 %r122, %r121;\n\t" 
                    "mov.f64 %r121, %r120;\n\t" 
                    "mov.f64 %r120, %r119;\n\t" 
                    "mov.f64 %r119, %r118;\n\t" 
                    "mov.f64 %r118, %r117;\n\t" 
                    "mov.f64 %r117, %r116;\n\t" 
                    "mov.f64 %r116, %r115;\n\t" 
                    "mov.f64 %r115, %r30;\n\t" 
                    "mov.f64 %r30, %r129;\n\t" 
                    "mov.f64 %r129, %r128;\n\t" 
                    "mov.f64 %r128, %r127;\n\t" 
                    "mov.f64 %r127, %r126;\n\t" 
                    "mov.f64 %r126, %r125;\n\t" 
                    "mov.f64 %r125, %r124;\n\t" 
                    "mov.f64 %r124, %r123;\n\t" 
                    "mov.f64 %r123, %r122;\n\t" 
                    "mov.f64 %r122, %r121;\n\t" 
                    "mov.f64 %r121, %r120;\n\t" 
                    "mov.f64 %r120, %r119;\n\t" 
                    "mov.f64 %r119, %r118;\n\t" 
                    "mov.f64 %r118, %r117;\n\t" 
                    "mov.f64 %r117, %r116;\n\t" 
                    "mov.f64 %r116, %r115;\n\t" 
                    "mov.f64 %r115, %r30;\n\t" 
                    "mov.f64 %r30, %r129;\n\t" 
                    "mov.f64 %r129, %r128;\n\t" 
                    "mov.f64 %r128, %r127;\n\t" 
                    "mov.f64 %r127, %r126;\n\t" 
                    "mov.f64 %r126, %r125;\n\t" 
                    "mov.f64 %r125, %r124;\n\t" 
                    "mov.f64 %r124, %r123;\n\t" 
                    "mov.f64 %r123, %r122;\n\t" 
                    "mov.f64 %r122, %r121;\n\t" 
                    "mov.f64 %r121, %r120;\n\t" 
                    "mov.f64 %r120, %r119;\n\t" 
                    "mov.f64 %r119, %r118;\n\t" 
                    "mov.f64 %r118, %r117;\n\t" 
                    "mov.f64 %r117, %r116;\n\t" 
                    "mov.f64 %r116, %r115;\n\t" 
                    "mov.f64 %r115, %r30;\n\t" 
                    );
        }
    }
    __syncthreads();

    //    if ((blockDim.x * blockIdx.x + threadIdx.x) == 0)
    *D = I1;

    __syncthreads();
}

void usage() {
    std::cout << "Usage ./binary <num_blocks> <num_threads_per_block> <iterations>" "threads active per warp" << std::endl;
}

int main(int argc, char **argv)
{
    if (argc != 5) {
        usage();
        exit(1);
    }

    int num_blocks = atoi(argv[1]);
    int num_threads_per_block = atoi(argv[2]);
    int iterations = atoi(argv[3]);
    int divergence = atoi(argv[4]);

//    h_A = new float(2.0);
//    h_B = new float(3.0);
//    h_C = new float(4.0);

//    cudaMalloc((void**)&d_A, sizeof(float));
//    cudaMalloc((void**)&d_B, sizeof(float));
//    cudaMalloc((void**)&d_C, sizeof(float));
    cudaMalloc((void**)&d_res, sizeof(float));
  
//    cudaMemcpy(d_A, h_A, sizeof(float), cudaMemcpyHostToDevice);
//    cudaMemcpy(d_B, h_B, sizeof(float), cudaMemcpyHostToDevice);
//    cudaMemcpy(d_C, h_C, sizeof(float), cudaMemcpyHostToDevice);
     
    cudaEvent_t start, stop;
    float time;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    cudaProfilerStart();
    
//    compute<<<num_blocks, num_threads_per_block>>>(d_A, d_B, d_C, d_res, iterations);
    compute<<<num_blocks, num_threads_per_block>>>(d_res, iterations, divergence);

    cudaProfilerStop();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);

    std::cout << "GPU Elapsed Time = " << time << std::endl;
  
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaDeviceSynchronize();

    cudaMemcpy(h_res, d_res, sizeof(float), cudaMemcpyDeviceToHost);

    return 0;
}
