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
                " .reg .s32 %r111;\n\t"
                " .reg .s32 %r112;\n\t"
                " .reg .s32 %r113;\n\t"
                " .reg .s32 %r114;\n\t"
                " .reg .s32 %r115;\n\t"
                " .reg .s32 %r116;\n\t"
                " .reg .s32 %r117;\n\t"
                " .reg .s32 %r118;\n\t"
                " .reg .s32 %r119;\n\t"
                " .reg .s32 %r120;\n\t"
                " .reg .s32 %r121;\n\t"
                " .reg .s32 %r122;\n\t"
                " .reg .s32 %r123;\n\t"
                " .reg .s32 %r124;\n\t"
                " .reg .s32 %r125;\n\t"
                " .reg .s32 %r126;\n\t"
                " .reg .s32 %r127;\n\t"
                " .reg .s32 %r128;\n\t"
                "mov.s32 %r112, 44;\n\t"
                "mov.s32 %r113, %r112;\n\t"
                "mov.s32 %r114, 22;\n\t"
                "mov.s32 %r115, 33;\n\t"
                "mov.s32 %r116, 123;\n\t"
                "mov.s32 %r117, 242;\n\t"
                "mov.s32 %r118, 334;\n\t"
                "mov.s32 %r119, 562;\n\t"
                "mov.s32 %r120, 256;\n\t"
                "mov.s32 %r121, 156;\n\t"
                "mov.s32 %r122, 256;\n\t"
                "mov.s32 %r123, 556;\n\t"
                "mov.s32 %r124, 856;\n\t"
                "mov.s32 %r125, 356;\n\t"
                "mov.s32 %r126, 556;\n\t"
                "mov.s32 %r127, 656;\n\t"
                "mov.s32 %r128, 56;\n\t"

                );
        for (int k = 0; k < n; k++) {
            __asm volatile (
                    "bfe.s32 %r113, %r111, %r113, %r112;\n\t"
                    "bfe.s32 %r114, %r111, %r114, %r112;\n\t"
                    "bfe.s32 %r115, %r111, %r115, %r112;\n\t"
                    "bfe.s32 %r116, %r111, %r116, %r112;\n\t"
                    "bfe.s32 %r117, %r111, %r117, %r112;\n\t"
                    "bfe.s32 %r118, %r111, %r118, %r112;\n\t"
                    "bfe.s32 %r119, %r111, %r119, %r112;\n\t"
                    "bfe.s32 %r120, %r111, %r120, %r112;\n\t"
                    "bfe.s32 %r121, %r111, %r121, %r112;\n\t"
                    "bfe.s32 %r122, %r111, %r122, %r112;\n\t"
                    "bfe.s32 %r123, %r111, %r123, %r112;\n\t"
                    "bfe.s32 %r124, %r111, %r124, %r112;\n\t"
                    "bfe.s32 %r125, %r111, %r125, %r112;\n\t"
                    "bfe.s32 %r126, %r111, %r126, %r112;\n\t"
                    "bfe.s32 %r127, %r111, %r127, %r112;\n\t"
                    "bfe.s32 %r128, %r111, %r128, %r112;\n\t"
                    "bfe.s32 %r113, %r111, %r113, %r112;\n\t"
                    "bfe.s32 %r114, %r111, %r114, %r112;\n\t"
                    "bfe.s32 %r115, %r111, %r115, %r112;\n\t"
                    "bfe.s32 %r116, %r111, %r116, %r112;\n\t"
                    "bfe.s32 %r117, %r111, %r117, %r112;\n\t"
                    "bfe.s32 %r118, %r111, %r118, %r112;\n\t"
                    "bfe.s32 %r119, %r111, %r119, %r112;\n\t"
                    "bfe.s32 %r120, %r111, %r120, %r112;\n\t"
                    "bfe.s32 %r121, %r111, %r121, %r112;\n\t"
                    "bfe.s32 %r122, %r111, %r122, %r112;\n\t"
                    "bfe.s32 %r123, %r111, %r123, %r112;\n\t"
                    "bfe.s32 %r124, %r111, %r124, %r112;\n\t"
                    "bfe.s32 %r125, %r111, %r125, %r112;\n\t"
                    "bfe.s32 %r126, %r111, %r126, %r112;\n\t"
                    "bfe.s32 %r127, %r111, %r127, %r112;\n\t"
                    "bfe.s32 %r128, %r111, %r128, %r112;\n\t"
                    "bfe.s32 %r113, %r111, %r113, %r112;\n\t"
                    "bfe.s32 %r114, %r111, %r114, %r112;\n\t"
                    "bfe.s32 %r115, %r111, %r115, %r112;\n\t"
                    "bfe.s32 %r116, %r111, %r116, %r112;\n\t"
                    "bfe.s32 %r117, %r111, %r117, %r112;\n\t"
                    "bfe.s32 %r118, %r111, %r118, %r112;\n\t"
                    "bfe.s32 %r119, %r111, %r119, %r112;\n\t"
                    "bfe.s32 %r120, %r111, %r120, %r112;\n\t"
                    "bfe.s32 %r121, %r111, %r121, %r112;\n\t"
                    "bfe.s32 %r122, %r111, %r122, %r112;\n\t"
                    "bfe.s32 %r123, %r111, %r123, %r112;\n\t"
                    "bfe.s32 %r124, %r111, %r124, %r112;\n\t"
                    "bfe.s32 %r125, %r111, %r125, %r112;\n\t"
                    "bfe.s32 %r126, %r111, %r126, %r112;\n\t"
                    "bfe.s32 %r127, %r111, %r127, %r112;\n\t"
                    "bfe.s32 %r128, %r111, %r128, %r112;\n\t"
                    "bfe.s32 %r113, %r111, %r113, %r112;\n\t"
                    "bfe.s32 %r114, %r111, %r114, %r112;\n\t"
                    "bfe.s32 %r115, %r111, %r115, %r112;\n\t"
                    "bfe.s32 %r116, %r111, %r116, %r112;\n\t"
                    "bfe.s32 %r117, %r111, %r117, %r112;\n\t"
                    "bfe.s32 %r118, %r111, %r118, %r112;\n\t"
                    "bfe.s32 %r119, %r111, %r119, %r112;\n\t"
                    "bfe.s32 %r120, %r111, %r120, %r112;\n\t"
                    "bfe.s32 %r121, %r111, %r121, %r112;\n\t"
                    "bfe.s32 %r122, %r111, %r122, %r112;\n\t"
                    "bfe.s32 %r123, %r111, %r123, %r112;\n\t"
                    "bfe.s32 %r124, %r111, %r124, %r112;\n\t"
                    "bfe.s32 %r125, %r111, %r125, %r112;\n\t"
                    "bfe.s32 %r126, %r111, %r126, %r112;\n\t"
                    "bfe.s32 %r127, %r111, %r127, %r112;\n\t"
                    "bfe.s32 %r128, %r111, %r128, %r112;\n\t"
                    "bfe.s32 %r113, %r111, %r113, %r112;\n\t"
                    "bfe.s32 %r114, %r111, %r114, %r112;\n\t"
                    "bfe.s32 %r115, %r111, %r115, %r112;\n\t"
                    "bfe.s32 %r116, %r111, %r116, %r112;\n\t"
                    "bfe.s32 %r117, %r111, %r117, %r112;\n\t"
                    "bfe.s32 %r118, %r111, %r118, %r112;\n\t"
                    "bfe.s32 %r119, %r111, %r119, %r112;\n\t"
                    "bfe.s32 %r120, %r111, %r120, %r112;\n\t"
                    "bfe.s32 %r121, %r111, %r121, %r112;\n\t"
                    "bfe.s32 %r122, %r111, %r122, %r112;\n\t"
                    "bfe.s32 %r123, %r111, %r123, %r112;\n\t"
                    "bfe.s32 %r124, %r111, %r124, %r112;\n\t"
                    "bfe.s32 %r125, %r111, %r125, %r112;\n\t"
                    "bfe.s32 %r126, %r111, %r126, %r112;\n\t"
                    "bfe.s32 %r127, %r111, %r127, %r112;\n\t"
                    "bfe.s32 %r128, %r111, %r128, %r112;\n\t"
                    "bfe.s32 %r113, %r111, %r113, %r112;\n\t"
                    "bfe.s32 %r114, %r111, %r114, %r112;\n\t"
                    "bfe.s32 %r115, %r111, %r115, %r112;\n\t"
                    "bfe.s32 %r116, %r111, %r116, %r112;\n\t"
                    "bfe.s32 %r117, %r111, %r117, %r112;\n\t"
                    "bfe.s32 %r118, %r111, %r118, %r112;\n\t"
                    "bfe.s32 %r119, %r111, %r119, %r112;\n\t"
                    "bfe.s32 %r120, %r111, %r120, %r112;\n\t"
                    "bfe.s32 %r121, %r111, %r121, %r112;\n\t"
                    "bfe.s32 %r122, %r111, %r122, %r112;\n\t"
                    "bfe.s32 %r123, %r111, %r123, %r112;\n\t"
                    "bfe.s32 %r124, %r111, %r124, %r112;\n\t"
                    "bfe.s32 %r125, %r111, %r125, %r112;\n\t"
                    "bfe.s32 %r126, %r111, %r126, %r112;\n\t"
                    "bfe.s32 %r127, %r111, %r127, %r112;\n\t"
                    "bfe.s32 %r128, %r111, %r128, %r112;\n\t"
                    "bfe.s32 %r113, %r111, %r113, %r112;\n\t"
                    "bfe.s32 %r114, %r111, %r114, %r112;\n\t"
                    "bfe.s32 %r115, %r111, %r115, %r112;\n\t"
                    "bfe.s32 %r116, %r111, %r116, %r112;\n\t"
                    "bfe.s32 %r117, %r111, %r117, %r112;\n\t"
                    "bfe.s32 %r118, %r111, %r118, %r112;\n\t"
                    "bfe.s32 %r119, %r111, %r119, %r112;\n\t"
                    "bfe.s32 %r120, %r111, %r120, %r112;\n\t"
                    "bfe.s32 %r121, %r111, %r121, %r112;\n\t"
                    "bfe.s32 %r122, %r111, %r122, %r112;\n\t"
                    "bfe.s32 %r123, %r111, %r123, %r112;\n\t"
                    "bfe.s32 %r124, %r111, %r124, %r112;\n\t"
                    "bfe.s32 %r125, %r111, %r125, %r112;\n\t"
                    "bfe.s32 %r126, %r111, %r126, %r112;\n\t"
                    "bfe.s32 %r127, %r111, %r127, %r112;\n\t"
                    "bfe.s32 %r128, %r111, %r128, %r112;\n\t"
                    "bfe.s32 %r113, %r111, %r113, %r112;\n\t"
                    "bfe.s32 %r114, %r111, %r114, %r112;\n\t"
                    "bfe.s32 %r115, %r111, %r115, %r112;\n\t"
                    "bfe.s32 %r116, %r111, %r116, %r112;\n\t"
                    "bfe.s32 %r117, %r111, %r117, %r112;\n\t"
                    "bfe.s32 %r118, %r111, %r118, %r112;\n\t"
                    "bfe.s32 %r119, %r111, %r119, %r112;\n\t"
                    "bfe.s32 %r120, %r111, %r120, %r112;\n\t"
                    "bfe.s32 %r121, %r111, %r121, %r112;\n\t"
                    "bfe.s32 %r122, %r111, %r122, %r112;\n\t"
                    "bfe.s32 %r123, %r111, %r123, %r112;\n\t"
                    "bfe.s32 %r124, %r111, %r124, %r112;\n\t"
                    "bfe.s32 %r125, %r111, %r125, %r112;\n\t"
                    "bfe.s32 %r126, %r111, %r126, %r112;\n\t"
                    "bfe.s32 %r127, %r111, %r127, %r112;\n\t"
                    "bfe.s32 %r128, %r111, %r128, %r112;\n\t"
                    "bfe.s32 %r113, %r111, %r113, %r112;\n\t"
                    "bfe.s32 %r114, %r111, %r114, %r112;\n\t"
                    "bfe.s32 %r115, %r111, %r115, %r112;\n\t"
                    "bfe.s32 %r116, %r111, %r116, %r112;\n\t"
                    "bfe.s32 %r117, %r111, %r117, %r112;\n\t"
                    "bfe.s32 %r118, %r111, %r118, %r112;\n\t"
                    "bfe.s32 %r119, %r111, %r119, %r112;\n\t"
                    "bfe.s32 %r120, %r111, %r120, %r112;\n\t"
                    "bfe.s32 %r121, %r111, %r121, %r112;\n\t"
                    "bfe.s32 %r122, %r111, %r122, %r112;\n\t"
                    "bfe.s32 %r123, %r111, %r123, %r112;\n\t"
                    "bfe.s32 %r124, %r111, %r124, %r112;\n\t"
                    "bfe.s32 %r125, %r111, %r125, %r112;\n\t"
                    "bfe.s32 %r126, %r111, %r126, %r112;\n\t"
                    "bfe.s32 %r127, %r111, %r127, %r112;\n\t"
                    "bfe.s32 %r128, %r111, %r128, %r112;\n\t"
                    "bfe.s32 %r113, %r111, %r113, %r112;\n\t"
                    "bfe.s32 %r114, %r111, %r114, %r112;\n\t"
                    "bfe.s32 %r115, %r111, %r115, %r112;\n\t"
                    "bfe.s32 %r116, %r111, %r116, %r112;\n\t"
                    "bfe.s32 %r117, %r111, %r117, %r112;\n\t"
                    "bfe.s32 %r118, %r111, %r118, %r112;\n\t"
                    "bfe.s32 %r119, %r111, %r119, %r112;\n\t"
                    "bfe.s32 %r120, %r111, %r120, %r112;\n\t"
                    "bfe.s32 %r121, %r111, %r121, %r112;\n\t"
                    "bfe.s32 %r122, %r111, %r122, %r112;\n\t"
                    "bfe.s32 %r123, %r111, %r123, %r112;\n\t"
                    "bfe.s32 %r124, %r111, %r124, %r112;\n\t"
                    "bfe.s32 %r125, %r111, %r125, %r112;\n\t"
                    "bfe.s32 %r126, %r111, %r126, %r112;\n\t"
                    "bfe.s32 %r127, %r111, %r127, %r112;\n\t"
                    "bfe.s32 %r128, %r111, %r128, %r112;\n\t"
                    "bfe.s32 %r113, %r111, %r113, %r112;\n\t"
                    "bfe.s32 %r114, %r111, %r114, %r112;\n\t"
                    "bfe.s32 %r115, %r111, %r115, %r112;\n\t"
                    "bfe.s32 %r116, %r111, %r116, %r112;\n\t"
                    "bfe.s32 %r117, %r111, %r117, %r112;\n\t"
                    "bfe.s32 %r118, %r111, %r118, %r112;\n\t"
                    "bfe.s32 %r119, %r111, %r119, %r112;\n\t"
                    "bfe.s32 %r120, %r111, %r120, %r112;\n\t"
                    "bfe.s32 %r121, %r111, %r121, %r112;\n\t"
                    "bfe.s32 %r122, %r111, %r122, %r112;\n\t"
                    "bfe.s32 %r123, %r111, %r123, %r112;\n\t"
                    "bfe.s32 %r124, %r111, %r124, %r112;\n\t"
                    "bfe.s32 %r125, %r111, %r125, %r112;\n\t"
                    "bfe.s32 %r126, %r111, %r126, %r112;\n\t"
                    "bfe.s32 %r127, %r111, %r127, %r112;\n\t"
                    "bfe.s32 %r128, %r111, %r128, %r112;\n\t"
                    "bfe.s32 %r113, %r111, %r113, %r112;\n\t"
                    "bfe.s32 %r114, %r111, %r114, %r112;\n\t"
                    "bfe.s32 %r115, %r111, %r115, %r112;\n\t"
                    "bfe.s32 %r116, %r111, %r116, %r112;\n\t"
                    "bfe.s32 %r117, %r111, %r117, %r112;\n\t"
                    "bfe.s32 %r118, %r111, %r118, %r112;\n\t"
                    "bfe.s32 %r119, %r111, %r119, %r112;\n\t"
                    "bfe.s32 %r120, %r111, %r120, %r112;\n\t"
                    "bfe.s32 %r121, %r111, %r121, %r112;\n\t"
                    "bfe.s32 %r122, %r111, %r122, %r112;\n\t"
                    "bfe.s32 %r123, %r111, %r123, %r112;\n\t"
                    "bfe.s32 %r124, %r111, %r124, %r112;\n\t"
                    "bfe.s32 %r125, %r111, %r125, %r112;\n\t"
                    "bfe.s32 %r126, %r111, %r126, %r112;\n\t"
                    "bfe.s32 %r127, %r111, %r127, %r112;\n\t"
                    "bfe.s32 %r128, %r111, %r128, %r112;\n\t"
                    "bfe.s32 %r113, %r111, %r113, %r112;\n\t"
                    "bfe.s32 %r114, %r111, %r114, %r112;\n\t"
                    "bfe.s32 %r115, %r111, %r115, %r112;\n\t"
                    "bfe.s32 %r116, %r111, %r116, %r112;\n\t"
                    "bfe.s32 %r117, %r111, %r117, %r112;\n\t"
                    "bfe.s32 %r118, %r111, %r118, %r112;\n\t"
                    "bfe.s32 %r119, %r111, %r119, %r112;\n\t"
                    "bfe.s32 %r120, %r111, %r120, %r112;\n\t"
                    "bfe.s32 %r121, %r111, %r121, %r112;\n\t"
                    "bfe.s32 %r122, %r111, %r122, %r112;\n\t"
                    "bfe.s32 %r123, %r111, %r123, %r112;\n\t"
                    "bfe.s32 %r124, %r111, %r124, %r112;\n\t"
                    "bfe.s32 %r125, %r111, %r125, %r112;\n\t"
                    "bfe.s32 %r126, %r111, %r126, %r112;\n\t"
                    "bfe.s32 %r127, %r111, %r127, %r112;\n\t"
                    "bfe.s32 %r128, %r111, %r128, %r112;\n\t"
                    "bfe.s32 %r113, %r111, %r113, %r112;\n\t"
                    "bfe.s32 %r114, %r111, %r114, %r112;\n\t"
                    "bfe.s32 %r115, %r111, %r115, %r112;\n\t"
                    "bfe.s32 %r116, %r111, %r116, %r112;\n\t"
                    "bfe.s32 %r117, %r111, %r117, %r112;\n\t"
                    "bfe.s32 %r118, %r111, %r118, %r112;\n\t"
                    "bfe.s32 %r119, %r111, %r119, %r112;\n\t"
                    "bfe.s32 %r120, %r111, %r120, %r112;\n\t"
                    "bfe.s32 %r121, %r111, %r121, %r112;\n\t"
                    "bfe.s32 %r122, %r111, %r122, %r112;\n\t"
                    "bfe.s32 %r123, %r111, %r123, %r112;\n\t"
                    "bfe.s32 %r124, %r111, %r124, %r112;\n\t"
                    "bfe.s32 %r125, %r111, %r125, %r112;\n\t"
                    "bfe.s32 %r126, %r111, %r126, %r112;\n\t"
                    "bfe.s32 %r127, %r111, %r127, %r112;\n\t"
                    "bfe.s32 %r128, %r111, %r128, %r112;\n\t"
                    "bfe.s32 %r113, %r111, %r113, %r112;\n\t"
                    "bfe.s32 %r114, %r111, %r114, %r112;\n\t"
                    "bfe.s32 %r115, %r111, %r115, %r112;\n\t"
                    "bfe.s32 %r116, %r111, %r116, %r112;\n\t"
                    "bfe.s32 %r117, %r111, %r117, %r112;\n\t"
                    "bfe.s32 %r118, %r111, %r118, %r112;\n\t"
                    "bfe.s32 %r119, %r111, %r119, %r112;\n\t"
                    "bfe.s32 %r120, %r111, %r120, %r112;\n\t"
                    "bfe.s32 %r121, %r111, %r121, %r112;\n\t"
                    "bfe.s32 %r122, %r111, %r122, %r112;\n\t"
                    "bfe.s32 %r123, %r111, %r123, %r112;\n\t"
                    "bfe.s32 %r124, %r111, %r124, %r112;\n\t"
                    "bfe.s32 %r125, %r111, %r125, %r112;\n\t"
                    "bfe.s32 %r126, %r111, %r126, %r112;\n\t"
                    "bfe.s32 %r127, %r111, %r127, %r112;\n\t"
                    "bfe.s32 %r128, %r111, %r128, %r112;\n\t"
                    "bfe.s32 %r113, %r111, %r113, %r112;\n\t"
                    "bfe.s32 %r114, %r111, %r114, %r112;\n\t"
                    "bfe.s32 %r115, %r111, %r115, %r112;\n\t"
                    "bfe.s32 %r116, %r111, %r116, %r112;\n\t"
                    "bfe.s32 %r117, %r111, %r117, %r112;\n\t"
                    "bfe.s32 %r118, %r111, %r118, %r112;\n\t"
                    "bfe.s32 %r119, %r111, %r119, %r112;\n\t"
                    "bfe.s32 %r120, %r111, %r120, %r112;\n\t"
                    "bfe.s32 %r121, %r111, %r121, %r112;\n\t"
                    "bfe.s32 %r122, %r111, %r122, %r112;\n\t"
                    "bfe.s32 %r123, %r111, %r123, %r112;\n\t"
                    "bfe.s32 %r124, %r111, %r124, %r112;\n\t"
                    "bfe.s32 %r125, %r111, %r125, %r112;\n\t"
                    "bfe.s32 %r126, %r111, %r126, %r112;\n\t"
                    "bfe.s32 %r127, %r111, %r127, %r112;\n\t"
                    "bfe.s32 %r128, %r111, %r128, %r112;\n\t"
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
