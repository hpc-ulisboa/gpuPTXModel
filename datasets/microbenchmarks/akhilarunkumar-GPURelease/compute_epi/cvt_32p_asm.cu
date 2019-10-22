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
                " .reg .s32 %r129;\n\t"
                " .reg .s32 %r30;\n\t"
                " .reg .s32 %r31;\n\t"
                " .reg .s32 %r32;\n\t"
                " .reg .s32 %r33;\n\t"
                " .reg .s32 %r34;\n\t"
                " .reg .s32 %r35;\n\t"
                " .reg .s32 %r36;\n\t"
                " .reg .s32 %r37;\n\t"
                " .reg .s32 %r38;\n\t"
                " .reg .s32 %r39;\n\t"
                " .reg .s32 %r40;\n\t"
                " .reg .s32 %r41;\n\t"
                " .reg .s32 %r42;\n\t"
                " .reg .s32 %r43;\n\t"
                " .reg .s32 %r44;\n\t"
                " .reg .s32 %r45;\n\t"
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
                "mov.f64 %r112, 4.4;\n\t"
                "mov.f64 %r113, %r112;\n\t"
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
                "mov.f64 %r128, 5.6;\n\t"

                );
        for (int k = 0; k < n; k++) {
            __asm volatile (
                    "cvt.rni.s32.f64 %r30, %r113;\n\t"
                    "cvt.rm.f64.s32 %r113, %r30;\n\t"
                    "cvt.rni.s32.f64 %r31, %r114;\n\t"
                    "cvt.rm.f64.s32 %r114, %r31;\n\t"
                    "cvt.rni.s32.f64 %r32, %r115;\n\t"
                    "cvt.rm.f64.s32 %r115, %r32;\n\t"
                    "cvt.rni.s32.f64 %r33, %r116;\n\t"
                    "cvt.rm.f64.s32 %r116, %r33;\n\t"
                    "cvt.rni.s32.f64 %r34, %r117;\n\t"
                    "cvt.rm.f64.s32 %r117, %r34;\n\t"
                    "cvt.rni.s32.f64 %r35, %r118;\n\t"
                    "cvt.rm.f64.s32 %r118, %r35;\n\t"
                    "cvt.rni.s32.f64 %r36, %r119;\n\t"
                    "cvt.rm.f64.s32 %r119, %r36;\n\t"
                    "cvt.rni.s32.f64 %r37, %r120;\n\t"
                    "cvt.rm.f64.s32 %r120, %r37;\n\t"
                    "cvt.rni.s32.f64 %r38, %r121;\n\t"
                    "cvt.rm.f64.s32 %r121, %r38;\n\t"
                    "cvt.rni.s32.f64 %r39, %r122;\n\t"
                    "cvt.rm.f64.s32 %r122, %r39;\n\t"
                    "cvt.rni.s32.f64 %r40, %r123;\n\t"
                    "cvt.rm.f64.s32 %r123, %r40;\n\t"
                    "cvt.rni.s32.f64 %r41, %r124;\n\t"
                    "cvt.rm.f64.s32 %r124, %r41;\n\t"
                    "cvt.rni.s32.f64 %r42, %r125;\n\t"
                    "cvt.rm.f64.s32 %r125, %r42;\n\t"
                    "cvt.rni.s32.f64 %r43, %r126;\n\t"
                    "cvt.rm.f64.s32 %r126, %r43;\n\t"
                    "cvt.rni.s32.f64 %r44, %r127;\n\t"
                    "cvt.rm.f64.s32 %r127, %r44;\n\t"
                    "cvt.rni.s32.f64 %r45, %r128;\n\t"
                    "cvt.rm.f64.s32 %r128, %r45;\n\t"
                    "cvt.rni.s32.f64 %r30, %r113;\n\t"
                    "cvt.rm.f64.s32 %r113, %r30;\n\t"
                    "cvt.rni.s32.f64 %r31, %r114;\n\t"
                    "cvt.rm.f64.s32 %r114, %r31;\n\t"
                    "cvt.rni.s32.f64 %r32, %r115;\n\t"
                    "cvt.rm.f64.s32 %r115, %r32;\n\t"
                    "cvt.rni.s32.f64 %r33, %r116;\n\t"
                    "cvt.rm.f64.s32 %r116, %r33;\n\t"
                    "cvt.rni.s32.f64 %r34, %r117;\n\t"
                    "cvt.rm.f64.s32 %r117, %r34;\n\t"
                    "cvt.rni.s32.f64 %r35, %r118;\n\t"
                    "cvt.rm.f64.s32 %r118, %r35;\n\t"
                    "cvt.rni.s32.f64 %r36, %r119;\n\t"
                    "cvt.rm.f64.s32 %r119, %r36;\n\t"
                    "cvt.rni.s32.f64 %r37, %r120;\n\t"
                    "cvt.rm.f64.s32 %r120, %r37;\n\t"
                    "cvt.rni.s32.f64 %r38, %r121;\n\t"
                    "cvt.rm.f64.s32 %r121, %r38;\n\t"
                    "cvt.rni.s32.f64 %r39, %r122;\n\t"
                    "cvt.rm.f64.s32 %r122, %r39;\n\t"
                    "cvt.rni.s32.f64 %r40, %r123;\n\t"
                    "cvt.rm.f64.s32 %r123, %r40;\n\t"
                    "cvt.rni.s32.f64 %r41, %r124;\n\t"
                    "cvt.rm.f64.s32 %r124, %r41;\n\t"
                    "cvt.rni.s32.f64 %r42, %r125;\n\t"
                    "cvt.rm.f64.s32 %r125, %r42;\n\t"
                    "cvt.rni.s32.f64 %r43, %r126;\n\t"
                    "cvt.rm.f64.s32 %r126, %r43;\n\t"
                    "cvt.rni.s32.f64 %r44, %r127;\n\t"
                    "cvt.rm.f64.s32 %r127, %r44;\n\t"
                    "cvt.rni.s32.f64 %r45, %r128;\n\t"
                    "cvt.rm.f64.s32 %r128, %r45;\n\t"
                    "cvt.rni.s32.f64 %r30, %r113;\n\t"
                    "cvt.rm.f64.s32 %r113, %r30;\n\t"
                    "cvt.rni.s32.f64 %r31, %r114;\n\t"
                    "cvt.rm.f64.s32 %r114, %r31;\n\t"
                    "cvt.rni.s32.f64 %r32, %r115;\n\t"
                    "cvt.rm.f64.s32 %r115, %r32;\n\t"
                    "cvt.rni.s32.f64 %r33, %r116;\n\t"
                    "cvt.rm.f64.s32 %r116, %r33;\n\t"
                    "cvt.rni.s32.f64 %r34, %r117;\n\t"
                    "cvt.rm.f64.s32 %r117, %r34;\n\t"
                    "cvt.rni.s32.f64 %r35, %r118;\n\t"
                    "cvt.rm.f64.s32 %r118, %r35;\n\t"
                    "cvt.rni.s32.f64 %r36, %r119;\n\t"
                    "cvt.rm.f64.s32 %r119, %r36;\n\t"
                    "cvt.rni.s32.f64 %r37, %r120;\n\t"
                    "cvt.rm.f64.s32 %r120, %r37;\n\t"
                    "cvt.rni.s32.f64 %r38, %r121;\n\t"
                    "cvt.rm.f64.s32 %r121, %r38;\n\t"
                    "cvt.rni.s32.f64 %r39, %r122;\n\t"
                    "cvt.rm.f64.s32 %r122, %r39;\n\t"
                    "cvt.rni.s32.f64 %r40, %r123;\n\t"
                    "cvt.rm.f64.s32 %r123, %r40;\n\t"
                    "cvt.rni.s32.f64 %r41, %r124;\n\t"
                    "cvt.rm.f64.s32 %r124, %r41;\n\t"
                    "cvt.rni.s32.f64 %r42, %r125;\n\t"
                    "cvt.rm.f64.s32 %r125, %r42;\n\t"
                    "cvt.rni.s32.f64 %r43, %r126;\n\t"
                    "cvt.rm.f64.s32 %r126, %r43;\n\t"
                    "cvt.rni.s32.f64 %r44, %r127;\n\t"
                    "cvt.rm.f64.s32 %r127, %r44;\n\t"
                    "cvt.rni.s32.f64 %r45, %r128;\n\t"
                    "cvt.rm.f64.s32 %r128, %r45;\n\t"
                    "cvt.rni.s32.f64 %r30, %r113;\n\t"
                    "cvt.rm.f64.s32 %r113, %r30;\n\t"
                    "cvt.rni.s32.f64 %r31, %r114;\n\t"
                    "cvt.rm.f64.s32 %r114, %r31;\n\t"
                    "cvt.rni.s32.f64 %r32, %r115;\n\t"
                    "cvt.rm.f64.s32 %r115, %r32;\n\t"
                    "cvt.rni.s32.f64 %r33, %r116;\n\t"
                    "cvt.rm.f64.s32 %r116, %r33;\n\t"
                    "cvt.rni.s32.f64 %r34, %r117;\n\t"
                    "cvt.rm.f64.s32 %r117, %r34;\n\t"
                    "cvt.rni.s32.f64 %r35, %r118;\n\t"
                    "cvt.rm.f64.s32 %r118, %r35;\n\t"
                    "cvt.rni.s32.f64 %r36, %r119;\n\t"
                    "cvt.rm.f64.s32 %r119, %r36;\n\t"
                    "cvt.rni.s32.f64 %r37, %r120;\n\t"
                    "cvt.rm.f64.s32 %r120, %r37;\n\t"
                    "cvt.rni.s32.f64 %r38, %r121;\n\t"
                    "cvt.rm.f64.s32 %r121, %r38;\n\t"
                    "cvt.rni.s32.f64 %r39, %r122;\n\t"
                    "cvt.rm.f64.s32 %r122, %r39;\n\t"
                    "cvt.rni.s32.f64 %r40, %r123;\n\t"
                    "cvt.rm.f64.s32 %r123, %r40;\n\t"
                    "cvt.rni.s32.f64 %r41, %r124;\n\t"
                    "cvt.rm.f64.s32 %r124, %r41;\n\t"
                    "cvt.rni.s32.f64 %r42, %r125;\n\t"
                    "cvt.rm.f64.s32 %r125, %r42;\n\t"
                    "cvt.rni.s32.f64 %r43, %r126;\n\t"
                    "cvt.rm.f64.s32 %r126, %r43;\n\t"
                    "cvt.rni.s32.f64 %r44, %r127;\n\t"
                    "cvt.rm.f64.s32 %r127, %r44;\n\t"
                    "cvt.rni.s32.f64 %r45, %r128;\n\t"
                    "cvt.rm.f64.s32 %r128, %r45;\n\t"
                    "cvt.rni.s32.f64 %r30, %r113;\n\t"
                    "cvt.rm.f64.s32 %r113, %r30;\n\t"
                    "cvt.rni.s32.f64 %r31, %r114;\n\t"
                    "cvt.rm.f64.s32 %r114, %r31;\n\t"
                    "cvt.rni.s32.f64 %r32, %r115;\n\t"
                    "cvt.rm.f64.s32 %r115, %r32;\n\t"
                    "cvt.rni.s32.f64 %r33, %r116;\n\t"
                    "cvt.rm.f64.s32 %r116, %r33;\n\t"
                    "cvt.rni.s32.f64 %r34, %r117;\n\t"
                    "cvt.rm.f64.s32 %r117, %r34;\n\t"
                    "cvt.rni.s32.f64 %r35, %r118;\n\t"
                    "cvt.rm.f64.s32 %r118, %r35;\n\t"
                    "cvt.rni.s32.f64 %r36, %r119;\n\t"
                    "cvt.rm.f64.s32 %r119, %r36;\n\t"
                    "cvt.rni.s32.f64 %r37, %r120;\n\t"
                    "cvt.rm.f64.s32 %r120, %r37;\n\t"
                    "cvt.rni.s32.f64 %r38, %r121;\n\t"
                    "cvt.rm.f64.s32 %r121, %r38;\n\t"
                    "cvt.rni.s32.f64 %r39, %r122;\n\t"
                    "cvt.rm.f64.s32 %r122, %r39;\n\t"
                    "cvt.rni.s32.f64 %r40, %r123;\n\t"
                    "cvt.rm.f64.s32 %r123, %r40;\n\t"
                    "cvt.rni.s32.f64 %r41, %r124;\n\t"
                    "cvt.rm.f64.s32 %r124, %r41;\n\t"
                    "cvt.rni.s32.f64 %r42, %r125;\n\t"
                    "cvt.rm.f64.s32 %r125, %r42;\n\t"
                    "cvt.rni.s32.f64 %r43, %r126;\n\t"
                    "cvt.rm.f64.s32 %r126, %r43;\n\t"
                    "cvt.rni.s32.f64 %r44, %r127;\n\t"
                    "cvt.rm.f64.s32 %r127, %r44;\n\t"
                    "cvt.rni.s32.f64 %r45, %r128;\n\t"
                    "cvt.rm.f64.s32 %r128, %r45;\n\t"
                    "cvt.rni.s32.f64 %r30, %r113;\n\t"
                    "cvt.rm.f64.s32 %r113, %r30;\n\t"
                    "cvt.rni.s32.f64 %r31, %r114;\n\t"
                    "cvt.rm.f64.s32 %r114, %r31;\n\t"
                    "cvt.rni.s32.f64 %r32, %r115;\n\t"
                    "cvt.rm.f64.s32 %r115, %r32;\n\t"
                    "cvt.rni.s32.f64 %r33, %r116;\n\t"
                    "cvt.rm.f64.s32 %r116, %r33;\n\t"
                    "cvt.rni.s32.f64 %r34, %r117;\n\t"
                    "cvt.rm.f64.s32 %r117, %r34;\n\t"
                    "cvt.rni.s32.f64 %r35, %r118;\n\t"
                    "cvt.rm.f64.s32 %r118, %r35;\n\t"
                    "cvt.rni.s32.f64 %r36, %r119;\n\t"
                    "cvt.rm.f64.s32 %r119, %r36;\n\t"
                    "cvt.rni.s32.f64 %r37, %r120;\n\t"
                    "cvt.rm.f64.s32 %r120, %r37;\n\t"
                    "cvt.rni.s32.f64 %r38, %r121;\n\t"
                    "cvt.rm.f64.s32 %r121, %r38;\n\t"
                    "cvt.rni.s32.f64 %r39, %r122;\n\t"
                    "cvt.rm.f64.s32 %r122, %r39;\n\t"
                    "cvt.rni.s32.f64 %r40, %r123;\n\t"
                    "cvt.rm.f64.s32 %r123, %r40;\n\t"
                    "cvt.rni.s32.f64 %r41, %r124;\n\t"
                    "cvt.rm.f64.s32 %r124, %r41;\n\t"
                    "cvt.rni.s32.f64 %r42, %r125;\n\t"
                    "cvt.rm.f64.s32 %r125, %r42;\n\t"
                    "cvt.rni.s32.f64 %r43, %r126;\n\t"
                    "cvt.rm.f64.s32 %r126, %r43;\n\t"
                    "cvt.rni.s32.f64 %r44, %r127;\n\t"
                    "cvt.rm.f64.s32 %r127, %r44;\n\t"
                    "cvt.rni.s32.f64 %r45, %r128;\n\t"
                    "cvt.rm.f64.s32 %r128, %r45;\n\t"
                    "cvt.rni.s32.f64 %r30, %r113;\n\t"
                    "cvt.rm.f64.s32 %r113, %r30;\n\t"
                    "cvt.rni.s32.f64 %r31, %r114;\n\t"
                    "cvt.rm.f64.s32 %r114, %r31;\n\t"
                    "cvt.rni.s32.f64 %r32, %r115;\n\t"
                    "cvt.rm.f64.s32 %r115, %r32;\n\t"
                    "cvt.rni.s32.f64 %r33, %r116;\n\t"
                    "cvt.rm.f64.s32 %r116, %r33;\n\t"
                    "cvt.rni.s32.f64 %r34, %r117;\n\t"
                    "cvt.rm.f64.s32 %r117, %r34;\n\t"
                    "cvt.rni.s32.f64 %r35, %r118;\n\t"
                    "cvt.rm.f64.s32 %r118, %r35;\n\t"
                    "cvt.rni.s32.f64 %r36, %r119;\n\t"
                    "cvt.rm.f64.s32 %r119, %r36;\n\t"
                    "cvt.rni.s32.f64 %r37, %r120;\n\t"
                    "cvt.rm.f64.s32 %r120, %r37;\n\t"
                    "cvt.rni.s32.f64 %r38, %r121;\n\t"
                    "cvt.rm.f64.s32 %r121, %r38;\n\t"
                    "cvt.rni.s32.f64 %r39, %r122;\n\t"
                    "cvt.rm.f64.s32 %r122, %r39;\n\t"
                    "cvt.rni.s32.f64 %r40, %r123;\n\t"
                    "cvt.rm.f64.s32 %r123, %r40;\n\t"
                    "cvt.rni.s32.f64 %r41, %r124;\n\t"
                    "cvt.rm.f64.s32 %r124, %r41;\n\t"
                    "cvt.rni.s32.f64 %r42, %r125;\n\t"
                    "cvt.rm.f64.s32 %r125, %r42;\n\t"
                    "cvt.rni.s32.f64 %r43, %r126;\n\t"
                    "cvt.rm.f64.s32 %r126, %r43;\n\t"
                    "cvt.rni.s32.f64 %r44, %r127;\n\t"
                    "cvt.rm.f64.s32 %r127, %r44;\n\t"
                    "cvt.rni.s32.f64 %r45, %r128;\n\t"
                    "cvt.rm.f64.s32 %r128, %r45;\n\t"
                    "cvt.rni.s32.f64 %r30, %r113;\n\t"
                    "cvt.rm.f64.s32 %r113, %r30;\n\t"
                    "cvt.rni.s32.f64 %r31, %r114;\n\t"
                    "cvt.rm.f64.s32 %r114, %r31;\n\t"
                    "cvt.rni.s32.f64 %r32, %r115;\n\t"
                    "cvt.rm.f64.s32 %r115, %r32;\n\t"
                    "cvt.rni.s32.f64 %r33, %r116;\n\t"
                    "cvt.rm.f64.s32 %r116, %r33;\n\t"
                    "cvt.rni.s32.f64 %r34, %r117;\n\t"
                    "cvt.rm.f64.s32 %r117, %r34;\n\t"
                    "cvt.rni.s32.f64 %r35, %r118;\n\t"
                    "cvt.rm.f64.s32 %r118, %r35;\n\t"
                    "cvt.rni.s32.f64 %r36, %r119;\n\t"
                    "cvt.rm.f64.s32 %r119, %r36;\n\t"
                    "cvt.rni.s32.f64 %r37, %r120;\n\t"
                    "cvt.rm.f64.s32 %r120, %r37;\n\t"
                    "cvt.rni.s32.f64 %r38, %r121;\n\t"
                    "cvt.rm.f64.s32 %r121, %r38;\n\t"
                    "cvt.rni.s32.f64 %r39, %r122;\n\t"
                    "cvt.rm.f64.s32 %r122, %r39;\n\t"
                    "cvt.rni.s32.f64 %r40, %r123;\n\t"
                    "cvt.rm.f64.s32 %r123, %r40;\n\t"
                    "cvt.rni.s32.f64 %r41, %r124;\n\t"
                    "cvt.rm.f64.s32 %r124, %r41;\n\t"
                    "cvt.rni.s32.f64 %r42, %r125;\n\t"
                    "cvt.rm.f64.s32 %r125, %r42;\n\t"
                    "cvt.rni.s32.f64 %r43, %r126;\n\t"
                    "cvt.rm.f64.s32 %r126, %r43;\n\t"
                    "cvt.rni.s32.f64 %r44, %r127;\n\t"
                    "cvt.rm.f64.s32 %r127, %r44;\n\t"
                    "cvt.rni.s32.f64 %r45, %r128;\n\t"
                    "cvt.rm.f64.s32 %r128, %r45;\n\t"
                    );
        }
    }
//    __syncthreads();

    //    if ((blockDim.x * blockIdx.x + threadIdx.x) == 0)
    *D = I1;

//    __syncthreads();
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

    cudaDeviceSynchronize();

    cudaProfilerStop();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);

    std::cout << "GPU Elapsed Time = " << time << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(h_res, d_res, sizeof(float), cudaMemcpyDeviceToHost);

    return 0;
}
