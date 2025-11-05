#ifndef UTILS_H
#define UTILS_H

#include <cuda_runtime.h>

#define TIME_KERNEL(...)                                                       \
  do {                                                                         \
    cudaEvent_t start, stop;                                                   \
    cudaEventCreate(&start);                                                   \
    cudaEventCreate(&stop);                                                    \
    cudaEventRecord(start);                                                    \
    __VA_ARGS__;                                                               \
    cudaEventRecord(stop);                                                     \
    cudaEventSynchronize(stop);                                                \
    float ms = 0.0f;                                                           \
    cudaEventElapsedTime(&ms, start, stop);                                    \
    printf("Time taken by %s: %.3f ms\n", #__VA_ARGS__, ms);                   \
    cudaEventDestroy(start);                                                   \
    cudaEventDestroy(stop);                                                    \
  } while (0)

#endif
