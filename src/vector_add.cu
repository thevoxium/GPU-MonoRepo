
#include <stdio.h>
#include <math.h>
#include "../include/utils.h"

__global__ void vectorAdd(float* a, float* b, float* c, int N){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N){
    c[idx] = b[idx] + a[idx];
  }
}

int main(){
  int N = 1 << 20;
  int size = N * sizeof(float);

  float* a = (float*) malloc(size);
  float* b = (float*) malloc(size);
  float* c = (float*) malloc(size);
  float* ref = (float*) malloc(size);

  for(int i = 0; i < N; i++){
    a[i] = i;
    b[i] = 2 * i;
    ref[i] = a[i] + b[i];
  }

  float* da;
  float* db;
  float* dc;

  cudaMalloc((void**)&da, size);
  cudaMalloc((void**)&db, size);
  cudaMalloc((void**)&dc, size);

  cudaMemcpy(da, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(db, b, size, cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(256, 1, 1);
  dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x);

  int warmup = 10;
  int iters = 100;
  for(int i = 0; i < warmup; i++){
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(da, db, dc, N);
  }
  cudaDeviceSynchronize();

  float total_time = 0.0f;
  for(int i = 0; i < iters; i++){
    float ms = 0.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(da, db, dc, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    total_time += ms;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  cudaMemcpy(c, dc, size, cudaMemcpyDeviceToHost);

  double max_rel_err = 0.0;
  for(int i = 0; i < N; i++){
    double rel_err = fabs((c[i] - ref[i]) / ref[i]);
    if (rel_err > max_rel_err) max_rel_err = rel_err;
  }

  printf("Average time over %d iterations: %f ms\n", iters, total_time / iters);
  printf("Max relative error: %e\n", max_rel_err);

  for(int i = 0; i < 5; i++){
    printf("%f, ", c[i]);
  }
  printf("\n");

  cudaFree(da);
  cudaFree(db);
  cudaFree(dc);
  free(a);
  free(b);
  free(c);
  free(ref);

  return 0;
}
