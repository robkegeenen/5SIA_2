#include "eeg.h"

__device__ double atomicAdd(double* address, double val){
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do{
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
  }while(assumed != old);
  return __longlong_as_double(old);
}

__global__ void apen_correlation (int np, int32_t *x, unsigned int m, double r, unsigned int *result1, unsigned int *result2, unsigned int length){
  bool set1, set2;
  unsigned int globalId = (blockIdx.x * blockDim.x) + threadIdx.x;
  unsigned int i = globalId / length;
  unsigned int j = globalId % length;
  unsigned int k;
  if((i < length) && (j < length)){
    set1 = false;
    set2 = false;
    for(k = 0; k < m; k++){
      if(abs(x[i + k] - x[j + k]) > r){
        set1 = true;
        set2 = true;
        break;
      }
    }
    if(abs(x[i + k] - x[j + k]) > r){
      set2 = true;
    }
    if(!set1){
      atomicAdd(result1, 1);
    }
    if((i < length - 1) && (j < length - 1)){
      if(!set2){
        atomicAdd(result2, 1);
      }
    }
  }
}

void apen(int np, int32_t *x, float *a, unsigned int m, double r, int blocksize){
  unsigned int *dev_inter1, inter1, *dev_inter2, inter2;
  int32_t *dev_x;
  int length = np - m + 1;
  unsigned int threads = (length > blocksize) ? blocksize : length;
  unsigned int blocks = (length > blocksize) ? (((length * length) + blocksize - 1) / blocksize) : length;
  cudaCheckError(cudaMalloc(&dev_x, np*sizeof(int32_t)));
  cudaCheckError(cudaMalloc(&dev_inter1, sizeof(unsigned int)));
  cudaCheckError(cudaMalloc(&dev_inter2, sizeof(unsigned int)));
  cudaCheckError(cudaMemcpy(dev_x, x, np*sizeof(int32_t), cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemset(dev_inter1, 0x00, sizeof(unsigned int)));
  cudaCheckError(cudaMemset(dev_inter2, 0x00, sizeof(unsigned int)));
  apen_correlation<<<blocks, threads>>>(np, dev_x, m, r, dev_inter1, dev_inter2, length);
  cudaCheckError(cudaDeviceSynchronize());
  cudaCheckError(cudaMemcpy(&inter1, dev_inter1, sizeof(unsigned int), cudaMemcpyDeviceToHost));
  cudaCheckError(cudaMemcpy(&inter2, dev_inter2, sizeof(unsigned int), cudaMemcpyDeviceToHost));
  cudaCheckError(cudaFree(dev_x));
  cudaCheckError(cudaFree(dev_inter1));
  cudaCheckError(cudaFree(dev_inter2));
  *a = log(((double)inter1 / ((double)(length * length))) / ((double)inter2 / ((double)((length - 1) * (length - 1)))));
}
