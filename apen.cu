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

__global__ void apen_correlation (int np, int32_t *x, unsigned int m, double r, unsigned int *result, unsigned int length){
  bool set;
  unsigned int globalId = (blockIdx.x * blockDim.x) + threadIdx.x;
  unsigned int i = globalId / length;
  unsigned int j = globalId % length;
  if((i >= length) || (j >= length)){
    return;
  }
  set = false;
  for(unsigned int k = 0; k < m; k++){
    if(abs(x[i + k] - x[j + k]) > r){
      set = true;
      break;
    }
  }
  if(!set){
    atomicAdd(result, 1);
  }
}

void apen(int np, int32_t *x, float *a, unsigned int m, double r, int blocksize){
  unsigned int *dev_inter1, inter1, *dev_inter2, inter2;
  int32_t *dev_x;
  int length1 = np - (m + 0) + 1;
  int length2 = np - (m + 1) + 1;
  unsigned int threads1 = (length1 > blocksize) ? blocksize : length1;
  unsigned int blocks1 = (length1 > blocksize) ? (((length1 * length1) + blocksize - 1) / blocksize) : length1;
  unsigned int threads2 = (length2 > blocksize) ? blocksize : length2;
  unsigned int blocks2 = (length2 > blocksize) ? (((length2 * length2) + blocksize - 1) / blocksize) : length2;
  cudaStream_t stream1, stream2; //Only helps a little bit
  cudaCheckError(cudaStreamCreate(&stream1));
  cudaCheckError(cudaStreamCreate(&stream2));
  cudaCheckError(cudaMalloc(&dev_x, np*sizeof(int32_t)));
  cudaCheckError(cudaMalloc(&dev_inter1, sizeof(unsigned int)));
  cudaCheckError(cudaMalloc(&dev_inter2, sizeof(unsigned int)));
  cudaCheckError(cudaMemcpy(dev_x, x, np*sizeof(int32_t), cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemset(dev_inter1, 0x00, sizeof(unsigned int)));
  cudaCheckError(cudaMemset(dev_inter2, 0x00, sizeof(unsigned int)));
  apen_correlation<<<blocks1, threads1, 0, stream1>>>(np, dev_x, m + 0, r, dev_inter1, length1);
  apen_correlation<<<blocks2, threads2, 0, stream2>>>(np, dev_x, m + 1, r, dev_inter2, length2);
  cudaCheckError(cudaDeviceSynchronize());
  cudaCheckError(cudaMemcpy(&inter1, dev_inter1, sizeof(unsigned int), cudaMemcpyDeviceToHost));
  cudaCheckError(cudaMemcpy(&inter2, dev_inter2, sizeof(unsigned int), cudaMemcpyDeviceToHost));
  cudaCheckError(cudaFree(dev_x));
  cudaCheckError(cudaFree(dev_inter1));
  cudaCheckError(cudaFree(dev_inter2));
  *a = log(((double)inter1 / ((double)(length1 * length1))) / ((double)inter2 / ((double)(length2 * length2))));
}
