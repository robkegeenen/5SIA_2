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

__global__
void apen_correlation (int np, int32_t *x, unsigned int m, double r, double *result)
{
  unsigned int i = blockIdx.x;
  bool set;
  unsigned int count;
  if(i == 0){
    *result = 0;
  }
  //for(unsigned int i = 0; i <= np - m; i++){
  count = 0;
  for(unsigned int j = 0; j <= np - m; j++){
    set = false;
    for(unsigned int k = 0; k < m; k++){
      if(abs(x[i + k] - x[j + k]) > r){
        set = true;
        break;
      }
    }
    if(!set){
      count++;
    }
  }
  //sum += ((double)count) / ((double)np - m + 1);
  atomicAdd(result, ((double)count) / ((double)np - m + 1));
  //}
}

void apen(int np, int32_t *x, float *a, unsigned int m, double r)
{
  double *dev_inter1, inter1, *dev_inter2, inter2;
  int32_t *dev_x;
  int length1 = np - (m + 0) + 1;
  int length2 = np - (m + 1) + 1;
  cudaCheckError(cudaMalloc(&dev_x, np*sizeof(int32_t)));
  cudaCheckError(cudaMalloc(&dev_inter1, sizeof(double)));
  cudaCheckError(cudaMalloc(&dev_inter2, sizeof(double)));
  cudaCheckError(cudaMemcpy(dev_x, x, np*sizeof(int32_t), cudaMemcpyHostToDevice));
  apen_correlation<<<length1, 1>>>(np, dev_x, m + 0, r, dev_inter1);
  apen_correlation<<<length2, 1>>>(np, dev_x, m + 1, r, dev_inter2);
  cudaCheckError(cudaDeviceSynchronize());
  cudaCheckError(cudaMemcpy(&inter1, dev_inter1, sizeof(double), cudaMemcpyDeviceToHost));
  cudaCheckError(cudaMemcpy(&inter2, dev_inter2, sizeof(double), cudaMemcpyDeviceToHost));
  cudaCheckError(cudaFree(dev_x));
  cudaCheckError(cudaFree(dev_inter1));
  cudaCheckError(cudaFree(dev_inter2));
  *a = log((inter1 / ((double)length1)) / (inter2 / ((double)length2)));
}
