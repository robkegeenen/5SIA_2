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
  bool set;
  extern __shared__ unsigned int count[];
  __shared__ double sum;
  sum = 0;
  unsigned int j = threadIdx.x;
  unsigned int i = blockIdx.x;
  //printf("BlockID: %d\n", i);
  //for(unsigned int i = 0; i <= np - m; i++){
  count[i] = 0;
  //for(unsigned int j = 0; j <= np - m; j++){
  __syncthreads();
  set = false;
  for(unsigned int k = 0; k < m; k++){
    if(abs(x[i + k] - x[j + k]) > r){
      set = true;
       if(blockIdx.x == 0){
        //printf("Test A\n");
      }
      break;
    }
  }
  if(blockIdx.x == 0){
  printf("Test C: %s\n", set ? "true" : "false");
}
  if(!set){
    //count[i]++;
    atomicAdd(&count[i], 1);
    if(blockIdx.x == 0){
      //printf("Test B\n");
      //printf("B: %d\n", count[i]);
    }
  }

  
  //}
  __syncthreads();
  if(blockIdx.x == 0 && threadIdx.x == 0){
    printf("Count: %d\n", count[i]);
  }
  if(j == 0){
    atomicAdd(&sum, ((double)count[i]) / ((double)np - m + 1));
  }
  //sum += ((double)count) / ((double)np - m + 1);
  
  //}
  __syncthreads();
  if(i == 0 && j == 0){
    *result = sum / ((double)np - m + 1);
  }
}

void apen(int np, int32_t *x, float *a, unsigned int m, double r)
{
  double *dev_inter1, inter1, *dev_inter2, inter2;
  int32_t *dev_x;
  cudaCheckError(cudaMalloc(&dev_x, np*sizeof(int32_t)));
  cudaCheckError(cudaMalloc(&dev_inter1, sizeof(double)));
  cudaCheckError(cudaMalloc(&dev_inter2, sizeof(double)));
  cudaCheckError(cudaMemcpy(dev_x, x, np*sizeof(int32_t), cudaMemcpyHostToDevice));
  apen_correlation<<<(np - m + 1), (np - m + 1), (np - m + 1)*sizeof(unsigned int)>>>(np, dev_x, m, r, dev_inter1);
  apen_correlation<<<(np - m + 1), (np - m + 1), (np - m + 1)*sizeof(unsigned int)>>>(np, dev_x, m + 1, r, dev_inter2);
  cudaCheckError(cudaDeviceSynchronize());
  cudaCheckError(cudaMemcpy(&inter1, dev_inter1, sizeof(double), cudaMemcpyDeviceToHost));
  cudaCheckError(cudaMemcpy(&inter2, dev_inter2, sizeof(double), cudaMemcpyDeviceToHost));
  cudaCheckError(cudaFree(dev_x));
  cudaCheckError(cudaFree(dev_inter1));
  cudaCheckError(cudaFree(dev_inter2));
  *a = log(inter1 / inter2);
}
