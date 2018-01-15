#include "eeg.h"

__global__
void apen_correlation (int np, int32_t *x, unsigned int m, double r, double *result)
{
  bool set;
  unsigned int count;
  double sum = 0;

  for (unsigned int i = 0; i <= np - (m + 1) + 1; i++) {
    count = 0;
    for (unsigned int j = 0; j <= np - (m + 1) + 1; j++) {
      set = false;

      for (unsigned int k = 0; k < m; k++) {
        if (abs(x[i + k] - x[j + k]) > r) {
          set = true;
          break;
        }
      }
      if (set == false) count++;
    }
    sum += ((double) count) / ((double) np - m + 1);
  }

  *result = sum / ((double) np - m + 1);
}

void apen(int np, int32_t *x, float *a, unsigned int m, double r)
{
  // Based on: https://nl.mathworks.com/matlabcentral/fileexchange/26546-approximate-entropy
  float A;
  double *inter1, *inter2;

  inter1 = (double*)malloc(sizeof(double));
  inter2 = (double*)malloc(sizeof(double));

  *inter1 = 16;
  *inter2 = 2;
  
  cudaMallocManaged(&x, np*sizeof(int32_t));
  //cudaMallocManaged(&inter1, sizeof(double));
  //cudaMallocManaged(&inter2, sizeof(double));

  //MEMCPY
  
  
  //apen_correlation<<<1, 1>>>(np, x, m, r, inter1);
  //apen_correlation<<<1, 1>>>(np, x, m + 1, r, inter2);
  cudaDeviceSynchronize();

  cudaFree(x);

  A = log(*inter1 / *inter2);

  //Convert to fixed point
  *a = A;
}
