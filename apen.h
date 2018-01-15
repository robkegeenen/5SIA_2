#ifndef APEN_H
#define APEN_H

#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <time.h>

//__global__ void apen_correlation (int np, int32_t *x, unsigned int m, double r, double *result);
void apen(int np, int32_t *x, float *a, unsigned int m, double r);

#endif
