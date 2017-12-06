#ifndef EEG_H
#define EEG_H

#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "bw0_int.h"
#include "stafeature.h"
#include "p2p.h"
#include "apen.h"
#include "hurst.h"
#include "fft.h"

#define CHANNELS 23
#define DATAPOINTS 256
#define FEATURE_LENGTH 14
#define FS 100

void read_data(int32_t x[CHANNELS][DATAPOINTS], int nc, int np);
void run_channel(int np, int32_t *x, float *features);

#endif
