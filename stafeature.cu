#include "eeg.h"

// Based on code by: Mohammad Tahghighi
int32_t abssum(int np, int32_t *x)
{
    int i;
    int32_t s = 0;

    for (i = 0; i < np; i++) {
        s += abs(x[i]);
    }

    return s;
}

float average(int np, int32_t *x)
{
    int i;
    int32_t s = 0;

    for (i = 0; i < np; i++) {
        s += x[i];
    }

    return ((float) s) / ((float) np);
}

__global__
void gpu_average(int32_t *x, float *y)
{

    // The id of this thread within our block
    unsigned int threadId = threadIdx.x;

    // The global id if this thread.
    // Since we launch np threads in total, each id maps to one unique element in x
    unsigned int globalId = blockIdx.x*blockDim.x + threadIdx.x;

    // Lets first copy the data from global GPU memory to shared memory
    // Shared memory is only accesible within a threadblock, but it is much faster to access than global memory
    // Note that by having the keyword "extern" and empty brackets [], the size of the array will be determined at runtime
    // Of you statically know the size of the shared memory array, it vould
    extern __shared__ int32_t blockData[];
    blockData[threadId]=x[globalId];

    // We insert a syncthreads here, to make sure every thread has copied valid data from global to local memory
    // Otherwise we potentially risk accessing uninitialized data in the shared memory
    __syncthreads();

    // The next step is summation of the elements in out blockData
    // The summation is done in a tree like fashion, as illustrated below
    // 0 1 2 3 4 5 6 7  (number of parallel summations)
    // |/  |/  |/  |/   (4)
    // 1   5   9   13
    // |__/    |__/     (2)
    // 6       22
    // |______/         (1)
    // 28
    for(unsigned int s=1;s<blockDim.x;s*=2){
        // Because the amount of work reduces, we use the threadId to determine which threads get to execute the summation
        if (threadId % (2*s) == 0 ){
            blockData[threadId] += blockData[threadId+s];
        }

        // For each layer of the tree, we have to make sure all threads finish their computations
        // otherwise we could read unsummed results
        __syncthreads();
    }

    //we let 1 selected thread per block write out our local sum to the global memory
    if(threadId==0){
        //each block has one summation
        y[blockIdx.x]=blockData[threadId];
    }
    __syncthreads();

    //Finally one thread (globally) does the final summations and average calculation
    //Note that we could have also chosen to do this on the CPU, if we copy back all the partial sums in y if there are only a couply of this that is probably faster
    //If there are many partial results, it might be smarter to also sum this in a parallel fashion using a single threadblock, instead of using only 1 thread as we are doing here.
    if(globalId==0){

        //"allocate" a local register to hold the sum
        int32_t sum=0;

        //loop over all blocks in the grid, and sum their results from global mem into local register "sum"
        for(unsigned int block=0;block<gridDim.x;block++)
            sum+=y[block];

        //divide by the total number of elements (equal the total number of threads)
        float avg=((float)sum)/((float)(gridDim.x * blockDim.x));

        //store the final average to global memory y[0]
        //This result will be fetched by the CPU later
        y[0]=avg;
    }

    //this will return the control to the CPU once all threads finish (reach this point)
    return;
}

float variance(int np, int32_t *x, float avg)
{
    int i;
    float s = 0;

    // Variance = Sum((x - avg)^2)
    for (i = 0; i < np; i++) {
        float tmp = x[i] - avg;
        s += (tmp * tmp);
    }

    return s / ((float) np);
}

float stddev(int np, int32_t *x, float avg)
{
    // Stddev = sqrt(variance)
    float var = variance(np, x, avg);
    return sqrt(var);
}

int mean_crosstimes(int np, int32_t *x, float avg)
{
    int i;
    bool negative = x[0] < avg;
    int count = 0;

    // Count number of zero crossings for (x - avg)
    for (i = 0; i < np; i++) {
        if (negative) {
            if (x[i] > avg) {
                negative = false;
                count++;
            }
        } else {
            if (x[i] < avg) {
                negative = true;
                count++;
            }
        }
    }

    return count;
}

void stafeature(int np, int32_t *x, float *sta)
{
    // Returns sta = [mean, std, abssum, mean_crosstimes)


    #ifdef CPU_ONLY
    //original CPU code
    float avg = average(np, x);
    sta[0] = avg;
    sta[1] = stddev(np, x, avg);
    sta[2] = abssum(np, x);
    sta[3] = mean_crosstimes(np, x, avg);
    #else

    //GPU code

    /*
     *  Our strategy to calculate the average in parallel is to split the input array into into a number of blocks (numBlocks).
     *  Each block contains then np/numBlocks elements
     *  These blocks will be mapped to the Streaming Multiprocessors of the GPU.
     *  For each block we calculate the sum(!) for that block
     *  Finally the sums of all the blocks are added by a single thread, and divided by the total number of elements to obtain the average
    */

    //NOTE: take care np is a multiple of numBlocks for this example
    int numBlocks=4;
    int threadsPerBlock=np/numBlocks; //i.e., this should have remainder==0

    //start by allocating room for array "x" on the global memory of the GPU
    int32_t* device_x;
    cudaMalloc(&device_x, np*sizeof(int32_t));

    //also allocate room for the answer
    float* device_y;
    //Note that room is allocated in global memory for the sum of *each* threadblock
    //The final result will however be stored in the first position of this array
    cudaMalloc(&device_y, numBlocks*sizeof(float));

    //Now copy array "x" from the CPU to the GPU
    cudaMemcpy(device_x,x, np*sizeof(int32_t), cudaMemcpyHostToDevice);

    //Compute the average on the GPU
    gpu_average<<<numBlocks,threadsPerBlock>>>(device_x, device_y);

    //copy result from GPU global memory to CPU memory
    float avg;
    //NOTE: we only copy back the first element of the y array, since this hold the final average
    cudaMemcpy(&avg,device_y, 1*sizeof(float), cudaMemcpyDeviceToHost);

    //free the memory on the GPU
    //Hint: if you do not free the memory, the values will be preserved between multiple kernel calls!
    //For example, you could keep the calculated average in GPU memory, and use it in the calculation of stdev on the GPU
    cudaFree(device_x);
    cudaFree(device_y);

    //calculate all other features on the CPU for now
    sta[0] = avg;
    sta[1] = stddev(np, x, avg);
    sta[2] = abssum(np, x);
    sta[3] = mean_crosstimes(np, x, avg);

    #endif
}
