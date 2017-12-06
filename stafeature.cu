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
void gpu_average(int32_t *x, int32_t *y)
{

    // The id of this thread within our block
    unsigned int threadId = threadIdx.x;

    // The global id if this thread.
    // Since we launch np threads in total, each id maps to one unique element in x
    unsigned int globalId = blockIdx.x*blockDim.x + threadIdx.x;

    // NOTE: for debugging you can print directly from the GPU device
    // however, if you print a lot, or the GPU encounters a serious error, this might fail
    // also performance is horrible, so make sure to disable it for benchmarking
    //printf("Hello from global thread %d, with local id %d\n",globalId,threadId);

    // Lets first copy the data from global GPU memory to shared memory
    // Shared memory is only accesible within a threadblock, but it is much faster to access than global memory
    // Note that by having the keyword "extern" and empty brackets [], the size of the array will be determined at runtime.
    // You will however have to pass the size in bytes as a 3rd argument to the kernel call (see the "sharedMemBytes" variable)
    // If you statically know the size of the shared memory array, it is probably faster to use that.
    // Because of this I would recommend using defines to calculate all these kind of sizes, although here for readability this is used
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

        #ifdef DEBUG
        //example debugging, print the partial sum of each block with the block id
        printf("Sum of Block %d: %d\n",blockIdx.x, blockData[0]);
        #endif

        //each block has one summation
        y[blockIdx.x]=blockData[0];
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

    //Because in this setup the amount of required shared memory is only known at runtime,
    //the number required shared memory bytes need to be passed as a 3rd argument to the kernel call later on
    //also see the remarks in the gpu_average code
    int sharedMemBytes = threadsPerBlock*sizeof(int32_t);

    //variable for holding return values of cuda functions
    cudaError_t err;

    //start by allocating room for array "x" on the global memory of the GPU
    int32_t* device_x;
    err=cudaMalloc(&device_x, np*sizeof(int32_t));

    //Here we check for errors of this cuda call
    //See eeg.h for the implementation of this error check (it's not a cuda default function)
    cudaCheckError(err);

    //also allocate room for the answer
    int32_t* device_blocksums;
    //Note that room is allocated in global memory for the sum of *each* threadblock
    //The final result will however be stored in the first position of this array
    //also, we allocate as floats here, but use as int32_t internally at some point which is not very pretty, but should work ;)
    err=cudaMalloc(&device_blocksums, numBlocks*sizeof(float));
    cudaCheckError(err);

    //Now copy array "x" from the CPU to the GPU
    err=cudaMemcpy(device_x,x, np*sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaCheckError(err);

    //Compute the average on the GPU

    gpu_average<<<numBlocks,threadsPerBlock, sharedMemBytes>>>(device_x, device_blocksums);
    //We use "peekatlasterror" since a kernel launch does not return a cudaError_t
    cudaCheckError(cudaPeekAtLastError());

    //copy result from GPU global memory to CPU memory
    int32_t blocksums[numBlocks];
    //NOTE: we only copy back the first element of the y array, since this hold the final average
    err=cudaMemcpy(blocksums,device_blocksums, numBlocks*sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckError(err);

    //Now add all the block sums on the CPU
    //if you have many blocks, you might consider mapping this to a threadblock on the GPU of course
    int32_t sum;
    for(uint32_t blk=0;blk<numBlocks;blk++)
        sum+=blocksums[blk];
    float avg = (float)(sum)/(float)(np);

    //free the memory on the GPU
    //Hint: if you do not free the memory, the values will be preserved between multiple kernel calls!
    //For example, you could keep the calculated average in GPU memory, and use it in the calculation of stdev on the GPU
    err=cudaFree(device_x);
    cudaCheckError(err);
    err=cudaFree(device_blocksums);
    cudaCheckError(err);

    printf("Average: %f\n",avg);
    printf("CPU reference: %f\n",average(np, x));
    int tsum=0;
    for(int b=0;b<numBlocks;b++){
        int sum=0;
        for (int i=0;i<threadsPerBlock;i++)
            sum+=x[b*threadsPerBlock+i];
        printf("CPU Block %d sum: %d\n",b,sum);
        tsum+=sum;
    }
    printf("CPU total sum: %d\n",tsum);


    //calculate all other features on the CPU for now
    sta[0] = avg;
    sta[1] = stddev(np, x, avg);
    sta[2] = abssum(np, x);
    sta[3] = mean_crosstimes(np, x, avg);

    #endif
}
