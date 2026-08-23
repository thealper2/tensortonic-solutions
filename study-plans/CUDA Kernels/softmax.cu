#include <cuda_runtime.h>
#include <cfloat>
#include <math.h>

__global__ void softmax_kernel(const float* input, float* output, int N) {
    extern __shared__ float shared[];
    
    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    
    float max_val = -FLT_MAX;
    for (int i = tid; i < N; i += blockSize) {
        float val = input[i];
        if (val > max_val) {
            max_val = val;
        }
    }
    shared[tid] = max_val;
    __syncthreads();
    
    for (int stride = blockSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (shared[tid + stride] > shared[tid]) {
                shared[tid] = shared[tid + stride];
            }
        }
        __syncthreads();
    }
    float global_max = shared[0];
    __syncthreads();
    
    float sum_val = 0.0f;
    for (int i = tid; i < N; i += blockSize) {
        sum_val += expf(input[i] - global_max);
    }
    shared[tid] = sum_val;
    __syncthreads();
    
    for (int stride = blockSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    float global_sum = shared[0];
    __syncthreads();
    
    for (int i = tid; i < N; i += blockSize) {
        output[i] = expf(input[i] - global_max) / global_sum;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int numBlocks = 1;
    size_t sharedMemSize = threads * sizeof(float);
    softmax_kernel<<<numBlocks, threads, sharedMemSize>>>(input, output, N);
    cudaDeviceSynchronize();
}