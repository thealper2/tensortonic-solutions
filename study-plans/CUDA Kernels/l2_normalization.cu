#include <cuda_runtime.h>
#include <math.h>

__global__ void reduce_sq_sum(const float* input, float* sumv, int N) {
    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    float sum = 0.0f;
    for (int i = blockIdx.x * blockSize + tid; i < N; i += blockSize * gridDim.x) {
        sum += input[i] * input[i];
    }
    shared[tid] = sum;
    __syncthreads();

    for (int stride = blockSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(sumv, shared[0]);
    }
}

__global__ void divide_by_sqrt(const float* input, float* output, const float* sumv, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float inv_norm = rsqrtf(*sumv);
        output[idx] = input[idx] * inv_norm;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    float* d_sum;
    cudaMalloc(&d_sum, sizeof(float));
    cudaMemset(d_sum, 0, sizeof(float));

    size_t shared_mem = threads * sizeof(float);
    reduce_sq_sum<<<blocks, threads, shared_mem>>>(input, d_sum, N);
    cudaDeviceSynchronize();

    divide_by_sqrt<<<blocks, threads>>>(input, output, d_sum, N);
    cudaDeviceSynchronize();

    cudaFree(d_sum);
}