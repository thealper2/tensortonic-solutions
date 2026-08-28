#include <cuda_runtime.h>
#include <math.h>

__global__ void sum_abs_kernel(const float* input, float* sum, int N) {
    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    float partial = 0.0f;
    for (int i = blockIdx.x * blockSize + tid; i < N; i += blockSize * gridDim.x) {
        partial += fabsf(input[i]);
    }
    shared[tid] = partial;
    __syncthreads();

    for (int stride = blockSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(sum, shared[0]);
    }
}

__global__ void divide_kernel(const float* input, float* output, float* sum, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = input[idx] / sum[0];
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    float* d_sum;
    cudaMalloc(&d_sum, sizeof(float));
    cudaMemset(d_sum, 0, sizeof(float));

    size_t shared_mem = threads * sizeof(float);
    sum_abs_kernel<<<blocks, threads, shared_mem>>>(input, d_sum, N);
    cudaDeviceSynchronize();

    divide_kernel<<<blocks, threads>>>(input, output, d_sum, N);
    cudaDeviceSynchronize();

    cudaFree(d_sum);
}
