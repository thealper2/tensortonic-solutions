#include <cuda_runtime.h>

__global__ void dot_kernel(const float* A, const float* B, float* result, int N) {
    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    float sum = 0.0f;
    for (int i = blockIdx.x * blockSize + tid; i < N; i += blockSize * gridDim.x) {
        sum += A[i] * B[i];
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
        atomicAdd(result, shared[0]);
    }
}

extern "C" void solve(const float* A, const float* B, float* result, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    cudaMemset(result, 0, sizeof(float));
    dot_kernel<<<blocks, threads, threads * sizeof(float)>>>(A, B, result, N);
    cudaDeviceSynchronize();
}