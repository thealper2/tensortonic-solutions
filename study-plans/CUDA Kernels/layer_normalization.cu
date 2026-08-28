#include <cuda_runtime.h>
#include <math.h>

__global__ void layer_norm_kernel(const float* input, const float* gamma, const float* beta, float* output, int M, int N, float eps) {
    extern __shared__ float shared[];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    float val = 0.0f;
    if (tid < N) {
        val = input[row * N + tid];
    }
    shared[tid] = val;
    shared[tid + blockSize] = val * val;
    __syncthreads();

    for (int stride = blockSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    float sum = shared[0];

    __syncthreads();
    if (tid < N) {
        shared[tid] = shared[tid + blockSize];
    } else {
        shared[tid] = 0.0f;
    }
    __syncthreads();

    for (int stride = blockSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    float sum_sq = shared[0];
    __syncthreads();

    if (tid < N) {
        shared[tid] = input[row * N + tid];
    }
    __syncthreads();

    float mean = sum / N;
    float var = sum_sq / N - mean * mean;
    float inv_std = rsqrtf(var + eps);

    if (tid < N) {
        float normalized = (shared[tid] - mean) * inv_std;
        output[row * N + tid] = normalized * gamma[tid] + beta[tid];
    }
}

extern "C" void solve(const float* input, const float* gamma, const float* beta, float* output, int M, int N, float eps) {
    int threads = 1;
    while (threads < N) {
        threads <<= 1;
    }
    if (threads > 1024) {
        threads = 1024;
    }
    dim3 blocks(M);
    size_t shared_mem = 2 * threads * sizeof(float);
    layer_norm_kernel<<<blocks, threads, shared_mem>>>(input, gamma, beta, output, M, N, eps);
    cudaDeviceSynchronize();
}