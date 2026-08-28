#include <cuda_runtime.h>
#include <math.h>

__global__ void rms_norm_kernel(const float* input, const float* gamma, float* output, int M, int N, float eps) {
    extern __shared__ float shared[];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    float val = 0.0f;
    float val_sq = 0.0f;
    if (tid < N) {
        val = input[row * N + tid];
        val_sq = val * val;
    }
    shared[tid] = val_sq;
    __syncthreads();

    for (int stride = blockSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    float sum_sq = shared[0];
    __syncthreads();

    float mean_sq = sum_sq / N;
    float inv_rms = rsqrtf(mean_sq + eps);

    if (tid < N) {
        float val = input[row * N + tid];
        output[row * N + tid] = val * inv_rms * gamma[tid];
    }
}

extern "C" void solve(const float* input, const float* gamma, float* output, int M, int N, float eps) {
    int threads = 1;
    while (threads < N) {
        threads <<= 1;
    }
    if (threads > 1024) {
        threads = 1024;
    }
    dim3 blocks(M);
    size_t shared_mem = threads * sizeof(float);
    rms_norm_kernel<<<blocks, threads, shared_mem>>>(input, gamma, output, M, N, eps);
    cudaDeviceSynchronize();
}