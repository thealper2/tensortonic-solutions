#include <cuda_runtime.h>

__global__ void mean_variance_kernel(const float* input, float* sum_out, float* sum_sq_out, int N) {
    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (int i = blockIdx.x * blockSize + tid; i < N; i += blockSize * gridDim.x) {
        float val = input[i];
        sum += val;
        sum_sq += val * val;
    }
    shared[tid] = sum;
    shared[tid + blockSize] = sum_sq;
    __syncthreads();

    for (int stride = blockSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
            shared[tid + blockSize] += shared[tid + stride + blockSize];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(sum_out, shared[0]);
        atomicAdd(sum_sq_out, shared[blockSize]);
    }
}

extern "C" void solve(const float* input, float* mean_out, float* var_out, int N) {
    float* d_sum;
    float* d_sum_sq;
    cudaMalloc(&d_sum, sizeof(float));
    cudaMalloc(&d_sum_sq, sizeof(float));
    cudaMemset(d_sum, 0, sizeof(float));
    cudaMemset(d_sum_sq, 0, sizeof(float));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    size_t shared_mem = 2 * threads * sizeof(float);

    mean_variance_kernel<<<blocks, threads, shared_mem>>>(input, d_sum, d_sum_sq, N);
    cudaDeviceSynchronize();

    float sum, sum_sq;
    cudaMemcpy(&sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&sum_sq, d_sum_sq, sizeof(float), cudaMemcpyDeviceToHost);

    float mean = sum / N;
    float var = sum_sq / N - mean * mean;

    cudaMemcpy(mean_out, &mean, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(var_out, &var, sizeof(float), cudaMemcpyHostToDevice);

    cudaFree(d_sum);
    cudaFree(d_sum_sq);
}