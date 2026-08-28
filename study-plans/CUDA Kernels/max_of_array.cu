#include <cuda_runtime.h>
#include <float.h>

__global__ void max_kernel(const float* input, float* partial_max, int N) {
    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    float max_val = -FLT_MAX;
    for (int i = blockIdx.x * blockSize + tid; i < N; i += blockSize * gridDim.x) {
        if (input[i] > max_val) {
            max_val = input[i];
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

    if (tid == 0) {
        partial_max[blockIdx.x] = shared[0];
    }
}

__global__ void reduce_max_kernel(const float* partial_max, float* result, int num_blocks) {
    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    float max_val = -FLT_MAX;
    for (int i = blockIdx.x * blockSize + tid; i < num_blocks; i += blockSize * gridDim.x) {
        if (partial_max[i] > max_val) {
            max_val = partial_max[i];
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

    if (tid == 0) {
        float old_val, new_val;
        do {
            old_val = *result;
            new_val = (shared[0] > old_val) ? shared[0] : old_val;
        } while (atomicCAS((unsigned int*)result,
                           __float_as_uint(old_val),
                           __float_as_uint(new_val)) != __float_as_uint(old_val));
    }
}

extern "C" void solve(const float* input, float* result, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    float* d_partial;
    cudaMalloc(&d_partial, blocks * sizeof(float));

    float neg_inf = -FLT_MAX;
    cudaMemset(d_partial, 0xFF, blocks * sizeof(float));
    cudaMemcpy(d_partial, &neg_inf, sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(result, &neg_inf, sizeof(float), cudaMemcpyHostToDevice);

    size_t shared_mem = threads * sizeof(float);
    max_kernel<<<blocks, threads, shared_mem>>>(input, d_partial, N);
    cudaDeviceSynchronize();

    int reduce_blocks = (blocks + threads - 1) / threads;
    reduce_max_kernel<<<reduce_blocks, threads, shared_mem>>>(d_partial, result, blocks);
    cudaDeviceSynchronize();

    cudaFree(d_partial);
}