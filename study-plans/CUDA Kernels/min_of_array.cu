#include <cuda_runtime.h>
#include <float.h>

__global__ void init_result(float* result) {
    result[0] = FLT_MAX;
}

__global__ void min_kernel(const float* input, float* result, int N) {
    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    float min_val = FLT_MAX;
    for (int i = blockIdx.x * blockSize + tid; i < N; i += blockSize * gridDim.x) {
        if (input[i] < min_val) {
            min_val = input[i];
        }
    }
    shared[tid] = min_val;
    __syncthreads();

    for (int stride = blockSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (shared[tid + stride] < shared[tid]) {
                shared[tid] = shared[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        float old_val, new_val;
        do {
            old_val = *result;
            new_val = (shared[0] < old_val) ? shared[0] : old_val;
        } while (atomicCAS((unsigned int*)result,
                          __float_as_uint(old_val),
                          __float_as_uint(new_val)) != __float_as_uint(old_val));
    }
}

extern "C" void solve(const float* input, float* result, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    init_result<<<1, 1>>>(result);
    cudaDeviceSynchronize();

    size_t shared_mem = threads * sizeof(float);
    min_kernel<<<blocks, threads, shared_mem>>>(input, result, N);
    cudaDeviceSynchronize();
}
