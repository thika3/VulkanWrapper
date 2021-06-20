#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include <algorithm>

#include "cuda_91.h"

__global__ void cuda91_kernel(uint8_t *data, int size, int channels) {
    const size_t stride = gridDim.x * blockDim.x;
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < size;
         tid += stride) {
        uint8_t *p = data + tid * channels;
        p[0] = 255 - p[0];
        p[1] = 255 - p[1];
        p[2] = 255 - p[2];
    }
}

void LaunchCuda91Kernel(uint8_t *data, int size, int channels, int device) {
    cudaDeviceProp prop = {};
    cudaSetDevice(device);
    cudaGetDeviceProperties(&prop, device);
    int threads = prop.warpSize;

    int blocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, cuda91_kernel,
                                                  prop.warpSize, 0);
    blocks *= prop.multiProcessorCount;
    blocks = std::min(blocks, int((size + threads - 1) / threads));

    cuda91_kernel<<<blocks, threads>>>(data, size, channels);
}