#ifndef HELPER_CUDA_VKW_H
#define HELPER_CUDA_VKW_H

#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __DRIVER_TYPES_H__
template <typename T>
void __checkCudaErrors(T result, const char *const func, const char *const file,
                       const int line) {
    if (result) {
        fprintf(stderr, "[CUDA Error] %s:L%d code=%d(%s) func=%s\n", file, line,
                static_cast<unsigned int>(result), cudaGetErrorName(result),
                func);
        exit(EXIT_FAILURE);
    }
}
#define checkCudaErrors(val) __checkCudaErrors((val), #val, __FILE__, __LINE__)

inline void __getLastCudaError(const char *error_msg, const char *file,
                               const int line) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA Last Error] %s:L%d code=%d(%s) msg=%s", file,
                line, static_cast<int>(err), cudaGetErrorString(err),
                error_msg);
        exit(EXIT_FAILURE);
    }
}
#define getLastCudaError(msg) __getLastCudaError(msg, __FILE__, __LINE__)

#endif

#endif