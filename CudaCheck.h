#pragma once

#include <cuda_runtime.h>
#include <cufft.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                                   \
    do {                                                                                   \
        cudaError_t _e = (call);                                                           \
        if (_e != cudaSuccess) {                                                           \
            std::fprintf(stderr, "[CUDA] error %d (%s) at %s:%d in %s\n",                  \
                         (int)_e, cudaGetErrorString(_e), __FILE__, __LINE__, #call);      \
            std::abort();                                                                  \
        }                                                                                  \
    } while (0)

#define CUFFT_CHECK(call)                                                                  \
    do {                                                                                   \
        cufftResult _e = (call);                                                           \
        if (_e != CUFFT_SUCCESS) {                                                         \
            std::fprintf(stderr, "[cuFFT] error %d at %s:%d in %s\n",                      \
                         (int)_e, __FILE__, __LINE__, #call);                              \
            std::abort();                                                                  \
        }                                                                                  \
    } while (0)

#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())
