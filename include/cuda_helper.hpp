// Copyright 2021 Pierre Talbot, Frédéric Pinel

#ifndef CUDA_HELPER_HPP
#define CUDA_HELPER_HPP

#include <cstdio>
#include <iostream>
#include <cassert>

#ifdef __NVCC__
  #define CUDA __device__ __host__
  #define CUDA_DEVICE __device__
  #define CUDA_VAR __device__ __managed__
  #define CUDA_GLOBAL __global__

  #define CUDIE(result) { \
    cudaError_t e = (result); \
    if (e != cudaSuccess) { \
      printf("%s:%d CUDA runtime error %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    }}

  #define CUDIE0() CUDIE(cudaGetLastError())
#else
  #define CUDA
  #define CUDA_DEVICE
  #define CUDA_VAR
  #define CUDA_GLOBAL
#endif


template<typename T> CUDA T min(T a, T b) { return a<=b ? a : b; }
template<typename T> CUDA T max(T a, T b) { return a>=b ? a : b; }

CUDA static constexpr int limit_min() noexcept { return -__INT_MAX__ - 1; }
CUDA static constexpr int limit_max() noexcept { return __INT_MAX__; }

#ifdef DEBUG
  #define TRACE
  #define LOG(X) X
#else
  #define LOG(X)
#endif

#ifdef TRACE
  #define INFO(X) X
#else
  #define INFO(X)
#endif

template<typename T> CUDA void swap(T* a, T* b) {
  T c = *a;
  *a = *b;
  *b = c;
}

#endif // CUDA_HELPER_HPP
