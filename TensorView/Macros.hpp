#pragma once

#include <array>
#include <complex>
#include <concepts>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#define TENSOR_VERSION_MAJOR 2
#define TENSOR_VERSION_MINOR 0

// #define TENSOR_DEBUG_PRINT_MALLOC

#ifdef TENSOR_USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#define TENSOR_HOST_DEVICE __host__ __device__
#else
#define TENSOR_HOST_DEVICE
#endif

#if defined(FFTW3_H) && !defined(TENSOR_USE_FFTW)
#define TENSOR_USE_FFTW
#endif

#ifdef TENSOR_USE_FFTW
#include <fftw3.h>
#endif

#ifndef NDEBUG
#define TENSOR_CONSTEXPR
#else
#define TENSOR_CONSTEXPR constexpr
#endif

#define TENSOR_FUNC TENSOR_HOST_DEVICE TENSOR_CONSTEXPR inline

namespace tensor
{
#ifdef TENSOR_USE_64BIT_INDEX
   using index_t = uint64_t;
#else
   using index_t = uint32_t;
#endif

   template <typename T>
   concept IndexLike = std::convertible_to<T, index_t>;
} // namespace tensor

#ifdef __CUDA_ARCH__
#define TENSOR_DEVICE_CODE
#endif

#ifdef TENSOR_DEVICE_CODE
// Device-only implementation
#define __TENSOR_ASSERT_IMPL(cond, msg)                                      \
   do                                                                        \
   {                                                                         \
      if (!(cond))                                                           \
      {                                                                      \
         printf("TensorView Device Errror at %s:%d:\n", __FILE__, __LINE__); \
         msg;                                                                \
         asm("trap;");                                                       \
      }                                                                      \
   } while (0)
#else
// Host-only implementation
#define __TENSOR_ASSERT_IMPL(cond, msg)                              \
   do                                                                \
   {                                                                 \
      if (!(cond))                                                   \
      {                                                              \
         printf("TensorView Error at %s:%d:\n", __FILE__, __LINE__); \
         msg;                                                        \
         std::abort();                                               \
      }                                                              \
   } while (0)
#endif

// Debug-only assertion (disabled with NDEBUG)
#ifndef NDEBUG
#define TENSOR_DEBUG_ASSERT(cond, msg) __TENSOR_ASSERT_IMPL(cond, msg)
#else
#define TENSOR_DEBUG_ASSERT(cond, msg) ((void)0)
#endif

// Always-on lightweight check
#define TENSOR_CHECK(cond, msg) __TENSOR_ASSERT_IMPL(cond, msg)
