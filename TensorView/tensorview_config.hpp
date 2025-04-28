#ifndef __TENSOR_VIEW_CONFIG_HPP__
#define __TENSOR_VIEW_CONFIG_HPP__

#include <type_traits>
#include <cstdio>
#include <stdexcept>
#include <array>
#include <vector>
#include <concepts>

#ifdef TENSOR_USE_CUDA
#include <cuda.h>
#define TENSOR_HOST_DEVICE __host__ __device__
#else
#define TENSOR_HOST_DEVICE
#endif

#ifdef TENSOR_ALWAYS_MUTABLE
#define TENSOR_CONST_QUAL(type) type
#else
#define TENSOR_CONST_QUAL(type) const type
#endif

#ifdef TENSOR_DEBUG
#define TENSOR_CONSTEXPR
#else
#define TENSOR_CONSTEXPR constexpr
#endif

#define TENSOR_FUNC TENSOR_HOST_DEVICE TENSOR_CONSTEXPR inline

namespace tensor
{
#ifdef TENSOR_USE_CUDA
  /// @brief type for indexing into tensor views. In cuda, unsigned integers are generally preferred.
  using index_t = int;
#else
  /// @brief type for indexing into tensor views.
  using index_t = unsigned long;
#endif
}

#define TENSOR_INT_LIKE std::convertible_to<index_t> // concept for index-like types

#endif
