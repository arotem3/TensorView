#ifndef __TENSOR_VIEW_ERRORS_HPP__
#define __TENSOR_VIEW_ERRORS_HPP__

#include "tensorview_config.hpp"

namespace tensor
{
  /// @brief terminates program/throws an out_of_range error.
  inline void tensor_out_of_range(const char *message)
  {
#ifdef TENSOR_USE_CUDA
    printf("TensorView: out of range error: %s\n", message);
    assert(false);
#else
    throw std::out_of_range(std::string("TensorView: out of range error: ") + message);
#endif
  }

  /// @brief terminates program/throws exception with message indicating bad memory access.
  inline void tensor_bad_memory_access()
  {
#ifdef TENSOR_USE_CUDA
    printf("TensorView: attempting to dereference nullptr.");
    assert(false);
#else
    throw std::runtime_error("TensorView: attempting to dereference nullptr.");
#endif
  }

  /// @brief terminates program/throws exception with message indicating that the tensor shape is illogical.
  inline void tensor_bad_shape()
  {
#ifdef TENSOR_USE_CUDA
    printf("TensorView: all dimensions must be strictly positive.");
    assert(false);
#else
    throw std::logic_error("TensorView: all dimensions must be strictly positive.");
#endif
  }
} // namespace tensor

#endif