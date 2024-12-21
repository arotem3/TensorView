#ifndef __TENSOR_VIEW_NAMED_TENSOR_HPP__
#define __TENSOR_VIEW_NAMED_TENSOR_HPP__

#include "tensorview_config.hpp"
#include "DynamicTensorView.hpp"
#include "FixedTensorView.hpp"
#include "Tensor.hpp"
#include "FixedTensor.hpp"

namespace tensor
{
  /// @brief specialization of `TensorView` when `Dim == 1`
  template <typename scalar>
  using vector_view = TensorView<scalar, 1>;

  /// @brief specialization of `TensorView` when `Dim == 2`
  template <typename scalar>
  using matrix_view = TensorView<scalar, 2>;

  /// @brief specialization of `TensorView` when `Dim == 3`
  template <typename scalar>
  using cube_view = TensorView<scalar, 3>;

  template <typename scalar, size_t Shape0, size_t Shape1>
  using fixed_matrix_view = FixedTensorView<scalar, Shape0, Shape1>;

  template <typename scalar, size_t Shape0, size_t Shape1, size_t Shape2>
  using fixed_cube_view = FixedTensorView<scalar, Shape0, Shape1, Shape2>;

  template <typename scalar, size_t Shape0, size_t Shape1>
  using fixed_matrix = FixedTensor<scalar, Shape0, Shape1>;

  template <typename scalar, size_t Shape0, size_t Shape1, size_t Shape2>
  using fixed_cube = FixedTensor<scalar, Shape0, Shape1, Shape2>;

  template <typename scalar, typename Allocator = std::allocator<scalar>>
  using Vector = Tensor<scalar, 1, Allocator>;

  template <typename scalar, typename Allocator = std::allocator<scalar>>
  using Matrix = Tensor<scalar, 2, Allocator>;

  template <typename scalar, typename Allocator = std::allocator<scalar>>
  using Cube = Tensor<scalar, 3, Allocator>;
} // namespace tensor

#endif
