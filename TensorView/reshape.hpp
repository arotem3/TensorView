#ifndef __TENSOR_VIEW_RESHAPE_HPP__
#define __TENSOR_VIEW_RESHAPE_HPP__

#include "tensorview_config.hpp"
#include "DynamicTensorView.hpp"
#include "FixedTensorView.hpp"
#include "Tensor.hpp"
#include "FixedTensor.hpp"

namespace tensor
{
  /// @brief wraps an array in a `TensorView`. Same as declaring a new
  /// `TensorView< sizeof...(Sizes), scalar >( data, shape... ).`
  /// @tparam scalar type of array
  /// @tparam ...Sizes sequence of `index_t`
  /// @param[in] data the array
  /// @param[in] ...shape the shape of the tensor
  template <typename scalar, TENSOR_INT_LIKE... Sizes>
  TENSOR_FUNC auto reshape(scalar *data, Sizes... shape)
  {
    return TensorView<scalar, sizeof...(Sizes)>(data, shape...);
  }

  /// @brief Returns new TensorView with new shape but points to same data.
  template <typename scalar, size_t Rank, TENSOR_INT_LIKE... Sizes>
  TENSOR_FUNC auto reshape(const TensorView<scalar, Rank> &tensor, Sizes... shape)
  {
    return reshape(tensor.data(), std::forward<Sizes>(shape)...);
  }

  /// @brief Returns new TensorView with new shape but points to same data.
  template <typename scalar, size_t Rank, TENSOR_INT_LIKE... Sizes>
  TENSOR_FUNC auto reshape(TensorView<scalar, Rank> &tensor, Sizes... shape)
  {
    return reshape(tensor.data(), std::forward<Sizes>(shape)...);
  }

  /// @brief Returns new TensorView with new shape but points to same data.
  template <typename scalar, size_t... Shape, TENSOR_INT_LIKE... Sizes>
  TENSOR_FUNC auto reshape(FixedTensorView<scalar, Shape...> &tensor, Sizes... shape)
  {
    return reshape(tensor.data(), std::forward<Sizes>(shape)...);
  }

  /// @brief Returns new TensorView with new shape but points to same data.
  template <typename scalar, size_t... Shape, TENSOR_INT_LIKE... Sizes>
  TENSOR_FUNC auto reshape(const FixedTensorView<scalar, Shape...> &tensor, Sizes... shape)
  {
    return reshape(tensor.data(), std::forward<Sizes>(shape)...);
  }

  /// @brief Returns new TensorView with new shape but points to same data.
  template <typename scalar, size_t... Shape, TENSOR_INT_LIKE... Sizes>
  TENSOR_FUNC auto reshape(FixedTensor<scalar, Shape...> &tensor, Sizes... shape)
  {
    return reshape(tensor.data(), std::forward<Sizes>(shape)...);
  }

  /// @brief Returns new TensorView with new shape but points to same data.
  template <typename scalar, size_t... Shape, TENSOR_INT_LIKE... Sizes>
  TENSOR_FUNC auto reshape(const FixedTensor<scalar, Shape...> &tensor, Sizes... shape)
  {
    return reshape(tensor.data(), std::forward<Sizes>(shape)...);
  }

  /// @brief Returns new TensorView with new shape but points to same data.
  template <typename scalar, size_t Rank, TENSOR_INT_LIKE... Sizes>
  TENSOR_FUNC auto reshape(const Tensor<scalar, Rank> &tensor, Sizes... shape)
  {
    return reshape(tensor.data(), std::forward<Sizes>(shape)...);
  }

  /// @brief Returns new TensorView with new shape but points to same data.
  template <typename scalar, size_t Rank, TENSOR_INT_LIKE... Sizes>
  TENSOR_FUNC auto reshape(Tensor<scalar, Rank> &tensor, Sizes... shape)
  {
    return reshape(tensor.data(), std::forward<Sizes>(shape)...);
  }
} // namespace tensor

#endif
