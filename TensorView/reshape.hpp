#ifndef __TENSOR_VIEW_RESHAPE_HPP__
#define __TENSOR_VIEW_RESHAPE_HPP__

#include "tensorview_config.hpp"
#include "DynamicTensorView.hpp"
#include "FixedTensorView.hpp"
#include "Tensor.hpp"
#include "FixedTensor.hpp"

namespace tensor
{
  /**
   * @brief wraps an array in a `TensorView`. Same as declaring a new
   * `TensorView< sizeof...(Sizes), scalar >( data, shape... ).`
   * @tparam scalar type of array
   * @tparam ...Sizes sequence of `index_t`
   * @param[in] data the array
   * @param[in] ...shape the shape of the tensor
   * @return a `TensorView` of the array
   */
  template <typename scalar, TENSOR_INT_LIKE... Sizes>
  TENSOR_FUNC auto reshape(scalar *data, Sizes... shape)
  {
    return TensorView<scalar, sizeof...(Sizes)>(data, shape...);
  }

  /**
   * @brief creates a view of any tensor-like object with new shape.
   * @tparam TensorType type of the tensor
   * @tparam ...Sizes sequence of `index_t`
   * @param[in] tensor the tensor
   * @param[in] ...shape the shape of the tensor
   * @return a `TensorView` of the tensor
   */
  template <typename TensorType, TENSOR_INT_LIKE... Sizes>
  TENSOR_FUNC auto reshape(TensorType &tensor, Sizes... shape)
  {
    return reshape(tensor.data(), std::forward<Sizes>(shape)...);
  }

  /**
   * @brief creates a view of any tensor-like object preserving the shape.
   * @tparam TensorType type of the tensor
   * @param[in] tensor the tensor
   * @return a `TensorView` of the tensor
   */
  template <typename TensorType>
  inline auto make_view(TensorType &tensor)
  {
    constexpr size_t order = TensorType::order();
    using value_type = typename TensorType::value_type;
    using view_value_type = std::conditional_t<std::is_const_v<TensorType>, const value_type, value_type>;
    return TensorView<view_value_type, order>(tensor);
  }
} // namespace tensor

#endif
