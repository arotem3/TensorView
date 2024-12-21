#ifndef __TENSOR_VIEW_FIXED_TENSOR_HPP__
#define __TENSOR_VIEW_FIXED_TENSOR_HPP__

#include "tensorview_config.hpp"
#include "FixedTensorShape.hpp"
#include "BaseTensor.hpp"

namespace tensor
{
  /// @brief high dimensional tensor with dimensions known at compile time.
  ///
  /// @details Fixed tensors use stack arrays and can be constructed in cuda
  /// __device__ code and passed as arguments to __global__ kernel.
  ///
  /// @tparam scalar type of tensor elements
  /// @tparam ...Shape shape of the tensor
  template <typename scalar, size_t... Shape>
  class FixedTensor : public details::BaseTensor<details::FixedTensorShape<Shape...>, std::array<scalar, (1 * ... * Shape)>>
  {
  public:
    using base_tensor = details::BaseTensor<details::FixedTensorShape<Shape...>, std::array<scalar, (1 * ... * Shape)>>;
    using pointer = typename base_tensor::pointer;
    using const_pointer = typename base_tensor::const_pointer;

    TENSOR_FUNC FixedTensor()
        : base_tensor(details::FixedTensorShape<Shape...>(), std::array<scalar, (1 * ... * Shape)>{scalar()}) {}

    /// @brief implicit conversion to scalar*
    TENSOR_FUNC operator pointer()
    {
      return this->container.data();
    }

    /// @brief implicit conversion to scalar*
    TENSOR_FUNC operator const_pointer() const
    {
      return this->container.data();
    }

    /// @brief returns the externally managed array
    TENSOR_FUNC pointer data()
    {
      return this->container.data();
    }

    /// @brief returns the externally managed array
    TENSOR_FUNC const_pointer data() const
    {
      return this->container.data();
    }
  };

} // namespace tensor

#endif
