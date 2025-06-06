#ifndef __TENSOR_VIEW_FIXED_TENSOR_HPP__
#define __TENSOR_VIEW_FIXED_TENSOR_HPP__

#include "tensorview_config.hpp"
#include "FixedTensorShape.hpp"
#include "BaseTensor.hpp"
#include "TensorTraits.hpp"

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
    using pointer = typename base_tensor::value_type *;
    using const_pointer = TENSOR_CONST_QUAL(typename base_tensor::value_type) *;

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
      return this->_container.data();
    }

    /// @brief returns the externally managed array
    TENSOR_FUNC const_pointer data() const
    {
      return this->_container.data();
    }

  private:
    friend struct details::tensor_traits<FixedTensor<scalar, Shape...>>;
  };
} // namespace tensor

namespace tensor::details
{
  template <typename scalar, size_t... Shape>
  struct tensor_traits<FixedTensor<scalar, Shape...>>
  {
    using tensor_type = FixedTensor<scalar, Shape...>;
    using value_type = scalar;
    using shape_type = typename tensor_type::shape_type;
    using container_type = typename tensor_type::container_type;

    static constexpr bool is_contiguous = true;

    static shape_type shape(const tensor_type &tensor)
    {
      return tensor._shape;
    }

    static const container_type &container(const tensor_type &tensor)
    {
      return tensor._container;
    }

    static container_type &container(tensor_type &tensor)
    {
      return tensor._container;
    }

    static container_type container(tensor_type &&tensor)
    {
      return std::move(tensor._container);
    }
  };
} // namespace tensor::details

#endif
