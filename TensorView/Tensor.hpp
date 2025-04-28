#ifndef __TENSOR_VIEW_TENSOR_HPP__
#define __TENSOR_VIEW_TENSOR_HPP__

#include "tensorview_config.hpp"
#include "DynamicTensorShape.hpp"

namespace tensor
{
  template <typename scalar, size_t Rank> class TensorView;

  /// @brief tensor type which manages its own memory (dynamically)
  ///
  /// @details Unlike the view type, this type dynamically allocates memory. For
  /// cuda applications, this object may allocate device memory (via an
  /// appropriate allocator), but the tensor object exists exclusively on the
  /// host. However, a view of the tensor can be safely passed to the device.
  ///
  /// @tparam scalar the type of elements in the tensor e.g. float
  /// @tparam Allocator an allocator for managing memory
  /// @tparam Rank the order of the tensor, e.g. 2 for a matrix
  template <typename scalar, size_t Rank, typename Allocator = std::allocator<scalar>>
  class Tensor : public details::BaseTensor<details::DynamicTensorShape<Rank>, std::vector<scalar, Allocator>>
  {
  public:
    using base_tensor = details::BaseTensor<details::DynamicTensorShape<Rank>, std::vector<scalar, Allocator>>;
    using shape_type = details::DynamicTensorShape<Rank>;
    using container_type = std::vector<scalar, Allocator>;

    using pointer = typename base_tensor::pointer;
    using const_pointer = typename base_tensor::const_pointer;

    template <TENSOR_INT_LIKE... Sizes>
    inline explicit Tensor(Sizes... shape) : base_tensor(shape_type(shape...), container_type((1 * ... * shape))) {}

    inline Tensor() : base_tensor(shape_type(), container_type(0)) {}

    template <TENSOR_INT_LIKE... Sizes>
    TENSOR_FUNC Tensor &reshape(Sizes... new_shape)
    {
      this->_shape.reshape(std::forward<Sizes>(new_shape)...);
      this->container.resize(this->_shape.size());

      return *this;
    }

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
  
    /// @brief cast to TensorView
    TENSOR_FUNC operator TensorView<scalar, Rank>()
    {
      return TensorView<scalar, Rank>(*this);
    }

    TENSOR_FUNC operator TensorView<const scalar, Rank>() const
    {
      return TensorView<const scalar, Rank>(*this);
    }

  private:
    template <typename T, size_t R> friend class TensorView;
  };

  /// @brief returns a Tensor of the specified shape.
  template <typename scalar, typename Allocator = std::allocator<scalar>, TENSOR_INT_LIKE... Sizes>
  TENSOR_FUNC auto make_tensor(Sizes... shape)
  {
    constexpr size_t rank = sizeof...(Sizes);
    return Tensor<scalar, rank, Allocator>(shape...);
  }
} // namespace tensor

#endif
