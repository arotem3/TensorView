#ifndef __TENSOR_VIEW_DYNAMIC_TENSOR_VIEW_HPP__
#define __TENSOR_VIEW_DYNAMIC_TENSOR_VIEW_HPP__

#include "tensorview_config.hpp"
#include "DynamicTensorShape.hpp"
#include "ViewContainer.hpp"
#include "BaseTensor.hpp"

namespace tensor
{
  /// @brief provides read/write access to an externally managed array with
  /// high dimensional indexing.
  /// @tparam scalar the type of array, e.g. double, int, etc.
  /// @tparam Rank the tensor dimension, e.g. 2 for a matrix
  template <typename scalar, size_t Rank>
  class TensorView : public details::BaseTensor<details::DynamicTensorShape<Rank>, details::ViewContainer<scalar>>
  {
  public:
    using base_tensor = details::BaseTensor<details::DynamicTensorShape<Rank>, details::ViewContainer<scalar>>;
    using shape_type = details::DynamicTensorShape<Rank>;
    using container_type = details::ViewContainer<scalar>;
    using pointer = typename base_tensor::pointer;
    using const_pointer = typename base_tensor::const_pointer;

    template <TENSOR_INT_LIKE... Sizes>
    TENSOR_FUNC explicit TensorView(scalar *data, Sizes... shape) : base_tensor(shape_type(shape...), container_type(data)) {}

    TENSOR_FUNC TensorView() : base_tensor(shape_type(), container_type(nullptr)) {}

    template <TENSOR_INT_LIKE... Sizes>
    TENSOR_FUNC TensorView &reshape(Sizes... new_shape)
    {
      this->_shape.reshape(std::forward<Sizes>(new_shape)...);
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
  };

} // namespace tensor

#endif
