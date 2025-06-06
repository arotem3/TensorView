#ifndef __TENSOR_VIEW_FIXED_TENSOR_VIEW_HPP__
#define __TENSOR_VIEW_FIXED_TENSOR_VIEW_HPP__

#include "tensorview_config.hpp"
#include "FixedTensorShape.hpp"
#include "ViewContainer.hpp"
#include "BaseTensor.hpp"
#include "TensorTraits.hpp"

namespace tensor
{
  /// @brief high dimensional view for tensors with dimensions known at compile time.
  /// @tparam scalar type of tensor elements
  /// @tparam ...Shape shape of the tensor
  template <typename scalar, size_t... Shape>
  class FixedTensorView : public details::BaseTensor<details::FixedTensorShape<Shape...>, details::ViewContainer<scalar>>
  {
  public:
    using base_tensor = details::BaseTensor<details::FixedTensorShape<Shape...>, details::ViewContainer<scalar>>;
    using shape_type = details::FixedTensorShape<Shape...>;
    using container_type = details::ViewContainer<scalar>;
    using pointer = typename base_tensor::value_type *;
    using const_pointer = TENSOR_CONST_QUAL(typename base_tensor::value_type) *;

    static constexpr bool is_contiguous()
    {
      return true;
    }

    TENSOR_FUNC explicit FixedTensorView(scalar *data) : base_tensor(shape_type{}, container_type(data)) {}

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
    friend struct details::tensor_traits<FixedTensorView<scalar, Shape...>>;
  };
} // namespace tensor

namespace tensor::details
{
  template <typename scalar, size_t... Shape>
  struct tensor_traits<FixedTensorView<scalar, Shape...>>
  {
    using tensor_type = FixedTensorView<scalar, Shape...>;
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
