#ifndef __TENSOR_VIEW_TENSOR_HPP__
#define __TENSOR_VIEW_TENSOR_HPP__

#include "tensorview_config.hpp"
#include "DynamicTensorShape.hpp"
#include "TensorTraits.hpp"

namespace tensor
{
  template <typename scalar, size_t Rank>
  class TensorView;

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

    using pointer = typename base_tensor::value_type *;
    using const_pointer = TENSOR_CONST_QUAL(typename base_tensor::value_type) *;

    template <TENSOR_INT_LIKE... Sizes>
    inline explicit Tensor(Sizes... shape) : base_tensor(shape_type(shape...), container_type((1 * ... * shape))) {}

    inline Tensor() : base_tensor(shape_type(), container_type(0)) {}

    template <TENSOR_INT_LIKE... Sizes>
    TENSOR_FUNC Tensor &reshape(Sizes... new_shape)
    {
      this->_shape.reshape(std::forward<Sizes>(new_shape)...);
      this->_container.resize(this->_shape.size());

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
      return this->_container.data();
    }

    /// @brief returns the externally managed array
    TENSOR_FUNC const_pointer data() const
    {
      return this->_container.data();
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
    template <typename T, size_t R>
    friend class TensorView;

    friend struct details::tensor_traits<Tensor<scalar, Rank, Allocator>>;
  };

  /// @brief returns a Tensor of the specified shape.
  template <typename scalar, typename Allocator = std::allocator<scalar>, TENSOR_INT_LIKE... Sizes>
  TENSOR_FUNC auto make_tensor(Sizes... shape)
  {
    constexpr size_t rank = sizeof...(Sizes);
    return Tensor<scalar, rank, Allocator>(shape...);
  }

  namespace details
  {
    template <typename value_type_out, typename TensorType, size_t... I>
    auto make_tensor_like_impl(const TensorType &tensor, std::index_sequence<I...>)
    {
      constexpr size_t order = TensorType::order();
      static_assert(order == sizeof...(I), "Invalid number of indices");
      return make_tensor<value_type_out>(tensor.shape(I)...);
    }
  } // namespace details

  /// @brief returns a Tensor with the same shape as the input tensor.
  template <typename value_type_out = void, typename TensorType>
  auto make_tensor_like(const TensorType &tensor)
  {
    constexpr size_t order = TensorType::order();
    using value_type = std::conditional_t<std::is_same_v<value_type_out, void>, std::decay_t<typename TensorType::value_type>, value_type_out>;
    return details::make_tensor_like_impl<value_type>(tensor, std::make_index_sequence<order>());
  }


  template <typename Tout = void, typename T, typename Allocator>
  auto make_tensor_like(const std::vector<T, Allocator> &vec)
  {
    using value_type = std::conditional_t<std::is_same_v<Tout, void>, std::decay_t<T>, Tout>;
    return make_tensor<value_type>(vec.size());
  }

  template <typename Tout = void, typename T, size_t N>
  auto make_tensor_like(const std::array<T, N> &)
  {
    using value_type = std::conditional_t<std::is_same_v<Tout, void>, std::decay_t<T>, Tout>;
    return make_tensor<value_type>(N);
  }
} // namespace tensor

namespace tensor::details
{
  template <typename scalar, size_t Rank>
  struct tensor_traits<tensor::Tensor<scalar, Rank>>
  {
    using tensor_type = tensor::Tensor<scalar, Rank>;
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
