#ifndef __TENSOR_VIEW_DYNAMIC_TENSOR_VIEW_HPP__
#define __TENSOR_VIEW_DYNAMIC_TENSOR_VIEW_HPP__

#include "tensorview_config.hpp"
#include "DynamicTensorShape.hpp"
#include "ViewContainer.hpp"
#include "BaseTensor.hpp"

namespace tensor
{
  template <typename scalar, size_t Rank, typename Allocator>
  class Tensor;

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

    /// @brief construct a TensorView from another TensorView
    template <typename T, size_t R>
      requires(R <= Rank)
    TENSOR_FUNC TensorView(const TensorView<T, R> &tensor)
        : base_tensor(shape_type(), container_type(tensor.data()))
    {
      if constexpr (Rank == R)
        this->_shape = tensor._shape;
      else
        match_shape(tensor, std::make_index_sequence<R>{});
    }

    template <typename T>
    TENSOR_FUNC TensorView(TensorView<T, Rank> &&tensor)
        : base_tensor(tensor._shape, container_type(tensor.data()))
    {
    }

    template <typename T, size_t R>
      requires(R <= Rank)
    TENSOR_FUNC TensorView(TensorView<T, R> &tensor)
        : base_tensor(shape_type(), container_type(tensor.data()))
    {
      if constexpr (Rank == R)
        this->_shape = tensor._shape;
      else
        match_shape(tensor, std::make_index_sequence<R>{});
    }

    /// @brief assign a TensorView to another TensorView
    template <typename T, size_t R>
      requires(R <= Rank)
    TENSOR_FUNC TensorView &operator=(const TensorView<T, R> &tensor)
    {
      this->container = tensor.data();
      if constexpr (Rank == R)
        this->_shape = tensor._shape;
      else
        match_shape(tensor, std::make_index_sequence<R>{});
      return *this;
    }

    template <typename T>
    TENSOR_FUNC TensorView &operator=(TensorView<T, Rank> &&tensor)
    {
      this->container = tensor.data();
      this->_shape = tensor._shape;
      return *this;
    }

    template <typename T, size_t R>
      requires(R <= Rank)
    TENSOR_FUNC TensorView &operator=(TensorView<T, R> &tensor)
    {
      this->container = tensor.data();
      if constexpr (Rank == R)
        this->_shape = tensor._shape;
      else
        match_shape(tensor, std::make_index_sequence<R>{});
      return *this;
    }

    /**
     * @brief construct a TensorView from any tensor-like object.
     *
     * @details Caveat: the order of the other tensor must be less than or equal to the order of the TensorView.
     */
    template <typename TensorType>
    TENSOR_FUNC TensorView(TensorType &tensor) : base_tensor(shape_type(), container_type(tensor.data()))
    {
      constexpr size_t other_rank = TensorType::order();
      match_shape(tensor, std::make_index_sequence<other_rank>{});
    }

    template <typename TensorType>
    TensorView(TensorType &&tensor) = delete; // avoids dangling references

    template <typename T, typename Allocator>
    TENSOR_FUNC TensorView(const std::vector<T, Allocator> &vec)
        : base_tensor(shape_type(vec.size()), container_type(vec.data()))
    {
    }

    template <typename T, typename Allocator>
    TENSOR_FUNC TensorView(std::vector<T, Allocator> &vec)
        : base_tensor(shape_type(vec.size()), container_type(vec.data()))
    {
    }

    template <typename T, typename Allocator>
    TENSOR_FUNC TensorView &operator=(const std::vector<T, Allocator> &vec)
    {
      this->container = vec.data();
      this->_shape.reshape(vec.size());
      return *this;
    }

    template <typename T, typename Allocator>
    TENSOR_FUNC TensorView &operator=(std::vector<T, Allocator> &vec)
    {
      this->container = vec.data();
      this->_shape.reshape(vec.size());
      return *this;
    }

    template <typename T, typename Allocator>
    TENSOR_FUNC TensorView(std::vector<T, Allocator> &&vec) = delete; // avoids dangling references

    template <typename T, typename Allocator>
    TENSOR_FUNC TensorView &operator=(std::vector<T, Allocator> &&vec) = delete; // avoids dangling references

    /**
     * @brief construct a TensorView from any tensor-like object.
     *
     * @details Caveat: the order of the other tensor must be less than or equal to the order of the TensorView.
     */
    template <typename TensorType>
    TENSOR_FUNC TensorView &operator=(TensorType &tensor)
    {
      constexpr size_t R = TensorType::order();
      match_shape(tensor, std::make_index_sequence<R>{});
      this->container = tensor.data();

      return *this;
    }

    template <typename TensorType>
    TensorView &operator=(TensorType &&tensor) = delete; // avoids dangling references

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

  private:
    template <typename T, size_t R, typename Allocator>
    friend class Tensor;

    template <typename T, size_t R>
    friend class TensorView;

    template <typename TensorType, size_t... I>
    TENSOR_FUNC void match_shape(TensorType &tensor, std::index_sequence<I...>)
    {
      this->_shape.reshape(tensor.shape(I)...);
    }
  };

} // namespace tensor

#endif
