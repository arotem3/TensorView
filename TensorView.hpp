#ifndef __TENSOR_VIEW_HPP__
#define __TENSOR_VIEW_HPP__

#include <type_traits>
#include <cstdio>
#include <stdexcept>
#include <array>
#include <vector>

#ifdef TENSOR_USE_CUDA
#include <cuda.h>
#define TENSOR_HOST_DEVICE __host__ __device__
#else
#define TENSOR_HOST_DEVICE
#endif

#ifdef TENSOR_ALWAYS_MUTABLE
#define TENSOR_CONST_QUAL(type) type
#else
#define TENSOR_CONST_QUAL(type) const type
#endif

#ifdef TENSOR_DEBUG
#define TENSOR_CONSTEXPR
#else
#define TENSOR_CONSTEXPR constexpr
#endif

#if __cplusplus >= 202002L
#include <concepts>
#define TENSOR_INT_LIKE std::convertible_to<int>
#else
#define TENSOR_INT_LIKE typename
#endif

#define TENSOR_FUNC TENSOR_HOST_DEVICE TENSOR_CONSTEXPR inline

namespace tensor
{

#ifdef TENSOR_USE_CUDA
  /// @brief type for indexing into tensor views. In cuda, unsigned integers are generally preferred.
  using index_t = int;
#else
  /// @brief type for indexing into tensor views.
  using index_t = unsigned long;
#endif

  /// @brief terminates program/throws an out_of_range error.
  inline void tensor_out_of_range()
  {
#ifdef TENSOR_USE_CUDA
    fprintf(stderr, "TensorView out of range error.\n");
    std::abort();
#else
    throw std::out_of_range("TensorView out of range error.\n");
#endif
  }

  /// @brief terminates program/throws exception with message indicating bad memory access.
  inline void tensor_bad_memory_access()
  {
#ifdef TENSOR_USE_CUDA
    fprintf(stderr, "TensorView attempting to dereference nullptr.");
    std::abort();
#else
    throw std::runtime_error("TensorView attempting to dereference nullptr.");
#endif
  }

  namespace details
  {
    template <size_t Rank>
    class DynamicTensorShape
    {
    public:
      template <TENSOR_INT_LIKE... Shape>
      TENSOR_FUNC DynamicTensorShape(Shape... shape_) : len((1 * ... * shape_)), _shape{(index_t)shape_...}
      {
        static_assert(Rank > 0, "DynamicTensorShape must have a non-zero rank");
        static_assert(sizeof...(shape_) == Rank, "wrong number of dimensions specified for DynanicTensorShape.");
#ifdef TENSOR_DEBUG
        if (((shape_ < 0) || ... || false))
        {
#ifdef TENSOR_USE_CUDA
          fprintf(stderr, "DynamicTensorShape: all dimensions must be strictly positive.");
          std::abort();
#else
          throw std::runtime_error("DynamicTensorShape: all dimensions must be strictly positive.");
#endif
        }
#endif
      }

      template <TENSOR_INT_LIKE... Indices>
      TENSOR_FUNC index_t operator()(Indices... indices) const
      {
        static_assert(sizeof...(Indices) == Rank, "wrong number of indices.");
        return linear_index(std::forward<Indices>(indices)...);
      }

      TENSOR_FUNC index_t size() const
      {
        return len;
      }

      TENSOR_FUNC index_t shape(index_t d) const
      {
        return _shape[d];
      }

    private:
      index_t len;
      std::array<index_t, Rank> _shape;

      template <index_t Dim = 0, TENSOR_INT_LIKE... Indices>
      TENSOR_FUNC index_t linear_index(index_t index, Indices... indices) const
      {
#ifdef TENSOR_DEBUG
        if (index < 0 || index >= _shape[Dim])
          tensor_out_of_range();
#endif
        if constexpr (Dim + 1 < Rank)
          return index + _shape[Dim] * linear_index<Dim + 1>(std::forward<Indices>(indices)...);
        else
          return index;
      }
    };

    template <size_t... Shape>
    class FixedTensorShape
    {
    public:
      FixedTensorShape() = default;

      template <TENSOR_INT_LIKE... Indices>
      TENSOR_FUNC index_t operator()(Indices... indices) const
      {
        static_assert(sizeof...(Indices) == rank, "wrong number of indices.");
        return linear_index<Shape...>(std::forward<Indices>(indices)...);
      }

      static TENSOR_FUNC index_t size()
      {
        return len;
      }

      static TENSOR_FUNC index_t shape(index_t d)
      {
#ifdef TENSOR_DEBUG
        if (d < 0 || d >= rank)
          tensor_out_of_range();
#endif
        constexpr index_t _shape[] = {Shape...};
        return _shape[d];
      }

    private:
      static constexpr index_t len = (1 * ... * Shape);
      static constexpr index_t rank = sizeof...(Shape);

      template <index_t N, index_t... Ns, TENSOR_INT_LIKE... Indices>
      static TENSOR_FUNC index_t linear_index(index_t index, Indices... indices)
      {
#ifdef TENSOR_DEBUG
        if (index < 0 || index >= N)
          tensor_out_of_range();
#endif
        if constexpr (sizeof...(Indices) == 0)
          return index;
        else
          return index + N * linear_index<Ns...>(std::forward<Indices>(indices)...);
      }
    };

    template <typename T>
    class ViewContainer
    {
    public:
      using value_type = T;
      using reference = T &;
      using const_reference = TENSOR_CONST_QUAL(T) &;
      using pointer = T *;
      using const_pointer = TENSOR_CONST_QUAL(T) *;

      TENSOR_FUNC ViewContainer(pointer data = nullptr) : ptr{data} {}

      TENSOR_FUNC const_pointer data() const
      {
        return ptr;
      }

      TENSOR_FUNC pointer data()
      {
        return ptr;
      }

      TENSOR_FUNC reference operator[](index_t index)
      {
#ifdef TENSOR_DEBUG
        if (ptr == nullptr)
          tensor_bad_memory_access();
#endif
        return ptr[index];
      }

      TENSOR_FUNC const_pointer operator[](index_t index) const
      {
#ifdef TENSOR_DEBUG
        if (ptr == nullptr)
          tensor_bad_memory_access();
#endif
        return ptr[index];
      }

    private:
      pointer ptr;
    };

    template <typename Shape, typename Container>
    class __TensorType
    {
    public:
      using value_type = typename Container::value_type;
      using reference = typename Container::reference;
      using const_reference = typename Container::reference;
      using pointer = typename Container::pointer;
      using const_pointer = typename Container::const_pointer;

      __TensorType() = default;
      ~__TensorType() = default;

      template <typename _Container, TENSOR_INT_LIKE... Sizes>
      TENSOR_FUNC __TensorType(_Container data, Sizes... sizes) : _shape(sizes...), container(data) {}

      /// @brief shallow copy.
      __TensorType(const __TensorType &) = default;
      __TensorType &operator=(const __TensorType &) = default;

      /// @brief move
      __TensorType(__TensorType &&) = default;
      __TensorType &operator=(__TensorType &&) = default;

      /// @brief high dimensional read/write access.
      /// @tparam ...Indices convertible to `index_t`
      /// @param ...ids indices
      /// @return reference to data at the index.
      template <TENSOR_INT_LIKE... Indices>
      TENSOR_FUNC reference at(Indices... indices)
      {
        return container[_shape(std::forward<Indices>(indices)...)];
      }

      /// @brief high dimensional read/write access.
      /// @tparam ...Indices convertible to `index_t`
      /// @param ...ids indices
      /// @return reference to data at the index.
      template <TENSOR_INT_LIKE... Indices>
      TENSOR_FUNC const_reference at(Indices... indices) const
      {
        return container[_shape(std::forward<Indices>(indices)...)];
      }

      /// @brief high dimensional read/write access.
      /// @tparam ...Indices convertible to `index_t`
      /// @param ...ids indices
      /// @return reference to data at the index.
      template <TENSOR_INT_LIKE... Indices>
      TENSOR_FUNC reference operator()(Indices... indices)
      {
        return container[_shape(std::forward<Indices>(indices)...)];
      }

      /// @brief high dimensional read/write access.
      /// @tparam ...Indices convertible to `index_t`
      /// @param ...ids indices
      /// @return reference to data at the index.
      template <TENSOR_INT_LIKE... Indices>
      TENSOR_FUNC const_reference operator()(Indices... indices) const
      {
        return container[_shape(std::forward<Indices>(indices)...)];
      }

      /// @brief linear index access.
      TENSOR_FUNC reference operator[](index_t index)
      {
#ifdef TENSOR_DEBUG
        if (index < 0 || index >= _shape.size())
          tensor_out_of_range();
#endif
        return container[index];
      }

      /// @brief linear index access.
      TENSOR_FUNC const_reference operator[](index_t index) const
      {
#ifdef TENSOR_DEBUG
        if (index < 0 || index >= _shape.size())
          tensor_out_of_range();
#endif
        return container[index];
      }

      /// @brief implicit conversion to scalar*
      TENSOR_FUNC operator pointer()
      {
        return container.data();
      }

      /// @brief implicit conversion to scalar*
      TENSOR_FUNC operator const_pointer() const
      {
        return container.data();
      }

      /// @brief returns the externally managed array
      TENSOR_FUNC pointer data()
      {
        return container.data();
      }

      /// @brief returns the externally managed array
      TENSOR_FUNC const_pointer data() const
      {
        return container.data();
      }

      /// @brief returns pointer to start of tensor.
      TENSOR_FUNC pointer begin()
      {
        return container.data();
      }

      /// @brief returns pointer to start of tensor.
      TENSOR_FUNC const_pointer begin() const
      {
        return container.data();
      }

      /// @brief returns pointer to the element following the last element of tensor.
      TENSOR_FUNC pointer end()
      {
        return begin() + _shape.size();
      }

      /// @brief returns pointer to the element following the last element of tensor.
      TENSOR_FUNC const_pointer end() const
      {
        return container.data() + _shape.size();
      }

      /// @brief returns total number of elements in the tensor.
      TENSOR_FUNC index_t size() const
      {
        return _shape.size();
      }

      /// @brief returns the size of the tensor along dimension d.
      TENSOR_FUNC index_t shape(index_t d) const
      {
        return _shape.shape(d);
      }

    private:
      Shape _shape;
      Container container;
    };
  }; // namespace details

  /// @brief provides read/write access to an externally managed array with
  /// high dimensional indexing.
  /// @tparam scalar the type of array, e.g. double, int, etc.
  /// @tparam Rank the tensor dimension, e.g. 2 for a matrix
  template <typename scalar, size_t Rank>
  using TensorView = details::__TensorType<details::DynamicTensorShape<Rank>, details::ViewContainer<scalar>>;

  /// @brief high dimensional view for tensors with dimensions known at compile time.
  /// @tparam scalar type of tensor elements
  /// @tparam ...Shape shape of the tensor
  template <typename scalar, size_t... Shape>
  using FixedTensorView = details::__TensorType<details::FixedTensorShape<Shape...>, details::ViewContainer<scalar>>;

  /// @brief high dimensional tensor with dimensions known at compile time.
  ///
  /// @details Fixed tensors use stack arrays and can be constructed in cuda
  /// __device__ code and passed as arguments to __global__ kernel.
  ///
  /// @tparam scalar type of tensor elements
  /// @tparam ...Shape shape of the tensor
  template <typename scalar, size_t... Shape>
  using FixedTensor = details::__TensorType<details::FixedTensorShape<Shape...>, std::array<scalar, sizeof...(Shape)>>;

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
  using Tensor = details::__TensorType<details::DynamicTensorShape<Rank>, std::vector<scalar, Allocator>>;

  /// @brief returns a Tensor of the specified shape.
  template <typename scalar, typename Allocator = std::allocator<scalar>, TENSOR_INT_LIKE... Sizes>
  TENSOR_FUNC auto make_tensor(Sizes... shape)
  {
    constexpr size_t rank = sizeof...(Sizes);
    const index_t n = (1 * ... * shape);
    return Tensor<scalar, rank, Allocator>(n, shape...);
  }

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
  template <typename ShapeType, typename ContainerType, TENSOR_INT_LIKE... Sizes>
  TENSOR_FUNC auto reshape(const details::__TensorType<ShapeType, ContainerType> &tensor, Sizes... shape)
  {
    return reshape(tensor.data(), std::forward<Sizes>(shape)...);
  }

  /// @brief Returns new TensorView with new shape but points to same data.
  template <typename ShapeType, typename ContainerType, TENSOR_INT_LIKE... Sizes>
  TENSOR_FUNC auto reshape(details::__TensorType<ShapeType, ContainerType> &tensor, Sizes... shape)
  {
    return reshape(tensor.data(), std::forward<Sizes>(shape)...);
  }

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
  using FixedMatrixView = FixedTensorView<scalar, Shape0, Shape1>;

  template <typename scalar, size_t Shape0, size_t Shape1, size_t Shape2>
  using FixedCubeView = FixedTensorView<scalar, Shape0, Shape1, Shape2>;

  template <typename scalar, size_t Shape0, size_t Shape1>
  using FixedMatrix = FixedTensor<scalar, Shape0, Shape1>;

  template <typename scalar, size_t Shape0, size_t Shape1, size_t Shape2>
  using FixedCube = FixedTensor<scalar, Shape0, Shape1, Shape2>;

  template <typename scalar, typename Allocator = std::allocator<scalar>>
  using Vector = Tensor<scalar, 1, Allocator>;

  template <typename scalar, typename Allocator = std::allocator<scalar>>
  using Matrix = Tensor<scalar, 2, Allocator>;

  template <typename scalar, typename Allocator = std::allocator<scalar>>
  using Cube = Tensor<scalar, 3, Allocator>;
} // namespace tensor

#endif
