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
    fprintf(stderr, "TensorView: out of range error.\n");
    std::abort();
#else
    throw std::out_of_range("TensorView: out of range error.\n");
#endif
  }

  /// @brief terminates program/throws exception with message indicating bad memory access.
  inline void tensor_bad_memory_access()
  {
#ifdef TENSOR_USE_CUDA
    fprintf(stderr, "TensorView: attempting to dereference nullptr.");
    std::abort();
#else
    throw std::runtime_error("TensorView: attempting to dereference nullptr.");
#endif
  }

  /// @brief terminates program/throws exception with message indicating that the tensor shape is illogical.
  inline void tensor_bad_shape()
  {
#ifdef TENSOR_USE_CUDA
    fprintf(stderr, "TensorView: all dimensions must be strictly positive.");
    std::abort();
#else
    throw std::logic_error("TensorView: all dimensions must be strictly positive.");
#endif
  }

  struct span
  {
  public:
    index_t begin;
    index_t end;
    index_t stride;

    constexpr explicit span(index_t Begin, index_t End, index_t inc = 1) : begin{Begin}, end{End}, stride{inc} {}

    TENSOR_FUNC index_t size() const
    {
      return (end - begin) / stride;
    }
  };

  struct all
  {
  };

  template <typename scalar, size_t Rank>
  class SubView;

  namespace details
  {
    constexpr span operator+(const span &x, index_t i)
    {
      return span(x.begin + i, x.end + i, x.stride);
    }

    constexpr span operator+(index_t i, const span &x)
    {
      return span(x.begin + i, x.end + i, x.stride);
    }

    constexpr span operator*(index_t s, const span &x)
    {
      return span(s * x.begin, s * x.end, s * x.stride);
    }

    constexpr std::array<span, 2> operator+(const span &x, const span &y)
    {
      return {x, y};
    }

    template <size_t N, size_t... I>
    constexpr std::array<span, N> add(index_t i, const std::array<span, N> &spans, std::index_sequence<I...>)
    {
      return {i + spans[0], spans[I + 1]...};
    }

    template <size_t N, size_t... I>
    constexpr std::array<span, N> scale(index_t s, const std::array<span, N> &spans, std::index_sequence<I...>)
    {
      return {(s * spans[I])...};
    }

    template <size_t N, size_t... I>
    constexpr std::array<span, N> operator+(index_t i, const std::array<span, N> &spans)
    {
      static_assert(N > 0);
      return add(i, spans, std::make_index_sequence<N - 1>{});
    }

    template <size_t N, size_t... I>
    constexpr std::array<span, N> operator+(const std::array<span, N> &spans, index_t i)
    {
      static_assert(N > 0);
      return add(i, spans, std::make_index_sequence<N - 1>{});
    }

    template <size_t N, size_t... I>
    constexpr std::array<span, N> operator*(index_t s, const std::array<span, N> &spans)
    {
      return scale(s, spans, std::make_index_sequence<N>{});
    }

    template <size_t N, size_t... I>
    constexpr std::array<span, N + 1> concat(const span &x, const std::array<span, N> &spans, std::index_sequence<I...>)
    {
      return {x, spans[I]...};
    }

    template <size_t N, size_t... I>
    constexpr std::array<span, N + 1> concat(const std::array<span, N> &spans, const span &x, std::index_sequence<I...>)
    {
      return {spans[I]..., x};
    }

    template <size_t N, size_t M, size_t... I, size_t... J>
    constexpr std::array<span, N + M> concat(const std::array<span, N> &A, const std::array<span, M> &B, std::index_sequence<I...>, std::index_sequence<J...>)
    {
      return {A[I]..., B[J]...};
    }

    template <size_t N>
    constexpr std::array<span, N + 1> operator+(const span &x, const std::array<span, N> &spans)
    {
      return concat(x, spans, std::make_index_sequence<N>{});
    }

    template <size_t N>
    constexpr std::array<span, N + 1> operator+(const std::array<span, N> &spans, const span &x)
    {
      return concat(spans, x, std::make_index_sequence<N>{});
    }

    template <size_t N, size_t M>
    constexpr auto operator+(const std::array<span, N> &a, const std::array<span, M> &b)
    {
      return concat(a, b, std::make_index_sequence<N>{}, std::make_index_sequence<M>{});
    }

    template <size_t N>
    constexpr index_t offset(const std::array<span, N> &spans)
    {
      index_t begin = 0;
      for (index_t d = 0; d < N; ++d)
      {
        begin += spans[d].begin;
      }
      return begin;
    }

    template <size_t Rank>
    class DynamicTensorShape
    {
    public:
      template <TENSOR_INT_LIKE... Shape>
      TENSOR_FUNC DynamicTensorShape(Shape... shape_) : len((1 * ... * shape_)), _shape{(index_t)shape_...}
      {
        static_assert(Rank > 0, "DynamicTensorShape must have a non-zero rank");
        static_assert(sizeof...(shape_) == Rank, "wrong number of dimensions specified for DynamicTensorShape.");
#ifdef TENSOR_DEBUG
        if (((shape_ <= 0) || ... || false))
          tensor_bad_shape();
#endif
      }

      TENSOR_FUNC DynamicTensorShape() : len{0}, _shape{} {}

      static constexpr index_t order()
      {
        return Rank;
      }

      template <typename... Indices>
      TENSOR_FUNC auto operator()(Indices... indices) const
      {
        static_assert(sizeof...(Indices) == Rank, "wrong number of indices.");
        return compute_index(std::forward<Indices>(indices)...);
      }

      TENSOR_FUNC index_t size() const
      {
        return len;
      }

      TENSOR_FUNC index_t shape(index_t d) const
      {
        return _shape[d];
      }

      template <TENSOR_INT_LIKE... Sizes>
      TENSOR_FUNC void reshape(Sizes... new_shape)
      {
        static_assert(sizeof...(Sizes) == Rank, "wrong number of dimensions.");
#ifdef TENSOR_DEBUG
        if (((new_shape <= 0) || ... || false))
          tensor_bad_shape();
#endif
        _shape = {(index_t)new_shape...};
        len = (1 * ... * new_shape);
      }

    private:
      index_t len;
      std::array<index_t, Rank> _shape;

      template <index_t Dim = 0, typename... Indices>
      TENSOR_FUNC auto compute_index(index_t index, Indices... indices) const
      {
#ifdef TENSOR_DEBUG
        if (index < 0 || index >= _shape[Dim])
          tensor_out_of_range();
#endif
        if constexpr (Dim + 1 < Rank)
          return index + _shape[Dim] * compute_index<Dim + 1>(std::forward<Indices>(indices)...);
        else
          return index;
      }

      template <index_t Dim = 0, typename... Indices>
      TENSOR_FUNC auto compute_index(span x, Indices... indices) const
      {
#ifdef TENSOR_DEBUG
        if (x.begin < 0 || x.end >= _shape[Dim])
          tensor_out_of_range();
#endif
        if constexpr (Dim + 1 < Rank)
          return x + _shape[Dim] * compute_index<Dim + 1>(std::forward<Indices>(indices)...);
        else
          return x;
      }

      template <index_t Dim = 0, typename... Indices>
      TENSOR_FUNC auto compute_index(all, Indices... indices) const
      {
        if constexpr (Dim + 1 < Rank)
          return span(0, _shape[Dim]) + _shape[Dim] * compute_index<Dim + 1>(std::forward<Indices>(indices)...);
        else
          return span(0, _shape[Dim]);
      }
    };

    template <size_t... Shape>
    class FixedTensorShape
    {
    public:
      FixedTensorShape() = default;

      template <typename... Indices>
      TENSOR_FUNC auto operator()(Indices... indices) const
      {
        static_assert(sizeof...(Indices) == rank, "wrong number of indices.");
        return compute_index<Shape...>(std::forward<Indices>(indices)...);
      }

      static constexpr index_t order()
      {
        return rank;
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

      template <index_t N, index_t... Ns, typename... Indices>
      static TENSOR_FUNC auto compute_index(index_t index, Indices... indices)
      {
#ifdef TENSOR_DEBUG
        if (index < 0 || index >= N)
          tensor_out_of_range();
#endif
        if constexpr (sizeof...(Indices) == 0)
          return index;
        else
          return index + N * compute_index<Ns...>(std::forward<Indices>(indices)...);
      }

      template <index_t N, index_t... Ns, typename... Indices>
      static TENSOR_FUNC auto compute_index(span x, Indices... indices)
      {
#ifdef TENSOR_DEBUG
        if (x.begin < 0 || x.end >= N)
          tensor_out_of_range();
#endif
        if constexpr (sizeof...(Indices) == 0)
          return x;
        else
          return x + N * compute_index<Ns...>(std::forward<Indices>(indices)...);
      }

      template <index_t N, index_t... Ns, typename... Indices>
      static TENSOR_FUNC auto compute_index(all, Indices... indices)
      {
        if constexpr (sizeof...(Indices) == 0)
          return span(0, N);
        else
          return span(0, N) + N * compute_index<Ns...>(std::forward<Indices>(indices)...);
      }
    };

    template <size_t Rank>
    struct StridedShape
    {
    public:
      TENSOR_FUNC StridedShape(const std::array<span, Rank> &spans)
      {
        len = 1;
        for (index_t d = 0; d < Rank; ++d)
        {
          _shape[d] = spans[d].size();
          strides[d] = spans[d].stride;
          len *= _shape[d];
        }
      }

      static constexpr index_t order()
      {
        return Rank;
      }

      template <typename... Indices>
      TENSOR_FUNC auto operator()(Indices... indices) const
      {
        return compute_index(std::forward<Indices>(indices)...);
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
      std::array<index_t, Rank> strides;

      template <index_t Dim = 0, typename... Indices>
      TENSOR_FUNC auto compute_index(index_t index, Indices... indices) const
      {
#ifdef TENSOR_DEBUG
        if (index < 0 || index >= _shape[Dim])
          tensor_out_of_range();
#endif
        if constexpr (Dim + 1 < Rank)
          return strides[Dim] * index + compute_index<Dim + 1>(std::forward<Indices>(indices)...);
        else
          return strides[Dim] * index;
      }

      template <index_t Dim = 0, typename... Indices>
      TENSOR_FUNC auto compute_index(span x, Indices... indices) const
      {
#ifdef TENSOR_DEBUG
        if (x.begin < 0 || x.end >= _shape[Dim])
          tensor_out_of_range();
#endif
        if constexpr (Dim + 1 < Rank)
          return strides[Dim] * x + compute_index<Dim + 1>(std::forward<Indices>(indices)...);
        else
          return strides[Dim] * x;
      }

      template <index_t Dim = 0, typename... Indices>
      TENSOR_FUNC auto compute_index(all, Indices... indices) const
      {
        if constexpr (Dim + 1 < Rank)
          return strides[Dim] * span(0, _shape[Dim]) + compute_index<Dim + 1>(std::forward<Indices>(indices)...);
        else
          return strides[Dim] * span(0, _shape[Dim]);
      }
    };

    template <>
    struct StridedShape<1>
    {
    public:
      TENSOR_FUNC StridedShape(const span &x) : len{x.size()}, stride{x.stride} {}

      static constexpr index_t order()
      {
        return 1;
      }

      TENSOR_FUNC index_t operator()(index_t i) const
      {
#ifdef TENSOR_DEBUG
        if (i < 0 || i >= len)
          tensor_out_of_range();
#endif
        return stride * i;
      }

      TENSOR_FUNC span operator()(span x) const
      {
#ifdef TENSOR_DEBUG
        if (x.begin < 0 || x.end >= len)
          tensor_out_of_range();
#endif
        return stride * x;
      }

      TENSOR_FUNC span operator()(all) const
      {
        return stride * span(0, len);
      }

      TENSOR_FUNC index_t size() const
      {
        return len;
      }

      TENSOR_FUNC index_t shape(index_t) const
      {
        return len;
      }

    private:
      index_t len;
      index_t stride;
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
      using const_reference = typename Container::const_reference;
      using pointer = typename Container::pointer;
      using const_pointer = typename Container::const_pointer;

      __TensorType() = default;
      ~__TensorType() = default;

      explicit TENSOR_FUNC __TensorType(Shape shape_, Container container_) : _shape(shape_), container(container_) {}

      /// @brief shallow copy.
      __TensorType(const __TensorType &) = default;
      __TensorType &operator=(const __TensorType &) = default;

      /// @brief move
      __TensorType(__TensorType &&) = default;
      __TensorType &operator=(__TensorType &&) = default;

      /// @brief returns the order of the tensor, i.e. the number of dimensions
      /// of the tensor.
      static constexpr index_t order()
      {
        return Shape::order();
      }

      /**
       * @brief high dimensional read/write access.
       *
       * @tparam Indices `span` or convertible to `index_t`
       * @param indices indices
       * @return if all indices are integers, the returns a reference to the
       * data at the index. If any indices are `span` then returns a TensorView
       * of the range specified by the spans.
       */
      template <typename... Indices>
      TENSOR_FUNC decltype(auto) at(Indices... indices)
      {
        return subview(_shape(std::forward<Indices>(indices)...));
      }

      /**
       * @brief high dimensional read/write access.
       *
       * @tparam Indices `span` or convertible to `index_t`
       * @param indices indices
       * @return if all indices are integers, the returns a reference to the
       * data at the index. If any indices are `span` then returns a TensorView
       * of the range specified by the spans.
       */
      template <typename... Indices>
      TENSOR_FUNC decltype(auto) at(Indices... indices) const
      {
        return subview(_shape(std::forward<Indices>(indices)...));
      }

      /**
       * @brief high dimensional read/write access.
       *
       * @tparam Indices `span` or convertible to `index_t`
       * @param indices indices
       * @return if all indices are integers, the returns a reference to the
       * data at the index. If any indices are `span` then returns a TensorView
       * of the range specified by the spans.
       */
      template <typename... Indices>
      TENSOR_FUNC decltype(auto) operator()(Indices... indices)
      {
        return subview(_shape(std::forward<Indices>(indices)...));
      }

      /**
       * @brief high dimensional read/write access.
       *
       * @tparam Indices `span` or convertible to `index_t`
       * @param indices indices
       * @return if all indices are integers, the returns a reference to the
       * data at the index. If any indices are `span` then returns a TensorView
       * of the range specified by the spans.
       */
      template <typename... Indices>
      TENSOR_FUNC decltype(auto) operator()(Indices... indices) const
      {
        return subview(_shape(std::forward<Indices>(indices)...));
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

    protected:
      Shape _shape;
      Container container;

      TENSOR_FUNC reference subview(index_t index)
      {
        return container[index];
      }

      TENSOR_FUNC const_reference subview(index_t index) const
      {
        return container[index];
      }

      template <size_t N>
      TENSOR_FUNC auto subview(const std::array<span, N> &spans)
      {
        return SubView<value_type, N>(data(), spans);
      }

      template <size_t N>
      TENSOR_FUNC auto subview(const std::array<span, N> &spans) const
      {
        return SubView<const value_type, N>(data(), spans);
      }

      TENSOR_FUNC auto subview(const span &s)
      {
        return SubView<value_type, 1>(data(), s);
      }

      TENSOR_FUNC auto subview(const span &s) const
      {
        return SubView<const value_type, 1>(data(), s);
      }
    };
  }; // namespace details

  /// @brief provides read/write access to an externally managed array with
  /// high dimensional indexing.
  /// @tparam scalar the type of array, e.g. double, int, etc.
  /// @tparam Rank the tensor dimension, e.g. 2 for a matrix
  template <typename scalar, size_t Rank>
  class TensorView : public details::__TensorType<details::DynamicTensorShape<Rank>, details::ViewContainer<scalar>>
  {
  public:
    template <TENSOR_INT_LIKE... Sizes>
    TENSOR_FUNC explicit TensorView(scalar *data, Sizes... shape) : base_tensor(shape_type(shape...), container_type(data)) {}

    template <TENSOR_INT_LIKE... Sizes>
    TENSOR_FUNC void reshape(Sizes... new_shape)
    {
      this->_shape.reshape(std::forward<Sizes>(new_shape)...);
    }

  private:
    using base_tensor = details::__TensorType<details::DynamicTensorShape<Rank>, details::ViewContainer<scalar>>;
    using shape_type = details::DynamicTensorShape<Rank>;
    using container_type = details::ViewContainer<scalar>;
  };

  /// @brief high dimensional view for tensors with dimensions known at compile time.
  /// @tparam scalar type of tensor elements
  /// @tparam ...Shape shape of the tensor
  template <typename scalar, size_t... Shape>
  class FixedTensorView : public details::__TensorType<details::FixedTensorShape<Shape...>, details::ViewContainer<scalar>>
  {
  public:
    TENSOR_FUNC explicit FixedTensorView(scalar *data) : base_tensor(shape_type{}, container_type(data)) {}

  private:
    using base_tensor = details::__TensorType<details::FixedTensorShape<Shape...>, details::ViewContainer<scalar>>;
    using shape_type = details::FixedTensorShape<Shape...>;
    using container_type = details::ViewContainer<scalar>;
  };

  /// @brief high dimensional sub-view into tensors
  /// @tparam scalar type of tensor elements
  /// @tparam Rank the dimension of the subview
  template <typename scalar, size_t Rank>
  class SubView : public details::__TensorType<details::StridedShape<Rank>, details::ViewContainer<scalar>>
  {
  public:
    TENSOR_FUNC explicit SubView(scalar *data, const std::array<span, Rank> &spans) : base_tensor(shape_type(spans), container_type(data + details::offset(spans))) {}

  private:
    using base_tensor = details::__TensorType<details::StridedShape<Rank>, details::ViewContainer<scalar>>;
    using shape_type = details::StridedShape<Rank>;
    using container_type = details::ViewContainer<scalar>;
  };

  template <typename scalar>
  class SubView<scalar, 1> : public details::__TensorType<details::StridedShape<1>, details::ViewContainer<scalar>>
  {
  public:
    TENSOR_FUNC explicit SubView(scalar *data, const span &s) : base_tensor(shape_type(s), container_type(data + s.begin)) {}

  private:
    using base_tensor = details::__TensorType<details::StridedShape<1>, details::ViewContainer<scalar>>;
    using shape_type = details::StridedShape<1>;
    using container_type = details::ViewContainer<scalar>;
  };

  /// @brief high dimensional tensor with dimensions known at compile time.
  ///
  /// @details Fixed tensors use stack arrays and can be constructed in cuda
  /// __device__ code and passed as arguments to __global__ kernel.
  ///
  /// @tparam scalar type of tensor elements
  /// @tparam ...Shape shape of the tensor
  template <typename scalar, size_t... Shape>
  class FixedTensor : public details::__TensorType<details::FixedTensorShape<Shape...>, std::array<scalar, sizeof...(Shape)>>
  {
  };

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
  class Tensor : public details::__TensorType<details::DynamicTensorShape<Rank>, std::vector<scalar, Allocator>>
  {
  public:
    template <TENSOR_INT_LIKE... Sizes>
    inline explicit Tensor(Sizes... shape) : base_tensor(shape_type(shape...), container_type((1 * ... * shape))) {}

    inline Tensor() : base_tensor(shape_type(), container_type(0)) {}

    template <TENSOR_INT_LIKE... Sizes>
    TENSOR_FUNC void reshape(Sizes... new_shape)
    {
      this->_shape.reshape(std::forward<Sizes>(new_shape)...);
      this->container.resize(this->_shape.size());
    }

  private:
    using base_tensor = details::__TensorType<details::DynamicTensorShape<Rank>, std::vector<scalar, Allocator>>;
    using shape_type = details::DynamicTensorShape<Rank>;
    using container_type = std::vector<scalar, Allocator>;
  };

  /// @brief returns a Tensor of the specified shape.
  template <typename scalar, typename Allocator = std::allocator<scalar>, TENSOR_INT_LIKE... Sizes>
  TENSOR_FUNC auto make_tensor(Sizes... shape)
  {
    constexpr size_t rank = sizeof...(Sizes);
    return Tensor<scalar, rank, Allocator>(shape...);
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
