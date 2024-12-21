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
  inline void tensor_out_of_range(const char *message)
  {
#ifdef TENSOR_USE_CUDA
    printf("TensorView: out of range error: %s\n", message);
    assert(false);
#else
    throw std::out_of_range(std::string("TensorView: out of range error: ") + message);
#endif
  }

  /// @brief terminates program/throws exception with message indicating bad memory access.
  inline void tensor_bad_memory_access()
  {
#ifdef TENSOR_USE_CUDA
    printf("TensorView: attempting to dereference nullptr.");
    assert(false);
#else
    throw std::runtime_error("TensorView: attempting to dereference nullptr.");
#endif
  }

  /// @brief terminates program/throws exception with message indicating that the tensor shape is illogical.
  inline void tensor_bad_shape()
  {
#ifdef TENSOR_USE_CUDA
    printf("TensorView: all dimensions must be strictly positive.");
    assert(false);
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

      TENSOR_FUNC index_t operator[](index_t index) const
      {
#ifdef TENSOR_DEBUG
        if (index < 0 || index >= len)
        {
          char msg[100];
          snprintf(msg, sizeof(msg), "linear index = %ld is out of range for tensor with size %ld.", index, len);
          tensor_out_of_range(msg);
        }
#endif
        return index;
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
        {
          char msg[100];
          snprintf(msg, sizeof(msg), "Index %ld is out of range for dimension %ld with size %ld.", index, Dim, _shape[Dim]);
          tensor_out_of_range(msg);
        }
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
        if (x.begin < 0 || x.end > _shape[Dim])
        {
          char msg[100];
          snprintf(msg, sizeof(msg), "span( %ld, %ld ) is out of range for dimension %ld with size %ld.", x.begin, x.end, Dim, _shape[Dim]);
          tensor_out_of_range(msg);
        }
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

      TENSOR_FUNC index_t operator[](index_t index) const
      {
#ifdef TENSOR_DEBUG
        if (index < 0 || index >= len)
        {
          char msg[100];
          snprintf(msg, sizeof(msg), "linear index = %ld is out of range for tensor with size %ld.", index, len);
          tensor_out_of_range(msg);
        }
#endif
        return index;
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
        {
          char msg[100];
          snprintf(msg, sizeof(msg), "shape index = %ld is out of range for tensor of rank %ld.", d, rank);
          tensor_out_of_range(msg);
        }
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
        {
          char msg[100];
          snprintf(msg, sizeof(msg), "Index %ld is out of range for dimension with size %ld.", index, N);
          tensor_out_of_range(msg);
        }
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
        if (x.begin < 0 || x.end > N)
        {
          char msg[100];
          snprintf(msg, sizeof(msg), "span( %ld, %ld ) is out of range for dimension with size %ld.", x.begin, x.end, N);
          tensor_out_of_range(msg);
        }
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

      TENSOR_FUNC index_t operator[](index_t index) const
      {
#ifdef TENSOR_DEBUG
        if (index < 0 || index >= len)
        {
          char msg[100];
          snprintf(msg, sizeof(msg), "linear index = %ld is out of range for tensor with size %ld.", index, len);
          tensor_out_of_range(msg);
        }
#endif
        return index * strides[0];
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
        {
          char msg[100];
          snprintf(msg, sizeof(msg), "Index %ld is out of range for dimension %ld with size %ld.", index, Dim, _shape[Dim]);
          tensor_out_of_range(msg);
        }
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
        if (x.begin < 0 || x.end > _shape[Dim])
        {
          char msg[100];
          snprintf(msg, sizeof(msg), "span( %ld, %ld ) is out of range for dimension %ld with size %ld.", x.begin, x.end, Dim, _shape[Dim]);
          tensor_out_of_range(msg);
        }
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
        {
          char msg[100];
          snprintf(msg, sizeof(msg), "Index = %ld is out of range for 1D tensor with size %ld.", i, len);
          tensor_out_of_range(msg);
        }
#endif
        return stride * i;
      }

      TENSOR_FUNC span operator()(span x) const
      {
#ifdef TENSOR_DEBUG
        if (x.begin < 0 || x.end >= len)
        {
          char msg[100];
          snprintf(msg, sizeof(msg), "span( %ld, %ld ) is out of range for 1D tensor with size %ld.", x.begin, x.end, len);
          tensor_out_of_range(msg);
        }
#endif
        return stride * x;
      }

      TENSOR_FUNC span operator()(all) const
      {
        return stride * span(0, len);
      }

      TENSOR_FUNC index_t operator[](index_t index) const
      {
#ifdef TENSOR_DEBUG
        if (index < 0 || index >= len)
        {
          char msg[100];
          snprintf(msg, sizeof(msg), "linear index = %ld is out of range for tensor with size %ld.", index, len);
          tensor_out_of_range(msg);
        }
#endif
        return index * stride;
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

      TENSOR_FUNC const_reference operator[](index_t index) const
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

    template <typename Container>
    auto make_view(Container &container)
    {
      using value_type = typename Container::value_type;
      return details::ViewContainer<value_type>(container.data());
    }

    template <typename Container>
    auto make_view(const Container &container)
    {
      return details::ViewContainer<const typename Container::value_type>(container.data());
    }

    template <typename Shape, typename Container>
    class __TensorType
    {
    public:
      using value_type = typename Container::value_type;
      using reference = typename Container::reference;
      using const_reference = typename Container::const_reference;
      using pointer = typename Container::pointer;
      using const_pointer = typename Container::const_pointer;

      template <typename ViewType>
      class Iterator
      {
      public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = typename Container::value_type;
        using difference_type = std::ptrdiff_t;
        using pointer = typename std::conditional<std::is_const<ViewType>::value, const value_type *, value_type *>::type;
        using reference = typename std::conditional<std::is_const<ViewType>::value, const value_type &, value_type &>::type;

        TENSOR_FUNC Iterator(Shape shape_, ViewType &&view_, index_t pos)
            : _pos(pos), _shape(shape_), _view(view_) {}

        TENSOR_FUNC reference operator*()
        {
          return _view[_shape[_pos]];
        }

        TENSOR_FUNC pointer operator->()
        {
          return &_view[_shape[_pos]];
        }

        TENSOR_FUNC reference operator[](difference_type n)
        {
          return _view[_shape[_pos + n]];
        }

        TENSOR_FUNC Iterator &operator++()
        {
          ++_pos;
          return *this;
        }

        TENSOR_FUNC Iterator operator++(int)
        {
          iterator tmp = *this;
          ++_pos;
          return tmp;
        }

        TENSOR_FUNC Iterator &operator--()
        {
          --_pos;
          return *this;
        }

        TENSOR_FUNC Iterator operator--(int)
        {
          iterator tmp = *this;
          --_pos;
          return tmp;
        }

        TENSOR_FUNC Iterator operator+(difference_type n) const
        {
          return iterator(_shape, _view, _pos + n);
        }

        TENSOR_FUNC Iterator operator-(difference_type n) const
        {
          return iterator(_shape, _view, _pos - n);
        }

        TENSOR_FUNC Iterator &operator+=(difference_type n)
        {
          _pos += n;
          return *this;
        }

        TENSOR_FUNC Iterator &operator-=(difference_type n)
        {
          _pos -= n;
          return *this;
        }

        TENSOR_FUNC difference_type operator-(const Iterator &other) const
        {
          return _pos - other._pos;
        }

        TENSOR_FUNC bool operator==(const Iterator &other) const
        {
          return _pos == other._pos;
        }

        TENSOR_FUNC bool operator!=(const Iterator &other) const
        {
          return _pos != other._pos;
        }

        TENSOR_FUNC bool operator<(const Iterator &other) const
        {
          return _pos < other._pos;
        }

        TENSOR_FUNC bool operator>(const Iterator &other) const
        {
          return _pos > other._pos;
        }

        TENSOR_FUNC bool operator<=(const Iterator &other) const
        {
          return _pos <= other._pos;
        }

        TENSOR_FUNC bool operator>=(const Iterator &other) const
        {
          return _pos >= other._pos;
        }

      private:
        index_t _pos;
        Shape _shape;
        ViewType _view;
      };

      using iterator = Iterator<decltype(make_view(std::declval<Container &>()))>;
      using const_iterator = Iterator<decltype(make_view(std::declval<const Container &>()))>;
      using reverse_iterator = std::reverse_iterator<iterator>;
      using const_reverse_iterator = std::reverse_iterator<const_iterator>;

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
        return container[_shape[index]];
      }

      /// @brief linear index access.
      TENSOR_FUNC const_reference operator[](index_t index) const
      {
        return container[_shape[index]];
      }

      /// @brief returns pointer to start of tensor.
      TENSOR_FUNC iterator begin()
      {
        return iterator(_shape, make_view<Container>(container), 0);
      }

      /// @brief returns pointer to start of tensor.
      TENSOR_FUNC const_iterator begin() const
      {
        return const_iterator(_shape, make_view<const Container>(container), 0);
      }

      /// @brief returns pointer to the element following the last element of tensor.
      TENSOR_FUNC iterator end()
      {
        return iterator(_shape, make_view<Container>(container), size());
      }

      /// @brief returns pointer to the element following the last element of tensor.
      TENSOR_FUNC const_iterator end() const
      {
        return const_iterator(_shape, make_view<const Container>(container), size());
      }

      /// @brief returns reverse iterator to the end of the tensor.
      TENSOR_FUNC reverse_iterator rbegin()
      {
        return reverse_iterator(end());
      }

      /// @brief returns reverse iterator to the end of the tensor.
      TENSOR_FUNC const_reverse_iterator rbegin() const
      {
        return const_reverse_iterator(end());
      }

      /// @brief returns reverse iterator to the start of the tensor.
      TENSOR_FUNC reverse_iterator rend()
      {
        return reverse_iterator(begin());
      }

      /// @brief returns reverse iterator to the start of the tensor.
      TENSOR_FUNC const_reverse_iterator rend() const
      {
        return const_reverse_iterator(begin());
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
        return SubView<value_type, N>(make_view(container), spans);
      }

      template <size_t N>
      TENSOR_FUNC auto subview(const std::array<span, N> &spans) const
      {
        return SubView<const value_type, N>(make_view(container), spans);
      }

      TENSOR_FUNC auto subview(const span &s)
      {
        return SubView<value_type, 1>(make_view(container), s);
      }

      TENSOR_FUNC auto subview(const span &s) const
      {
        return SubView<const value_type, 1>(make_view(container), s);
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
    using base_tensor = details::__TensorType<details::DynamicTensorShape<Rank>, details::ViewContainer<scalar>>;
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

  /// @brief high dimensional view for tensors with dimensions known at compile time.
  /// @tparam scalar type of tensor elements
  /// @tparam ...Shape shape of the tensor
  template <typename scalar, size_t... Shape>
  class FixedTensorView : public details::__TensorType<details::FixedTensorShape<Shape...>, details::ViewContainer<scalar>>
  {
  public:
    using base_tensor = details::__TensorType<details::FixedTensorShape<Shape...>, details::ViewContainer<scalar>>;
    using shape_type = details::FixedTensorShape<Shape...>;
    using container_type = details::ViewContainer<scalar>;
    using pointer = typename base_tensor::pointer;
    using const_pointer = typename base_tensor::const_pointer;

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
      return this->container.data();
    }

    /// @brief returns the externally managed array
    TENSOR_FUNC const_pointer data() const
    {
      return this->container.data();
    }
  };

  /// @brief high dimensional sub-view into tensors
  /// @tparam scalar type of tensor elements
  /// @tparam Rank the dimension of the subview
  template <typename scalar, size_t Rank>
  class SubView : public details::__TensorType<details::StridedShape<Rank>, details::ViewContainer<scalar>>
  {
  public:
    TENSOR_FUNC explicit SubView(details::ViewContainer<scalar> view, const std::array<span, Rank> &spans) : base_tensor(shape_type(spans), container_type(view.data() + details::offset(spans))) {}

  private:
    using base_tensor = details::__TensorType<details::StridedShape<Rank>, details::ViewContainer<scalar>>;
    using shape_type = details::StridedShape<Rank>;
    using container_type = details::ViewContainer<scalar>;
  };

  template <typename scalar>
  class SubView<scalar, 1> : public details::__TensorType<details::StridedShape<1>, details::ViewContainer<scalar>>
  {
  public:
    TENSOR_FUNC explicit SubView(details::ViewContainer<scalar> view, const span &s) : base_tensor(shape_type(s), container_type(view.data() + s.begin)) {}

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
  class FixedTensor : public details::__TensorType<details::FixedTensorShape<Shape...>, std::array<scalar, (1 * ... * Shape)>>
  {
  public:
    using base_tensor = details::__TensorType<details::FixedTensorShape<Shape...>, std::array<scalar, (1 * ... * Shape)>>;
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
    using base_tensor = details::__TensorType<details::DynamicTensorShape<Rank>, std::vector<scalar, Allocator>>;
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
