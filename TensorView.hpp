#ifndef __TENSOR_VIEW_HPP__
#define __TENSOR_VIEW_HPP__

#include <type_traits>
#include <cstdio>
#include <stdexcept>

#ifdef TENSOR_USE_CUDA
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
  inline void tensor_view_out_of_range()
  {
#ifdef TENSOR_USE_CUDA
    fprintf(stderr, "TensorView out of range error.\n");
    std::abort();
#else
    throw std::out_of_range("TensorView out of range error.\n");
#endif
  }

  /// @brief terminates program/throws exception with message indicating bad memory access.
  inline void tensor_view_bad_memory_access()
  {
#ifdef TENSOR_USE_CUDA
    fprintf(stderr, "TensorView attempting to dereference nullptr.");
    std::abort();
#else
    throw std::runtime_error("TensorView attempting to dereference nullptr.");
#endif
  }

  /// @brief provides read/write access to an externally managed array with
  /// high dimensional indexing.
  /// @tparam scalar the type of array, e.g. double, int, etc.
  /// @tparam Rank the tensor dimension, e.g. 2 for a matrix
  template <typename scalar, size_t Rank>
  class TensorView
  {
  protected:
    template <index_t Dim = 0, TENSOR_INT_LIKE... Inds>
    TENSOR_FUNC index_t linear_index(index_t I, Inds... Is)
    {
#ifdef TENSOR_DEBUG
      if (I < 0 || I >= _shape[Dim])
        tensor_view_out_of_range();
#endif

      if constexpr (Dim + 1 < Rank)
        return I + _shape[Dim] * linear_index<Dim + 1>(std::forward<Inds>(Is)...);
      else
        return I;
    }

    index_t _shape[Rank];
    index_t len;
    scalar *ptr;

  public:
    using value_t = scalar;
    using ref_t = scalar &;
    using cref_t = TENSOR_CONST_QUAL(scalar) &;
    using ptr_t = scalar *;
    using cptr_t = TENSOR_CONST_QUAL(scalar) *;

    /// @brief empty tensor
    constexpr TensorView() = default;
    ~TensorView() = default;

    /// @brief (shallow) copy tensor. Copy points to the same data.
    /// @param[in] tensor to copy
    TensorView(const TensorView &) = default;

    /// @brief (shallow) copy tensor. Copy points to the same data.
    /// @param[in] tensor to copy
    /// @return `this`
    TensorView &operator=(const TensorView &) = default;

    /// @brief wrap externally managed array
    /// @tparam ...Sizes sequence of `index_t`
    /// @param[in] data_ externally managed array
    /// @param[in] ...shape_ shape of array as a sequence of `index_t`s
    template <TENSOR_INT_LIKE... Sizes>
    TENSOR_FUNC explicit TensorView(scalar *data_, Sizes... shape_) : _shape{(index_t)shape_...}, len{(index_t)(1 * ... * shape_)}, ptr(data_)
    {
      static_assert(Rank > 0, "TensorView must have a positive number of dimensions");
      static_assert(sizeof...(shape_) == Rank, "TensorView: wrong number of dimensions specified.");
#ifdef TENSOR_DEBUG
      if (((shape_ < 0) || ... || false))
      {
#ifdef TENSOR_USE_CUDA
        fprintf(stderr, "TensorView all dimensions must be strictly positive.");
        std::abort();
#else
        throw std::runtime_error("TensorView all dimensions must be strictly positive.");
#endif
      }
#endif
    }

    /// @brief high dimensional read/write access.
    /// @tparam ...Indices sequence of `index_t`
    /// @param[in] ...ids indices
    /// @return reference to data at index (`...ids`)
    template <TENSOR_INT_LIKE... Indices>
    TENSOR_FUNC ref_t at(Indices... ids)
    {
      static_assert(sizeof...(ids) == Rank, "TensorView: wrong number of indices specified.");

#ifdef TENSOR_DEBUG
      if (ptr == nullptr)
        tensor_view_bad_memory_access();
#endif

      return ptr[linear_index(std::forward<Indices>(ids)...)];
    }

    /// @brief high dimensional read-only access.
    /// @tparam ...Indices sequence of `index_t`
    /// @param[in] ...ids indices
    /// @return const reference to data at index (`...ids`)
    template <TENSOR_INT_LIKE... Indices>
    TENSOR_FUNC cref_t at(Indices... ids) const
    {
      static_assert(sizeof...(ids) == Rank, "TensorViewWrong number of indices specified.");

#ifdef TENSOR_DEBUG
      if (ptr == nullptr)
        tensor_view_bad_memory_access();
#endif

      return ptr[linear_index(std::forward<Indices>(ids)...)];
    }

    /// @brief high dimensional read/write access.
    /// @tparam ...Indices sequence of `index_t`
    /// @param[in] ...ids indices
    /// @return reference to data at index (`...ids`)
    template <TENSOR_INT_LIKE... Indices>
    TENSOR_FUNC ref_t operator()(Indices... ids)
    {
      return at(std::forward<Indices>(ids)...);
    }

    /// @brief high dimensional read-only access.
    /// @tparam ...Indices sequence of `index_t`
    /// @param[in] ...ids indices
    /// @return const reference to data at index (`...ids`)
    template <TENSOR_INT_LIKE... Indices>
    TENSOR_FUNC cref_t operator()(Indices... ids) const
    {
      return at(std::forward<Indices>(ids)...);
    }

    /// @brief linear indexing. read/write access.
    /// @param[in] idx flattened index
    /// @return reference to data at linear index `idx`.
    TENSOR_FUNC ref_t operator[](index_t idx)
    {
#ifdef TENSOR_DEBUG
      if (ptr == nullptr)
        tensor_view_bad_memory_access();
      if (idx < 0 || idx >= len)
        tensor_view_out_of_range();
#endif

      return ptr[idx];
    }

    /// @brief linear indexing. read only access.
    /// @param[in] idx flattened index
    /// @return const reference to data at linear index `idx`.
    TENSOR_FUNC cref_t operator[](index_t idx) const
    {
#ifdef TENSOR_DEBUG
      if (ptr == nullptr)
        tensor_view_bad_memory_access();
      if (idx < 0 || idx >= len)
        tensor_view_out_of_range();
#endif

      return ptr[idx];
    }

    /// @brief implicit conversion to scalar* where the returned pointer is
    /// the one managed by the tensor.
    TENSOR_FUNC operator ptr_t()
    {
      return ptr;
    }

    /// @brief implicit conversion to scalar* where the returned pointer is
    /// the one managed by the tensor.
    TENSOR_FUNC operator cptr_t() const
    {
      return ptr;
    }

    /// @brief returns the externally managed array
    TENSOR_FUNC ptr_t data()
    {
      return ptr;
    }

    /// @brief returns read-only pointer to the externally managed array
    TENSOR_FUNC cptr_t data() const
    {
      return ptr;
    }

    /// @brief returns pointer to start of tensor.
    TENSOR_FUNC ptr_t begin()
    {
      return ptr;
    }

    /// @brief returns pointer to the element following the last element of tensor.
    TENSOR_FUNC ptr_t end()
    {
      return ptr + len;
    }

    /// @brief returns pointer to start of tensor.
    TENSOR_FUNC cptr_t begin() const
    {
      return ptr;
    }

    /// @brief returns pointer to the element following the last element of tensor.
    TENSOR_FUNC cptr_t end() const
    {
      return ptr + len;
    }

    /// @brief returns the shape of the tensor. Has length `Rank`
    TENSOR_FUNC const index_t *shape() const
    {
      return _shape;
    }

    /// @brief returns the size of the tensor along dimension d.
    TENSOR_FUNC index_t shape(index_t d) const
    {
#ifdef TENSOR_DEBUG
      if (d < 0 || d >= Rank)
        tensor_view_out_of_range();
#endif
      return _shape[d];
    }

    /// @brief returns total number of elements in the tensor.
    TENSOR_FUNC index_t size() const
    {
      return len;
    }
  };

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

  /// @brief reshape `TensorView`. Returns new TensorView with new shape
  /// but points to same data.
  /// @tparam scalar type of array
  /// @tparam ...Sizes sequence of `index_t`
  /// @param[in] tensor the array
  /// @param[in] ...shape the shape of the tensor
  template <typename scalar, size_t Rank, TENSOR_INT_LIKE... Sizes>
  TENSOR_FUNC auto reshape(const TensorView<scalar, Rank> &tensor, Sizes... shape)
  {
    return reshape(tensor.data(), std::forward<Sizes>(shape)...);
  }

  /// @brief reshape `TensorView`. Returns new TensorView with new shape
  /// but points to same data.
  /// @tparam scalar type of array
  /// @tparam ...Sizes sequence of `index_t`
  /// @param[in] tensor the array
  /// @param[in] ...shape the shape of the tensor
  template <typename scalar, size_t Rank, TENSOR_INT_LIKE... Sizes>
  TENSOR_FUNC auto reshape(TensorView<scalar, Rank> &tensor, Sizes... shape)
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

  /// @brief high dimensional indexing for tensors with dimensions known at compile time.
  /// @tparam scalar type of tensor elements
  /// @tparam ...Shape shape of the tensor
  template <typename scalar, size_t... Shape>
  class FixedTensorView
  {
  private:
    static constexpr index_t len = (1 * ... * Shape);
    static constexpr index_t rank = sizeof...(Shape);

    scalar *ptr;

    template <index_t N, index_t... Ns, TENSOR_INT_LIKE... Is>
    static TENSOR_FUNC index_t linear_index(index_t idx, Is... ids)
    {
#ifdef TENSOR_DEBUG
      if (idx < 0 || idx >= N)
        tensor_view_out_of_range();
#endif

      if constexpr (sizeof...(Is) == 0)
        return idx;
      else
        return idx + N * linear_index<Ns...>(std::forward<Is>(ids)...);
    }

  public:
    using value_t = scalar;
    using ref_t = scalar &;
    using cref_t = TENSOR_CONST_QUAL(scalar) &;
    using ptr_t = scalar *;
    using cptr_t = TENSOR_CONST_QUAL(scalar) *;

    TENSOR_FUNC FixedTensorView(scalar *data = nullptr) : ptr{data}
    {
      static_assert(((Shape > 0) && ... && true), "FixedTensorView dimensions must be strictly positive.");
      static_assert(rank > 0, "FixedTensorView must have at least one dimension.");
    }

    ~FixedTensorView() = default;

    /// @brief shallow copy.
    FixedTensorView(const FixedTensorView &) = default;
    FixedTensorView &operator=(const FixedTensorView &) = default;

    /// @brief high dimensional read/write access.
    /// @tparam ...Indices convertible to `index_t`
    /// @param ...ids indices
    /// @return reference to data at the index.
    template <TENSOR_INT_LIKE... Indices>
    TENSOR_FUNC ref_t at(Indices... ids)
    {
      static_assert(sizeof...(Indices) == rank, "FixedTensorView: wrong number of indices.");

#ifdef TENSOR_DEBUG
      if (ptr == nullptr)
        tensor_view_bad_memory_access();
#endif
      return ptr[linear_index<Shape...>(std::forward<Indices>(ids)...)];
    }

    /// @brief high dimensional read/write access.
    /// @tparam ...Indices convertible to `index_t`
    /// @param ...ids indices
    /// @return reference to data at the index.
    template <TENSOR_INT_LIKE... Indices>
    TENSOR_FUNC cref_t at(Indices... ids) const
    {
      static_assert(sizeof...(Indices) == rank, "FixedTensorView: wrong number of indices.");

#ifdef TENSOR_DEBUG
      if (ptr == nullptr)
        tensor_view_bad_memory_access();
#endif
      return ptr[linear_index<Shape...>(std::forward<Indices>(ids)...)];
    }

    /// @brief high dimensional read/write access.
    /// @tparam ...Indices convertible to `index_t`
    /// @param ...ids indices
    /// @return reference to data at the index.
    template <TENSOR_INT_LIKE... Indices>
    TENSOR_FUNC ref_t operator()(Indices... ids)
    {
      return at(std::forward<Indices>(ids)...);
    }

    /// @brief high dimensional read/write access.
    /// @tparam ...Indices convertible to `index_t`
    /// @param ...ids indices
    /// @return reference to data at the index.
    template <TENSOR_INT_LIKE... Indices>
    TENSOR_FUNC cref_t operator()(Indices... ids) const
    {
      return at(std::forward<Indices>(ids)...);
    }

    /// @brief linear index access.
    TENSOR_FUNC ref_t operator[](index_t idx)
    {
#ifdef TENSOR_DEBUG
      if (ptr == nullptr)
        tensor_view_bad_memory_access();
      if (idx < 0 || idx >= len)
        tensor_view_out_of_range();
#endif
      return ptr[idx];
    }

    /// @brief linear index access.
    TENSOR_FUNC cref_t operator[](index_t idx) const
    {
#ifdef TENSOR_DEBUG
      if (ptr == nullptr)
        tensor_view_bad_memory_access();
      if (idx < 0 || idx >= len)
        tensor_view_out_of_range();
#endif
      return ptr[idx];
    }

    /// @brief implicit conversion to scalar*
    TENSOR_FUNC operator ptr_t()
    {
      return ptr;
    }

    /// @brief implicit conversion to scalar*
    TENSOR_FUNC operator cptr_t() const
    {
      return ptr;
    }

    /// @brief returns the externally managed array
    TENSOR_FUNC ptr_t data()
    {
      return ptr;
    }

    /// @brief returns the externally managed array
    TENSOR_FUNC cptr_t data() const
    {
      return ptr;
    }

    /// @brief returns pointer to start of tensor.
    TENSOR_FUNC ptr_t begin()
    {
      return ptr;
    }

    /// @brief returns pointer to start of tensor.
    TENSOR_FUNC cptr_t begin() const
    {
      return ptr;
    }

    /// @brief returns pointer to the element following the last element of tensor.
    TENSOR_FUNC ptr_t end()
    {
      return ptr + len;
    }

    /// @brief returns pointer to the element following the last element of tensor.
    TENSOR_FUNC cptr_t end() const
    {
      return ptr + len;
    }

    /// @brief returns total number of elements in the tensor.
    static TENSOR_FUNC index_t size()
    {
      return len;
    }

    /// @brief returns the size of the tensor along dimension d.
    static TENSOR_FUNC index_t shape(index_t d)
    {
#ifdef TENSOR_DEBUG
      if (d < 0 || d >= rank)
        tensor_view_out_of_range();
#endif
      constexpr index_t _shape[] = {Shape...};
      return _shape[d];
    }
  };

  /// @brief reshapes a FixedTensorView to a TensorView of the specified shape
  template <typename scalar, size_t... Shape, TENSOR_INT_LIKE... Sizes>
  TENSOR_FUNC auto reshape(const FixedTensorView<scalar, Shape...> &tensor, Sizes... shape)
  {
    return reshape(tensor.data(), shape...);
  }

  /// @brief reshapes a FixedTensorView to a TensorView of the specified shape
  template <typename scalar, size_t... Shape, TENSOR_INT_LIKE... Sizes>
  TENSOR_FUNC auto reshape(FixedTensorView<scalar, Shape...> &tensor, Sizes... shape)
  {
    return reshape(tensor.data(), shape...);
  }

  template <typename scalar, size_t Shape0, size_t Shape1>
  using FixedMatrixView = FixedTensorView<scalar, Shape0, Shape1>;

  template <typename scalar, size_t Shape0, size_t Shape1, size_t Shape2>
  using FixedCubeView = FixedTensorView<scalar, Shape0, Shape1, Shape2>;
} // namespace tensor

#endif
