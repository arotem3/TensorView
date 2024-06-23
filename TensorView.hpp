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
  template <size_t Rank, typename scalar>
  class TensorView
  {
  protected:
    template <index_t Dim = 0, typename... Sizes>
    TENSOR_HOST_DEVICE inline index_t initialize_shape(index_t I, Sizes... Is)
    {
      if constexpr (std::is_unsigned_v<index_t>)
        if (I < 0)
          tensor_view_out_of_range();

      _shape[Dim] = I;

      if constexpr (Dim + 1 < Rank)
        return I * initialize_shape<Dim + 1>(std::forward<Sizes>(Is)...);
      else
        return I;
    }

    template <index_t Dim = 0, typename... Inds>
    TENSOR_HOST_DEVICE inline index_t linear_index(index_t I, Inds... Is)
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
    /// @brief empty tensor
    TensorView() = default;
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
    template <typename... Sizes>
    TENSOR_HOST_DEVICE inline explicit TensorView(scalar *data_, Sizes... shape_) : ptr(data_)
    {
      static_assert(Rank > 0, "TensorView must have a positive number of dimensions");
      static_assert(sizeof...(shape_) == Rank, "TensorView: wrong number of dimensions specified.");

      len = initialize_shape(std::forward<Sizes>(shape_)...);
    }

    /// @brief high dimensional read/write access.
    /// @tparam ...Indices sequence of `index_t`
    /// @param[in] ...ids indices
    /// @return reference to data at index (`...ids`)
    template <typename... Indices>
    TENSOR_HOST_DEVICE inline scalar &at(Indices... ids)
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
    template <typename... Indices>
        TENSOR_HOST_DEVICE inline TENSOR_CONST_QUAL(scalar) & at(Indices... ids) const
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
    template <typename... Indices>
    TENSOR_HOST_DEVICE inline scalar &operator()(Indices... ids)
    {
      return at(std::forward<Indices>(ids)...);
    }

    /// @brief high dimensional read-only access.
    /// @tparam ...Indices sequence of `index_t`
    /// @param[in] ...ids indices
    /// @return const reference to data at index (`...ids`)
    template <typename... Indices>
    TENSOR_HOST_DEVICE inline TENSOR_CONST_QUAL(scalar) &operator()(Indices... ids) const
    {
      return at(std::forward<Indices>(ids)...);
    }

    /// @brief linear indexing. read/write access.
    /// @param[in] idx flattened index
    /// @return reference to data at linear index `idx`.
    TENSOR_HOST_DEVICE inline scalar &operator[](index_t idx)
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
    TENSOR_HOST_DEVICE inline const TENSOR_CONST_QUAL(scalar) &operator[](index_t idx) const
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
    TENSOR_HOST_DEVICE inline
    operator scalar *()
    {
      return ptr;
    }

    /// @brief implicit conversion to scalar* where the returned pointer is
    /// the one managed by the tensor.
    TENSOR_HOST_DEVICE inline
        operator TENSOR_CONST_QUAL(scalar) *
        () const
    {
      return ptr;
    }

    /// @brief returns the externally managed array
    TENSOR_HOST_DEVICE inline scalar *data()
    {
      return ptr;
    }

    /// @brief returns read-only pointer to the externally managed array
    TENSOR_HOST_DEVICE inline TENSOR_CONST_QUAL(scalar) * data() const
    {
      return ptr;
    }

    /// @brief returns pointer to start of tensor.
    TENSOR_HOST_DEVICE inline scalar *begin()
    {
      return ptr;
    }

    /// @brief returns pointer to the element following the last element of tensor.
    TENSOR_HOST_DEVICE inline scalar *end()
    {
      return ptr + len;
    }

    /// @brief returns pointer to start of tensor.
    TENSOR_HOST_DEVICE inline TENSOR_CONST_QUAL(scalar) * begin() const
    {
      return ptr;
    }

    /// @brief returns pointer to the element following the last element of tensor.
    TENSOR_HOST_DEVICE inline TENSOR_CONST_QUAL(scalar) * end() const
    {
      return ptr + len;
    }

    /// @brief returns the shape of the tensor. Has length `Rank`
    TENSOR_HOST_DEVICE inline constexpr const index_t *shape() const
    {
      return _shape;
    }

    /// @brief returns the size of the tensor along dimension d.
    TENSOR_HOST_DEVICE inline constexpr index_t shape(index_t d) const
    {
#ifdef TENSOR_DEBUG
      if (d < 0 || d >= Rank)
        tensor_view_out_of_range();
#endif
      return _shape[d];
    }

    /// @brief returns total number of elements in the tensor.
    TENSOR_HOST_DEVICE inline constexpr index_t size() const
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
  template <typename scalar, typename... Sizes>
  TENSOR_HOST_DEVICE inline TensorView<sizeof...(Sizes), scalar> reshape(scalar *data, Sizes... shape)
  {
    return TensorView<sizeof...(Sizes), scalar>(data, shape...);
  }

  /// @brief reshape `TensorView`. Returns new TensorView with new shape
  /// but points to same data.
  /// @tparam scalar type of array
  /// @tparam ...Sizes sequence of `index_t`
  /// @param[in] tensor the array
  /// @param[in] ...shape the shape of the tensor
  template <typename scalar, size_t Rank, typename... Sizes>
  TENSOR_HOST_DEVICE inline TensorView<sizeof...(Sizes), scalar> reshape(TensorView<Rank, scalar> tensor, Sizes... shape)
  {
    return reshape(tensor.data(), std::forward<Sizes>(shape)...);
  }

  /// @brief specialization of `TensorView` when `Dim == 1`
  template <typename scalar>
  using vector_view = TensorView<1, scalar>;

  /// @brief specialization of `TensorView` when `Dim == 2`
  template <typename scalar>
  using matrix_view = TensorView<2, scalar>;

  /// @brief specialization of `TensorView` when `Dim == 3`
  template <typename scalar>
  using cube_view = TensorView<3, scalar>;
} // namespace tensor

#endif
