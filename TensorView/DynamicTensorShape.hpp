#ifndef __TENSOR_VIEW_DYNAMIC_TENSOR_SHAPE_HPP__
#define __TENSOR_VIEW_DYNAMIC_TENSOR_SHAPE_HPP__

#include "tensorview_config.hpp"
#include "errors.hpp"
#include "span.hpp"

namespace tensor::details
{
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

} // namespace tensor

#endif
