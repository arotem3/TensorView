#ifndef __TENSOR_VIEW_STRIDED_SHAPE_HPP__
#define __TENSOR_VIEW_STRIDED_SHAPE_HPP__

#include "tensorview_config.hpp"
#include "errors.hpp"
#include "span.hpp"

namespace tensor::details
{
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
      if (index >= len)
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
      if (index >= _shape[Dim])
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
      if (x.end > _shape[Dim])
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
      if (i >= len)
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
      if (x.end >= len)
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
      if (index >= len)
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

} // namespace tensor

#endif
