#ifndef __TENSOR_VIEW_FIXED_TENSOR_SHAPE_HPP__
#define __TENSOR_VIEW_FIXED_TENSOR_SHAPE_HPP__

#include "tensorview_config.hpp"
#include "errors.hpp"
#include "span.hpp"

namespace tensor::details
{
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
      if (index >= len)
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
      if (d >= rank)
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
      if (index >= N)
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
      if (x.end > N)
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

} // namespace tensor

#endif
