#ifndef __TENSOR_VIEW_SPAN_HPP__
#define __TENSOR_VIEW_SPAN_HPP__

#include "TensorView/tensorview_config.hpp"

namespace tensor
{
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

  namespace details
  {
    // offset a span by an index
    constexpr span operator+(const span &x, index_t i)
    {
      return span(x.begin + i, x.end + i, x.stride);
    }

    // offset a span by an index
    constexpr span operator+(index_t i, const span &x)
    {
      return span(x.begin + i, x.end + i, x.stride);
    }

    // scale a span
    constexpr span operator*(index_t s, const span &x)
    {
      return span(s * x.begin, s * x.end, s * x.stride);
    }

    // combine two spans into a multidiensional span
    constexpr std::array<span, 2> operator+(const span &x, const span &y)
    {
      return {x, y};
    }

    // offset multidimensional span by an index -- implementation
    template <size_t N, size_t... I>
    constexpr std::array<span, N> add(index_t i, const std::array<span, N> &spans, std::index_sequence<I...>)
    {
      return {i + spans[0], spans[I + 1]...};
    }

    // scale multidimensional span -- implementation
    template <size_t N, size_t... I>
    constexpr std::array<span, N> scale(index_t s, const std::array<span, N> &spans, std::index_sequence<I...>)
    {
      return {(s * spans[I])...};
    }

    // offset multidimensional span by an index
    template <size_t N, size_t... I>
    constexpr std::array<span, N> operator+(index_t i, const std::array<span, N> &spans)
    {
      static_assert(N > 0);
      return add(i, spans, std::make_index_sequence<N - 1>{});
    }

    // offset multidimensional span by an index
    template <size_t N, size_t... I>
    constexpr std::array<span, N> operator+(const std::array<span, N> &spans, index_t i)
    {
      static_assert(N > 0);
      return add(i, spans, std::make_index_sequence<N - 1>{});
    }

    // scale multidimensional span
    template <size_t N, size_t... I>
    constexpr std::array<span, N> operator*(index_t s, const std::array<span, N> &spans)
    {
      return scale(s, spans, std::make_index_sequence<N>{});
    }

    // combine two spans into a multidimensional span -- implementation
    template <size_t N, size_t... I>
    constexpr std::array<span, N + 1> concat(const span &x, const std::array<span, N> &spans, std::index_sequence<I...>)
    {
      return {x, spans[I]...};
    }

    // combine two spans into a multidimensional span -- implementation
    template <size_t N, size_t... I>
    constexpr std::array<span, N + 1> concat(const std::array<span, N> &spans, const span &x, std::index_sequence<I...>)
    {
      return {spans[I]..., x};
    }

    // combine two spans into a multidimensional span -- implementation
    template <size_t N, size_t M, size_t... I, size_t... J>
    constexpr std::array<span, N + M> concat(const std::array<span, N> &A, const std::array<span, M> &B, std::index_sequence<I...>, std::index_sequence<J...>)
    {
      return {A[I]..., B[J]...};
    }

    // combine two spans into a multidimensional span
    template <size_t N>
    constexpr std::array<span, N + 1> operator+(const span &x, const std::array<span, N> &spans)
    {
      return concat(x, spans, std::make_index_sequence<N>{});
    }

    // combine two spans into a multidimensional span
    template <size_t N>
    constexpr std::array<span, N + 1> operator+(const std::array<span, N> &spans, const span &x)
    {
      return concat(spans, x, std::make_index_sequence<N>{});
    }

    // combine two spans into a multidimensional span
    template <size_t N, size_t M>
    constexpr auto operator+(const std::array<span, N> &a, const std::array<span, M> &b)
    {
      return concat(a, b, std::make_index_sequence<N>{}, std::make_index_sequence<M>{});
    }

    // compute offset of a multidimensional span
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
  } // namespace details

} // namespace tensor

#endif