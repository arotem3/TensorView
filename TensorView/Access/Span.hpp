#pragma once
#include "TensorView/Macros.hpp"

namespace tensor
{
   // a Span represents a range of indices with a given stride
   struct Span
   {
   public:
      index_t begin;
      index_t end;
      index_t stride;

      constexpr explicit Span(index_t Begin, index_t End, index_t inc = 1) : begin{Begin}, end{End}, stride{inc} {}

      TENSOR_FUNC index_t size() const
      {
         if (begin >= end)
            return 0;
         return (end - begin) / stride;
      }
   };

   // represents All indices in a dimension
   struct All
   {
   };

   template <size_t numDims>
   using MultiSpan = std::array<Span, numDims>;
} // namespace tensor

namespace tensor::details
{
   // offset a Span by an index
   constexpr Span operator+(const Span &x, index_t i)
   {
      return Span(x.begin + i, x.end + i, x.stride);
   }

   // offset a Span by an index
   constexpr Span operator+(index_t i, const Span &x)
   {
      return Span(x.begin + i, x.end + i, x.stride);
   }

   // scale a Span
   constexpr Span operator*(index_t s, const Span &x)
   {
      return Span(s * x.begin, s * x.end, s * x.stride);
   }

   // combine two spans into a multidiensional Span
   constexpr MultiSpan<2> operator+(const Span &x, const Span &y)
   {
      return {x, y};
   }

   // offset multidimensional Span by an index -- implementation
   template <size_t N, size_t... I>
   constexpr MultiSpan<N> add(index_t i, const MultiSpan<N> &spans, std::index_sequence<I...>)
   {
      return {i + spans[0], spans[I + 1]...};
   }

   // scale multidimensional Span -- implementation
   template <size_t N, size_t... I>
   constexpr MultiSpan<N> scale(index_t s, const MultiSpan<N> &spans, std::index_sequence<I...>)
   {
      return {(s * spans[I])...};
   }

   // offset multidimensional Span by an index
   template <size_t N, size_t... I>
   constexpr MultiSpan<N> operator+(index_t i, const MultiSpan<N> &spans)
   {
      static_assert(N > 0);
      return add(i, spans, std::make_index_sequence<N - 1>{});
   }

   // offset multidimensional Span by an index
   template <size_t N, size_t... I>
   constexpr MultiSpan<N> operator+(const MultiSpan<N> &spans, index_t i)
   {
      static_assert(N > 0);
      return add(i, spans, std::make_index_sequence<N - 1>{});
   }

   // scale multidimensional Span
   template <size_t N, size_t... I>
   constexpr MultiSpan<N> operator*(index_t s, const MultiSpan<N> &spans)
   {
      return scale(s, spans, std::make_index_sequence<N>{});
   }

   // combine two spans into a multidimensional Span -- implementation
   template <size_t N, size_t... I>
   constexpr MultiSpan<N + 1> concat(const Span &x, const MultiSpan<N> &spans, std::index_sequence<I...>)
   {
      return {x, spans[I]...};
   }

   // combine two spans into a multidimensional Span -- implementation
   template <size_t N, size_t... I>
   constexpr MultiSpan<N + 1> concat(const MultiSpan<N> &spans, const Span &x, std::index_sequence<I...>)
   {
      return {spans[I]..., x};
   }

   // combine two spans into a multidimensional Span -- implementation
   template <size_t N, size_t M, size_t... I, size_t... J>
   constexpr MultiSpan<N + M> concat(const MultiSpan<N> &A, const MultiSpan<M> &B, std::index_sequence<I...>,
                                     std::index_sequence<J...>)
   {
      return {A[I]..., B[J]...};
   }

   // combine two spans into a multidimensional Span
   template <size_t N>
   constexpr MultiSpan<N + 1> operator+(const Span &x, const MultiSpan<N> &spans)
   {
      return concat(x, spans, std::make_index_sequence<N>{});
   }

   // combine two spans into a multidimensional Span
   template <size_t N>
   constexpr MultiSpan<N + 1> operator+(const MultiSpan<N> &spans, const Span &x)
   {
      return concat(spans, x, std::make_index_sequence<N>{});
   }

   // combine two spans into a multidimensional Span
   template <size_t N, size_t M>
   constexpr auto operator+(const MultiSpan<N> &a, const MultiSpan<M> &b)
   {
      return concat(a, b, std::make_index_sequence<N>{}, std::make_index_sequence<M>{});
   }

   // compute offset of a multidimensional Span
   template <size_t N>
   constexpr index_t offset(const MultiSpan<N> &spans)
   {
      index_t begin = 0;
      for (index_t d = 0; d < N; ++d)
      {
         begin += spans[d].begin;
      }
      return begin;
   }
} // namespace tensor::details
