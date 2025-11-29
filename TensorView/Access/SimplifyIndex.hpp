#pragma once
#include "TensorView/Access/Span.hpp"
#include "TensorView/Macros.hpp"

namespace tensor::details
{
   TENSOR_FUNC index_t simplifyIndex(index_t index, [[maybe_unused]] index_t dim, [[maybe_unused]] index_t dim_size)
   {
      TENSOR_DEBUG_ASSERT(index < dim_size, printf("Index %ju is out of range for dimension %ju with size %ju.\n",
                                                   static_cast<uintmax_t>(index), static_cast<uintmax_t>(dim),
                                                   static_cast<uintmax_t>(dim_size)));

      return index;
   }

   TENSOR_FUNC const Span &simplifyIndex(const Span &s, [[maybe_unused]] index_t dim, [[maybe_unused]] index_t dim_size)
   {
      TENSOR_DEBUG_ASSERT(
          s.end <= dim_size && s.begin <= s.end,
          printf("Span( %ju, %ju ) is out of range for dimension %ju with size %ju.\n", static_cast<uintmax_t>(s.begin),
                 static_cast<uintmax_t>(s.end), static_cast<uintmax_t>(dim), static_cast<uintmax_t>(dim_size)));

      TENSOR_DEBUG_ASSERT(s.stride >= 1, printf("Span stride %ju is invalid for dimension %ju; must be >= 1.\n",
                                                static_cast<uintmax_t>(s.stride), static_cast<uintmax_t>(dim)));

      return s;
   }

   constexpr Span simplifyIndex(All, index_t, index_t dim_size)
   {
      return Span(0, dim_size);
   }
} // namespace tensor::details
