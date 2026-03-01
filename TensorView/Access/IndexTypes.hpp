#pragma once
#include <variant>

#include "TensorView/Macros.hpp"

namespace tensor
{
   /**
    * @brief Represents all indices along a dimension.
    */
   struct All
   {
      constexpr index_t operator[](index_t i) const
      {
         return i;
      }
   };

   /**
    * @brief Represents a range of indices along a dimension.
    */
   struct Range
   {
      index_t begin;
      index_t end;
      index_t stride;

      constexpr Range(index_t b, index_t e, index_t s = 1) : begin(b), end(e), stride(s) {}
      constexpr Range(index_t e) : begin(0), end(e), stride(1) {}

      constexpr index_t size() const
      {
         if (begin >= end)
            return 0;
         return (end - begin + stride - 1) / stride;
      }

      constexpr index_t operator[](index_t i) const
      {
         return begin + i * stride;
      }
   };

   /**
    * @brief Variant type that can hold any index type (All, Range, or a single index).
    */
   using IndexVariant = std::variant<All, Range, index_t>;
} // namespace tensor