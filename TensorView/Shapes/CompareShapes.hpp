#pragma once
#include "TensorView/Macros.hpp"

namespace tensor::details
{
   /**
    * @brief Checks if two tensors have compatible shapes for broadcasting.
    * 0-dimensional expressions (scalars) are compatible with any shape (they broadcast).
    * For non-scalar tensors:
    * - The leading dimensions must match exactly
    * - Any trailing dimensions (in the higher-dimensional tensor) must be singleton (1)
    */
   template <typename TensorA, typename TensorB>
   constexpr bool sameShape(const TensorA &a, const TensorB &b)
   {
      // 0-dimensional expressions (scalars) are compatible with anything
      if (a.numDims() == 0 || b.numDims() == 0)
         return true;

      const index_t n = std::min(a.numDims(), b.numDims());

      for (index_t i = 0; i < n; ++i)
      {
         if (a.shape(i) != b.shape(i))
            return false;
      }

      for (index_t i = n; i < a.numDims(); ++i)
      {
         if (a.shape(i) != 1)
            return false;
      }

      for (index_t i = n; i < b.numDims(); ++i)
      {
         if (b.shape(i) != 1)
            return false;
      }

      return true;
   }
} // namespace tensor::details

#define TENSOR_REQUIRE_EQUAL_SHAPES(a, b)                    \
   TENSOR_CHECK(details::sameShape(a, b), {                  \
      printf("%s: A shape (", __func__);                     \
      for (index_t i = 0; i < a.numDims(); ++i)              \
         printf("%ju,", static_cast<uintmax_t>(a.shape(i))); \
      printf(") does not match B shape (");                  \
      for (index_t i = 0; i < b.numDims(); ++i)              \
         printf("%ju,", static_cast<uintmax_t>(b.shape(i))); \
      printf(").\n");                                        \
   });
