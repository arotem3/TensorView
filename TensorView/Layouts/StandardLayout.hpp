#pragma once
#include "TensorView/Macros.hpp"

namespace tensor
{
   enum class LinearOrder
   {
      F, // Fortran order (column-major)
      C  // C order (row-major)
   };
}

namespace tensor::details
{
   template <size_t NumDims>
   constexpr index_t computeFIndex(const std::array<index_t, NumDims> &dimensions,
                                   const std::array<index_t, NumDims> &indices)
   {
      index_t l = 0;

      for (index_t d = NumDims; d-- > 0;)
      {
         TENSOR_DEBUG_ASSERT(
             indices[d] < dimensions[d],
             printf("Index %jd out of bounds for dimension %jd of size %jd\n", static_cast<uintmax_t>(indices[d]),
                    static_cast<uintmax_t>(d), static_cast<uintmax_t>(dimensions[d])));

         l = indices[d] + dimensions[d] * l;
      }

      return l;
   }

   template <size_t NumDims>
   constexpr index_t computeCIndex(const std::array<index_t, NumDims> &dimensions,
                                   const std::array<index_t, NumDims> &indices)
   {
      index_t l = 0;
      for (index_t d = 0; d < NumDims; ++d)
      {
         TENSOR_DEBUG_ASSERT(
             indices[d] < dimensions[d],
             printf("Index %jd out of bounds for dimension %jd of size %jd\n", static_cast<uintmax_t>(indices[d]),
                    static_cast<uintmax_t>(d), static_cast<uintmax_t>(dimensions[d])));

         l = indices[d] + dimensions[d] * l;
      }

      return l;
   }

   template <index_t NumDims, LinearOrder Order>
   class StandardLayout
   {
   public:
      std::array<index_t, NumDims> dimensions;

      static constexpr index_t numDims()
      {
         return NumDims;
      }

      constexpr index_t shape(index_t dim) const
      {
         TENSOR_DEBUG_ASSERT(dim < NumDims, printf("Dimension %jd out of bounds for layout with %jd dimensions\n",
                                                   static_cast<uintmax_t>(dim), static_cast<uintmax_t>(NumDims)));

         return dimensions[dim];
      }

      constexpr index_t offset() const
      {
         return 0;
      }

      constexpr index_t size() const
      {
         index_t s = 1;
         for (index_t d = 0; d < NumDims; ++d)
            s *= dimensions[d];
         return s;
      }

      constexpr index_t extent() const
      {
         return size() + offset();
      }

      constexpr index_t at(const std::array<index_t, NumDims> &indices) const
      {
         if constexpr (Order == LinearOrder::F)
            return computeFIndex<NumDims>(dimensions, indices);
         else
            return computeCIndex<NumDims>(dimensions, indices);
      }

      template <IndexLike... Indices>
      constexpr index_t at(Indices... indices) const
      {
         static_assert(sizeof...(Indices) == NumDims, "Number of indices must match number of dimensions in layout.");
         return at({static_cast<index_t>(indices)...});
      }
   };

   template <index_t NumDims>
   using FLayout = StandardLayout<NumDims, LinearOrder::F>;

   template <index_t NumDims>
   using CLayout = StandardLayout<NumDims, LinearOrder::C>;
} // namespace tensor::details
