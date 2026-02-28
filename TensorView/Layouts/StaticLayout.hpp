#pragma once
#include "TensorView/Layouts/StandardLayout.hpp"
#include "TensorView/Macros.hpp"

namespace tensor::details
{
   template <LinearOrder Order, index_t... Dimensions>
   struct StaticLayout
   {
      static constexpr index_t numDims()
      {
         return sizeof...(Dimensions);
      }

      constexpr index_t shape(index_t dim) const
      {
         TENSOR_DEBUG_ASSERT(dim < numDims(),
                             printf("Dimension %jd out of bounds for StaticLayout with %jd dimensions\n",
                                    static_cast<uintmax_t>(dim), static_cast<uintmax_t>(numDims())));
         constexpr index_t dims[] = {Dimensions...};
         return dims[dim];
      }

      static constexpr index_t size()
      {
         return (1 * ... * Dimensions);
      }

      constexpr index_t offset() const
      {
         return 0;
      }

      constexpr index_t extent() const
      {
         return size();
      }

      constexpr index_t at(const std::array<index_t, sizeof...(Dimensions)> &indices) const
      {
         if constexpr (Order == LinearOrder::F)
            return computeFIndex({Dimensions...}, indices);
         else
            return computeCIndex({Dimensions...}, indices);
      }

      template <IndexLike... Indices>
      constexpr index_t at(Indices... indices) const
      {
         static_assert(sizeof...(Indices) == sizeof...(Dimensions),
                       "Number of indices must match number of dimensions in layout.");
         return at({static_cast<index_t>(indices)...});
      }
   };

   template <index_t... Dimensions>
   using FStaticLayout = StaticLayout<LinearOrder::F, Dimensions...>;

   template <index_t... Dimensions>
   using CStaticLayout = StaticLayout<LinearOrder::C, Dimensions...>;
} // namespace tensor::details
