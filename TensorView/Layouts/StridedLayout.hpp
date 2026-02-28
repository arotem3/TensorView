#pragma once
#include "TensorView/Layouts/StandardLayout.hpp"
#include "TensorView/Layouts/StridedLayout.hpp"
#include "TensorView/Macros.hpp"

namespace tensor::details
{
   // flat index always assumes F-layout
   template <typename Layout>
   constexpr auto computeProductIndex(const Layout &layout, index_t index)
   {
      TENSOR_DEBUG_ASSERT(index < layout.size(),
                          printf("Linear index %jd out of bounds for layout of size %jd\n",
                                 static_cast<uintmax_t>(index), static_cast<uintmax_t>(layout.size())));

      constexpr index_t N = Layout::numDims();
      std::array<index_t, N> indices;

      for (index_t d = 0; d < N; ++d)
      {
         indices[d] = index % layout.shape(d);
         index /= layout.shape(d);
      }

      return indices;
   }

   struct StridedDimension
   {
      index_t size = 0;
      index_t stride = 1;
   };

   template <index_t NumDims>
   struct StridedLayout
   {
      std::array<StridedDimension, NumDims> dimensions;
      index_t start = 0;

      static constexpr index_t numDims()
      {
         return NumDims;
      }

      constexpr index_t offset() const
      {
         return start;
      }

      constexpr index_t shape(index_t dim) const
      {
         TENSOR_DEBUG_ASSERT(dim < NumDims, printf("Dimension %jd out of bounds for layout with %jd dimensions\n",
                                                   static_cast<uintmax_t>(dim), static_cast<uintmax_t>(NumDims)));

         return dimensions[dim].size;
      }

      constexpr index_t size() const
      {
         index_t s = 1;
         for (index_t d = 0; d < NumDims; ++d)
            s *= dimensions[d].size;
         return s;
      }

      constexpr index_t extent() const
      {
         const index_t sz = size();
         if (sz == 0)
            return start;
         return at(computeProductIndex(*this, sz - 1)) + 1; // last element + 1
      }

      constexpr index_t at(const std::array<index_t, NumDims> &indices) const
      {
         index_t l = start;
         for (index_t d = 0; d < NumDims; ++d)
         {
            TENSOR_DEBUG_ASSERT(
                indices[d] < dimensions[d].size,
                printf("Index %jd out of bounds for dimension %jd of size %jd\n", static_cast<uintmax_t>(indices[d]),
                       static_cast<uintmax_t>(d), static_cast<uintmax_t>(dimensions[d].size)));

            l += indices[d] * dimensions[d].stride;
         }
         return l;
      }

      template <IndexLike... Indices>
      constexpr index_t at(Indices... indices) const
         requires(sizeof...(Indices) == NumDims)
      {
         return at({static_cast<index_t>(indices)...});
      }
   };

   template <index_t TargetDims, index_t SourceDims = TargetDims>
   constexpr StridedLayout<TargetDims> makeStridedLayoutFrom(const StridedLayout<SourceDims> &layout)
   {
      static_assert(SourceDims <= TargetDims,
                    "Source layout must have less than or equal dimensions than target layout.");

      StridedLayout<TargetDims> strided;
      for (index_t d = 0; d < TargetDims; ++d)
         strided.dimensions[d] = (d < SourceDims) ? layout.dimensions[d] : StridedDimension{1, 1};
      strided.start = layout.start;

      return strided;
   }

   template <index_t TargetDims, index_t SourceDims = TargetDims>
   constexpr StridedLayout<TargetDims> makeStridedLayoutFrom(const FLayout<SourceDims> &layout)
   {
      static_assert(SourceDims <= TargetDims,
                    "Source layout must have less than or equal dimensions than target layout.");

      StridedLayout<TargetDims> strided;

      index_t stride = 1;
      for (index_t d = 0; d < TargetDims; ++d)
      {
         strided.dimensions[d] = (d < SourceDims) ? StridedDimension{layout.shape(d), stride} : StridedDimension{1, 1};
         stride *= layout.shape(d);
      }
      strided.start = 0;

      return strided;
   }

   template <index_t TargetDims, index_t... Dimensions>
   constexpr StridedLayout<TargetDims> makeStridedLayoutFrom(const FStaticLayout<Dimensions...> &layout)
   {
      constexpr index_t SourceDims = sizeof...(Dimensions);
      static_assert(SourceDims <= TargetDims,
                    "Source layout must have less than or equal dimensions than target layout.");

      StridedLayout<TargetDims> strided;

      index_t stride = 1;
      for (index_t d = 0; d < TargetDims; ++d)
      {
         strided.dimensions[d] = (d < SourceDims) ? StridedDimension{layout.shape(d), stride} : StridedDimension{1, 1};
         stride *= layout.shape(d);
      }
      strided.start = 0;

      return strided;
   }

   template <index_t TargetDims, index_t SourceDims = TargetDims>
   constexpr StridedLayout<TargetDims> makeStridedLayoutFrom(const CLayout<SourceDims> &layout)
   {
      static_assert(SourceDims <= TargetDims,
                    "Source layout must have less than or equal dimensions than target layout.");

      StridedLayout<TargetDims> strided;

      index_t stride = 1;
      for (index_t d = TargetDims; d-- > 0;)
      {
         strided.dimensions[d] = (d < SourceDims) ? StridedDimension{layout.shape(d), stride} : StridedDimension{1, 1};
         stride *= layout.shape(d);
      }
      strided.start = 0;

      return strided;
   }

   template <index_t TargetDims, index_t... Dimensions>
   constexpr StridedLayout<TargetDims> makeStridedLayoutFrom(const CStaticLayout<Dimensions...> &layout)
   {
      constexpr index_t SourceDims = sizeof...(Dimensions);
      static_assert(SourceDims == TargetDims,
                    "Source layout must have less than or equal dimensions than target layout.");

      StridedLayout<TargetDims> strided;

      index_t stride = 1;
      for (index_t d = TargetDims; d-- > 0;)
      {
         strided.dimensions[d] = (d < SourceDims) ? StridedDimension{layout.shape(d), stride} : StridedDimension{1, 1};
         stride *= layout.shape(d);
      }
      strided.start = 0;

      return strided;
   }
} // namespace tensor::details
