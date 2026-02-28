#pragma once
#include "TensorView/Access/AccessPattern.hpp"
#include "TensorView/Macros.hpp"

namespace tensor::details
{
   // Fortran-contiguous or C-contiguous standard shape
   template <index_t NumDims, LinearOrder Order>
   using StandardPattern = AccessPattern<StandardLayout<NumDims, Order>, AllProduct<NumDims>>;

   template <index_t NumDims>
   using FPattern = StandardPattern<NumDims, LinearOrder::F>;

   template <index_t NumDims>
   using CPattern = StandardPattern<NumDims, LinearOrder::C>;

   template <index_t NumDims, size_t N>
      requires(static_cast<size_t>(NumDims) == N)
   struct IsFContiguous<AccessPattern<FLayout<NumDims>, AllProduct<N>>> : std::true_type
   {};

   template <index_t NumDims, LinearOrder Order, size_t N>
      requires(static_cast<size_t>(NumDims) == N)
   struct IsStandardPattern<AccessPattern<StandardLayout<NumDims, Order>, AllProduct<N>>> : std::true_type
   {
      static constexpr LinearOrder order = Order;
   };

   template <index_t NumDims, LinearOrder Order, IndexLike... Dimensions>
   constexpr auto makeStandardPattern(Dimensions... dimensions)
   {
      constexpr index_t N = sizeof...(Dimensions);
      static_assert(N <= NumDims, "Number of dimensions must be less than or equal to NumDims.");

      index_t dims[] = {static_cast<index_t>(dimensions)...};
      StandardLayout<NumDims, Order> layout;
      for (index_t d = 0; d < NumDims; ++d)
         layout.dimensions[d] = (d < N) ? dims[d] : 1;

      return StandardPattern<NumDims, Order>(std::move(layout));
   }

   template <index_t NumDims, IndexLike... Dimensions>
   constexpr auto makeFPattern(Dimensions... dimensions)
   {
      return makeStandardPattern<NumDims, LinearOrder::F>(dimensions...);
   }

   template <IndexLike... Dimensions>
   constexpr auto makeFPattern(Dimensions... dimensions)
   {
      constexpr index_t N = sizeof...(Dimensions);
      return FPattern<N>(FLayout<N>{{static_cast<index_t>(dimensions)...}});
   }

   template <index_t NumDims, IndexLike... Dimensions>
   constexpr auto makeCPattern(Dimensions... dimensions)
   {
      return makeStandardPattern<NumDims, LinearOrder::C>(dimensions...);
   }

   template <IndexLike... Dimensions>
   constexpr auto makeCPattern(Dimensions... dimensions)
   {
      constexpr index_t N = sizeof...(Dimensions);
      return CPattern<N>(CLayout<N>{{static_cast<index_t>(dimensions)...}});
   }

   template <index_t NumDims, LinearOrder Order, typename ShapeType>
   constexpr auto makeStandardPatternLike(const ShapeType &shape)
   {
      static_assert(ShapeType::numDims() <= NumDims,
                    "Source shape must have less than or equal dimensions than target shape.");

      StandardLayout<NumDims, Order> layout;
      for (index_t d = 0; d < NumDims; ++d)
         layout.dimensions[d] = (d < ShapeType::numDims()) ? shape.shape(d) : 1;

      return StandardPattern<NumDims, Order>(std::move(layout));
   }

   template <typename ShapeType, LinearOrder Order>
   constexpr auto makeStandardPatternLike(const ShapeType &shape)
   {
      constexpr index_t N = ShapeType::numDims();
      return makeStandardPatternLike<N, Order>(shape);
   }

   template <index_t TargetDims, LinearOrder Order, typename SourcePattern>
      requires(is_standard_pattern<SourcePattern> && IsStandardPattern<SourcePattern>::order == Order)
   constexpr auto makeStandardPatternFrom(const SourcePattern &shape)
   {
      return makeStandardPatternLike<TargetDims, Order>(shape);
   }
} // namespace tensor::details
