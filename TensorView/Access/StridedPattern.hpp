#pragma once
#include "TensorView/Access/AccessPattern.hpp"
#include "TensorView/Access/StandardPattern.hpp"
#include "TensorView/Macros.hpp"

namespace tensor::details
{
   template <index_t NumDims>
   using StridedPattern = AccessPattern<StridedLayout<NumDims>, AllProduct<NumDims>>;

   template <index_t NumDims, typename Layout, typename IndexSet>
   constexpr auto makeStridedPatternLike(const AccessPattern<Layout, IndexSet> &shape)
   {
      auto [layout, index_set] = unpackAccessPattern(shape);
      return makeAccessPattern(makeStridedLayoutFrom<NumDims>(layout), index_set);
   }

   template <index_t NumDims, typename Layout, typename IndexSet>
   constexpr auto makeStridedPatternFrom(const AccessPattern<Layout, IndexSet> &shape)
   {
      auto [layout, index_set] = unpackAccessPattern(shape);
      return makeAccessPattern(makeStridedLayoutFrom<NumDims>(layout), index_set);
   }

   template <typename Layout, typename IndexSet>
   constexpr auto makeStridedPatternFrom(const AccessPattern<Layout, IndexSet> &shape)
   {
      auto [layout, index_set] = unpackAccessPattern(shape);
      return makeAccessPattern(makeStridedLayoutFrom<Layout::numDims()>(layout), index_set);
   }

   template <typename ShapeType>
   struct IsStridedPattern : std::false_type
   {};

   template <index_t NumDims>
   struct IsStridedPattern<AccessPattern<StridedLayout<NumDims>, AllProduct<NumDims>>> : std::true_type
   {};

   template <typename ShapeType>
   inline constexpr bool is_strided_pattern = IsStridedPattern<ShapeType>::value;
} // namespace tensor::details
