#pragma once
#include "TensorView/Access/AccessPattern.hpp"
#include "TensorView/Access/StandardPattern.hpp"
#include "TensorView/Access/StaticPattern.hpp"
#include "TensorView/Access/StridedPattern.hpp"
#include "TensorView/Macros.hpp"

namespace tensor::details
{
   /**
    * @brief Creates a TargetPattern with the same dimensions of the SourcePattern.
    * Weaker than makePatternFrom since the memory layout need not be preserved.
    */
   template <typename TargetPattern, typename SourcePattern>
   auto makePatternLike(const SourcePattern &shape)
   {
      constexpr index_t tN = TargetPattern::numDims();
      constexpr index_t sN = SourcePattern::numDims();

      static_assert(tN >= sN, "Target shape must have at least as many dimensions as source shape.");

      if constexpr (std::is_same_v<TargetPattern, SourcePattern>)
         return shape;
      else if constexpr (is_standard_pattern<TargetPattern>)
         return makeStandardPatternLike<tN, IsStandardPattern<TargetPattern>::order>(shape);
      else if constexpr (is_strided_pattern<TargetPattern>)
         return makeStridedPatternLike<tN>(shape);
      else if constexpr (is_static_pattern<TargetPattern>)
      {
         TENSOR_CHECK(sameShape(shape, TargetPattern{}),
                      { printf("Cannot make static pattern from shape with different dimensions.\n"); });
         return TargetPattern{};
      }
      else
      {
         static_assert(tN == -1, "Unsupported target shape type in makePatternLike.");
      }
   }

   /**
    * @brief Creates a TargetPattern with the same dimensions and memory layout of the SourcePattern.
    * Stronger than makePatternLike since the memory layout is preserved.
    */
   template <typename TargetPattern, typename SourcePattern>
   auto makePatternFrom(const SourcePattern &shape)
   {
      constexpr index_t tN = TargetPattern::numDims();
      constexpr index_t sN = SourcePattern::numDims();

      static_assert(tN >= sN, "Target shape must have at least as many dimensions as source shape.");

      if constexpr (std::is_same_v<TargetPattern, SourcePattern>)
         return shape;
      else if constexpr (is_standard_pattern<TargetPattern>)
         return makeStandardPatternFrom<tN, IsStandardPattern<TargetPattern>::order>(shape);
      else if constexpr (is_strided_pattern<TargetPattern>)
         return makeStridedPatternFrom<tN>(shape);
      else if constexpr (is_static_pattern<TargetPattern> && is_static_pattern<SourcePattern> &&
                         IsStaticPattern<SourcePattern>::order == IsStaticPattern<TargetPattern>::order)
      {
         TENSOR_CHECK(sameShape(shape, TargetPattern{}),
                      { printf("Cannot make static pattern from shape with different dimensions.\n"); });
         return TargetPattern{};
      }
      else if constexpr (is_static_pattern<TargetPattern> && is_standard_pattern<SourcePattern> &&
                         IsStandardPattern<SourcePattern>::order == IsStaticPattern<TargetPattern>::order)
      {
         TENSOR_CHECK(sameShape(shape, TargetPattern{}),
                      { printf("Cannot make static pattern from shape with different dimensions.\n"); });
         return TargetPattern{};
      }
      else
      {
         static_assert(tN == -1, "Unsupported target shape type in makePatternFrom.");
      }
   }

   template <typename TargetPattern, IndexLike... Dimensions>
   constexpr TargetPattern makePattern(Dimensions... dims)
   {
      constexpr index_t tN = TargetPattern::numDims();
      constexpr index_t sN = sizeof...(Dimensions);

      static_assert(tN >= sN, "Target shape must have at least as many dimensions as source shape.");

      if constexpr (std::is_same_v<TargetPattern, CPattern<tN>>)
         return makeCPattern<tN>(dims...);
      else if constexpr (std::is_same_v<TargetPattern, FPattern<tN>>)
         return makeFPattern<tN>(dims...);
      else
      {
         static_assert(tN == -1, "Unsupported target shape type in makePattern.");
      }
   }
} // namespace tensor::details
