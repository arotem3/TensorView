#pragma once
#include "TensorView/Access/CartesianIndexSet.hpp"
#include "TensorView/Access/Compose.hpp"
#include "TensorView/Access/IndexTypes.hpp"
#include "TensorView/Access/SimplifyIndexExpr.hpp"
#include "TensorView/Layouts/StaticLayout.hpp"
#include "TensorView/Layouts/StridedLayout.hpp"
#include "TensorView/Macros.hpp"

namespace tensor::details
{
   /**
    * @brief Optimized overload for AllProduct on contiguous StandardLayout.
    */
   template <index_t NumDims, size_t N>
   constexpr index_t computeLinearIndex(index_t index, const FLayout<NumDims> &layout, const AllProduct<N> &)
   {
      static_assert(static_cast<size_t>(NumDims) == N,
                    "AllProduct dimensionality must match StandardLayout dimensionality.");

      TENSOR_DEBUG_ASSERT(index < layout.size(),
                          printf("Linear index %jd out of bounds for layout of size %jd\n",
                                 static_cast<uintmax_t>(index), static_cast<uintmax_t>(layout.size())));

      return layout.offset() + index;
   }

   /**
    * @brief Optimized overload for AllProduct on contiguous StaticLayout.
    */
   template <index_t... Dimensions, size_t N>
   constexpr index_t computeLinearIndex(index_t index, const FStaticLayout<Dimensions...> &layout,
                                        const AllProduct<N> &)
   {
      static_assert(sizeof...(Dimensions) == N, "AllProduct dimensionality must match StaticLayout dimensionality.");

      TENSOR_DEBUG_ASSERT(index < layout.size(),
                          printf("Linear index %jd out of bounds for layout of size %jd\n",
                                 static_cast<uintmax_t>(index), static_cast<uintmax_t>(layout.size())));

      return layout.offset() + index;
   }

   /**
    * @brief Optimized overload for AllProduct - no composition needed, just return offset + index.
    */
   template <typename Layout, size_t N>
   constexpr index_t computeLinearIndex(index_t index, const Layout &layout, const AllProduct<N> &)
   {
      TENSOR_DEBUG_ASSERT(index < layout.size(),
                          printf("Linear index %jd out of bounds for layout of size %jd\n",
                                 static_cast<uintmax_t>(index), static_cast<uintmax_t>(layout.size())));

      return layout.at(computeProductIndex(layout, index));
   }

   /**
    * @brief Helper implementation to compose indices from an array.
    */
   template <size_t N, size_t... I>
   auto composeIndicesArrayImpl(const CartesianIndexSet<N> &index_set, const std::array<index_t, N> &indices,
                                std::index_sequence<I...>)
   {
      return std::make_tuple(compose(index_set[I], indices[I])...);
   }

   /**
    * @brief Composes a CartesianIndexSet with an array of indices.
    * Returns a tuple of IndexVariants (All, Range, or index_t).
    */
   template <size_t N>
   auto composeIndices(const CartesianIndexSet<N> &index_set, const std::array<index_t, N> &indices)
   {
      return composeIndicesArrayImpl(index_set, indices, std::make_index_sequence<N>{});
   }

   // General case for computing linear index from a CartesianIndexSet
   template <typename Layout, size_t N>
   constexpr index_t computeLinearIndex(index_t index, const Layout &layout, const CartesianIndexSet<N> &index_set)
   {
      auto indices = computeProductIndex(layout, index);
      auto composed = composeIndices(index_set, indices);
      return simplifyIndexExpr(layout, composed);
   }
} // namespace tensor::details
