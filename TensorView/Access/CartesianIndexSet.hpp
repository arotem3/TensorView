#pragma once
#include <array>

#include "TensorView/Access/IndexTypes.hpp"
#include "TensorView/Macros.hpp"

namespace tensor::details
{
   /**
    * @brief CartesianIndexSet is now a simple array of IndexVariants.
    */
   template <size_t N>
   using CartesianIndexSet = std::array<tensor::IndexVariant, N>;

   /**
    * @brief Creates a CartesianIndexSet from a tuple.
    */
   template <typename... IndexSets>
   auto makeCartesianIndexSet(const std::tuple<IndexSets...> &indexSets)
   {
      return makeCartesianIndexSetFromTuple(indexSets, std::index_sequence_for<IndexSets...>{});
   }

   template <typename... IndexSets>
   auto makeCartesianIndexSet(std::tuple<IndexSets...> &&indexSets)
   {
      return makeCartesianIndexSetFromTuple(std::move(indexSets), std::index_sequence_for<IndexSets...>{});
   }

   /**
    * @brief AllProduct is an empty marker type representing all dimensions as All.
    * This enables zero-overhead optimization when all indices are "All".
    */
   template <size_t N>
   struct AllProduct
   {};

} // namespace tensor::details
