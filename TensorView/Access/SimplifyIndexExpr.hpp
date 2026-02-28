#pragma once
#include "TensorView/Access/CartesianIndexSet.hpp"
#include "TensorView/Access/Compose.hpp"
#include "TensorView/Access/makePattern.hpp"
#include "TensorView/Layouts/StridedLayout.hpp"
#include "TensorView/Macros.hpp"

namespace tensor::details
{
   // Marker type for a dimension that has been collapsed during simplification.
   // Used only for type discrimination in simplifyIndexExprImpl, never composed.
   struct CollapsedDimMarker
   {};

   template <typename T>
   constexpr index_t getLowerBound(const T &index)
   {
      if constexpr (IndexLike<T>)
         return static_cast<index_t>(index);
      else if constexpr (std::is_same_v<T, Range>)
         return index.begin;
      else if constexpr (std::is_same_v<T, All>)
         return 0;
      else
      {
         TENSOR_UNREACHABLE();
         return index_t{0};
      }
   }

   template <typename IndexType>
      requires requires(IndexType idx) { idx.size(); }
   constexpr bool validateSize(const IndexType &index, index_t dimension_size)
   {
      return index.size() == dimension_size;
   }

   constexpr bool validateSize(All, index_t)
   {
      return true;
   }

   constexpr bool validateSize(index_t, index_t)
   {
      return true; // A single index is always valid (will be validated at runtime elsewhere)
   }

   /**
    * @brief Simplifies the given index in the context of a strided layout dimension.
    * If the index has an offset then the offset is set accordingly and the index is shifted to start from zero.
    * If the index has a stride or size different from the dimension, it sets the dimension accordingly and returns
    * returns an equivalent index which is not strided.
    */
   constexpr All simplifyIndexInStridedLayout(index_t &offset, StridedDimension &dimension, const Range &range)
   {
      TENSOR_CHECK(
          range.end <= dimension.size,
          printf("Range( %jd, %jd ) out of bounds for dimension size %jd\n", static_cast<uintmax_t>(range.begin),
                 static_cast<uintmax_t>(range.end), static_cast<uintmax_t>(dimension.size)));

      offset = range.begin;
      dimension.size = range.size();
      dimension.stride *= range.stride;
      return All{};
   }

   constexpr CollapsedDimMarker simplifyIndexInStridedLayout(index_t &offset, StridedDimension &, index_t index)
   {
      offset = index;
      return CollapsedDimMarker{};
   }

   constexpr All simplifyIndexInStridedLayout(index_t &offset, StridedDimension &, All)
   {
      offset = 0;
      return All{};
   }

   template <typename T>
   using SimplifiedIndexType = decltype(simplifyIndexInStridedLayout(
       std::declval<index_t &>(), std::declval<StridedDimension &>(), std::declval<const T &>()));

   template <typename IndexTuple, size_t... I>
   consteval index_t countDims(std::index_sequence<I...>)
   {
      using simpl = decltype(std::make_tuple(SimplifiedIndexType<std::tuple_element_t<I, IndexTuple>>{}...));
      return (0 + ... + (std::is_same_v<std::tuple_element_t<I, simpl>, CollapsedDimMarker> ? 0 : 1));
   }

   template <typename Layout, typename IndexTuple, size_t... I>
   constexpr bool validateSizes(const Layout &layout, const IndexTuple &indices, std::index_sequence<I...>)
   {
      constexpr index_t N = Layout::numDims();
      static_assert(sizeof...(I) == N, "Number of indices must match number of layout dimensions.");

      return (validateSize(std::get<I>(indices), layout.shape(I)) && ... && true);
   }

   /**
    * @brief Check if all elements of an IndexTuple are All (compile-time check on tuple element types).
    */
   template <typename IndexTuple, size_t... I>
   consteval bool allAreAll(std::index_sequence<I...>)
   {
      return (std::is_same_v<std::tuple_element_t<I, IndexTuple>, All> && ... && true);
   }

   template <typename Layout, typename IndexTuple, size_t... I>
   constexpr auto simplifyIndexExprImpl(const Layout &embedding, const IndexTuple &indices, std::index_sequence<I...>)
   {
      constexpr index_t InputDims = Layout::numDims();
      static_assert(sizeof...(I) == InputDims, "Number of indices must match number of layout dimensions.");

      constexpr index_t OutputDims = countDims<IndexTuple>(std::index_sequence<I...>{});

      const auto st = makeStridedLayoutFrom<InputDims>(embedding);
      std::array<index_t, InputDims> offsets = {0};
      StridedLayout<OutputDims> collapsed;

      // Process each dimension: simplify and collect non-collapsed indices
      index_t i = 0, dim = 0;
      auto processed_index_set = std::tuple_cat([&]() {
         StridedDimension sdim = st.dimensions[i];
         auto index = simplifyIndexInStridedLayout(offsets[i++], sdim, std::get<I>(indices));

         if constexpr (std::is_same_v<decltype(index), CollapsedDimMarker>)
            return std::tuple<>{};
         else
         {
            collapsed.dimensions[dim++] = sdim;
            return std::make_tuple(std::move(index));
         }
      }()...);

      collapsed.start = embedding.offset() + st.at(offsets);

      TENSOR_DEBUG_ASSERT(
          validateSizes(collapsed, processed_index_set, std::make_index_sequence<OutputDims>{}),
          printf("The dimensions of the processed index set do not match the collapsed layout dimensions.\n"));

      if constexpr (allAreAll<decltype(processed_index_set)>(std::make_index_sequence<OutputDims>{}))
         return makeAccessPattern(std::move(collapsed), AllProduct<OutputDims>{});
      else
         return makeAccessPattern(std::move(collapsed), makeCartesianIndexSet(std::move(processed_index_set)));
   }

   /**
    * @brief Constructs a shape by simplifying the given index tuple in the context of the embedding layout.
    */
   template <typename Layout, typename IndexTuple>
   constexpr auto simplifyIndexExpr(const Layout &embedding, const IndexTuple &indices)
   {
      constexpr index_t InputDims = Layout::numDims();
      using seq = std::make_index_sequence<InputDims>;
      constexpr index_t OutputDims = countDims<IndexTuple>(seq{});

      if constexpr (allAreAll<IndexTuple>(seq{}))
         return makeAccessPattern(embedding, AllProduct<OutputDims>{});
      else if constexpr (OutputDims == 0)
         return std::apply([&](auto... idx) { return embedding.at(idx...); }, indices);
      else
         return simplifyIndexExprImpl(embedding, indices, seq{});
   }
} // namespace tensor::details
