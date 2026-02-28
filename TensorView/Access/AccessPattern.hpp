#pragma once
#include "TensorView/Access/CartesianIndexSet.hpp"
#include "TensorView/Access/Compose.hpp"
#include "TensorView/Access/IndexTypes.hpp"
#include "TensorView/Access/SimplifyIndexExpr.hpp"
#include "TensorView/Access/computeLinearIndex.hpp"
#include "TensorView/Access/makePattern.hpp"
#include "TensorView/Macros.hpp"

namespace tensor::details
{
   /**
    * @brief Helper to compose CartesianIndexSet with variadic indices.
    */
   template <size_t N, typename... Indices>
   auto composeIndices(const CartesianIndexSet<N> &index_set, Indices &&...indices)
      requires(sizeof...(Indices) == N)
   {
      return composeIndicesTupleImpl(index_set, std::forward_as_tuple(indices...), std::make_index_sequence<N>{});
   }

   template <size_t N, typename IndexTuple, size_t... I>
   auto composeIndicesTupleImpl(const CartesianIndexSet<N> &index_set, const IndexTuple &indices,
                                std::index_sequence<I...>)
   {
      return std::make_tuple(compose(index_set[I], std::get<I>(indices))...);
   }

   // Forward declaration
   template <typename Layout, typename IndexSet>
   struct AccessPattern;

   /**
    * @brief Specialization for CartesianIndexSet - standard composition path.
    */
   template <typename Layout, size_t N>
   struct AccessPattern<Layout, CartesianIndexSet<N>>
   {
   private:
      Layout layout;
      CartesianIndexSet<N> index_set;

   public:
      static constexpr index_t NumDims = Layout::numDims();

      static_assert(NumDims == std::tuple_size_v<CartesianIndexSet<N>>,
                    "Layout and IndexSet must have the same number of dimensions in AccessPattern.");

      static constexpr index_t numDims()
      {
         return NumDims;
      }

      constexpr index_t shape(index_t dim) const
      {
         return layout.shape(dim);
      }

      constexpr index_t size() const
      {
         return layout.size();
      }

      constexpr index_t extent() const
      {
         return layout.extent();
      }

      constexpr index_t offset() const
      {
         return layout.offset();
      }

      constexpr index_t operator[](index_t i) const
      {
         return computeLinearIndex(i, layout, index_set);
      }

      template <typename... Indices>
         requires(sizeof...(Indices) == NumDims)
      auto at(Indices &&...indices) const
      {
         return simplifyIndexExpr(layout, composeIndices(index_set, std::forward<Indices>(indices)...));
      }

      AccessPattern() = default;
      ~AccessPattern() = default;
      AccessPattern(const AccessPattern &) = default;
      AccessPattern(AccessPattern &&) = default;
      AccessPattern &operator=(const AccessPattern &) = default;
      AccessPattern &operator=(AccessPattern &&) = default;

      explicit AccessPattern(Layout &&layout_, CartesianIndexSet<N> &&index_set_)
          : layout(std::move(layout_)), index_set(std::move(index_set_))
      {}

   public:
      friend auto unpackAccessPattern(AccessPattern<Layout, CartesianIndexSet<N>> &&pattern)
      {
         return std::make_tuple(std::move(pattern.layout), std::move(pattern.index_set));
      }

      friend auto unpackAccessPattern(const AccessPattern<Layout, CartesianIndexSet<N>> &pattern)
      {
         return std::make_tuple(std::ref(pattern.layout), std::ref(pattern.index_set));
      }

      friend auto unpackAccessPattern(AccessPattern<Layout, CartesianIndexSet<N>> &pattern)
      {
         return std::make_tuple(std::ref(pattern.layout), std::ref(pattern.index_set));
      }
   };

   /**
    * @brief Specialization of AccessPattern for AllProduct - optimized with zero overhead.
    * When all indices are All, we don't need index composition or any extra storage.
    */
   template <typename Layout, size_t N>
   struct AccessPattern<Layout, AllProduct<N>>
   {
   private:
      Layout layout;

   public:
      static constexpr index_t NumDims = Layout::numDims();

      static_assert(NumDims == N, "Layout and AllProduct must have the same number of dimensions.");

      static constexpr index_t numDims()
      {
         return NumDims;
      }

      constexpr index_t shape(index_t dim) const
      {
         return layout.shape(dim);
      }

      constexpr index_t size() const
      {
         return layout.size();
      }

      constexpr index_t extent() const
      {
         return layout.extent();
      }

      constexpr index_t offset() const
      {
         return layout.offset();
      }

      /**
       * @brief Optimized operator[] for AllProduct - just returns the linear index as-is.
       */
      constexpr index_t operator[](index_t i) const
      {
         return computeLinearIndex(i, layout, AllProduct<N>{});
      }

      /**
       * @brief at() for AllProduct - compose given indices with All{} and simplify.
       * Since all indices in AllProduct are All, composing with given indices yields the given indices.
       */
      template <typename... Indices>
         requires(sizeof...(Indices) == NumDims)
      auto at(Indices &&...indices) const
      {
         // Create a CartesianIndexSet from the given indices
         auto composed = std::make_tuple(std::forward<Indices>(indices)...);
         return simplifyIndexExpr(layout, composed);
      }

      AccessPattern() = default;
      ~AccessPattern() = default;
      AccessPattern(const AccessPattern &) = default;
      AccessPattern(AccessPattern &&) = default;
      AccessPattern &operator=(const AccessPattern &) = default;
      AccessPattern &operator=(AccessPattern &&) = default;

      explicit AccessPattern(Layout &&layout_) : layout(std::move(layout_)) {}

   public:
      friend auto unpackAccessPattern(AccessPattern<Layout, AllProduct<N>> &&pattern)
      {
         return std::make_tuple(std::move(pattern.layout), AllProduct<N>{});
      }

      friend auto unpackAccessPattern(const AccessPattern<Layout, AllProduct<N>> &pattern)
      {
         return std::make_tuple(std::ref(pattern.layout), AllProduct<N>{});
      }

      friend auto unpackAccessPattern(AccessPattern<Layout, AllProduct<N>> &pattern)
      {
         return std::make_tuple(std::ref(pattern.layout), AllProduct<N>{});
      }
   };

   template <typename Shape>
   struct IsFContiguous : std::false_type
   {};

   template <typename Shape>
   inline constexpr bool is_contiguous_access_pattern = IsFContiguous<Shape>::value;

   template <typename Shape>
   struct IsStandardPattern : std::false_type
   {};

   template <typename Shape>
   inline constexpr bool is_standard_pattern = IsStandardPattern<Shape>::value;
} // namespace tensor::details
