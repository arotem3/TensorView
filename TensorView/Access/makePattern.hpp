#pragma once
#include <TensorView/Access/CartesianIndexSet.hpp>
#include <TensorView/Macros.hpp>

namespace tensor::details
{
   template <typename Layout, typename IndexSet>
   struct AccessPattern;

   template <typename LayoutType, size_t N>
   auto makeAccessPattern(LayoutType &&layout, CartesianIndexSet<N> indices)
   {
      using Layout = std::remove_cvref_t<LayoutType>;
      static_assert(Layout::numDims() == N, "Layout and IndexSet must have same number of dimensions.");

      if constexpr (Layout::numDims() == 0)
         return layout.offset();
      else
         return AccessPattern<Layout, CartesianIndexSet<N>>{std::forward<LayoutType>(layout), std::move(indices)};
   }

   template <typename LayoutType, size_t N>
   auto makeAccessPattern(LayoutType &&layout, AllProduct<N>)
   {
      using Layout = std::remove_cvref_t<LayoutType>;
      static_assert(Layout::numDims() == N, "Layout and IndexSet must have same number of dimensions.");

      if constexpr (Layout::numDims() == 0)
         return layout.offset();
      else
      {
         return AccessPattern<Layout, AllProduct<N>>{std::forward<LayoutType>(layout)};
      }
   }
} // namespace tensor::details
