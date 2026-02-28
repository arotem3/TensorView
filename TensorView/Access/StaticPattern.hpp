#pragma once
#include "TensorView/Access/AccessPattern.hpp"
#include "TensorView/Layouts/StaticLayout.hpp"
#include "TensorView/Macros.hpp"

namespace tensor::details
{
   template <LinearOrder Order, index_t... Dimensions>
   using StaticPattern = AccessPattern<StaticLayout<Order, Dimensions...>, AllProduct<sizeof...(Dimensions)>>;

   template <index_t... Dimensions>
   using FStaticPattern = StaticPattern<LinearOrder::F, Dimensions...>;

   template <index_t... Dimensions>
   using CStaticPattern = StaticPattern<LinearOrder::C, Dimensions...>;

   template <index_t... Dimensions>
   struct IsFContiguous<AccessPattern<FStaticLayout<Dimensions...>, AllProduct<sizeof...(Dimensions)>>> : std::true_type
   {};

   template <typename T>
   struct IsStaticPattern : std::false_type
   {};

   template <LinearOrder Order, index_t... Dimensions>
   struct IsStaticPattern<AccessPattern<StaticLayout<Order, Dimensions...>, AllProduct<sizeof...(Dimensions)>>>
       : std::true_type
   {
      static constexpr LinearOrder order = Order;
   };

   template <typename T>
   inline constexpr bool is_static_pattern = IsStaticPattern<T>::value;
} // namespace tensor::details
