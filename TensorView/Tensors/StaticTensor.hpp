#pragma once
#include "TensorView/Access/StaticPattern.hpp"
#include "TensorView/Containers/StaticContainer.hpp"
#include "TensorView/Tensors/TensorBase.hpp"

namespace tensor
{
   /**
    * @brief A tensor with static shape and static storage.
    */
   template <typename T, LinearOrder Order, index_t... Dims>
   using FCStaticTensor =
       details::TensorBase<details::StaticPattern<Order, Dims...>, details::StaticContainer<T, (1 * ... * Dims)>, true>;

   template <typename T, index_t... Dims>
   using FStaticTensor = FCStaticTensor<T, LinearOrder::F, Dims...>;

   template <typename T, index_t... Dims>
   using CStaticTensor = FCStaticTensor<T, LinearOrder::C, Dims...>;

   template <typename T, index_t... Dims>
   using StaticTensor = FStaticTensor<T, Dims...>;
} // namespace tensor
