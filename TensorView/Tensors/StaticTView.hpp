#pragma once
#include "TensorView/Shapes/StaticShape.hpp"
#include "TensorView/Tensors/RawView.hpp"

namespace tensor
{
   /**
    * @brief FCStaticView represents a non-owning raw view of a multi-dimensional array of elements of type T, with
    * Fortran or C contiguous layout.
    */
   template <typename T, LinearOrder Order, index_t... Dims>
   class FCStaticView : public details::RawView<details::StaticShape<Order, Dims...>, T, MemorySpace::Unspecified>
   {
   private:
      using base = details::RawView<details::StaticShape<Order, Dims...>, T, MemorySpace::Unspecified>;

   public:
      template <typename U>
      FCStaticView(U *ptr) : base({}, {ptr, base::shape_type::extent()})
      {
      }
   };

   template <typename T, index_t... Dims>
   using CStaticView = FCStaticView<T, LinearOrder::C, Dims...>;

   template <typename T, index_t... Dims>
   using FStaticView = FCStaticView<T, LinearOrder::F, Dims...>;

   template <typename T, index_t... Dims>
   using StaticView = FStaticView<T, Dims...>;
} // namespace tensor

namespace tensor::details
{
   template <typename T, LinearOrder Order, index_t... Dims>
   struct TensorTraits<FCStaticView<T, Order, Dims...>>
       : public TensorTraits<RawView<StaticShape<Order, Dims...>, T, MemorySpace::Unspecified>>
   {
   };
} // namespace tensor::details
