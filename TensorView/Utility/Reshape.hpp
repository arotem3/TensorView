#pragma once
#include "TensorView/Macros.hpp"
#include "TensorView/Tensors/StaticTView.hpp"
#include "TensorView/Tensors/StaticTensor.hpp"
#include "TensorView/Tensors/TView.hpp"
#include "TensorView/Tensors/Tensor.hpp"

namespace tensor::details
{
   template <LinearOrder Order, typename Pattern>
   consteval bool standardPatternCompatible()
   {
      if constexpr (is_standard_pattern<Pattern>)
         return (IsStandardPattern<Pattern>::order == Order);
      else if constexpr (is_static_pattern<Pattern>)
         return (IsStaticPattern<Pattern>::order == Order);
      else
         return false;
   }
} // namespace tensor::details

namespace tensor
{
   template <LinearOrder Order = LinearOrder::F, MemorySpace MemSpace = MemorySpace::Host, typename T,
             IndexLike... Dimensions>
   TENSOR_FUNC auto reshape(T *data, Dimensions... dims)
   {
      return FCTensorView<T, sizeof...(Dimensions), Order, MemSpace>(data, dims...);
   }

   template <LinearOrder Order = LinearOrder::F, typename TensorType, IndexLike... Dimensions>
   auto reshape(const TensorType &tensor, Dimensions... dims)
   {
      using traits = details::TensorTraits<std::remove_cvref_t<TensorType>>;
      using shape = details::StandardPattern<sizeof...(Dimensions), Order>;

      static_assert(details::standardPatternCompatible<Order, typename traits::shape_type>(),
                    "Tensor's shape is not compatible with the requested standard layout order.");

      shape s = details::makeStandardPattern<sizeof...(Dimensions), Order>(dims...);
      return details::makeView(std::move(s), traits::container(tensor));
   }

   template <typename T, LinearOrder Order = LinearOrder::F, MemorySpace MemSpace = MemorySpace::Unspecified,
             IndexLike... Sizes>
   TENSOR_FUNC auto reshape(std::vector<T> &vec, Sizes... dims)
   {
      return reshape<Order, MemSpace>(vec.data(), dims...);
   }

   template <typename T, LinearOrder Order = LinearOrder::F, MemorySpace MemSpace = MemorySpace::Unspecified,
             IndexLike... Sizes>
   TENSOR_FUNC auto reshape(const std::vector<T> &vec, Sizes... dims)
   {
      return reshape<Order, MemSpace>(vec.data(), dims...);
   }

   template <typename T, size_t N, LinearOrder Order = LinearOrder::F, MemorySpace MemSpace = MemorySpace::Unspecified,
             IndexLike... Sizes>
   TENSOR_FUNC auto reshape(std::array<T, N> &arr, Sizes... dims)
   {
      return reshape<Order, MemSpace>(arr.data(), dims...);
   }

   template <typename T, size_t N, LinearOrder Order = LinearOrder::F, MemorySpace MemSpace = MemorySpace::Unspecified,
             IndexLike... Sizes>
   TENSOR_FUNC auto reshape(const std::array<T, N> &arr, Sizes... dims)
   {
      return reshape<Order, MemSpace>(arr.data(), dims...);
   }

} // namespace tensor
