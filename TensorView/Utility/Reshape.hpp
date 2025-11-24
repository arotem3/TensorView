#pragma once
#include "TensorView/Macros.hpp"
#include "TensorView/Tensors/TView.hpp"
#include "TensorView/Tensors/Tensor.hpp"

namespace tensor
{
   template <LinearOrder Order = LinearOrder::F, MemorySpace MemSpace = MemorySpace::Host, typename T,
             TENSOR_INT_LIKE... Sizes>
   TENSOR_FUNC auto reshape(T *data, Sizes... dims)
   {
      using view = FCTensorView<T, sizeof...(Sizes), Order, MemSpace>;

      using traits = details::TensorTraits<view>;
      using shape = typename traits::shape_type;
      using container = typename traits::container_type;

      shape s(dims...);
      container c(data, s.size());

      return view(std::move(s), std::move(c));
   }

   template <typename T, index_t NumDims, LinearOrder Order, MemorySpace MemSpace, TENSOR_INT_LIKE... Sizes>
   TENSOR_FUNC auto reshape(FCTensorView<T, NumDims, Order, MemSpace> &tensor, Sizes... shape)
   {
      return reshape<Order, MemSpace>(tensor.data(), shape...);
   }

   template <typename T, index_t NumDims, LinearOrder Order, MemorySpace MemSpace, TENSOR_INT_LIKE... Sizes>
   TENSOR_FUNC auto reshape(const FCTensorView<T, NumDims, Order, MemSpace> &tensor, Sizes... shape)
   {
      return reshape<Order, MemSpace>(tensor.data(), shape...);
   }

   template <typename T, index_t NumDims, LinearOrder Order, MemorySpace MemSpace, TENSOR_INT_LIKE... Sizes>
   auto reshape(FCTensor<T, NumDims, Order, MemSpace> &tensor, Sizes... dims)
   {
      using view = PView<T, sizeof...(Sizes), Order, MemSpace>;

      using from_traits = details::TensorTraits<FCTensor<T, NumDims, Order, MemSpace>>;
      using to_traits = details::TensorTraits<view>;
      using shape = typename to_traits::shape_type;
      using container = typename to_traits::container_type;

      shape s(dims...);
      container c = details::ContainerTraits<container>::from(from_traits::container(tensor));
      return view(std::move(s), std::move(c));
   }

   template <typename T, index_t NumDims, LinearOrder Order, MemorySpace MemSpace, TENSOR_INT_LIKE... Sizes>
   auto reshape(const FCTensor<T, NumDims, Order, MemSpace> &tensor, Sizes... dims)
   {
      using view = PView<const T, sizeof...(Sizes), Order, MemSpace>;

      using from_traits = details::TensorTraits<FCTensor<T, NumDims, Order, MemSpace>>;
      using to_traits = details::TensorTraits<view>;
      using shape = typename to_traits::shape_type;
      using container = typename to_traits::container_type;

      shape s(dims...);
      container c = details::ContainerTraits<container>::from(from_traits::container(tensor));
      return view(std::move(s), std::move(c));
   }
} // namespace tensor
