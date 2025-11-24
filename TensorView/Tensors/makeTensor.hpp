#pragma once
#include "TensorView/Macros.hpp"
#include "TensorView/Tensors/Tensor.hpp"

namespace tensor
{
   /**
    * @brief creates a Tensor of the specified shape.
    */
   template <typename Scalar, LinearOrder Order = LinearOrder::F, MemorySpace MemSpace = MemorySpace::Host,
             TENSOR_INT_LIKE... Dims>
   auto makeTensor(Dims... dims)
   {
      constexpr size_t numDims = sizeof...(Dims);
      return FCTensor<Scalar, numDims, Order, MemSpace>(std::forward<Dims>(dims)...);
   }

   namespace details
   {
      template <typename value_type_out, LinearOrder Order, MemorySpace MemSpace, typename TensorType, size_t... I>
      auto makeTensorLikeImpl(const TensorType &tensor, std::index_sequence<I...>)
      {
         return makeTensor<value_type_out, Order, MemSpace>(tensor.shape(I)...);
      }
   } // namespace details

   /**
    * @brief creates a Tensor with the same shape as the input tensor.
    */
   template <typename value_type_out = void, LinearOrder Order = LinearOrder::F,
             MemorySpace MemSpace = MemorySpace::Host, typename TensorType>
   auto makeTensorLike(const TensorType &tensor)
   {
      using value_type = std::conditional_t<std::is_same_v<value_type_out, void>,
                                            std::decay_t<typename TensorType::value_type>, value_type_out>;
      constexpr size_t NumDims = TensorType::numDims();
      return details::makeTensorLikeImpl<value_type, Order, MemSpace>(tensor, std::make_index_sequence<NumDims>());
   }

} // namespace tensor
