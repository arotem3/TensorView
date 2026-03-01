#pragma once
#include "TensorView/Macros.hpp"
#include "TensorView/Tensors/Tensor.hpp"

namespace tensor
{
   /**
    * @brief creates a Tensor of the specified shape.
    */
   template <typename Scalar, LinearOrder Order = LinearOrder::F, MemorySpace MemSpace = MemorySpace::Host,
             IndexLike... Dims>
   auto makeTensor(Dims... dims)
   {
      constexpr size_t numDims = sizeof...(Dims);
      return FCTensor<Scalar, numDims, Order, MemSpace>(std::forward<Dims>(dims)...);
   }

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
      using shape_type = details::StandardPattern<NumDims, Order>;
      auto shape = details::makePatternLike<shape_type>(tensor);
      details::SharedContainer<value_type, MemSpace> container(shape.extent());
      return details::makeTensorBase<true>(std::move(shape), std::move(container));
   }

} // namespace tensor
