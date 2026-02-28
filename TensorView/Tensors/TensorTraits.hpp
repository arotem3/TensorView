#pragma once
#include "TensorView/Containers/ContainerTraits.hpp"
#include "TensorView/Macros.hpp"

namespace tensor::details
{
   template <typename TensorType>
   struct TensorTraits : std::false_type
   {};
   /*
      using tensor_type = TensorType;
      using value_type; // type of tensor elements
      using shape_type; // type of tensor shape
      using container_type; // type of tensor container

      // are all valid instances of this tensor type F-contiguous in memory?
      static constexpr bool contiguous();

      // are elements of this tensor mutable?
      static constexpr bool mutableElements();

      // get the shape of the tensor
      static shape_type shape(const tensor_type &tensor);

      // get a reference to the container of the tensor
      static const container_type &container(const tensor_type &tensor);
      static container_type &container(tensor_type &tensor);
      static container_type container(tensor_type &&tensor);
   */
} // namespace tensor::details