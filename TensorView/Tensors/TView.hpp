#pragma once
#include "TensorView/Access/StandardPattern.hpp"
#include "TensorView/Containers/ViewContainer.hpp"
#include "TensorView/Macros.hpp"
#include "TensorView/Tensors/TensorBase.hpp"

namespace tensor
{
   /**
    * @brief TensorView represents a non-owning raw view of a multi-dimensional array of elements of type T in the
    * specified memory space, with Fortran or C contiguous layout.
    *
    * @tparam T The type of elements in the tensor
    * @tparam numDims The number of dimensions of the tensor
    * @tparam Order the memory layout order (C or Fortran)
    * @tparam MemSpace the memory space where the tensor data is stored
    */
   template <typename T, index_t numDims, LinearOrder Order, MemorySpace MemSpace = MemorySpace::Host>
   using FCTensorView =
       details::TensorBase<details::StandardPattern<numDims, Order>, details::ViewContainer<T, MemSpace>, false>;

   template <typename T, index_t numDims, MemorySpace MemSpace = MemorySpace::Host>
   using FTensorView = FCTensorView<T, numDims, LinearOrder::F, MemSpace>;

   template <typename T, index_t numDims, MemorySpace MemSpace = MemorySpace::Host>
   using CTensorView = FCTensorView<T, numDims, LinearOrder::C, MemSpace>;

   template <typename T, index_t numDims, MemorySpace MemSpace = MemorySpace::Host>
   using TensorView = FTensorView<T, numDims, MemSpace>;
} // namespace tensor
