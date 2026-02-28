#pragma once
#include "TensorView/Access/StandardPattern.hpp"
#include "TensorView/Containers/OwningContainer.hpp"
#include "TensorView/Macros.hpp"
#include "TensorView/Tensors/TensorBase.hpp"

namespace tensor
{
   /**
    * @brief Tensor class representing a multi-dimensional array of elements of type Scalar in the specified memory
    * space which owns its data.
    *
    * @tparam Scalar the type of elements in the tensor e.g. float
    * @tparam NumDims The number of dimensions of the tensor, e.g., 2 for a matrix
    * @tparam Order the memory layout order (C or Fortran)
    * @tparam MemSpace the memory space where the tensor data is stored
    */
   template <typename Scalar, size_t NumDims, LinearOrder Order, MemorySpace MemSpace = MemorySpace::Host>
   using FCTensor =
       details::TensorBase<details::StandardPattern<NumDims, Order>, details::OwningContainer<Scalar, MemSpace>, true>;

   template <typename Scalar, size_t NumDims, MemorySpace MemSpace = MemorySpace::Host>
   using FTensor = FCTensor<Scalar, NumDims, LinearOrder::F, MemSpace>;

   template <typename Scalar, size_t NumDims, MemorySpace MemSpace = MemorySpace::Host>
   using CTensor = FCTensor<Scalar, NumDims, LinearOrder::C, MemSpace>;

   template <typename Scalar, size_t NumDims, MemorySpace MemSpace = MemorySpace::Host>
   using Tensor = FTensor<Scalar, NumDims, MemSpace>;

   template <typename Scalar, size_t NumDims, LinearOrder Order, MemorySpace MemSpace>
   using PView =
       details::TensorBase<details::StandardPattern<NumDims, Order>, details::OwningContainer<Scalar, MemSpace>, false>;
} // namespace tensor
