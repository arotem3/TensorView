#pragma once
#include <algorithm>
#include <type_traits>
#include <utility>

#include "TensorView/Tensors/makeTensor.hpp"

namespace tensor::details
{
   template <typename In>
   decltype(auto) toDevice(In &&in)
   {
#ifdef TENSOR_USE_CUDA
      using in_type = std::remove_cvref_t<In>;
      constexpr MemorySpace in_space = in_type::memorySpace();

      if constexpr (in_space == MemorySpace::Host)
      {
         auto managed = tensor::makeTensorLike<void, LinearOrder::F, MemorySpace::Managed>(in);
         std::ranges::copy(in.begin(), in.end(), managed.begin());
         tensor::synchronizeMemory(managed.data(), managed.size(), MemorySpace::Managed, MemorySpace::Device);
         return managed;
      }
      else
      {
         return std::forward<In>(in);
      }
#else
      return std::forward<In>(in);
#endif
   }
} // namespace tensor::details
