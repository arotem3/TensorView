#pragma once
#include "TensorView/Containers/OwningContainer.hpp"
#include "TensorView/Macros.hpp"
#include "TensorView/Shapes/CompareShapes.hpp"
#include "TensorView/Utility/Memory.hpp"

namespace tensor::details
{
#ifdef TENSOR_USE_CUDA
   template <typename Src, typename Dst>
   __global__ inline void copyTensorToArrayKernel(const Src src, Dst *dst)
   {
      index_t idx = blockIdx.x * blockDim.x + threadIdx.x;

      if (idx < src.size())
         dst[idx] = src[idx];
   }
#endif

   template <MemorySpace MemSpace, typename Src, typename Dst>
   inline void copyTensorToArray(const Src &src, Dst *dst)
   {
      if constexpr (MemSpace == MemorySpace::Host)
      {
         const index_t n = src.size();
         for (index_t i = 0; i < n; ++i)
            dst[i] = src[i];
      }
      else
      {
#ifdef TENSOR_USE_CUDA
         auto rview = src.makeRawView();

         const index_t n = rview.size();
         const index_t blockSize = 256;
         const index_t numBlocks = (n + blockSize - 1) / blockSize;

         copyTensorToArrayKernel<<<numBlocks, blockSize>>>(rview, dst);
         TENSOR_CHECK(cudaDeviceSynchronize() == cudaSuccess,
                      printf("cudaDeviceSynchronize failed after copyTensorToArray\n"));
#else
         static_assert(!sizeof(Src), "copyTensorToArray: Device memory space requested but CUDA not enabled.");
#endif
      }
   }

#ifdef TENSOR_USE_CUDA
   template <typename Src, typename Dst>
   __global__ inline void copyArrayToTensorKernel(const Src *src, Dst dst)
   {
      index_t idx = blockIdx.x * blockDim.x + threadIdx.x;

      if (idx < dst.size())
         dst[idx] = src[idx];
   }
#endif

   template <MemorySpace MemSpace, typename Src, typename Dst>
   inline void copyArrayToTensor(const Src *src, Dst &dst)
   {
      if constexpr (MemSpace == MemorySpace::Host)
      {
         const index_t n = dst.size();
         for (index_t i = 0; i < n; ++i)
            dst[i] = src[i];
      }
      else
      {
#ifdef TENSOR_USE_CUDA
         auto rview = dst.makeRawView();

         const index_t n = rview.size();
         const index_t blockSize = 256;
         const index_t numBlocks = (n + blockSize - 1) / blockSize;

         copyArrayToTensorKernel<<<numBlocks, blockSize>>>(src, rview);
         TENSOR_DEBUG_ASSERT(cudaDeviceSynchronize() == cudaSuccess,
                             printf("cudaDeviceSynchronize failed after copyArrayToTensor\n"));
#else
         static_assert(!sizeof(Src), "copyArrayToTensor: Device memory space requested but CUDA not enabled.");
#endif
      }
   }

#ifdef TENSOR_USE_CUDA
   template <MemorySpace MemSpace, typename Src, typename Dst>
   __global__ inline void copyTensorToTensorKernel(const Src src, Dst dst)
   {
      index_t idx = blockIdx.x * blockDim.x + threadIdx.x;

      if (idx < src.size())
         dst[idx] = src[idx];
   }
#endif

   template <typename Src, typename Dst>
   inline void copyTensorToTensor(const Src &src, Dst &dst)
   {
      using ct_src = ContainerTraits<typename std::decay_t<Src>::container_type>;
      using ct_dst = ContainerTraits<typename std::decay_t<Dst>::container_type>;

      static_assert(ct_dst::mutableElements(), "destination container must be mutable.");

      using src_t = typename ct_src::value_type;
      using dst_t = typename ct_dst::value_type;

      static_assert(std::is_convertible_v<src_t, dst_t>,
                    "source value_type must be convertible to destination value_type.");

      constexpr MemorySpace ms_src = ct_src::memorySpace();
      constexpr MemorySpace ms_dst = ct_dst::memorySpace();

      TENSOR_REQUIRE_EQUAL_SHAPES(src, dst);

#ifdef TENSOR_USE_CUDA
      // If the memory spaces are not compatible, use managed memory as an intermediate buffer.
      if constexpr (!compatibleMemorySpaces(ms_src, ms_dst))
      {
         auto tmp = allocate<MemorySpace::Managed, dst_t>(src.size());
         copyTensorToArray<ms_src>(src, tmp);
         copyArrayToTensor<ms_dst>(tmp, dst);
         deallocate<MemorySpace::Managed>(tmp);
         return;
      }
#endif
      // Direct copy since memory spaces are compatible.
      if constexpr (ms_src == MemorySpace::Host || ms_dst == MemorySpace::Host)
      {
         const index_t n = src.size();
         for (index_t i = 0; i < n; ++i)
            dst[i] = src[i];
      }
      else
      {
#ifdef TENSOR_USE_CUDA
         auto rview_src = src.makeRawView();
         auto rview_dst = dst.makeRawView();
         const index_t n = rview_src.size();
         const index_t blockSize = 256;
         const index_t numBlocks = (n + blockSize - 1) / blockSize;

         copyTensorToTensorKernel<<<numBlocks, blockSize>>>(rview_src, rview_dst);
         TENSOR_DEBUG_ASSERT(cudaDeviceSynchronize() == cudaSuccess,
                             printf("cudaDeviceSynchronize failed after copyTensorToTensor\n"));
#else
         static_assert(!sizeof(Src), "Device memory space requested but CUDA not enabled.");
#endif
      }
   }
} // namespace tensor::details