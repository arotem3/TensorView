#pragma once
#include "TensorView/Containers/SharedContainer.hpp"
#include "TensorView/Expressions/ExpressionTraits.hpp"
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
      if constexpr (MemSpace == MemorySpace::Host || MemSpace == MemorySpace::Unspecified)
      {
         std::ranges::copy(src.begin(), src.end(), dst);
      }
      else
      {
#ifdef TENSOR_USE_CUDA
         auto rview = src.raw();

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
      if constexpr (MemSpace == MemorySpace::Host || MemSpace == MemorySpace::Unspecified)
      {
         const index_t n = dst.size();
         for (index_t i = 0; i < n; ++i)
            dst[i] = src[i];
      }
      else
      {
#ifdef TENSOR_USE_CUDA
         auto rview = dst.raw();

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

   template <Expression Src, Expression Dst>
   inline void copyTensorToTensor(const Src &src, Dst &dst)
   {
      using src_t = typename std::remove_cvref_t<Src>::value_type;
      using dst_t = typename std::remove_cvref_t<Dst>::value_type;

      static_assert(std::is_convertible_v<src_t, dst_t>,
                    "source value_type must be convertible to destination value_type.");

      TENSOR_REQUIRE_EQUAL_SHAPES(src, dst);

#ifdef TENSOR_USE_CUDA
      constexpr MemorySpace ms_src = std::remove_cvref_t<Src>::memorySpace();
      constexpr MemorySpace ms_dst = std::remove_cvref_t<Dst>::memorySpace();

      if constexpr (!compatibleMemorySpaces(ms_src, ms_dst))
      {
         auto tmp = allocate<dst_t, MemorySpace::Managed>(src.size());
         copyTensorToArray<ms_src>(src, tmp);
         copyArrayToTensor<ms_dst>(tmp, dst);
         deallocate<dst_t, MemorySpace::Managed>(tmp);
         return;
      }
      else
      {
         if constexpr ((ms_src == MemorySpace::Host || ms_src == MemorySpace::Unspecified) &&
                       (ms_dst == MemorySpace::Host || ms_dst == MemorySpace::Unspecified))
         {
            std::ranges::copy(src.begin(), src.end(), dst.begin());
         }
         else
         {
            auto rview_src = src.raw();
            auto rview_dst = dst.raw();
            const index_t n = rview_src.size();
            const index_t blockSize = 256;
            const index_t numBlocks = (n + blockSize - 1) / blockSize;

            copyTensorToTensorKernel<MemorySpace::Managed><<<numBlocks, blockSize>>>(rview_src, rview_dst);
            TENSOR_DEBUG_ASSERT(cudaDeviceSynchronize() == cudaSuccess,
                                printf("cudaDeviceSynchronize failed after copyTensorToTensor\n"));
         }
      }
#else
      std::ranges::copy(src.begin(), src.end(), dst.begin());
#endif
   }
} // namespace tensor::details
