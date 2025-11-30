#pragma once

namespace tensor
{
   enum class LinearOrder
   {
      F, // Fortran order (column-major)
      C  // C order (row-major)
   };
} // namespace tensor

namespace tensor::details
{
   template <size_t NumDims>
   constexpr std::array<index_t, NumDims> CStrides(const std::array<index_t, NumDims> &shape)
   {
      std::array<index_t, NumDims> strides;
      index_t stride = 1;
      for (index_t i = NumDims; i-- > 0;)
      {
         strides[i] = stride;
         stride *= shape[i];
      }
      return strides;
   }

   template <size_t NumDims>
   constexpr std::array<index_t, NumDims> FStrides(const std::array<index_t, NumDims> &shape)
   {
      std::array<index_t, NumDims> strides;
      index_t stride = 1;
      for (index_t i = 0; i < NumDims; ++i)
      {
         strides[i] = stride;
         stride *= shape[i];
      }
      return strides;
   }
} // namespace tensor::details