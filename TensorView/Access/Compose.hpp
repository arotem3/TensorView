#pragma once
#include <variant>

#include "TensorView/Access/IndexTypes.hpp"
#include "TensorView/Macros.hpp"

namespace tensor::details
{
   /**
    * @brief Composes two index types, reducing to IndexVariant (All, Range, or index_t).
    * When composing Range with Range, computes the resulting Range directly.
    */
   template <typename Outer, typename Inner>
   constexpr auto composeImpl(const Outer &outer, const Inner &inner)
   {
      if constexpr (std::is_same_v<Inner, All>)
         return outer;
      else if constexpr (IndexLike<Outer>)
         return static_cast<index_t>(outer);
      else if constexpr (IndexLike<Inner>)
      {
         auto idx = static_cast<index_t>(inner);
         if constexpr (std::is_same_v<Outer, All>)
            return idx;
         else if constexpr (std::is_same_v<Outer, Range>)
            return outer[idx];
         else
         {
            TENSOR_UNREACHABLE();
            return index_t{0};
         }
      }
      else if constexpr (std::is_same_v<Inner, Range>)
      {
         if constexpr (std::is_same_v<Outer, All>)
            return inner;
         else if constexpr (std::is_same_v<Outer, Range>)
            return Range{outer.begin + inner.begin * outer.stride, outer.begin + inner.end * outer.stride,
                         outer.stride * inner.stride};
         else
         {
            TENSOR_UNREACHABLE();
            return index_t{0};
         }
      }
      else
      {
         TENSOR_UNREACHABLE();
         return index_t{0};
      }
   }

   /**
    * @brief Composes two IndexVariants.
    */
   inline tensor::IndexVariant compose(const tensor::IndexVariant &outer, const tensor::IndexVariant &inner)
   {
      return std::visit([](const auto &o, const auto &i) -> tensor::IndexVariant { return composeImpl(o, i); }, outer,
                        inner);
   }
} // namespace tensor::details