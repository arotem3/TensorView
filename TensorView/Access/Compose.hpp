#pragma once
#include <variant>

#include "TensorView/Access/IndexTypes.hpp"
#include "TensorView/Macros.hpp"

namespace tensor::details
{
   template <typename T>
   constexpr decltype(auto) passAsIndex(T &&value)
   {
      if constexpr (IndexLike<T>)
         return static_cast<index_t>(value);
      else
         return std::forward<T>(value);
   }

   template <typename Inner>
   constexpr auto compose(All, const Inner &inner)
   {
      return passAsIndex(inner);
   }

   template <typename Inner>
   constexpr auto compose(const Range &outer, const Inner &inner)
   {
      decltype(auto) i = passAsIndex(inner);
      using inner_t = std::decay_t<decltype(i)>;

      if constexpr (IndexLike<inner_t>)
         return outer[i];
      else if constexpr (std::is_same_v<inner_t, Range>)
         return Range{outer.begin + i.begin * outer.stride, outer.begin + i.end * outer.stride,
                      outer.stride * i.stride};
      else if constexpr (std::is_same_v<inner_t, All>)
         return outer;
      else
      {
         TENSOR_UNREACHABLE();
         return index_t{0};
      }
   }

   template <typename Inner>
   constexpr auto compose(const index_t &outer, const Inner &)
   {
      return passAsIndex(outer);
   }

   /**
    * @brief Composes two IndexVariants.
    */
   inline tensor::IndexVariant compose(const tensor::IndexVariant &outer, const tensor::IndexVariant &inner)
   {
      return std::visit([](const auto &o, const auto &i) -> tensor::IndexVariant { return compose(o, i); }, outer,
                        inner);
   }
} // namespace tensor::details