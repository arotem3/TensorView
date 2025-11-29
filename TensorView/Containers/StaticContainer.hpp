#pragma once
#include <array>

#include "TensorView/Containers/ContainerTraits.hpp"
#include "TensorView/Containers/ViewContainer.hpp"
#include "TensorView/Macros.hpp"

namespace tensor::details
{
   /**
    * @brief A container with static storage duration.
    */
   template <typename T, index_t N>
   class StaticContainer : public std::array<T, N>
   {
   public:
      constexpr index_t capacity() const
      {
         return N;
      }
   };

   template <typename T, index_t N>
   struct ContainerTraits<StaticContainer<T, N>>
   {
      using value_type = T;
      using container_type = StaticContainer<T, N>;
      using rview_type = ViewContainer<T, MemorySpace::Unspecified>;
      using rcview_type = ViewContainer<const T, MemorySpace::Unspecified>;

      static constexpr bool owning()
      {
         return true;
      }

      static constexpr bool mutableElements()
      {
         return std::assignable_from<T &, T>;
      }

      static constexpr MemorySpace memorySpace()
      {
         return MemorySpace::Unspecified;
      }

      static TENSOR_FUNC rview_type makeRView(container_type &x)
      {
         return rview_type(x.data(), x.capacity());
      }

      static TENSOR_FUNC rcview_type makeRCView(const container_type &x)
      {
         return rcview_type(x.data(), x.capacity());
      }

      static constexpr container_type from(const container_type &other)
      {
         return container_type(other);
      }
   };
} // namespace tensor::details
