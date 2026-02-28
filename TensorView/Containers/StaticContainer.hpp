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

      static constexpr auto makeRView(container_type &x)
      {
         using rview = ViewContainer<T, MemorySpace::Unspecified>;
         return rview(x.data(), x.capacity());
      }

      static constexpr auto makeRView(const container_type &x)
      {
         using rcview = ViewContainer<const T, MemorySpace::Unspecified>;
         return rcview(x.data(), x.capacity());
      }

      static void makeRView(container_type &&x) = delete; // cannot move static container to view

      static constexpr auto makeView(container_type &x)
      {
         return makeRView(x);
      }

      static constexpr auto makeView(const container_type &x)
      {
         return makeRView(x);
      }

      static void makeView(container_type &&x) = delete; // cannot move static container to view

      static constexpr container_type from(const container_type &other)
      {
         return container_type(other);
      }
   };

   template <typename T>
   struct IsStaticContainer : std::false_type
   {};

   template <typename T, index_t N>
   struct IsStaticContainer<StaticContainer<T, N>> : std::true_type
   {};

   template <typename T>
   inline constexpr bool is_static_container = IsStaticContainer<T>::value;
} // namespace tensor::details
