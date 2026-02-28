#pragma once
#include "TensorView/Macros.hpp"

namespace tensor::details
{
   template <typename T>
   struct ContainerTraits;
   /*
      using value_type;      // type of the elements stored in the container
      using container_type;  // the container type itself

      static constexpr bool owning();          // Does the container own its data?
      static constexpr bool mutableElements(); // Can the container's elements be modified?
      static constexpr MemorySpace memorySpace(); // Returns the memory space of the container

      // Functions to create views from the container
      static auto makeRView(container_type &x);
      static auto makeRView(const container_type &x);
      static auto makeView(container_type &x);
      static auto makeView(const container_type &x);
   */

   template <typename Container>
   struct IsRawContainer : std::false_type
   {};

   template <typename Container>
   inline constexpr bool is_raw_container = IsRawContainer<Container>::value;
} // namespace tensor::details
