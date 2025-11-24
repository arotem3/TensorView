#pragma once
#include "TensorView/Macros.hpp"

namespace tensor::details
{
   template <typename T>
   struct ContainerTraits;
   /*
      using value_type;      // type of the elements stored in the container
      using container_type;  // the container type itself
      using rview_type;      // "raw view" type. Mutable view, no persistence
      using rcview_type;     // "raw const view" type. Immutable view, no persistence
      using pview_type;      // "persistent view" type. Mutable view, with persistence
      using pcview_type;     // "persistent const view" type. Immutable view, with persistence

      static constexpr bool owning();          // Does the container own its data?
      static constexpr bool mutableElements(); // Can the container's elements be modified?
      static constexpr MemorySpace memorySpace(); // Returns the memory space of the container

      // Functions to create views from the container
      static rview_type makeRView(container_type &x);
      static rcview_type makeRCView(const container_type &x);
      static pview_type makePView(container_type &x);
      static pcview_type makePCView(const container_type &x);
   */
} // namespace tensor::details
