#pragma once
#include "TensorView/Macros.hpp"
#include "TensorView/Shapes/LinearOrder.hpp"

namespace tensor::details
{
   template <typename T>
   struct ShapeTraits;
   /*
      using shape_type; // the shape type itself

      static constexpr index_t numDims(); // number of dimensions
      static constexpr bool contiguous(); // are all valid instances of this shape contiguous in F-order?

      static shape_type from(OtherShapeType); // convert from another shape type if possible
   */
} // namespace tensor::details
