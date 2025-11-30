#pragma once
#include "TensorView/Access/Iterator.hpp"
#include "TensorView/Access/MakeView.hpp"
#include "TensorView/Access/SimplifyIndex.hpp"
#include "TensorView/Access/Span.hpp"
#include "TensorView/Containers/ContainerTraits.hpp"
#include "TensorView/Containers/OwningContainer.hpp"
#include "TensorView/Containers/StaticContainer.hpp"
#include "TensorView/Containers/ViewContainer.hpp"
#include "TensorView/Macros.hpp"
#include "TensorView/Shapes/CompareShapes.hpp"
#include "TensorView/Shapes/LinearOrder.hpp"
#include "TensorView/Shapes/ShapeTraits.hpp"
#include "TensorView/Shapes/StandardShape.hpp"
#include "TensorView/Shapes/StaticShape.hpp"
#include "TensorView/Shapes/StridedShape.hpp"
#include "TensorView/Tensors/PersistentView.hpp"
#include "TensorView/Tensors/RawView.hpp"
#include "TensorView/Tensors/StaticTView.hpp"
#include "TensorView/Tensors/StaticTensor.hpp"
#include "TensorView/Tensors/TView.hpp"
#include "TensorView/Tensors/Tensor.hpp"
#include "TensorView/Tensors/TensorTraits.hpp"
#include "TensorView/Tensors/makeTensor.hpp"
#include "TensorView/Utility/Copy.hpp"
#include "TensorView/Utility/InitializerTensor.hpp"
#include "TensorView/Utility/Memory.hpp"
#include "TensorView/Utility/Reshape.hpp"

namespace tensor
{
   template <typename scalar, MemorySpace MemSpace = MemorySpace::Host>
   using VectorView = TensorView<scalar, 1, MemSpace>;

   template <typename scalar, MemorySpace MemSpace = MemorySpace::Host>
   using MatrixView = TensorView<scalar, 2, MemSpace>;

   template <typename scalar, MemorySpace MemSpace = MemorySpace::Host>
   using CubeView = TensorView<scalar, 3, MemSpace>;

   template <typename scalar, MemorySpace MemSpace = MemorySpace::Host>
   using Vector = Tensor<scalar, 1, MemSpace>;

   template <typename scalar, MemorySpace MemSpace = MemorySpace::Host>
   using Matrix = Tensor<scalar, 2, MemSpace>;

   template <typename scalar, MemorySpace MemSpace = MemorySpace::Host>
   using Cube = Tensor<scalar, 3, MemSpace>;
} // namespace tensor
