#pragma once
#include "TensorView/Containers/ContainerTraits.hpp"
#include "TensorView/Containers/OwningContainer.hpp"
#include "TensorView/Containers/ViewContainer.hpp"
#include "TensorView/Expressions/ExpressionBase.hpp"
#include "TensorView/Expressions/ExpressionTraits.hpp"
#include "TensorView/LinearAlgebra/LAPACKView.hpp"
#include "TensorView/Macros.hpp"
#include "TensorView/Shapes/CompareShapes.hpp"
#include "TensorView/Tensors/StaticTView.hpp"
#include "TensorView/Tensors/StaticTensor.hpp"
#include "TensorView/Tensors/TView.hpp"
#include "TensorView/Tensors/Tensor.hpp"
#include "TensorView/Tensors/TensorTraits.hpp"
#include "TensorView/Tensors/makeTensor.hpp"
#include "TensorView/Utility/Copy.hpp"
#include "TensorView/Utility/InitializerTensor.hpp"
#include "TensorView/Utility/Memory.hpp"
#include "TensorView/Utility/PermuteDimensions.hpp"
#include "TensorView/Utility/Reshape.hpp"

#ifdef TENSOR_USE_FFTW
#include "TensorView/FFT/fft.hpp"
#endif

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
