#pragma once
#include "TensorView/Access/Span.hpp"
#include "TensorView/Containers/ContainerTraits.hpp"
#include "TensorView/Macros.hpp"
#include "TensorView/Shapes/ShapeTraits.hpp"
#include "TensorView/Shapes/StridedShape.hpp"
#include "TensorView/Tensors/TensorTraits.hpp"
#include "TensorView/Utility/Memory.hpp"

namespace tensor::details
{
   template <typename Shape, typename T, MemorySpace MemSpace>
   class RawView;

   template <typename Shape, typename T, MemorySpace MemSpace, bool Owner>
   class PersistentView;

   template <typename TensorType>
   constexpr decltype(auto) makeRawSubView(TensorType &&x, index_t index)
   {
      using traits = TensorTraits<std::remove_cvref_t<TensorType>>;
      return traits::container(x)[index];
   }

   template <typename TensorType>
   TENSOR_FUNC auto makeRawSubView(TensorType &&x, const Span &s)
   {
      using traits = TensorTraits<std::remove_cvref_t<TensorType>>;
      using T = typename traits::value_type;
      using value_type = std::conditional_t<std::is_const_v<std::remove_reference_t<TensorType>>, const T, T>;

      using ct = ContainerTraits<typename traits::container_type>;
      constexpr MemorySpace MemSpace = ct::memorySpace();

      using subview_type = RawView<StridedShape<1>, value_type, MemSpace>;
      using subtraits = TensorTraits<subview_type>;
      using subcontainer_type = typename subtraits::container_type;
      using subshape_type = typename subtraits::shape_type;

      using sct = ContainerTraits<subcontainer_type>;

      return subview_type(subshape_type(s), sct::from(traits::container(x)));
   }

   template <typename TensorType, size_t numDims>
   TENSOR_FUNC auto makeRawSubView(TensorType &&x, const MultiSpan<numDims> &ms)
   {
      using traits = TensorTraits<std::remove_cvref_t<TensorType>>;
      using T = typename traits::value_type;
      using value_type = std::conditional_t<std::is_const_v<std::remove_reference_t<TensorType>>, const T, T>;

      using ct = ContainerTraits<typename traits::container_type>;
      constexpr MemorySpace MemSpace = ct::memorySpace();

      using subview_type = RawView<StridedShape<numDims>, value_type, MemSpace>;
      using subtraits = TensorTraits<subview_type>;
      using subcontainer_type = typename subtraits::container_type;
      using subshape_type = typename subtraits::shape_type;

      using sct = ContainerTraits<subcontainer_type>;

      return subview_type(subshape_type(ms), sct::from(traits::container(x)));
   }

   template <typename TensorType>
   decltype(auto) makeSubView(TensorType &&x, index_t index)
   {
      using traits = TensorTraits<std::remove_cvref_t<TensorType>>;
      return traits::container(x)[index];
   }

   template <typename TensorType>
   auto makeSubView(TensorType &&x, const Span &s)
   {
      using traits = TensorTraits<std::remove_cvref_t<TensorType>>;
      using T = typename traits::value_type;
      using value_type = std::conditional_t<std::is_const_v<std::remove_reference_t<TensorType>>, const T, T>;

      using ct = ContainerTraits<typename traits::container_type>;
      constexpr MemorySpace MemSpace = ct::memorySpace();

      using subview_type = PersistentView<StridedShape<1>, value_type, MemSpace, false>;
      using subtraits = TensorTraits<subview_type>;
      using subcontainer_type = typename subtraits::container_type;
      using subshape_type = typename subtraits::shape_type;

      using sct = ContainerTraits<subcontainer_type>;

      return subview_type(subshape_type(s), sct::from(traits::container(x)));
   }

   template <typename TensorType, size_t numDims>
   auto makeSubView(TensorType &&x, const MultiSpan<numDims> &ms)
   {
      using traits = TensorTraits<std::remove_cvref_t<TensorType>>;
      using T = typename traits::value_type;
      using value_type = std::conditional_t<std::is_const_v<std::remove_reference_t<TensorType>>, const T, T>;

      using ct = ContainerTraits<typename traits::container_type>;
      constexpr MemorySpace MemSpace = ct::memorySpace();

      using subview_type = PersistentView<StridedShape<numDims>, value_type, MemSpace, false>;
      using subtraits = TensorTraits<subview_type>;
      using subcontainer_type = typename subtraits::container_type;
      using subshape_type = typename subtraits::shape_type;

      using sct = ContainerTraits<subcontainer_type>;
      return subview_type(subshape_type(ms), sct::from(traits::container(x)));
   }
} // namespace tensor::details
