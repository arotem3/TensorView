#pragma once
#include "TensorView/Macros.hpp"
#include "TensorView/Shapes/StridedShape.hpp"
#include "TensorView/Tensors/PersistentView.hpp"
#include "TensorView/Tensors/RawView.hpp"
#include "TensorView/Tensors/TView.hpp"

namespace tensor
{
   /**
    * @brief Returns a view of the given tensor-like object with its dimensions permuted according to the specified
    * order.
    *
    * @example
    * Tensor<double, 2> A = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    * std::vector<int> perm = {1, 0};
    * auto At = permuteDimensions(A, perm); // Transpose of A
    *
    * @param x The tensor-like object to permute.
    * @param perm A random access range specifying the new order of dimensions.
    * @return A view of the tensor with permuted dimensions.
    */
   template <typename TensorLike, std::ranges::random_access_range Permutation>
   auto permuteDimensions(TensorLike &&x, Permutation &&perm)
      requires(details::TensorTraits<std::decay_t<TensorLike>>::value)
   {
      using tensor_type = std::decay_t<TensorLike>;
      using traits = details::TensorTraits<tensor_type>;
      using value_type = typename traits::value_type;

      constexpr MemorySpace mem_space = traits::container_traits::memorySpace();
      constexpr index_t numDims = tensor_type::numDims();

      TENSOR_CHECK(perm.size() == numDims,
                   printf("Permutation size %ju must match the number of dimensions %ju in the tensor.\n",
                          static_cast<uintmax_t>(perm.size()), static_cast<uintmax_t>(numDims)));

      TENSOR_CHECK(
          [&]() constexpr
          {
             for (auto p : perm)
             {
                if (static_cast<index_t>(p) >= numDims || (std::is_signed_v<Permutation> && p < 0))
                   return false;
             }
             return true;
          }(),
          printf("Permutation indices must be within the valid range of dimensions."));

      TENSOR_CHECK(
          [=]() constexpr
          {
             uint_fast8_t seen[numDims] = {0};
             for (auto p : perm)
                if (seen[p]++ != 0)
                   return false;
             return true;
          }(),
          printf("Permutation indices must be unique and cover all dimensions."));

      using shape_traits = details::ShapeTraits<details::StridedShape<numDims>>;
      details::StridedShape<numDims> shp = shape_traits::from(x.shape());

      std::array<index_t, numDims> perm_shape, perm_strides;
      for (index_t i = 0; i < numDims; ++i)
      {
         index_t p = perm[i];
         perm_shape[i] = shp.shape(p);
         perm_strides[i] = shp.stride(p);
      }

      details::StridedShape<numDims> new_shp(std::move(perm_shape), std::move(perm_strides), shp.offset());

      using view =
          std::conditional_t<details::is_persistent_view_v<tensor_type>, SubView<value_type, numDims, mem_space>,
                             RawSubView<value_type, numDims, mem_space>>;

      using ct = details::ContainerTraits<typename view::container_type>;

      return view(std::move(new_shp), ct::from(traits::container(x)));
   }

   /**
    * @brief Returns a view of the given tensor-like object with its dimensions permuted according to the specified
    * order.
    *
    * @example
    * Tensor<double, 2> A = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    * auto At = permuteDimensions(A, 1, 0); // Transpose of A
    *
    * @param x The tensor-like object to permute.
    * @param perm A variadic list of dimension indices specifying the new order of dimensions.
    * @return A view of the tensor with permuted dimensions.
    */
   template <typename TensorLike, IndexLike... Permutation>
   auto permuteDimensions(TensorLike &&x, Permutation... perm)
      requires(details::TensorTraits<std::decay_t<TensorLike>>::value)
   {
      constexpr index_t num_perms = sizeof...(Permutation);
      constexpr index_t numDims = details::TensorTraits<std::decay_t<TensorLike>>::shape_traits::numDims();

      static_assert(num_perms == numDims,
                    "Number of permutation indices must match the number of dimensions in the tensor.");

      return permuteDimensions(std::forward<TensorLike>(x),
                               std::array<index_t, num_perms>{static_cast<index_t>(perm)...});
   }

   /**
    * @brief Returns a view of the given matrix-like object with its dimensions transposed.
    *
    * @example
    * Tensor<double, 2> A = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    * auto At = transpose(A); // Transpose of A
    *
    * @param x The matrix-like object to transpose.
    * @return A view of the matrix with transposed dimensions.
    */
   template <typename MatrixLike>
   auto transpose(MatrixLike &&x)
   {
      return permuteDimensions(std::forward<MatrixLike>(x), 1, 0);
   }
} // namespace tensor
