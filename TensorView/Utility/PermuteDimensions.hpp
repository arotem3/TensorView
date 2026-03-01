#pragma once
#include "TensorView/Access/StridedPattern.hpp"
#include "TensorView/Macros.hpp"
#include "TensorView/Tensors/TView.hpp"

namespace tensor
{
   namespace details
   {
      template <size_t N, IndexLike... I>
      constexpr bool validPemutation(I... i)
      {
         uint8_t seen[N] = {0};
         // (correct number of elements) && (all elements in range) && (all elements are unique)
         return (sizeof...(I) == N) &&
                ((static_cast<size_t>(i) < N && (std::is_unsigned_v<I> || i >= 0)) && ... && true) &&
                ((seen[i]++ == 0) && ... && true);
      }

      template <size_t N, IndexLike... Permutation>
      auto selectFromIndexSet(const CartesianIndexSet<N> &index_set, Permutation... p)
      {
         return CartesianIndexSet<N>{index_set[p]...};
      }

      template <size_t N, IndexLike... Permutation>
      auto selectFromIndexSet(AllProduct<N>, Permutation...)
      {
         return AllProduct<N>{};
      }
   } // namespace details

   template <typename TensorLike, IndexLike... Permutation>
   constexpr auto permuteDimensions(TensorLike &&x, Permutation... permutation)
   {
      using namespace tensor::details;

      using tensor_type = std::remove_cvref_t<TensorLike>;
      using traits = TensorTraits<tensor_type>;

      constexpr index_t numDims = tensor_type::numDims();

      TENSOR_CHECK(validPemutation<numDims>(permutation...),
                   printf("Permutation must consist of N unique elements in the range 0...N-1"));

      auto [layout, index_set] = unpackAccessPattern(makeStridedPatternFrom(x.shape()));

      StridedLayout<numDims> permuted_layout{{layout.dimensions[permutation]...}, layout.start};
      auto permuted_index_set = details::selectFromIndexSet(index_set, permutation...);

      return makeView(makeAccessPattern(std::move(permuted_layout), std::move(permuted_index_set)),
                      traits::container(std::forward<TensorLike>(x)));
   }

   template <typename TensorLike>
   constexpr auto permuteDimensions(TensorLike &&x,
                                    std::array<size_t, std::remove_cvref_t<TensorLike>::numDims()> permutation)
   {
      return std::apply([&](auto... p) { return permuteDimensions(std::forward<TensorLike>(x), p...); }, permutation);
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
   constexpr auto transpose(MatrixLike &&x)
   {
      return permuteDimensions(std::forward<MatrixLike>(x), 1, 0);
   }
} // namespace tensor
