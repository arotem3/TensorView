#pragma once
#include "TensorView/Macros.hpp"
#include "TensorView/Shapes/LinearOrder.hpp"
#include "TensorView/Tensors/RawView.hpp"
#include "TensorView/Tensors/TView.hpp"
#include "TensorView/Utility/Memory.hpp"

namespace tensor
{
   template <typename T>
   class LAPACKMatrix
   {
   public:
      using value_type = T;

      value_type *data;
      int rows;
      int cols;
      int ld;
      char trans; // 'N' for no transpose, 'T' for transpose, 'C' for conjugate transpose
      MemorySpace mem_space;

      ~LAPACKMatrix()
      {
         if (owner)
            deallocate(data, mem_space);
      }

      LAPACKMatrix(bool owning = false) : owner(owning) {}

   private:
      bool owner = false;
   };

   /**
    * @brief Returns an std::optional which contains a LAPACK-compatible view of the given tensor-like object if
    * possible. Otherwise, it returns an empty std::optional.
    */
   template <typename TensorLike>
   auto makeLAPACKMatrixView(TensorLike &&x)
   {
      using tensor_type = std::decay_t<TensorLike>;
      using traits = details::TensorTraits<tensor_type>;
      using value_type = typename traits::value_type;

      using return_type = std::optional<LAPACKMatrix<value_type>>;

      const MemorySpace mem_space = traits::container_traits::memorySpace();
      static_assert(tensor_type::numDims() == 2, "makeLAPACKMatrix only supports 2D tensor-like objects.");

      using view_type = RawSubView<value_type, 2, mem_space>; // Strided 2D view
      view_type view = x;
      auto &layout = view.shape();

      if (layout.stride(0) == 1 && layout.stride(1) >= layout.shape(0))
      {
         LAPACKMatrix<value_type> lapack_matrix;
         lapack_matrix.data = traits::container(x).data() + layout.offset();
         lapack_matrix.rows = layout.shape(0);
         lapack_matrix.cols = layout.shape(1);
         lapack_matrix.ld = layout.stride(1);
         lapack_matrix.trans = 'N';
         lapack_matrix.mem_space = mem_space;
         return std::make_optional(lapack_matrix);
      }
      else if (layout.stride(1) == 1 && layout.stride(0) >= layout.shape(1))
      {
         LAPACKMatrix<value_type> lapack_matrix;
         lapack_matrix.data = traits::container(x).data() + layout.offset();
         lapack_matrix.rows = layout.shape(0);
         lapack_matrix.cols = layout.shape(1);
         lapack_matrix.ld = layout.stride(0);
         lapack_matrix.trans = 'T';
         lapack_matrix.mem_space = mem_space;
         return std::make_optional(lapack_matrix);
      }
      else
      {
         return return_type{std::nullopt};
      }
   }

   /**
    * @brief Returns a LAPACK-compatible view of the given tensor-like object if possible.
    * Otherwise, it makes a contiguous copy in column-major order and returns a LAPACKMatrix
    * view of the copy.
    */
   template <typename TensorLike>
   auto makeLAPACKMatrixViewOrCopy(TensorLike &&x)
   {
      auto view = makeLAPACKMatrixView(std::forward<TensorLike>(x));

      if (view)
         return *view;
      else
      {
         using tensor_type = std::decay_t<TensorLike>;
         using traits = details::TensorTraits<tensor_type>;
         using value_type = typename traits::value_type;
         constexpr MemorySpace mem_space = traits::container_traits::memorySpace();

         LAPACKMatrix<value_type> copy(true);
         copy.rows = x.shape(0);
         copy.cols = x.shape(1);
         copy.ld = copy.rows;
         copy.trans = 'N';
         copy.mem_space = mem_space;
         copy.data = allocate<value_type, mem_space>(copy.rows * copy.cols);
         details::copyTensorToArray<mem_space>(std::forward<TensorLike>(x), copy.data);
         return copy;
      }
   }
} // namespace tensor
