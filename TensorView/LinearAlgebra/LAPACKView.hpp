#pragma once
#include "TensorView/Macros.hpp"
#include "TensorView/Tensors/TView.hpp"
#include "TensorView/Utility/Memory.hpp"
#include "TensorView/Utility/PermuteDimensions.hpp"

namespace tensor::details
{
   // Modern LAPACK-C API enums (LAPACK >= 3.6)
   // These match the values used by the LAPACK C interface
   enum class LAPACKLayout : int
   {
      RowMajor = 102, // LAPACK_ROW_MAJOR
      ColMajor = 101  // LAPACK_COL_MAJOR
   };

   enum class LAPACKTranspose : int
   {
      NoTrans = 111,  // LAPACK_NO_TRANS
      Trans = 112,    // LAPACK_TRANS
      ConjTrans = 113 // LAPACK_CONJ_TRANS
   };
} // namespace tensor::details

namespace tensor
{
   /**
    * @brief A LAPACK-compatible matrix view with modern LAPACK-C API support.
    *
    * This class provides a view of a matrix that is compatible with the modern LAPACK-C interface.
    * It supports both row-major and column-major layouts and different transpose states.
    *
    * @tparam T The scalar type (double, float, complex, etc.)
    */
   template <typename T>
   class LAPACKMatrix
   {
   public:
      using pointer = T *;

      pointer data;                       ///< Pointer to matrix data
      int rows;                           ///< Number of rows
      int cols;                           ///< Number of columns
      int ld;                             ///< Leading dimension
      details::LAPACKLayout layout;       ///< Memory layout (row-major or column-major)
      details::LAPACKTranspose transpose; ///< Transpose state (no trans, transpose, conjugate transpose)
      MemorySpace mem_space;              ///< Memory space (host or device)

      LAPACKMatrix(const LAPACKMatrix &other)
          : data(other.data),
            rows(other.rows),
            cols(other.cols),
            ld(other.ld),
            layout(other.layout),
            transpose(other.transpose),
            mem_space(other.mem_space),
            owner(other.owner)
      {
         if (owner && other.data)
         {
            const auto n = static_cast<size_t>(rows) * static_cast<size_t>(cols);
            data = allocate<T>(n, mem_space);
            details::copy_n(other.data, n, data);
         }
      }

      LAPACKMatrix &operator=(const LAPACKMatrix &other)
      {
         if (this == &other)
            return *this;

         if (owner && data)
            deallocate(data, mem_space);

         data = other.data;
         rows = other.rows;
         cols = other.cols;
         ld = other.ld;
         layout = other.layout;
         transpose = other.transpose;
         mem_space = other.mem_space;
         owner = other.owner;

         if (owner && other.data)
         {
            const auto n = static_cast<size_t>(rows) * static_cast<size_t>(cols);
            data = allocate<T>(n, mem_space);
            details::copy_n(other.data, n, data);
         }

         return *this;
      }

      LAPACKMatrix(LAPACKMatrix &&other) noexcept
          : data(other.data),
            rows(other.rows),
            cols(other.cols),
            ld(other.ld),
            layout(other.layout),
            transpose(other.transpose),
            mem_space(other.mem_space),
            owner(other.owner)
      {
         other.data = nullptr;
         other.rows = 0;
         other.cols = 0;
         other.ld = 0;
         other.owner = false;
      }

      LAPACKMatrix &operator=(LAPACKMatrix &&other) noexcept
      {
         if (this == &other)
            return *this;

         if (owner && data)
            deallocate(data, mem_space);

         data = other.data;
         rows = other.rows;
         cols = other.cols;
         ld = other.ld;
         layout = other.layout;
         transpose = other.transpose;
         mem_space = other.mem_space;
         owner = other.owner;

         other.data = nullptr;
         other.rows = 0;
         other.cols = 0;
         other.ld = 0;
         other.owner = false;

         return *this;
      }

      ~LAPACKMatrix()
      {
         if (owner)
            deallocate(data, mem_space);
      }

      LAPACKMatrix(bool owning = false) noexcept
          : data(nullptr),
            rows(0),
            cols(0),
            ld(0),
            layout(details::LAPACKLayout::ColMajor),
            transpose(details::LAPACKTranspose::NoTrans),
            mem_space(MemorySpace::Host),
            owner(owning)
      {}

      /**
       * @brief Get the LAPACK layout value for C API calls
       * @return Integer value for LAPACK_ROW_MAJOR or LAPACK_COL_MAJOR
       */
      int getLayoutValue() const noexcept
      {
         return static_cast<int>(layout);
      }

      /**
       * @brief Get the LAPACK transpose value for C API calls
       * @return Integer value for LAPACK_NO_TRANS, LAPACK_TRANS, or LAPACK_CONJ_TRANS
       */
      int getTransposeValue() const noexcept
      {
         return static_cast<int>(transpose);
      }

      /**
       * @brief Get the effective number of rows (considering transpose state)
       * @return Number of rows as they would appear to LAPACK
       */
      int getEffectiveRows() const noexcept
      {
         return (transpose == details::LAPACKTranspose::NoTrans) ? rows : cols;
      }

      /**
       * @brief Get the effective number of columns (considering transpose state)
       * @return Number of columns as they would appear to LAPACK
       */
      int getEffectiveCols() const noexcept
      {
         return (transpose == details::LAPACKTranspose::NoTrans) ? cols : rows;
      }

      /**
       * @brief Convert this matrix view between F and C ordering metadata.
       *
       * This conversion does not move data and assumes the current metadata
       * is valid for the underlying storage.
       */
      void convertOrdering(LinearOrder desired_order) noexcept
      {
         const auto desired_layout =
             (desired_order == LinearOrder::F) ? details::LAPACKLayout::ColMajor : details::LAPACKLayout::RowMajor;

         if (layout == desired_layout)
            return;

         layout = desired_layout;

         if (transpose == details::LAPACKTranspose::NoTrans)
            transpose = details::LAPACKTranspose::Trans;
         else if (transpose == details::LAPACKTranspose::Trans)
            transpose = details::LAPACKTranspose::NoTrans;
      }

   private:
      bool owner = false;
   };

   /**
    * @brief Returns an std::optional which contains a LAPACK-compatible view of the given tensor-like object if
    * possible. Otherwise, it returns an empty std::optional.
    *
    * This function attempts to create a view of a 2D tensor-like object that is compatible with the modern
    * LAPACK-C interface. It accepts a desired ordering (F for column-major/Fortran or C for row-major)
    * and detects if the matrix can be provided in that ordering, optionally transposed.
    *
    * @tparam TensorLike The tensor-like type (must be 2D)
    * @param x The tensor-like object to create a view for
    * @param desired_order The desired LinearOrder (LinearOrder::F for column-major, LinearOrder::C for row-major)
    * @return An optional containing the LAPACKMatrix view if possible, otherwise std::nullopt
    */
   template <typename TensorLike>
   auto makeLAPACKMatrixView(TensorLike &&x, LinearOrder desired_order = LinearOrder::F)
   {
      using tensor_type = std::decay_t<TensorLike>;
      using traits = details::TensorTraits<tensor_type>;
      using value_type = typename traits::value_type;

      const MemorySpace mem_space = traits::container_traits::memorySpace();
      static_assert(tensor_type::numDims() == 2, "makeLAPACKMatrix only supports 2D tensor-like objects.");

      const auto &shape = x.shape();

      // Convert to strided layout for uniform access to stride information
      auto strided_layout = details::makeStridedLayoutFrom<2>(std::get<0>(unpackAccessPattern(shape)));

      // Compute offset from the full access pattern (layout + index set).
      // This preserves subview offsets regardless of whether they are encoded
      // in the layout start or in the index set composition.
      const size_t offset = static_cast<size_t>(shape.at(0, 0));

      const auto stride0 = strided_layout.dimensions[0].stride;
      const auto stride1 = strided_layout.dimensions[1].stride;
      const auto rows = strided_layout.shape(0);
      const auto cols = strided_layout.shape(1);
      auto *const base_data = traits::container(x).data() + offset;

      auto make_view = [&](index_t leading_dim, details::LAPACKLayout layout,
                           details::LAPACKTranspose trans) -> std::optional<LAPACKMatrix<value_type>> {
         LAPACKMatrix<value_type> lapack_matrix;
         lapack_matrix.data = base_data;
         lapack_matrix.rows = rows;
         lapack_matrix.cols = cols;
         lapack_matrix.ld = leading_dim;
         lapack_matrix.layout = layout;
         lapack_matrix.transpose = trans;
         lapack_matrix.mem_space = mem_space;
         return std::make_optional(lapack_matrix);
      };

      if (desired_order == LinearOrder::F)
      {
         // Fortran order: column-major (stride_row == 1)
         // Case 1: Already in column-major order
         if (stride0 == 1 && stride1 >= rows)
         {
            return make_view(stride1, details::LAPACKLayout::ColMajor, details::LAPACKTranspose::NoTrans);
         }
         // Case 2: In row-major order (can be viewed as transposed column-major)
         else if (stride1 == 1 && stride0 >= cols)
         {
            return make_view(stride0, details::LAPACKLayout::ColMajor, details::LAPACKTranspose::Trans);
         }
      }
      else if (desired_order == LinearOrder::C)
      {
         // C order: row-major (stride_col == 1)
         // Case 1: Already in row-major order
         if (stride1 == 1 && stride0 >= cols)
         {
            return make_view(stride0, details::LAPACKLayout::RowMajor, details::LAPACKTranspose::NoTrans);
         }
         // Case 2: In column-major order (can be viewed as transposed row-major)
         else if (stride0 == 1 && stride1 >= rows)
         {
            return make_view(stride1, details::LAPACKLayout::RowMajor, details::LAPACKTranspose::Trans);
         }
      }

      return std::optional<LAPACKMatrix<value_type>>{std::nullopt};
   }

   /**
    * @brief Returns a LAPACK-compatible view of the given tensor-like object if possible.
    * Otherwise, it makes a contiguous copy in the desired ordering and returns a LAPACKMatrix
    * view of the copy.
    *
    * This function guarantees that the returned matrix can be used with LAPACK routines.
    * If the input tensor is already LAPACK-compatible in the desired order, no copy is made.
    * Otherwise, a copy is made with the LAPACKMatrix taking ownership of the allocated memory.
    *
    * @tparam TensorLike The tensor-like type (must be 2D)
    * @param x The tensor-like object to create a view or copy for
    * @param desired_order The desired LinearOrder (LinearOrder::F for column-major, LinearOrder::C for row-major)
    * @return A LAPACKMatrix view (owning if a copy was made)
    */
   template <typename TensorLike>
   auto makeLAPACKMatrixViewOrCopy(TensorLike &&x, LinearOrder desired_order = LinearOrder::F)
   {
      auto view = makeLAPACKMatrixView(std::forward<TensorLike>(x), desired_order);

      if (view)
         return *view;

      using tensor_type = std::decay_t<TensorLike>;
      using traits = details::TensorTraits<tensor_type>;
      using value_type = typename traits::value_type;
      constexpr MemorySpace mem_space = traits::container_traits::memorySpace();

      LAPACKMatrix<value_type> copy(true);
      copy.rows = x.shape(0);
      copy.cols = x.shape(1);

      if (desired_order == LinearOrder::F)
      {
         copy.ld = copy.rows;
         copy.layout = details::LAPACKLayout::ColMajor;
      }
      else
      {
         copy.ld = copy.cols;
         copy.layout = details::LAPACKLayout::RowMajor;
      }

      copy.transpose = details::LAPACKTranspose::NoTrans;
      copy.mem_space = mem_space;
      copy.data = allocate<value_type>(static_cast<size_t>(copy.rows) * static_cast<size_t>(copy.cols), mem_space);

      if (desired_order == LinearOrder::F)
      {
         details::copyTensorToArray<mem_space>(std::forward<TensorLike>(x), copy.data);
      }
      else
      {
         // Native tensor iteration is F-order. Copying transpose(x) in F-order
         // yields a row-major contiguous representation of x.
         details::copyTensorToArray<mem_space>(transpose(std::forward<TensorLike>(x)), copy.data);
      }

      return copy;
   }
} // namespace tensor
