#include "test.hpp"

using namespace tensor;
using LAPACKTranspose = tensor::details::LAPACKTranspose;
using LAPACKLayout = tensor::details::LAPACKLayout;

static int testSimpleView()
{
   int n_failed = 0;

   Tensor<double, 2> A(3, 4);

   auto lapack_view = makeLAPACKMatrixView(A);

   if (!lapack_view)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " makeLAPACKMatrixView failed for contiguous matrix A."
                << std::endl;
      ++n_failed;
   }
   else
   {
      auto view = *lapack_view;
      if (view.rows != 3 || view.cols != 4 || view.ld != 3 || view.transpose != LAPACKTranspose::NoTrans ||
          view.data != A.data())
      {
         std::cout << "\t" << ColorText::red("[ ✗ ]")
                   << " makeLAPACKMatrixView returned incorrect LAPACKMatrix for matrix A." << std::endl;
         ++n_failed;
      }
      else
      {
         std::cout << "\t" << ColorText::green("[ ✓ ]")
                   << " makeLAPACKMatrixView returned correct LAPACKMatrix for matrix A." << std::endl;
      }
   }

   return n_failed;
}

static int testLDA()
{
   int n_failed = 0;

   Tensor<double, 2> A(5, 2);

   auto lapack_view = makeLAPACKMatrixView(A(Range(0, 3), All{}));

   if (!lapack_view)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " makeLAPACKMatrixView failed for a contiguous submatrix."
                << std::endl;
      ++n_failed;
   }
   else
   {
      auto view = *lapack_view;
      if (view.rows != 3 || view.cols != 2 || view.ld != 5 || view.transpose != LAPACKTranspose::NoTrans ||
          view.data != A.data())
      {
         std::cout << "\t" << ColorText::red("[ ✗ ]")
                   << " makeLAPACKMatrixView returned incorrect LAPACKMatrix for submatrix." << std::endl;
         ++n_failed;
      }
      else
      {
         std::cout << "\t" << ColorText::green("[ ✓ ]")
                   << " makeLAPACKMatrixView returned correct LAPACKMatrix for submatrix." << std::endl;
      }
   }

   return n_failed;
}

static int testTranspose()
{
   int n_failed = 0;

   Tensor<double, 2> A(4, 3);

   auto lapack_view = makeLAPACKMatrixView(transpose(A));

   if (!lapack_view)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " makeLAPACKMatrixView failed for transposed matrix A."
                << std::endl;
      ++n_failed;
   }
   else
   {
      auto view = *lapack_view;
      if (view.rows != 3 || view.cols != 4 || view.ld != 4 || view.transpose != LAPACKTranspose::Trans ||
          view.data != A.data())
      {
         std::cout << "\t" << ColorText::red("[ ✗ ]")
                   << " makeLAPACKMatrixView returned incorrect LAPACKMatrix for transposed matrix A." << std::endl;
         ++n_failed;
      }
      else
      {
         std::cout << "\t" << ColorText::green("[ ✓ ]")
                   << " makeLAPACKMatrixView returned correct LAPACKMatrix for transposed matrix A." << std::endl;
      }
   }
   return n_failed;
}

static int testLDATranspose()
{
   int n_failed = 0;

   Tensor<double, 2> A(6, 4);

   auto lapack_view = makeLAPACKMatrixView(transpose(A(Range(2, 5), All{})));

   if (!lapack_view)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " makeLAPACKMatrixView failed for transposed submatrix."
                << std::endl;
      ++n_failed;
   }
   else
   {
      auto view = *lapack_view;
      if (view.rows != 4 || view.cols != 3 || view.ld != 6 || view.transpose != LAPACKTranspose::Trans ||
          view.data != A.data() + 2)
      {
         std::cout << "\t" << ColorText::red("[ ✗ ]")
                   << " makeLAPACKMatrixView returned incorrect LAPACKMatrix for transposed submatrix." << std::endl;
         ++n_failed;
      }
      else
      {
         std::cout << "\t" << ColorText::green("[ ✓ ]")
                   << " makeLAPACKMatrixView returned correct LAPACKMatrix for transposed submatrix." << std::endl;
      }
   }
   return n_failed;
}

static int testFailureCase()
{
   int n_failed = 0;

   Tensor<double, 2> A(4, 4);

   auto lapack_view = makeLAPACKMatrixView(A(Range(0, 4, 2), All{}));

   if (lapack_view)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]")
                << " makeLAPACKMatrixView should have failed for non-contiguous submatrix." << std::endl;
      ++n_failed;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]")
                << " makeLAPACKMatrixView correctly failed for non-contiguous submatrix." << std::endl;
   }

   return n_failed;
}

static int testConvertOrdering()
{
   int n_failed = 0;

   Tensor<double, 2> A(3, 4);
   auto lapack_view = makeLAPACKMatrixView(A, LinearOrder::F);

   if (!lapack_view)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " makeLAPACKMatrixView failed in testConvertOrdering."
                << std::endl;
      return n_failed + 1;
   }

   auto view = *lapack_view;
   auto original_data = view.data;
   auto original_rows = view.rows;
   auto original_cols = view.cols;
   auto original_ld = view.ld;

   view.convertOrdering(LinearOrder::C);

   if (view.layout != LAPACKLayout::RowMajor || view.transpose != LAPACKTranspose::Trans ||
       view.data != original_data || view.rows != original_rows || view.cols != original_cols || view.ld != original_ld)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " convertOrdering(F->C) returned incorrect LAPACKMatrix metadata."
                << std::endl;
      ++n_failed;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " convertOrdering(F->C) returned correct LAPACKMatrix metadata."
                << std::endl;
   }

   view.convertOrdering(LinearOrder::F);

   if (view.layout != LAPACKLayout::ColMajor || view.transpose != LAPACKTranspose::NoTrans ||
       view.data != original_data || view.rows != original_rows || view.cols != original_cols || view.ld != original_ld)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " convertOrdering(C->F) returned incorrect LAPACKMatrix metadata."
                << std::endl;
      ++n_failed;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " convertOrdering(C->F) returned correct LAPACKMatrix metadata."
                << std::endl;
   }

   return n_failed;
}

static int testDesiredCOrderView()
{
   int n_failed = 0;

   Tensor<double, 2> A(3, 4);

   auto checkCase = [&](auto &&expr, LAPACKTranspose expected_transpose, int expected_rows, int expected_cols,
                        int expected_ld, const char *fail_msg, const char *mismatch_msg, const char *ok_msg) {
      auto lapack_view = makeLAPACKMatrixView(std::forward<decltype(expr)>(expr), LinearOrder::C);

      if (!lapack_view)
      {
         std::cout << "\t" << ColorText::red("[ ✗ ]") << " " << fail_msg << std::endl;
         ++n_failed;
         return;
      }

      const auto view = *lapack_view;
      if (view.layout != LAPACKLayout::RowMajor || view.transpose != expected_transpose || view.rows != expected_rows ||
          view.cols != expected_cols || view.ld != expected_ld || view.data != A.data())
      {
         std::cout << "\t" << ColorText::red("[ ✗ ]") << " " << mismatch_msg << std::endl;
         ++n_failed;
      }
      else
      {
         std::cout << "\t" << ColorText::green("[ ✓ ]") << " " << ok_msg << std::endl;
      }
   };

   checkCase(A, LAPACKTranspose::Trans, 3, 4, 3, "makeLAPACKMatrixView failed for C-order request on F-order matrix.",
             "makeLAPACKMatrixView returned incorrect C-order metadata for F-order matrix.",
             "makeLAPACKMatrixView returned correct C-order metadata for F-order matrix.");

   checkCase(transpose(A), LAPACKTranspose::NoTrans, 4, 3, 3,
             "makeLAPACKMatrixView failed for C-order request on transposed matrix.",
             "makeLAPACKMatrixView returned incorrect C-order metadata for transposed matrix.",
             "makeLAPACKMatrixView returned correct C-order metadata for transposed matrix.");

   checkCase(transpose(A(Range(0, 2), All{})), LAPACKTranspose::NoTrans, 4, 2, 3,
             "makeLAPACKMatrixView failed for C-order request on transposed submatrix.",
             "makeLAPACKMatrixView returned incorrect C-order metadata for transposed submatrix.",
             "makeLAPACKMatrixView returned correct C-order metadata for transposed submatrix.");

   return n_failed;
}

static int testConvertOrderingEdgeCases()
{
   int n_failed = 0;

   Tensor<double, 2> A(3, 4);
   auto lapack_view = makeLAPACKMatrixView(A, LinearOrder::C);

   if (!lapack_view)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " makeLAPACKMatrixView failed in testConvertOrderingEdgeCases."
                << std::endl;
      return n_failed + 1;
   }

   auto view = *lapack_view;
   auto original_data = view.data;
   auto original_rows = view.rows;
   auto original_cols = view.cols;
   auto original_ld = view.ld;

   // Idempotence: converting to same desired order should not change metadata
   view.convertOrdering(LinearOrder::C);
   if (view.layout != LAPACKLayout::RowMajor || view.transpose != LAPACKTranspose::Trans ||
       view.data != original_data || view.rows != original_rows || view.cols != original_cols || view.ld != original_ld)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " convertOrdering(C->C) unexpectedly changed metadata."
                << std::endl;
      ++n_failed;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " convertOrdering(C->C) preserved metadata." << std::endl;
   }

   // Toggle from Trans to NoTrans when converting C -> F
   view.convertOrdering(LinearOrder::F);
   if (view.layout != LAPACKLayout::ColMajor || view.transpose != LAPACKTranspose::NoTrans ||
       view.data != original_data || view.rows != original_rows || view.cols != original_cols || view.ld != original_ld)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]")
                << " convertOrdering(C->F) failed to toggle transpose metadata correctly." << std::endl;
      ++n_failed;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " convertOrdering(C->F) toggled transpose metadata correctly."
                << std::endl;
   }

   return n_failed;
}

static int testViewOrCopyViewCase()
{
   int n_failed = 0;

   Tensor<double, 2> matrix(3, 4);
   const auto matrix_view = makeLAPACKMatrixViewOrCopy(matrix, LinearOrder::F);

   if (matrix_view.layout != LAPACKLayout::ColMajor || matrix_view.transpose != LAPACKTranspose::NoTrans ||
       matrix_view.rows != 3 || matrix_view.cols != 4 || matrix_view.ld != 3 || matrix_view.data != matrix.data())
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]")
                << " makeLAPACKMatrixViewOrCopy returned incorrect view metadata for contiguous matrix." << std::endl;
      ++n_failed;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]")
                << " makeLAPACKMatrixViewOrCopy returned correct view metadata for contiguous matrix." << std::endl;
   }

   return n_failed;
}

static int testViewOrCopyCopyCase()
{
   int n_failed = 0;

   Tensor<double, 2> matrix(4, 4);
   for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j)
         matrix(i, j) = 10.0 * i + j;

   auto subview = matrix(Range(0, 4, 2), All{}); // non-contiguous along first dim, forces copy

   auto validateCopy = [&](auto m, LAPACKLayout expected_layout, int expected_ld, const char *meta_fail,
                           const char *data_fail, const char *ok_msg, auto index_fn) {
      if (m.layout != expected_layout || m.transpose != LAPACKTranspose::NoTrans || m.rows != 2 || m.cols != 4 ||
          m.ld != expected_ld)
      {
         std::cout << "\t" << ColorText::red("[ ✗ ]") << " " << meta_fail << std::endl;
         ++n_failed;
         return;
      }

      bool ok = true;
      for (int i = 0; i < 2 && ok; ++i)
         for (int j = 0; j < 4 && ok; ++j)
         {
            const double expected = matrix(2 * i, j);
            const double got = m.data[index_fn(i, j, m.ld)];
            if (got != expected)
               ok = false;
         }

      if (!ok)
      {
         std::cout << "\t" << ColorText::red("[ ✗ ]") << " " << data_fail << std::endl;
         ++n_failed;
      }
      else
      {
         std::cout << "\t" << ColorText::green("[ ✓ ]") << " " << ok_msg << std::endl;
      }
   };

   validateCopy(makeLAPACKMatrixViewOrCopy(subview, LinearOrder::F), LAPACKLayout::ColMajor, 2,
                "makeLAPACKMatrixViewOrCopy returned incorrect F-order copy metadata.",
                "makeLAPACKMatrixViewOrCopy produced incorrect F-order copy data.",
                "makeLAPACKMatrixViewOrCopy produced correct F-order copy data.",
                [](int i, int j, int ld) { return i + ld * j; });

   validateCopy(makeLAPACKMatrixViewOrCopy(subview, LinearOrder::C), LAPACKLayout::RowMajor, 4,
                "makeLAPACKMatrixViewOrCopy returned incorrect C-order copy metadata.",
                "makeLAPACKMatrixViewOrCopy produced incorrect C-order copy data.",
                "makeLAPACKMatrixViewOrCopy produced correct C-order copy data.",
                [](int i, int j, int ld) { return i * ld + j; });

   return n_failed;
}

int main()
{
   int n_failed = 0;

   n_failed += testSimpleView();
   n_failed += testLDA();
   n_failed += testTranspose();
   n_failed += testLDATranspose();
   n_failed += testFailureCase();
   n_failed += testConvertOrdering();
   n_failed += testDesiredCOrderView();
   n_failed += testConvertOrderingEdgeCases();
   n_failed += testViewOrCopyViewCase();
   n_failed += testViewOrCopyCopyCase();

   PRINT_RESULT(n_failed);

   return n_failed;
}