#include "test.hpp"

using namespace tensor;

static int test_simple_view()
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
      if (view.rows != 3 || view.cols != 4 || view.ld != 3 || view.trans != 'N' || view.data != A.data())
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

static int test_lda()
{
   int n_failed = 0;

   Tensor<double, 2> A(5, 2);

   auto lapack_view = makeLAPACKMatrixView(A(Span(0, 3), All{}));

   if (!lapack_view)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " makeLAPACKMatrixView failed for a contiguous submatrix."
                << std::endl;
      ++n_failed;
   }
   else
   {
      auto view = *lapack_view;
      if (view.rows != 3 || view.cols != 2 || view.ld != 5 || view.trans != 'N' || view.data != A.data())
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

static int test_transpose()
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
      if (view.rows != 3 || view.cols != 4 || view.ld != 4 || view.trans != 'T' || view.data != A.data())
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

static int test_lda_transpose()
{
   int n_failed = 0;

   Tensor<double, 2> A(6, 4);

   auto lapack_view = makeLAPACKMatrixView(transpose(A(Span(2, 5), All{})));

   if (!lapack_view)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " makeLAPACKMatrixView failed for transposed submatrix."
                << std::endl;
      ++n_failed;
   }
   else
   {
      auto view = *lapack_view;
      if (view.rows != 4 || view.cols != 3 || view.ld != 6 || view.trans != 'T' || view.data != A.data() + 2)
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

static int test_failure_case()
{
   int n_failed = 0;

   Tensor<double, 2> A(4, 4);

   auto lapack_view = makeLAPACKMatrixView(A(Span(0, 4, 2), All{}));

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

int main()
{
   int n_failed = 0;

   n_failed += test_simple_view();
   n_failed += test_lda();
   n_failed += test_transpose();
   n_failed += test_lda_transpose();
   n_failed += test_failure_case();

   PRINT_RESULT(n_failed);

   return n_failed;
}