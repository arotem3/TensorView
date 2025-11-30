#include "test.hpp"

using namespace tensor;

int test_trivial()
{
   Vector<int> v = {1, 2, 3, 4, 5};
   auto vp = permuteDimensions(v, 0);

   if (vp.shape(0) != 5)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Trivial permuteDimensions test failed! Incorrect shape."
                << std::endl;
      return 1;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Trivial permuteDimensions test passed! Correct shape."
                << std::endl;
   }

   return 0;
}

int test_transpose()
{
   int n_failed = 0;

   Tensor<double, 2> A = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};

   double ref[3][2] = {{1.0, 4.0}, {2.0, 5.0}, {3.0, 6.0}};

   auto At = transpose(A); // Transpose of A

   if (At.shape(0) != 3 || At.shape(1) != 2)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Transpose test failed! Incorrect shape." << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Transpose test passed! Correct shape." << std::endl;
   }

   bool correct = true;
   for (index_t i = 0; i < 3; ++i)
   {
      for (index_t j = 0; j < 2; ++j)
      {
         correct = correct && (At(i, j) == ref[i][j]);
      }
   }

   if (!correct)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Transpose test failed! Incorrect values." << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Transpose test passed! Correct values." << std::endl;
   }

   auto Att = transpose(At);

   if (Att.shape(0) != 2 || Att.shape(1) != 3)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Transpose of Transpose test failed! Incorrect shape."
                << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Transpose of Transpose test passed! Correct shape."
                << std::endl;
   }

   correct = true;
   for (index_t i = 0; i < 2; ++i)
   {
      for (index_t j = 0; j < 3; ++j)
      {
         correct = correct && (Att(i, j) == A(i, j));
      }
   }

   if (!correct)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Transpose of Transpose test failed! Incorrect values."
                << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Transpose of Transpose test passed! Correct values."
                << std::endl;
   }

   return n_failed;
}

int test_permute3d()
{
   int n_failed = 0;

   Tensor<int, 3> A(2, 3, 4);
   int val = 0;
   for (index_t i = 0; i < 2; ++i)
      for (index_t j = 0; j < 3; ++j)
         for (index_t k = 0; k < 4; ++k)
            A(i, j, k) = val++;

   auto Ap = permuteDimensions(A, 2, 0, 1); // from (2,3,4) to (4,2,3)

   if (Ap.shape(0) != 4 || Ap.shape(1) != 2 || Ap.shape(2) != 3)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " PermuteDimensions test failed! Incorrect shape." << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " PermuteDimensions test passed! Correct shape." << std::endl;
   }

   bool correct = true;
   for (index_t i = 0; i < 2; ++i)
      for (index_t j = 0; j < 3; ++j)
         for (index_t k = 0; k < 4; ++k)
            correct = correct && (Ap(k, i, j) == A(i, j, k));

   if (!correct)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " PermuteDimensions test failed! Incorrect values." << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " PermuteDimensions test passed! Correct values." << std::endl;
   }

   return n_failed;
}

int main()
{
   int n_failed = 0;

   n_failed += test_trivial();
   n_failed += test_transpose();
   n_failed += test_permute3d();

   PRINT_RESULT(n_failed);

   return n_failed;
}