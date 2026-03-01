#include "test.hpp"

using namespace tensor;

int testTrivial()
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

int testTranspose()
{
   int n_failed = 0;

   Matrix<double> A = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};

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

int testPermute3d()
{
   int n_failed = 0;

   Tensor<int, 3> A(2, 3, 4);
   int val = 0;
   for (index_t i = 0; i < 2; ++i)
      for (index_t j = 0; j < 3; ++j)
         for (index_t k = 0; k < 4; ++k)
            A(i, j, k) = val++;

   // auto Ap = permuteDimensions<2, 0, 1>(A); // from (2,3,4) to (4,2,3)
   auto Ap = permuteDimensions(A, 2, 0, 1);

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

int testPermute4d()
{
   int n_failed = 0;

   Tensor<int, 4> A(2, 3, 4, 5);
   int val = 0;
   for (index_t i = 0; i < 2; ++i)
      for (index_t j = 0; j < 3; ++j)
         for (index_t k = 0; k < 4; ++k)
            for (index_t l = 0; l < 5; ++l)
               A(i, j, k, l) = val++;

   // use the array version of permuteDimensions to test that as well
   auto Ap = permuteDimensions(A, {3, 1, 2, 0}); // from (2,3,4,5) to (5,3,4,2)

   if (Ap.shape(0) != 5 || Ap.shape(1) != 3 || Ap.shape(2) != 4 || Ap.shape(3) != 2)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " 4D PermuteDimensions test failed! Incorrect shape."
                << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " 4D PermuteDimensions test passed! Correct shape."
                << std::endl;
   }

   bool correct = true;
   for (index_t i = 0; i < 2; ++i)
      for (index_t j = 0; j < 3; ++j)
         for (index_t k = 0; k < 4; ++k)
            for (index_t l = 0; l < 5; ++l)
               correct = correct && (Ap(l, j, k, i) == A(i, j, k, l));

   if (!correct)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " 4D PermuteDimensions test failed! Incorrect values."
                << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " 4D PermuteDimensions test passed! Correct values."
                << std::endl;
   }

   return n_failed;
}

int testIdentityPermutation()
{
   int n_failed = 0;

   Tensor<int, 3> A(2, 3, 4);
   int val = 0;
   for (auto &elem : A)
      elem = val++;

   auto Ap = permuteDimensions(A, 0, 1, 2); // Identity permutation

   if (Ap.shape(0) != 2 || Ap.shape(1) != 3 || Ap.shape(2) != 4)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Identity permutation test failed! Shape changed." << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Identity permutation test passed! Shape preserved."
                << std::endl;
   }

   bool correct = true;
   for (index_t i = 0; i < 2; ++i)
      for (index_t j = 0; j < 3; ++j)
         for (index_t k = 0; k < 4; ++k)
            correct = correct && (Ap(i, j, k) == A(i, j, k));

   if (!correct)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Identity permutation test failed! Values changed." << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Identity permutation test passed! Values preserved."
                << std::endl;
   }

   return n_failed;
}

int testChainedPermutations()
{
   int n_failed = 0;

   Matrix<double> A = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};

   // Transpose twice should give back original
   auto At = transpose(A);
   auto Att = transpose(At);

   bool correct = true;
   for (index_t i = 0; i < 2; ++i)
      for (index_t j = 0; j < 3; ++j)
         correct = correct && (Att(i, j) == A(i, j));

   if (!correct)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Chained transpose test failed!" << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Chained transpose test passed!" << std::endl;
   }

   // 3D chained permutations
   Tensor<int, 3> B(2, 3, 4);
   int val = 0;
   for (auto &elem : B)
      elem = val++;

   auto Bp1 = permuteDimensions(B, 2, 0, 1);   // (2,3,4) -> (4,2,3)
   auto Bp2 = permuteDimensions(Bp1, 1, 2, 0); // (4,2,3) -> (2,3,4)

   correct = true;
   for (index_t i = 0; i < 2; ++i)
      for (index_t j = 0; j < 3; ++j)
         for (index_t k = 0; k < 4; ++k)
            correct = correct && (Bp2(i, j, k) == B(i, j, k));

   if (!correct)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Chained 3D permutation test failed!" << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Chained 3D permutation test passed!" << std::endl;
   }

   return n_failed;
}

int testPermutationModifications()
{
   int n_failed = 0;

   Tensor<int, 3> A(2, 3, 4);
   for (auto &elem : A)
      elem = 0;

   auto Ap = permuteDimensions(A, 1, 2, 0); // (2,3,4) -> (3,4,2)

   // Modify through permuted view
   for (index_t i = 0; i < 3; ++i)
      for (index_t j = 0; j < 4; ++j)
         for (index_t k = 0; k < 2; ++k)
            Ap(i, j, k) = 100 + i * 8 + j * 2 + k;

   // Verify changes in original
   bool correct = true;
   for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 3; ++j)
         for (int k = 0; k < 4; ++k)
            correct = correct && (A(i, j, k) == 100 + j * 8 + k * 2 + i);

   if (!correct)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Permutation modification test failed!" << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Permutation modification test passed!" << std::endl;
   }

   return n_failed;
}

int testReversePermutation()
{
   int n_failed = 0;

   Tensor<int, 3> A(2, 3, 4);
   int val = 0;
   for (auto &elem : A)
      elem = val++;

   auto Ap = permuteDimensions(A, 2, 1, 0); // Reverse all dimensions

   if (Ap.shape(0) != 4 || Ap.shape(1) != 3 || Ap.shape(2) != 2)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Reverse permutation test failed! Incorrect shape." << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Reverse permutation test passed! Correct shape." << std::endl;
   }

   bool correct = true;
   for (index_t i = 0; i < 2; ++i)
      for (index_t j = 0; j < 3; ++j)
         for (index_t k = 0; k < 4; ++k)
            correct = correct && (Ap(k, j, i) == A(i, j, k));

   if (!correct)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Reverse permutation test failed! Incorrect values."
                << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Reverse permutation test passed! Correct values."
                << std::endl;
   }

   return n_failed;
}

int testDifferentTypes()
{
   int n_failed = 0;

   // Test with TensorView
   TensorView<double, 3> tv(new double[24], 2, 3, 4);
   int val = 0;
   for (auto &elem : tv)
      elem = val++;

   auto tvp = permuteDimensions(tv, 1, 0, 2);

   bool correct = true;
   for (index_t i = 0; i < 2; ++i)
      for (index_t j = 0; j < 3; ++j)
         for (index_t k = 0; k < 4; ++k)
            correct = correct && (tvp(j, i, k) == tv(i, j, k));

   if (!correct)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " TensorView permutation test failed!" << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " TensorView permutation test passed!" << std::endl;
   }

   // Test with StaticTensor
   StaticTensor<int, 2, 3, 4> st;
   val = 0;
   for (auto &elem : st)
      elem = val++;

   auto stp = permuteDimensions(st, 2, 0, 1);

   correct = true;
   for (index_t i = 0; i < 2; ++i)
      for (index_t j = 0; j < 3; ++j)
         for (index_t k = 0; k < 4; ++k)
            correct = correct && (stp(k, i, j) == st(i, j, k));

   if (!correct)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " StaticTensor permutation test failed!" << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " StaticTensor permutation test passed!" << std::endl;
   }

   return n_failed;
}

int main()
{
   int n_failed = 0;

   std::cout << "\n=== Basic Permutation Tests ===" << std::endl;
   n_failed += testTrivial();
   n_failed += testTranspose();
   n_failed += testPermute3d();

   std::cout << "\n=== 4D Permutation Tests ===" << std::endl;
   n_failed += testPermute4d();

   std::cout << "\n=== Special Permutation Tests ===" << std::endl;
   n_failed += testIdentityPermutation();
   n_failed += testReversePermutation();

   std::cout << "\n=== Chained Permutation Tests ===" << std::endl;
   n_failed += testChainedPermutations();

   std::cout << "\n=== Modification Tests ===" << std::endl;
   n_failed += testPermutationModifications();

   std::cout << "\n=== Different Types Tests ===" << std::endl;
   n_failed += testDifferentTypes();

   PRINT_RESULT(n_failed);

   return n_failed;
}