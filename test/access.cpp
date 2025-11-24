#include "test.hpp"
using namespace tensor;

int test_tensorview_access()
{
   int n_failed = 0;

   double data[6] = {1, 2, 3, 4, 5, 6};
   FTensorView<double, 2> tensor_view(data, 2, 3);

   int linear_access_fails = 0;
   for (int i = 0; i < 6; i++)
      linear_access_fails += tensor_view[i] != data[i];

   if (linear_access_fails)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " FTensorView linear access test failed!" << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " FTensorView linear access test passed!" << std::endl;
   }

   int multi_access_fails = 0;
   for (int i = 0; i < 2; i++)
   {
      for (int j = 0; j < 3; j++)
      {
         multi_access_fails += tensor_view(i, j) != data[i + 2 * j];
      }
   }

   if (multi_access_fails)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " FTensorView multi-dimensional access test failed!" << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " FTensorView multi-dimensional access test passed!"
                << std::endl;
   }

   CTensorView<double, 2> c_tensor_view(data, 2, 3);

   linear_access_fails = 0;
   for (int i = 0; i < 6; i++)
      linear_access_fails += c_tensor_view[i] != data[i];

   if (linear_access_fails)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " CTensorView linear access test failed!" << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " CTensorView linear access test passed!" << std::endl;
   }

   multi_access_fails = 0;
   for (int i = 0; i < 2; i++)
   {
      for (int j = 0; j < 3; j++)
      {
         multi_access_fails += c_tensor_view(i, j) != data[i * 3 + j];
      }
   }

   if (multi_access_fails)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " CTensorView multi-dimensional access test failed!" << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " CTensorView multi-dimensional access test passed!"
                << std::endl;
   }

   return n_failed;
}

int main()
{
   int n_failed = 0;

   n_failed += test_tensorview_access();

   if (n_failed == 0)
   {
      std::cout << ColorText::green("access.cpp: All tests passed!") << std::endl;
   }
   else
   {
      std::cout << ColorText::red(std::format("access.cpp: {} tests failed.", n_failed)) << std::endl;
   }

   return n_failed;
}
