#include "test.hpp"
using namespace tensor;

template <typename lambda>
static int test_tensorview_access(std::string name, lambda &&init)
{
   int n_failed = 0;

   auto [data, view] = init();

   int linear_access_fails = 0;
   for (int i = 0; i < 6; i++)
      linear_access_fails += view[i] != data[i];

   if (linear_access_fails)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << name << " linear access test failed!" << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << name << " linear access test passed!" << std::endl;
   }

   int multi_access_fails = 0;
   for (int i = 0; i < 2; i++)
   {
      for (int j = 0; j < 3; j++)
      {
         multi_access_fails += view(i, j) != data[i + 2 * j];
      }
   }

   if (multi_access_fails)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << name << " multi-dimensional access test failed!" << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << name << " multi-dimensional access test passed!" << std::endl;
   }

   return n_failed;
}

int main()
{
   int n_failed = 0;

   n_failed += test_tensorview_access("TensorView",
                                      []()
                                      {
                                         std::vector<int> data = {0, 1, 2, 3, 4, 5};
                                         TensorView<int, 2> view(data.data(), 2, 3);
                                         return std::make_pair(std::move(data), view);
                                      });
   n_failed += test_tensorview_access("StaticView",
                                      []()
                                      {
                                         std::vector<int> data = {0, 1, 2, 3, 4, 5};
                                         StaticView<int, 2, 3> view(data.data());
                                         return std::make_pair(std::move(data), view);
                                      });

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
