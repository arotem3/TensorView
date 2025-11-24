#include "test.hpp"
using namespace tensor;

template <typename T>
int test_subview(T &x)
{
   int n_fails = 0;

   for (auto &val : x)
      val = rand();

   auto subview = x.at(All(), 2, Span(0, 1), Span(2, 4));

   if (subview.numDims() != 3)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " fancy indexing produced view with wrong number of dimensions."
                << " Expected 3, got " << subview.numDims() << "." << std::endl;
      n_fails++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]")
                << " fancy indexing produced view with correct number of dimensions." << std::endl;
   }

   bool mismatch_found = false;
   for (int i = 0; i < 5; ++i)
   {
      for (int j = 0; j < 1; ++j)
      {
         for (int k = 0; k < 2; ++k)
         {
            if (subview.at(i, j, k) != x.at(i, 2, j, 2 + k))
            {
               mismatch_found = true;
               break;
            }
         }
      }
   }

   if (mismatch_found)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " fancy indexing produced incorrect view." << std::endl;
      n_fails++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " fancy indexing produced incorrect view." << std::endl;
   }

   return n_fails;
}

int test_tensor_subview()
{
   auto tensor = makeTensor<double>(5, 10, 2, 5);
   return test_subview(tensor);
}

int test_tensorview_subview()
{
   double data[500];
   TensorView<double, 4> tensor_view(data, 5, 10, 2, 5);
   return test_subview(tensor_view);
}

int main()
{
   int n_fails = 0;

   n_fails += test_tensor_subview();
   n_fails += test_tensorview_subview();

   if (n_fails == 0)
   {
      std::cout << ColorText::green("subview.cpp: All tests passed.") << std::endl;
   }
   else
   {
      std::cout << ColorText::red(std::format("subview.cpp: {} tests failed.", n_fails)) << std::endl;
   }

   return n_fails;
}
