#include "test.hpp"
using namespace tensor;

template <typename T>
int test_subview_iterator(T &x)
{
   for (auto &val : x)
      val = rand();

   auto subview = x.at(4, All{});

   const int stride = 10;
   const int offset = 4;

   bool mismatch = false;
   int pos = 0;
   for (auto val : subview)
   {
      if (val != x.data()[offset + pos * stride])
      {
         mismatch = true;
         break;
      }
      pos++;
   }

   if (mismatch)
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Subview iterator test failed!" << std::endl;
   else
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Subview iterator test passed!" << std::endl;

   return mismatch;
}

int test_tensor_subview_iterator()
{
   Tensor<double, 2> tensor(10, 100);
   return test_subview_iterator(tensor);
}

int test_tensorview_subview_iterator()
{
   double data[1000];
   TensorView<double, 2> tensor_view(data, 10, 100);
   return test_subview_iterator(tensor_view);
}

int main()
{
   int n_fails = 0;

   n_fails += test_tensor_subview_iterator();
   n_fails += test_tensorview_subview_iterator();

   PRINT_RESULT(n_fails);
   return n_fails;
}
