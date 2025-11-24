#include "test.hpp"
using namespace tensor;

int test_tensor_reshape_inplace()
{
   int n_fails = 0;
   Tensor<double, 2> tensor(2, 3);

   tensor.reshape(3, 2);

   if (tensor.shape(0) != 3 || tensor.shape(1) != 2)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Failed to reshape Tensor inplace.\n";
      n_fails++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Successfully reshaped Tensor inplace.\n";
   }

   return n_fails;
}

int test_reshape_pointer()
{
   int n_fails = 0;
   double data[6] = {1, 2, 3, 4, 5, 6};

   auto reshaped_data = reshape(data, 2, 3);

   if (reshaped_data.shape(0) != 2 || reshaped_data.shape(1) != 3 || reshaped_data.data() != data)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Failed to reshape pointer into TensorView.\n";
      n_fails++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Successfully reshaped pointer into TensorView.\n";
   }

   return n_fails;
}

int test_reshape_tensorview()
{
   int n_fails = 0;
   double data[6] = {1, 2, 3, 4, 5, 6};
   TensorView<double, 2> tensor_view(data, 2, 3);

   auto reshaped_view = reshape(tensor_view, 3, 2);

   if (reshaped_view.shape(0) != 3 || reshaped_view.shape(1) != 2 || reshaped_view.data() != tensor_view.data())
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Failed to reshape TensorView into TensorView.\n";
      n_fails++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Successfully reshaped TensorView into TensorView.\n";
   }

   return n_fails;
}

int test_reshape_tensor()
{
   int n_fails = 0;
   Tensor<double, 2> tensor(2, 3);

   auto reshaped_view = reshape(tensor, 3, 2);

   if (reshaped_view.shape(0) != 3 || reshaped_view.shape(1) != 2 || reshaped_view.data() != tensor.data())
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Failed to reshape Tensor into PersistentView.\n";
      n_fails++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Successfully reshaped Tensor into PersistentView.\n";
   }

   return n_fails;
}

int main()
{
   int n_failed = 0;

   n_failed += test_tensor_reshape_inplace();
   n_failed += test_reshape_pointer();
   n_failed += test_reshape_tensorview();
   n_failed += test_reshape_tensor();

   if (n_failed == 0)
   {
      std::cout << ColorText::green("reshape.cpp: All tests passed!") << std::endl;
   }
   else
   {
      std::cout << ColorText::red(std::format("reshape.cpp: {} tests failed.", n_failed)) << std::endl;
   }

   return n_failed;
}
