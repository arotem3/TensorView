#include "test.hpp"
using namespace tensor;

static int test_tensor_reshape_inplace()
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

template <typename T>
static void *__data(T &obj)
{
   if constexpr (std::is_pointer_v<T>)
   {
      return static_cast<void *>(obj);
   }
   else
   {
      return static_cast<void *>(std::data(obj));
   }
}

template <typename lambda>
static int test_reshape(std::string name, lambda &&init)
{
   int n_fails = 0;

   auto data = init();
   auto reshaped = reshape(data, 2, 3);

   if (reshaped.shape(0) != 2 || reshaped.shape(1) != 3 || reshaped.data() != __data(data))
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Failed to reshape " << name << ".\n";
      n_fails++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Successfully reshaped " << name << ".\n";
   }

   return n_fails;
}

int main()
{
   int n_failed = 0;

   n_failed += test_tensor_reshape_inplace();
   n_failed += test_reshape("pointer", []() { return new double[6]; });
   n_failed += test_reshape("tensor", []() { return Tensor<double, 1>(6); });
   n_failed += test_reshape("TensorView", []() { return TensorView<double, 1>(new double[6], 6); });
   n_failed += test_reshape("std::array", []() { return std::array<double, 6>{}; });
   n_failed += test_reshape("std::vector", []() { return std::vector<double>(6); });
   n_failed += test_reshape("StaticTensor", []() { return StaticTensor<double, 6>(); });
   n_failed += test_reshape("StaticView", []() { return StaticView<double, 6>(new double[6]); });

   PRINT_RESULT(n_failed);
   return n_failed;
}
