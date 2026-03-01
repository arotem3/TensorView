#include "test.hpp"
using namespace tensor;

template <typename lambda>
static int testTensorViewAccess2d(std::string name, lambda &&init)
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

template <typename T>
static int test3dAccess(std::string name, T &&tensor)
{
   int n_failed = 0;

   // Initialize with known values using multidimensional access
   for (int i = 0; i < 3; ++i)
   {
      for (int j = 0; j < 4; ++j)
      {
         for (int k = 0; k < 2; ++k)
         {
            tensor(i, j, k) = i * 100 + j * 10 + k;
         }
      }
   }

   // Verify values
   bool mismatch = false;
   for (int i = 0; i < 3; ++i)
   {
      for (int j = 0; j < 4; ++j)
      {
         for (int k = 0; k < 2; ++k)
         {
            if (tensor(i, j, k) != i * 100 + j * 10 + k)
            {
               mismatch = true;
               break;
            }
         }
      }
   }

   if (mismatch)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << name << " 3D access test failed!" << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << name << " 3D access test passed!" << std::endl;
   }

   return n_failed;
}

int main()
{
   int n_failed = 0;

   std::cout << "\n=== 2D Linear and Multi-dimensional Access ===" << std::endl;
   n_failed += testTensorViewAccess2d("TensorView", []() {
      std::vector<int> data = {0, 1, 2, 3, 4, 5};
      TensorView<int, 2> view(data.data(), 2, 3);
      return std::make_pair(std::move(data), view);
   });
   n_failed += testTensorViewAccess2d("StaticView", []() {
      std::vector<int> data = {0, 1, 2, 3, 4, 5};
      StaticView<int, 2, 3> view(data.data());
      return std::make_pair(std::move(data), view);
   });
   n_failed += testTensorViewAccess2d("Tensor", []() {
      std::vector<int> data = {0, 1, 2, 3, 4, 5};
      Tensor<int, 2> tensor(2, 3);
      for (int i = 0; i < 6; ++i)
         tensor[i] = data[i];
      return std::make_pair(std::move(data), tensor);
   });

   std::cout << "\n=== 3D Access Tests ===" << std::endl;
   n_failed += test3dAccess("Tensor<int, 3>", makeTensor<int>(3, 4, 2));
   n_failed += test3dAccess("TensorView<int, 3>", TensorView<int, 3>(new int[24], 3, 4, 2));
   n_failed += test3dAccess("StaticTensor<int, 3, 4, 2>", StaticTensor<int, 3, 4, 2>());

   PRINT_RESULT(n_failed);
   return n_failed;
}
