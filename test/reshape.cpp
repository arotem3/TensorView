#include "test.hpp"
using namespace tensor;

static int testTensorReshapeInplace()
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

static int testTensorReshape3dTo2d()
{
   int n_fails = 0;
   Tensor<double, 3> tensor(2, 3, 4);

   // Initialize with values
   for (index_t i = 0; i < tensor.size(); ++i)
      tensor[i] = i;

   tensor.reshape(6, 4);

   if (tensor.shape(0) != 6 || tensor.shape(1) != 4)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Failed to reshape 3D->2D inplace.\n";
      n_fails++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Successfully reshaped 3D->2D inplace.\n";
   }

   // Verify data preservation
   bool data_preserved = true;
   for (int i = 0; i < 24; ++i)
   {
      if (tensor[i] != i)
      {
         data_preserved = false;
         break;
      }
   }

   if (!data_preserved)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Reshape 3D->2D did not preserve data.\n";
      n_fails++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Reshape 3D->2D preserved data correctly.\n";
   }

   return n_fails;
}

static int testTensorReshapeTo1d()
{
   int n_fails = 0;
   Tensor<double, 3> tensor(2, 3, 4);

   for (index_t i = 0; i < tensor.size(); ++i)
      tensor[i] = i * 10;

   tensor.reshape(24);

   if (tensor.shape(0) != 24)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Failed to reshape to 1D.\n";
      n_fails++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Successfully reshaped to 1D.\n";
   }

   // Verify data
   bool correct = true;
   for (int i = 0; i < 24; ++i)
   {
      if (tensor[i] != i * 10)
      {
         correct = false;
         break;
      }
   }

   if (!correct)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Reshape to 1D corrupted data.\n";
      n_fails++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Reshape to 1D preserved data.\n";
   }

   return n_fails;
}

static int testTensorMultipleReshapes()
{
   int n_fails = 0;
   Tensor<double, 2> tensor(6, 4);

   for (index_t i = 0; i < tensor.size(); ++i)
      tensor[i] = i;

   // First reshape (within 2D)
   tensor.reshape(3, 8);
   if (tensor.shape(0) != 3 || tensor.shape(1) != 8)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " First reshape failed.\n";
      n_fails++;
   }

   // Second reshape (within 2D)
   tensor.reshape(4, 6);
   if (tensor.shape(0) != 4 || tensor.shape(1) != 6)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Second reshape failed.\n";
      n_fails++;
   }

   // Third reshape (back to original within 2D)
   tensor.reshape(6, 4);
   if (tensor.shape(0) != 6 || tensor.shape(1) != 4)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Third reshape failed.\n";
      n_fails++;
   }

   // Verify data after multiple reshapes
   bool correct = true;
   for (int i = 0; i < 24; ++i)
   {
      if (tensor[i] != i)
      {
         correct = false;
         break;
      }
   }

   if (!correct)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Multiple reshapes corrupted data.\n";
      n_fails++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Multiple reshapes preserved data.\n";
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

template <typename T>
static int testReshape(std::string name, T data)
{
   int n_fails = 0;

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

template <typename T>
static int testReshapeDataPreservation(std::string name, T data)
{
   int n_fails = 0;

   // Initialize with known values
   for (int i = 0; i < 24; ++i)
      data[i] = i * 5;

   auto reshaped = reshape(data, 3, 8);

   // Check data preservation
   bool correct = true;
   for (int i = 0; i < 24; ++i)
   {
      if (reshaped[i] != i * 5)
      {
         correct = false;
         break;
      }
   }

   if (!correct)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Reshape of " << name << " did not preserve data.\n";
      n_fails++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Reshape of " << name << " preserved data.\n";
   }

   return n_fails;
}

template <typename T>
static int testReshapeDifferentDims(std::string name, T data)
{
   int n_fails = 0;

   // 1D to 3D
   auto reshaped_3d = reshape(data, 2, 3, 4);
   if (reshaped_3d.shape(0) != 2 || reshaped_3d.shape(1) != 3 || reshaped_3d.shape(2) != 4)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Failed to reshape " << name << " to 3D.\n";
      n_fails++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Successfully reshaped " << name << " to 3D.\n";
   }

   // 1D to 4D
   auto reshaped_4d = reshape(data, 2, 2, 3, 2);
   if (reshaped_4d.shape(0) != 2 || reshaped_4d.shape(1) != 2 || reshaped_4d.shape(2) != 3 || reshaped_4d.shape(3) != 2)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Failed to reshape " << name << " to 4D.\n";
      n_fails++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Successfully reshaped " << name << " to 4D.\n";
   }

   return n_fails;
}

int main()
{
   int n_failed = 0;

   std::cout << "\n=== Tensor Inplace Reshape Tests ===" << std::endl;
   n_failed += testTensorReshapeInplace();
   n_failed += testTensorReshape3dTo2d();
   n_failed += testTensorReshapeTo1d();
   n_failed += testTensorMultipleReshapes();

   std::cout << "\n=== Basic Reshape Tests ===" << std::endl;
   n_failed += testReshape("pointer", new double[6]);
   n_failed += testReshape("tensor", Tensor<double, 1>(6));
   n_failed += testReshape("TensorView", TensorView<double, 1>(new double[6], 6));
   n_failed += testReshape("StaticTensor", StaticTensor<double, 6>());
   n_failed += testReshape("StaticView", StaticView<double, 6>(new double[6]));
   n_failed += testReshape("std::array", std::array<double, 6>{});
   n_failed += testReshape("std::vector", std::vector<double>(6));

   std::cout << "\n=== Data Preservation Tests ===" << std::endl;
   n_failed += testReshapeDataPreservation("pointer", new double[24]);
   n_failed += testReshapeDataPreservation("std::vector", std::vector<double>(24));
   n_failed += testReshapeDataPreservation("Tensor", Tensor<double, 1>(24));

   std::cout << "\n=== Multi-dimensional Reshape Tests ===" << std::endl;
   n_failed += testReshapeDifferentDims("pointer", new double[24]);
   n_failed += testReshapeDifferentDims("std::vector", std::vector<double>(24));
   n_failed += testReshapeDifferentDims("Tensor", Tensor<double, 1>(24));

   PRINT_RESULT(n_failed);
   return n_failed;
}
