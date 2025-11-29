#include "test.hpp"

using namespace tensor;

static int test_tensorview_initialization()
{
   int n_failed = 0;
   double data[6] = {1, 2, 3, 4, 5, 6};

   TensorView<double, 2> t(data, 2, 3);

   if (t.shape(0) != 2 || t.shape(1) != 3)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " TensorView initialization failed: incorrect shape."
                << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " TensorView initialization passed." << std::endl;
   }

   if (t.size() != 6)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " TensorView initialization failed: incorrect size." << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " TensorView size check passed." << std::endl;
   }

   if (t.data() != data)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " TensorView initialization failed: incorrect data pointer."
                << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " TensorView data pointer check passed." << std::endl;
   }

   return n_failed;
}

static int test_tensor_initialization()
{
   int n_failed = 0;
   Tensor<double, 2> tensor(2, 3);

   if (tensor.shape(0) != 2 || tensor.shape(1) != 3)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Tensor initialization failed: incorrect shape." << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Tensor initialization passed." << std::endl;
   }

   if (tensor.size() != 6)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Tensor initialization failed: incorrect size." << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Tensor size check passed." << std::endl;
   }

   Tensor<double, 3> tensor3(2, 3);
   if (tensor3.shape(0) != 2 || tensor3.shape(1) != 3 || tensor3.shape(2) != 1)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]")
                << " Tensor initialization with trailing singleton dimension failed: incorrect shape." << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]")
                << " Tensor initialization with trailing singleton dimension passed." << std::endl;
   }

   return n_failed;
}

static int test_tensorview_conversion()
{
   int n_failed = 0;

   Tensor<double, 2> tensor(2, 3);

   auto correct_conversion = [&tensor](auto &tv)
   {
      return (tv.data() == tensor.data() && tv.size() == tensor.size() && tv.shape(0) == tensor.shape(0) &&
              tv.shape(1) == tensor.shape(1));
   };

#define __CHECK_CONV(expr, explain)                                                              \
   {                                                                                             \
      TensorView<double, 2> tv = expr;                                                           \
      if (!correct_conversion(tv))                                                               \
      {                                                                                          \
         std::cout << "\t" << ColorText::red("[ ✗ ]") << " Implicit conversion << " << explain   \
                   << " to non-const TensorView failed." << std::endl;                           \
         n_failed++;                                                                             \
      }                                                                                          \
      else                                                                                       \
      {                                                                                          \
         std::cout << "\t" << ColorText::green("[ ✓ ]") << " Implicit conversion << " << explain \
                   << " to non-const TensorView passed." << std::endl;                           \
      }                                                                                          \
                                                                                                 \
      TensorView<const double, 2> ctv = expr;                                                    \
      if (!correct_conversion(ctv))                                                              \
      {                                                                                          \
         std::cout << "\t" << ColorText::red("[ ✗ ]") << " Implicit conversion << " << explain   \
                   << " to const TensorView failed." << std::endl;                               \
         n_failed++;                                                                             \
      }                                                                                          \
      else                                                                                       \
      {                                                                                          \
         std::cout << "\t" << ColorText::green("[ ✓ ]") << " Implicit conversion << " << explain \
                   << " to const TensorView passed." << std::endl;                               \
      }                                                                                          \
   }

   __CHECK_CONV(tensor, "from Tensor");
   __CHECK_CONV(tensor.raw(), "from raw()");
   __CHECK_CONV(tensor.view(), "from view()");

   return n_failed;
}

static int test_pview_initialization()
{
   int n_failed = 0;
   Tensor<double, 2> tensor(2, 3);

   auto pview = tensor.view();

   if (pview.shape(0) != 2 || pview.shape(1) != 3)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " PersistentView initialization failed: incorrect shape."
                << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " PersistentView initialization passed." << std::endl;
   }

   if (pview.size() != 6)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " PersistentView initialization failed: incorrect size."
                << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " PersistentView size check passed." << std::endl;
   }

   if (pview.data() != tensor.data())
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " PersistentView initialization failed: incorrect data pointer."
                << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " PersistentView data pointer check passed." << std::endl;
   }

   auto pview_const = std::as_const(tensor).view();

   if (pview_const.shape(0) != 2 || pview_const.shape(1) != 3)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Const PersistentView initialization failed: incorrect shape."
                << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Const PersistentView initialization passed." << std::endl;
   }

   if (pview_const.size() != 6)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Const PersistentView initialization failed: incorrect size."
                << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Const PersistentView size check passed." << std::endl;
   }

   if (pview_const.data() != tensor.data())
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]")
                << " Const PersistentView initialization failed: incorrect data pointer." << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Const PersistentView data pointer check passed." << std::endl;
   }

   return n_failed;
}

static int test_static_tensor_initialization()
{
   int n_failed = 0;
   StaticTensor<double, 2, 3> stensor;

   if (stensor.shape(0) != 2 || stensor.shape(1) != 3)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " StaticTensor initialization failed: incorrect shape."
                << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " StaticTensor initialization passed." << std::endl;
   }

   if (stensor.size() != 6)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " StaticTensor initialization failed: incorrect size."
                << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " StaticTensor size check passed." << std::endl;
   }

   return n_failed;
}

static int test_static_view_initialization()
{
   int n_failed = 0;

   double data[6] = {1, 2, 3, 4, 5, 6};
   StaticView<double, 2, 3> sview(data);

   if (sview.shape(0) != 2 || sview.shape(1) != 3)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " StaticView initialization failed: incorrect shape."
                << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " StaticView initialization passed." << std::endl;
   }

   if (sview.size() != 6)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " StaticView initialization failed: incorrect size." << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " StaticView size check passed." << std::endl;
   }

   if (sview.data() != data)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " StaticView initialization failed: incorrect data pointer."
                << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " StaticView data pointer check passed." << std::endl;
   }

   return n_failed;
}

static int test_lowD_to_highD()
{
   int n_failed = 0;
   Tensor<double, 2> tensor(2, 3);

   TensorView<double, 3> view3d_of_2d = tensor;
   if (view3d_of_2d.shape(0) != 2 || view3d_of_2d.shape(1) != 3 || view3d_of_2d.shape(2) != 1)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]")
                << " TensorView initialization with trailing singleton dimension failed: incorrect shape." << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]")
                << " TensorView initialization with trailing singleton dimension passed." << std::endl;
   }

   Tensor<double, 3> tensor3d_from_2d = tensor;
   if (tensor3d_from_2d.shape(0) != 2 || tensor3d_from_2d.shape(1) != 3 || tensor3d_from_2d.shape(2) != 1)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]")
                << " Tensor initialization with trailing singleton dimension failed: incorrect shape." << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]")
                << " Tensor initialization with trailing singleton dimension passed." << std::endl;
   }

   PView<double, 3, LinearOrder::F, MemorySpace::Host> pview3d_of_2d = tensor;
   if (pview3d_of_2d.shape(0) != 2 || pview3d_of_2d.shape(1) != 3 || pview3d_of_2d.shape(2) != 1)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]")
                << " PersistentView initialization with trailing singleton dimension failed: incorrect shape."
                << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]")
                << " PersistentView initialization with trailing singleton dimension passed." << std::endl;
   }

   return n_failed;
}

static int test_copy_shape()
{
   int n_failed = 0;

   Tensor<double, 2> tensor(2, 3);

   auto like = makeTensorLike(tensor);
   if (like.shape(0) != tensor.shape(0) || like.shape(1) != tensor.shape(1))
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " makeTensorLike failed: shape mismatch." << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " makeTensorLike passed." << std::endl;
   }

   return n_failed;
}

int main()
{
   int n_failed = 0;

   n_failed += test_tensorview_initialization();
   n_failed += test_tensor_initialization();
   n_failed += test_tensorview_conversion();
   n_failed += test_pview_initialization();
   n_failed += test_static_tensor_initialization();
   n_failed += test_static_view_initialization();
   n_failed += test_lowD_to_highD();
   n_failed += test_copy_shape();

   if (n_failed == 0)
   {
      std::cout << ColorText::green("initialization.cpp: All tests passed!") << std::endl;
   }
   else
   {
      std::cout << ColorText::red(std::format("initialization.cpp: {} tests failed.", n_failed)) << std::endl;
   }

   return n_failed;
}
