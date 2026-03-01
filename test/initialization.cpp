#include <complex>

#include "test.hpp"

using namespace tensor;

static int testTensorViewInitialization()
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

static int testTensorInitialization()
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

static int testTensorViewConversion()
{
   int n_failed = 0;

   Tensor<double, 2> tensor(2, 3);

   auto correct_conversion = [&tensor](auto &tv) {
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

static int testPViewInitialization()
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

static int testStaticTensorInitialization()
{
   int n_failed = 0;
   StaticTensor<double, 2, 3> stensor{};

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

static int testStaticViewInitialization()
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

static int testLowDToHighD()
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

   return n_failed;
}

static int testCopyShape()
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

static int testTensorFromInitializer()
{
   int n_failed = 0;

   double data[2][3] = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
   Tensor<double, 2> tensor = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};

   if (tensor.shape(0) != 2 || tensor.shape(1) != 3)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]")
                << " Tensor initialization from initializer list failed: incorrect shape." << std::endl;
      n_failed++;
      return n_failed;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]")
                << " Tensor initialization from initializer list shape check passed." << std::endl;
   }

   bool correct = true;
   for (index_t i = 0; i < 2; ++i)
   {
      for (index_t j = 0; j < 3; ++j)
      {
         if (tensor(i, j) != data[i][j])
         {
            correct = false;
         }
      }
   }

   if (!correct)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]")
                << " Tensor initialization from initializer list failed: incorrect data." << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]")
                << " Tensor initialization from initializer list data check passed." << std::endl;
   }

   return n_failed;
}

template <typename lambda>
static int testCopyFromInitializer(std::string name, lambda &&init)
{
   int n_failed = 0;

   double data[2][3][4] = {{{1.0, 2.0, 3.0, 4.0}, {5.0, 6.0, 7.0, 8.0}, {9.0, 10.0, 11.0, 12.0}},
                           {{13.0, 14.0, 15.0, 16.0}, {17.0, 18.0, 19.0, 20.0}, {21.0, 22.0, 23.0, 24.0}}};
   auto t = init();

   t = {{{1.0, 2.0, 3.0, 4.0}, {5.0, 6.0, 7.0, 8.0}, {9.0, 10.0, 11.0, 12.0}},
        {{13.0, 14.0, 15.0, 16.0}, {17.0, 18.0, 19.0, 20.0}, {21.0, 22.0, 23.0, 24.0}}};

   if (t.shape(0) != 2 || t.shape(1) != 3 || t.shape(2) != 4)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ] ") << name << " copy from initializer list failed: incorrect shape."
                << std::endl;
      n_failed++;
      return n_failed;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ] ") << name << " copy from initializer list shape check passed."
                << std::endl;
   }

   bool correct = true;
   for (index_t i = 0; i < 2; ++i)
   {
      for (index_t j = 0; j < 3; ++j)
      {
         for (index_t k = 0; k < 4; ++k)
         {
            if (t(i, j, k) != data[i][j][k])
            {
               correct = false;
            }
         }
      }
   }

   if (!correct)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ] ") << name << " copy from initializer list failed: incorrect data."
                << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ] ") << name << " copy from initializer list data check passed."
                << std::endl;
   }

   return n_failed;
}

static int testSingleElementTensor()
{
   int n_failed = 0;

   Tensor<int, 1> t1(1);
   t1[0] = 42;

   if (t1.size() != 1 || t1[0] != 42)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Single element 1D tensor failed." << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Single element 1D tensor passed." << std::endl;
   }

   Tensor<int, 3> t3(1, 1, 1);
   t3(0, 0, 0) = 99;

   if (t3.size() != 1 || t3(0, 0, 0) != 99)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Single element 3D tensor failed." << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Single element 3D tensor passed." << std::endl;
   }

   return n_failed;
}

static int testDifferentDataTypes()
{
   int n_failed = 0;

   // Test float
   Tensor<float, 2> tf(2, 3);
   tf(0, 0) = 1.5f;
   if (tf(0, 0) != 1.5f)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Float tensor failed." << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Float tensor passed." << std::endl;
   }

   // Test int
   Tensor<int, 2> ti(2, 3);
   ti(0, 0) = 42;
   if (ti(0, 0) != 42)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Int tensor failed." << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Int tensor passed." << std::endl;
   }

   // Test complex numbers (if available)
   Tensor<std::complex<double>, 2> tc(2, 2);
   tc(0, 0) = std::complex<double>(1.0, 2.0);
   if (tc(0, 0) != std::complex<double>(1.0, 2.0))
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Complex tensor failed." << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Complex tensor passed." << std::endl;
   }

   return n_failed;
}

static int testLargeDimensions()
{
   int n_failed = 0;

   Tensor<int, 5> t5(2, 3, 4, 5, 6);
   if (t5.size() != 2 * 3 * 4 * 5 * 6)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " 5D tensor size check failed." << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " 5D tensor size check passed." << std::endl;
   }

   // Verify shapes
   if (t5.shape(0) != 2 || t5.shape(1) != 3 || t5.shape(2) != 4 || t5.shape(3) != 5 || t5.shape(4) != 6)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " 5D tensor shape check failed." << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " 5D tensor shape check passed." << std::endl;
   }

   // Test access
   t5(1, 2, 3, 4, 5) = 12345;
   if (t5(1, 2, 3, 4, 5) != 12345)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " 5D tensor access failed." << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " 5D tensor access passed." << std::endl;
   }

   return n_failed;
}

static int testMoveSemantics()
{
   int n_failed = 0;

   Tensor<int, 2> t1(2, 3);
   for (int i = 0; i < 6; ++i)
      t1[i] = i;

   auto *original_data = t1.data();

   // Move construct
   Tensor<int, 2> t2(std::move(t1));

   if (t2.data() != original_data)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Move constructor didn't transfer ownership." << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Move constructor passed." << std::endl;
   }

   // Verify data
   bool correct = true;
   for (int i = 0; i < 6; ++i)
   {
      if (t2[i] != i)
      {
         correct = false;
         break;
      }
   }

   if (!correct)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Move constructor corrupted data." << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Move constructor preserved data." << std::endl;
   }

   return n_failed;
}

template <typename lambda>
static int testCopy(std::string from, std::string to, lambda &&init)
{
   int n_failed = 0;

   auto [source, target] = init();

   target = source;

   bool correct = true;
   for (index_t i = 0; i < source.shape(0); ++i)
   {
      for (index_t j = 0; j < source.shape(1); ++j)
      {
         correct = correct && (source(i, j) == target(i, j));
      }
   }

   if (!correct)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Copy from " << from << " to " << to
                << " failed: incorrect data." << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Copy from " << from << " to " << to << " data check passed."
                << std::endl;
   }

   return n_failed;
}

int main()
{
   int n_failed = 0;

   std::cout << "\n=== Basic Initialization Tests ===" << std::endl;
   n_failed += testTensorViewInitialization();
   n_failed += testTensorInitialization();
   n_failed += testStaticTensorInitialization();
   n_failed += testStaticViewInitialization();

   std::cout << "\n=== View and Conversion Tests ===" << std::endl;
   n_failed += testTensorViewConversion();
   n_failed += testPViewInitialization();
   n_failed += testLowDToHighD();

   std::cout << "\n=== Shape and Copy Tests ===" << std::endl;
   n_failed += testCopyShape();

   std::cout << "\n=== Initializer List Tests ===" << std::endl;
   n_failed += testTensorFromInitializer();

   std::cout << "\n=== Copy from Initializer List Tests ===" << std::endl;
   n_failed += testCopyFromInitializer("Tensor", []() { return Tensor<double, 3>(); });
   n_failed += testCopyFromInitializer("CTensor", []() { return CTensor<double, 3>(); });
   n_failed += testCopyFromInitializer("PView", []() {
      Tensor<double, 3> t(2, 3, 4);
      return t.view();
   });
   n_failed += testCopyFromInitializer("StaticTensor", []() { return StaticTensor<double, 2, 3, 4>(); });
   n_failed += testCopyFromInitializer("TensorView", []() {
      TensorView<double, 3> tv(new double[24], 2, 3, 4);
      return tv;
   });
   n_failed += testCopyFromInitializer("StaticView", []() {
      StaticView<double, 2, 3, 4> sv(new double[24]);
      return sv;
   });

   std::cout << "\n=== Edge Case Tests ===" << std::endl;
   n_failed += testSingleElementTensor();
   n_failed += testDifferentDataTypes();
   n_failed += testLargeDimensions();
   n_failed += testMoveSemantics();

   std::cout << "\n=== Copy Between Different Types ===" << std::endl;

   n_failed += testCopy("Tensor<int>", "Tensor<double>", []() {
      Tensor<int, 2> src(2, 3);
      Tensor<double, 2> tgt(2, 3);
      for (auto &x : src)
         x = rand() % 100;
      return std::make_pair(src, tgt);
   });
   n_failed += testCopy("FTensor<int>", "CTensor<int>", []() {
      CTensor<int, 2> src(2, 3);
      CTensor<int, 2> tgt(2, 3);
      for (auto &x : src)
         x = rand() % 100;
      return std::make_pair(src, tgt);
   });
   n_failed += testCopy("PView<float>", "TensorView<double>", []() {
      Tensor<float, 2> src(2, 3);
      TensorView<double, 2> tgt(new double[6], 2, 3);
      for (auto &x : src)
         x = static_cast<float>(rand() % 100);
      return std::make_pair(src.view(), tgt);
   });
   n_failed += testCopy("TensorView<int>", "StaticTensor<double>", []() {
      TensorView<int, 2> src(new int[6], 2, 3);
      StaticTensor<double, 2, 3> tgt;
      for (auto &x : src)
         x = rand() % 100;
      return std::make_pair(src, tgt);
   });

   PRINT_RESULT(n_failed);
   return n_failed;
}
