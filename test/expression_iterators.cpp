#include "TensorView/Expressions/BinaryExpressions.hpp"
#include "TensorView/Expressions/ScalarExpr.hpp"
#include "test.hpp"

using namespace tensor;

template <typename T>
static int testExpressionIteratorTraits(const std::string &name)
{
   int n_fails = 0;

   // Test that expression is a range
   if (std::ranges::range<T>)
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " " << name << " is a range." << std::endl;
   }
   else
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " " << name << " is NOT a range." << std::endl;
      n_fails++;
   }

   // Test that const iterator is an input iterator
   using iterator_t = decltype(std::declval<const T &>().begin());
   if (std::input_iterator<iterator_t>)
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " " << name << "::const_iterator is an input iterator."
                << std::endl;
   }
   else
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " " << name << "::const_iterator is NOT an input iterator."
                << std::endl;
      n_fails++;
   }

   // Test that sentinel works with const iterator
   if (std::sentinel_for<tensor::details::TensorEndSentinel, iterator_t>)
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " " << name
                << "::TensorEndSentinel is a sentinel for const_iterator." << std::endl;
   }
   else
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " " << name
                << "::TensorEndSentinel is NOT a sentinel for const_iterator." << std::endl;
      n_fails++;
   }

   return n_fails;
}

static int testBinaryExpressionIteratorTraits()
{
   int n_fails = 0;

   auto a = makeTensor<float>(3, 4);
   auto b = makeTensor<float>(3, 4);

   for (index_t i = 0; i < a.size(); ++i)
   {
      a[i] = static_cast<float>(i + 1);
      b[i] = static_cast<float>(i * 2 + 10);
   }

   auto expr = a + b;

   std::cout << "\n=== Binary Expression Iterator Traits ===" << std::endl;
   n_fails += testExpressionIteratorTraits<decltype(expr)>("a + b");

   return n_fails;
}

static int testBinaryExpressionSubviewIteratorTraits()
{
   int n_fails = 0;

   auto a = makeTensor<float>(4, 5);
   auto b = makeTensor<float>(4, 5);

   for (index_t i = 0; i < a.size(); ++i)
   {
      a[i] = static_cast<float>(i + 1);
      b[i] = static_cast<float>(i * 2 + 10);
   }

   auto expr = a + b;
   auto subview = expr.at(Range(1, 3), All());

   std::cout << "\n=== Binary Expression Subview Iterator Traits ===" << std::endl;
   n_fails += testExpressionIteratorTraits<decltype(subview)>("(a + b).at(Range(1,3), All())");

   return n_fails;
}

static int testChainedExpressionIteratorTraits()
{
   int n_fails = 0;

   auto a = makeTensor<float>(3, 4);
   auto b = makeTensor<float>(3, 4);
   auto c = makeTensor<float>(3, 4);

   for (index_t i = 0; i < a.size(); ++i)
   {
      a[i] = static_cast<float>(i + 1);
      b[i] = static_cast<float>(i * 2 + 10);
      c[i] = static_cast<float>(i * 3 + 5);
   }

   auto expr = (a + b) * c;

   std::cout << "\n=== Chained Expression Iterator Traits ===" << std::endl;
   n_fails += testExpressionIteratorTraits<decltype(expr)>("(a + b) * c");

   return n_fails;
}

static int testScalarExpressionIteratorTraits()
{
   int n_fails = 0;

   auto a = makeTensor<float>(2, 3);

   for (index_t i = 0; i < a.size(); ++i)
   {
      a[i] = static_cast<float>(i + 1);
   }

   auto expr = a + 5.0f;

   std::cout << "\n=== Scalar Expression Iterator Traits ===" << std::endl;
   n_fails += testExpressionIteratorTraits<decltype(expr)>("a + 5.0f");

   return n_fails;
}

static int testBinaryExpressionIteratorSequence()
{
   int n_fails = 0;

   auto a = makeTensor<float>(2, 3);
   auto b = makeTensor<float>(2, 3);

   for (index_t i = 0; i < a.size(); ++i)
   {
      a[i] = static_cast<float>(i + 1);
      b[i] = static_cast<float>(i * 2);
   }

   auto expr = a + b;

   std::cout << "\n=== Binary Expression Iterator Sequence ===" << std::endl;

   // Test that we can iterate through the expression
   bool all_correct = true;
   index_t count = 0;
   for (const auto &value : expr)
   {
      float expected = a[count] + b[count];
      if (std::abs(value - expected) > 1e-6f)
      {
         std::cout << "\t" << ColorText::red("[ ✗ ]") << " Value mismatch at index " << count << ": expected "
                   << expected << ", got " << value << std::endl;
         all_correct = false;
         n_fails++;
         break;
      }
      count++;
   }

   if (all_correct && count == a.size())
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Iterator sequence traverses all " << count
                << " elements correctly." << std::endl;
   }
   else if (all_correct)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Iterator only traversed " << count << " elements, expected "
                << a.size() << std::endl;
      n_fails++;
   }

   return n_fails;
}

static int testSubviewExpressionIteratorSequence()
{
   int n_fails = 0;

   auto a = makeTensor<float>(4, 5);
   auto b = makeTensor<float>(4, 5);

   for (index_t i = 0; i < a.size(); ++i)
   {
      a[i] = static_cast<float>(i + 1);
      b[i] = static_cast<float>(i * 2);
   }

   auto expr = a + b;
   auto subview = expr.at(Range(1, 3), All());

   std::cout << "\n=== Subview Expression Iterator Sequence ===" << std::endl;

   // Test that we can iterate through the subview expression
   bool all_correct = true;
   index_t row_count = 0;
   for (index_t i = 1; i < 3; ++i)
   {
      for (index_t j = 0; j < 5; ++j)
      {
         float expected = a(i, j) + b(i, j);
         float actual = subview(row_count, j);
         if (std::abs(actual - expected) > 1e-6f)
         {
            std::cout << "\t" << ColorText::red("[ ✗ ]") << " Subview value mismatch at (" << row_count << ", " << j
                      << "): expected " << expected << ", got " << actual << std::endl;
            all_correct = false;
            n_fails++;
            break;
         }
      }
      if (!all_correct)
         break;
      row_count++;
   }

   if (all_correct)
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Subview iterator sequence traverses correctly." << std::endl;
   }

   return n_fails;
}

int testIterator()
{
   int n_failed = 0;

   Tensor<double, 2> a(3, 4);
   Tensor<double, 2> b(3, 4);

   for (index_t i = 0; i < a.size(); ++i)
   {
      a[i] = static_cast<double>(i * 3);
      b[i] = static_cast<double>(i * 7);
   }

   auto expr = a + b;

   index_t idx = 0;
   bool all_correct = true;
   for (auto val : expr)
   {
      double expected = a[idx] + b[idx];
      if (std::abs(val - expected) > 1e-10)
      {
         std::cout << "\t" << ColorText::red("[ ✗ ]") << " Iterator value mismatch at index " << idx << ": expected "
                   << expected << ", got " << val << std::endl;
         all_correct = false;
         n_failed++;
         break;
      }
      idx++;
   }

   if (idx != a.size())
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Iterator count mismatch: expected " << a.size() << ", got "
                << idx << std::endl;
      n_failed++;
   }
   else if (all_correct)
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Iterator: all elements correct with correct count"
                << std::endl;
   }

   return n_failed;
}

int testChainedExpressionIterator()
{
   int n_failed = 0;

   Tensor<float, 1> a(6);
   Tensor<float, 1> b(6);
   Tensor<float, 1> c(6);

   for (index_t i = 0; i < a.size(); ++i)
   {
      a[i] = static_cast<float>(i + 1);
      b[i] = static_cast<float>(i + 2);
      c[i] = static_cast<float>(i + 3);
   }

   auto expr = (a + b) * c;

   index_t idx = 0;
   bool all_correct = true;
   for (auto val : expr)
   {
      float expected = (a[idx] + b[idx]) * c[idx];
      if (std::abs(val - expected) > 1e-5f)
      {
         std::cout << "\t" << ColorText::red("[ ✗ ]") << " Chained iterator mismatch at index " << idx << ": expected "
                   << expected << ", got " << val << std::endl;
         all_correct = false;
         n_failed++;
         break;
      }
      idx++;
   }

   if (idx != a.size())
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Chained iterator count mismatch: expected " << a.size()
                << ", got " << idx << std::endl;
      n_failed++;
   }
   else if (all_correct)
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Chained expression iterator: all elements correct"
                << std::endl;
   }

   return n_failed;
}

int testBroadcastingIterator()
{
   int n_failed = 0;

   Tensor<double, 2> a(2, 3);
   Tensor<double, 3> b(2, 3, 1);

   for (index_t i = 0; i < a.size(); ++i)
   {
      a[i] = static_cast<double>(i * 2);
      b[i] = static_cast<double>(i * 5 + 10);
   }

   auto expr = a + b;

   index_t idx = 0;
   bool all_correct = true;
   for (auto val : expr)
   {
      double expected = a[idx] + b[idx];
      if (std::abs(val - expected) > 1e-10)
      {
         std::cout << "\t" << ColorText::red("[ ✗ ]") << " Broadcasting iterator mismatch at index " << idx
                   << ": expected " << expected << ", got " << val << std::endl;
         all_correct = false;
         n_failed++;
         break;
      }
      idx++;
   }

   if (idx != a.size())
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Broadcasting iterator count mismatch: expected " << a.size()
                << ", got " << idx << std::endl;
      n_failed++;
   }
   else if (all_correct)
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Broadcasting iterator: all elements correct" << std::endl;
   }

   return n_failed;
}

int main()
{
   int n_failed = 0;

   std::cout << "=== Iterator Traits Tests ===" << std::endl;
   n_failed += testBinaryExpressionIteratorTraits();
   n_failed += testBinaryExpressionSubviewIteratorTraits();
   n_failed += testChainedExpressionIteratorTraits();
   n_failed += testScalarExpressionIteratorTraits();
   n_failed += testBinaryExpressionIteratorSequence();
   n_failed += testSubviewExpressionIteratorSequence();

   std::cout << "\n=== Iterator Tests ===" << std::endl;
   n_failed += testIterator();
   n_failed += testChainedExpressionIterator();
   n_failed += testBroadcastingIterator();

   PRINT_RESULT(n_failed);
   return n_failed;
}
