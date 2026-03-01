#include <complex>

#include "TensorView/Expressions/BinaryExpressions.hpp"
#include "TensorView/Expressions/ScalarExpr.hpp"
#include "TensorView/Utility/Eval.hpp"
#include "test.hpp"

using namespace tensor;
using namespace tensor::details;

int testAdditionExpression()
{
   int n_failed = 0;

   Tensor<float, 2> a(2, 3);
   Tensor<float, 2> b(2, 3);

   for (index_t i = 0; i < a.size(); ++i)
   {
      a[i] = static_cast<float>(i * 2 + 1);
      b[i] = static_cast<float>(i * 3 + 5);
   }

   auto sum_expr = a + b;

   static_assert(is_expression<decltype(sum_expr)>, "sum_expr should be an expression");

   bool all_correct = true;
   for (index_t i = 0; i < a.size(); ++i)
   {
      float expected = a[i] + b[i];
      if (std::abs(sum_expr[i] - expected) > 1e-6f)
      {
         std::cout << "\t" << ColorText::red("[ ✗ ]") << " Addition mismatch at index " << i << ": expected "
                   << expected << ", got " << sum_expr[i] << std::endl;
         all_correct = false;
         n_failed++;
         break;
      }
   }

   if (all_correct)
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Addition expression: all elements correct" << std::endl;
   }

   return n_failed;
}

int testSubtractionExpression()
{
   int n_failed = 0;

   Tensor<int, 2> a(3, 4);
   Tensor<int, 2> b(3, 4);

   for (index_t i = 0; i < a.size(); ++i)
   {
      a[i] = static_cast<int>(i * 5 + 100);
      b[i] = static_cast<int>(i * 2 + 30);
   }

   auto diff_expr = a - b;

   bool all_correct = true;
   for (index_t i = 0; i < a.size(); ++i)
   {
      int expected = a[i] - b[i];
      if (diff_expr[i] != expected)
      {
         std::cout << "\t" << ColorText::red("[ ✗ ]") << " Subtraction mismatch at index " << i << ": expected "
                   << expected << ", got " << diff_expr[i] << std::endl;
         all_correct = false;
         n_failed++;
         break;
      }
   }

   if (all_correct)
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Subtraction expression: all elements correct" << std::endl;
   }

   return n_failed;
}

int testMultiplicationExpression()
{
   int n_failed = 0;

   Tensor<float, 1> a(5);
   Tensor<float, 1> b(5);

   for (index_t i = 0; i < a.size(); ++i)
   {
      a[i] = static_cast<float>(i + 1) * 1.5f;
      b[i] = static_cast<float>(i + 2) * 2.0f;
   }

   auto prod_expr = a * b;

   bool all_correct = true;
   for (index_t i = 0; i < a.size(); ++i)
   {
      float expected = a[i] * b[i];
      if (std::abs(prod_expr[i] - expected) > 1e-6f)
      {
         std::cout << "\t" << ColorText::red("[ ✗ ]") << " Multiplication mismatch at index " << i << ": expected "
                   << expected << ", got " << prod_expr[i] << std::endl;
         all_correct = false;
         n_failed++;
         break;
      }
   }

   if (all_correct)
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Multiplication expression: all elements correct" << std::endl;
   }

   return n_failed;
}

int testDivisionExpression()
{
   int n_failed = 0;

   Tensor<float, 1> a(4);
   Tensor<float, 1> b(4);

   for (index_t i = 0; i < a.size(); ++i)
   {
      a[i] = static_cast<float>(i + 1) * 10.0f;
      b[i] = static_cast<float>(i + 1) * 2.0f;
   }

   auto div_expr = a / b;

   bool all_correct = true;
   for (index_t i = 0; i < a.size(); ++i)
   {
      float expected = a[i] / b[i];
      if (std::abs(div_expr[i] - expected) > 1e-6f)
      {
         std::cout << "\t" << ColorText::red("[ ✗ ]") << " Division mismatch at index " << i << ": expected "
                   << expected << ", got " << div_expr[i] << std::endl;
         all_correct = false;
         n_failed++;
         break;
      }
   }

   if (all_correct)
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Division expression: all elements correct" << std::endl;
   }

   return n_failed;
}

int testChainedExpressions()
{
   int n_failed = 0;

   Tensor<float, 2> a(2, 3);
   Tensor<float, 2> b(2, 3);
   Tensor<float, 2> c(2, 3);

   for (index_t i = 0; i < a.size(); ++i)
   {
      a[i] = static_cast<float>(i + 1);
      b[i] = static_cast<float>(i + 2);
      c[i] = static_cast<float>(i + 3);
   }

   auto chained_expr = (a + b) * c;

   bool all_correct = true;
   for (index_t i = 0; i < a.size(); ++i)
   {
      float expected = (a[i] + b[i]) * c[i];
      if (std::abs(chained_expr[i] - expected) > 1e-6f)
      {
         std::cout << "\t" << ColorText::red("[ ✗ ]") << " Chained expression mismatch at index " << i << ": expected "
                   << expected << ", got " << chained_expr[i] << std::endl;
         all_correct = false;
         n_failed++;
         break;
      }
   }

   if (all_correct)
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Chained expression: all elements correct" << std::endl;
   }

   return n_failed;
}

int testMultiDimensionalAccess()
{
   int n_failed = 0;

   Tensor<float, 2> a(3, 4);
   Tensor<float, 2> b(3, 4);

   for (index_t i = 0; i < a.size(); ++i)
   {
      a[i] = static_cast<float>(i);
      b[i] = static_cast<float>(i * 2);
   }

   auto sum_expr = a + b;

   bool all_correct = true;
   for (index_t i = 0; i < 3; ++i)
   {
      for (index_t j = 0; j < 4; ++j)
      {
         float expected = a(i, j) + b(i, j);
         if (std::abs(sum_expr.at(i, j) - expected) > 1e-6f)
         {
            std::cout << "\t" << ColorText::red("[ ✗ ]") << " Expression at(" << i << ", " << j
                      << ") mismatch: expected " << expected << ", got " << sum_expr.at(i, j) << std::endl;
            all_correct = false;
            n_failed++;
            break;
         }
      }
      if (!all_correct)
         break;
   }

   if (all_correct)
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Multi-dimensional at() access: all elements correct"
                << std::endl;
   }

   all_correct = true;
   for (index_t i = 0; i < 3; ++i)
   {
      for (index_t j = 0; j < 4; ++j)
      {
         float expected = a(i, j) + b(i, j);
         if (std::abs(sum_expr(i, j) - expected) > 1e-6f)
         {
            std::cout << "\t" << ColorText::red("[ ✗ ]") << " Expression operator()(" << i << ", " << j << ") mismatch"
                      << std::endl;
            all_correct = false;
            n_failed++;
            break;
         }
      }
      if (!all_correct)
         break;
   }

   if (all_correct)
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Multi-dimensional operator() access: all elements correct"
                << std::endl;
   }

   return n_failed;
}

int testExpressionMetadata()
{
   int n_failed = 0;

   Tensor<int, 2> a(3, 4);
   Tensor<int, 2> b(3, 4);

   auto expr = a + b;

   if (expr.size() != 12)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Expression size() failed (expected 12, got " << expr.size()
                << ")" << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Expression size() correct" << std::endl;
   }

   if (expr.shape(0) != 3 || expr.shape(1) != 4)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Expression shape() failed" << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Expression shape() correct" << std::endl;
   }

   if (expr.numDims() != 2)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Expression numDims() failed (expected 2, got " << expr.numDims()
                << ")" << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Expression numDims() correct" << std::endl;
   }

   return n_failed;
}

int testScalarExprWithTensor()
{
   int n_failed = 0;

   Tensor<float, 2> a(2, 3);

   for (index_t i = 0; i < a.size(); ++i)
   {
      a[i] = static_cast<float>(i + 1);
   }

   auto scalar_expr = ScalarExpr<float>(5.0f);
   auto result = a + scalar_expr;

   bool all_correct = true;
   for (index_t i = 0; i < a.size(); ++i)
   {
      float expected = a[i] + 5.0f;
      if (std::abs(result[i] - expected) > 1e-6f)
      {
         std::cout << "\t" << ColorText::red("[ ✗ ]") << " ScalarExpr+Tensor mismatch at index " << i << ": expected "
                   << expected << ", got " << result[i] << std::endl;
         all_correct = false;
         n_failed++;
         break;
      }
   }

   if (all_correct)
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " ScalarExpr + Tensor: all elements correct" << std::endl;
   }

   return n_failed;
}

int testScalarExprProperties()
{
   int n_failed = 0;

   ScalarExpr<int> scalar_expr(42);

   if (scalar_expr.numDims() != 0)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " ScalarExpr numDims failed (expected 0, got "
                << scalar_expr.numDims() << ")" << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " ScalarExpr numDims correct (0)" << std::endl;
   }

   if (scalar_expr.size() != 1)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " ScalarExpr size failed (expected 1, got " << scalar_expr.size()
                << ")" << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " ScalarExpr size correct (1)" << std::endl;
   }

   if (scalar_expr.shape(0) != 1 || scalar_expr.shape(100) != 1)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " ScalarExpr shape failed" << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " ScalarExpr shape correct (always 1)" << std::endl;
   }

   if (scalar_expr[0] != 42 || scalar_expr[100] != 42)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " ScalarExpr operator[] failed" << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " ScalarExpr operator[] correct (always returns value)"
                << std::endl;
   }

   return n_failed;
}

int testScalarExprFromIndexing()
{
   int n_failed = 0;

   auto a = makeTensor<float>(3, 1);
   auto b = makeTensor<float>(3);

   for (int i = 0; i < 3; ++i)
   {
      b(i) = static_cast<float>(i * 10);
      for (int j = 0; j < 1; ++j)
      {
         a(i, j) = static_cast<float>(i * 100 + j);
      }
   }

   auto expr = a + b;

   if (expr.numDims() != 2 || expr.shape(0) != 3 || expr.shape(1) != 1)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " (3,1)+(3,): shape wrong" << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " (3,1)+(3,): broadcast shape correct" << std::endl;
   }

   float scalar_result = expr.at(1, 0);
   float expected = a(1, 0) + b(1);
   if (std::abs(scalar_result - expected) > 1e-6f)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " (3,1)+(3,) scalar indexing: value mismatch" << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " (3,1)+(3,) scalar indexing: value correct" << std::endl;
   }

   auto row_expr = expr.at(0, All());
   if (row_expr.numDims() != 1 || row_expr.size() != 1)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " (3,1)+(3,) row indexing: shape wrong" << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " (3,1)+(3,) row indexing: shape (1,) correct" << std::endl;
   }

   bool all_correct = true;
   for (int j = 0; j < 1; ++j)
   {
      float val = row_expr(j);
      float expected_val = a(0, j) + b(0);
      if (std::abs(val - expected_val) > 1e-6f)
      {
         all_correct = false;
         n_failed++;
         break;
      }
   }

   if (all_correct)
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " (3,1)+(3,) row operation: all elements correct" << std::endl;
   }
   else
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " (3,1)+(3,) row operation: element mismatch" << std::endl;
   }

   return n_failed;
}

int testScalarOperators()
{
   int n_failed = 0;

   auto a = makeTensor<float>(2, 3);

   for (int i = 0; i < 2; ++i)
   {
      for (int j = 0; j < 3; ++j)
      {
         a(i, j) = static_cast<float>(i * 3 + j + 1);
      }
   }

   auto expr_add = a + 5.0f;
   if (expr_add.numDims() != 2 || expr_add.size() != 6)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " tensor+scalar: shape wrong" << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " tensor+scalar: shape correct" << std::endl;
   }

   bool all_correct = true;
   for (int i = 0; i < 2; ++i)
   {
      for (int j = 0; j < 3; ++j)
      {
         float expected = a(i, j) + 5.0f;
         float actual = expr_add(i, j);
         if (std::abs(actual - expected) > 1e-6f)
         {
            all_correct = false;
            n_failed++;
            break;
         }
      }
      if (!all_correct)
         break;
   }

   if (all_correct)
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " tensor+scalar: all elements correct" << std::endl;
   }
   else
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " tensor+scalar: element mismatch" << std::endl;
   }

   auto expr_sub = 10.0 - a;
   all_correct = true;
   for (int i = 0; i < 2; ++i)
   {
      for (int j = 0; j < 3; ++j)
      {
         double expected = 10.0 - a(i, j);
         double actual = expr_sub(i, j);
         if (std::abs(actual - expected) > 1e-6)
         {
            all_correct = false;
            n_failed++;
            break;
         }
      }
      if (!all_correct)
         break;
   }

   if (all_correct)
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " scalar-tensor: all elements correct" << std::endl;
   }
   else
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " scalar-tensor: element mismatch" << std::endl;
   }

   auto expr_mul = a * 2;
   all_correct = true;
   for (int i = 0; i < 2; ++i)
   {
      for (int j = 0; j < 3; ++j)
      {
         int expected = (int)a(i, j) * 2;
         int actual = (int)expr_mul(i, j);
         if (actual != expected)
         {
            all_correct = false;
            n_failed++;
            break;
         }
      }
      if (!all_correct)
         break;
   }

   if (all_correct)
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " tensor*int: all elements correct" << std::endl;
   }
   else
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " tensor*int: element mismatch" << std::endl;
   }

   auto expr_div = a / 2.0f;
   all_correct = true;
   for (int i = 0; i < 2; ++i)
   {
      for (int j = 0; j < 3; ++j)
      {
         float expected = a(i, j) / 2.0f;
         float actual = expr_div(i, j);
         if (std::abs(actual - expected) > 1e-6f)
         {
            all_correct = false;
            n_failed++;
            break;
         }
      }
      if (!all_correct)
         break;
   }

   if (all_correct)
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " tensor/float: all elements correct" << std::endl;
   }
   else
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " tensor/float: element mismatch" << std::endl;
   }

   static_assert(std::is_same_v<decltype(expr_add)::value_type, float>);
   static_assert(std::is_same_v<decltype(expr_sub)::value_type, double>);
   static_assert(std::is_same_v<decltype(expr_mul)::value_type, float>);

   std::cout << "\t" << ColorText::green("[ ✓ ]") << " Promoted value_type correct" << std::endl;

   return n_failed;
}

int testExpressionOnSubview()
{
   int n_failed = 0;

   auto a = makeTensor<float>(4, 5);
   auto b = makeTensor<float>(4, 5);

   for (int i = 0; i < 4; ++i)
   {
      for (int j = 0; j < 5; ++j)
      {
         a(i, j) = i * 10 + j;
         b(i, j) = i * 100 + j * 10;
      }
   }

   auto sub_a = a.at(Range(1, 3), All());
   auto sub_b = b.at(Range(1, 3), All());

   auto expr = sub_a + sub_b;

   bool all_correct = true;
   for (int i = 0; i < 2; ++i)
   {
      for (int j = 0; j < 5; ++j)
      {
         float expected = a(i + 1, j) + b(i + 1, j);
         if (std::abs(expr(i, j) - expected) > 1e-6f)
         {
            std::cout << "\t" << ColorText::red("[ ✗ ]") << " Subview expression mismatch at (" << i << ", " << j
                      << "): expected " << expected << ", got " << expr(i, j) << std::endl;
            all_correct = false;
            n_failed++;
            break;
         }
      }
      if (!all_correct)
         break;
   }

   if (all_correct)
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Expression on subview: all elements correct" << std::endl;
   }

   return n_failed;
}

int testChainedExpressionWithSubviews()
{
   int n_failed = 0;

   auto a = makeTensor<float>(3, 4);
   auto b = makeTensor<float>(3, 4);
   auto c = makeTensor<float>(3, 4);

   for (int i = 0; i < 3; ++i)
   {
      for (int j = 0; j < 4; ++j)
      {
         a(i, j) = i + j;
         b(i, j) = i * 2 + j;
         c(i, j) = i * 3 + j;
      }
   }

   auto sub_a = a.at(All(), Range(1, 3));
   auto sub_b = b.at(All(), Range(1, 3));
   auto sub_c = c.at(All(), Range(1, 3));

   auto expr = (sub_a + sub_b) * sub_c;

   bool all_correct = true;
   for (int i = 0; i < 3; ++i)
   {
      for (int j = 0; j < 2; ++j)
      {
         float expected = (a(i, j + 1) + b(i, j + 1)) * c(i, j + 1);
         if (std::abs(expr(i, j) - expected) > 1e-6f)
         {
            std::cout << "\t" << ColorText::red("[ ✗ ]") << " Chained subview expression mismatch at (" << i << ", "
                      << j << "): expected " << expected << ", got " << expr(i, j) << std::endl;
            all_correct = false;
            n_failed++;
            break;
         }
      }
      if (!all_correct)
         break;
   }

   if (all_correct)
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Chained expression with subviews: all elements correct"
                << std::endl;
   }

   return n_failed;
}

int testSubviewOfExpression()
{
   int n_failed = 0;

   auto a = makeTensor<float>(4, 5);
   auto b = makeTensor<float>(4, 5);

   for (int i = 0; i < 4; ++i)
   {
      for (int j = 0; j < 5; ++j)
      {
         a(i, j) = i * 10 + j;
         b(i, j) = i * 100 + j * 10;
      }
   }

   auto expr = a + b;

   auto sub_expr = expr.at(Range(1, 3), All());

   bool all_correct = true;
   for (int i = 0; i < 2; ++i)
   {
      for (int j = 0; j < 5; ++j)
      {
         float expected = a(i + 1, j) + b(i + 1, j);
         if (std::abs(sub_expr(i, j) - expected) > 1e-6f)
         {
            std::cout << "\t" << ColorText::red("[ ✗ ]") << " Subview of expression mismatch at (" << i << ", " << j
                      << "): expected " << expected << ", got " << sub_expr(i, j) << std::endl;
            all_correct = false;
            n_failed++;
            break;
         }
      }
      if (!all_correct)
         break;
   }

   if (all_correct)
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Subview of expression: all elements correct" << std::endl;
   }

   return n_failed;
}

int testAssignExpressionToSubview()
{
   int n_failed = 0;

   // Create source tensors
   auto a = makeTensor<float>(4, 5);
   auto b = makeTensor<float>(4, 5);

   for (int i = 0; i < 4; ++i)
   {
      for (int j = 0; j < 5; ++j)
      {
         a(i, j) = i * 10 + j;
         b(i, j) = i * 100 + j * 10;
      }
   }

   // Create target tensor to hold the result
   auto result = makeTensor<float>(4, 5);
   for (int i = 0; i < 4; ++i)
   {
      for (int j = 0; j < 5; ++j)
      {
         result(i, j) = 0.0f; // Initialize to zero
      }
   }

   // Create an expression from source tensors
   auto expr = a + b;

   // Get a subview of the result tensor and assign the expression to it
   auto result_subview = result.at(Range(1, 3), All());
   result_subview = expr.at(Range(1, 3), All());

   // Verify the assignment worked correctly
   bool all_correct = true;
   for (int i = 0; i < 2; ++i)
   {
      for (int j = 0; j < 5; ++j)
      {
         float expected = a(i + 1, j) + b(i + 1, j);
         float actual = result(i + 1, j);
         if (std::abs(actual - expected) > 1e-6f)
         {
            std::cout << "\t" << ColorText::red("[ ✗ ]") << " Assign expression to subview failed at (" << (i + 1)
                      << ", " << j << "): expected " << expected << ", got " << actual << std::endl;
            all_correct = false;
            n_failed++;
            break;
         }
      }
      if (!all_correct)
         break;
   }

   // Verify that other regions of result tensor are unmodified
   if (all_correct)
   {
      for (int i = 0; i < 1; ++i)
      {
         for (int j = 0; j < 5; ++j)
         {
            if (std::abs(result(i, j) - 0.0f) > 1e-6f)
            {
               std::cout << "\t" << ColorText::red("[ ✗ ]") << " First row was modified: expected 0, got "
                         << result(i, j) << std::endl;
               all_correct = false;
               n_failed++;
               break;
            }
         }
         if (!all_correct)
            break;
      }
   }

   if (all_correct)
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Assign expression to subview: all elements correct"
                << std::endl;
   }

   return n_failed;
}

int testSameDimensionalBroadcasting()
{
   int n_failed = 0;

   Tensor<float, 2> a(3, 4);
   Tensor<float, 2> b(3, 4);

   for (index_t i = 0; i < a.size(); ++i)
   {
      a[i] = static_cast<float>(i + 1);
      b[i] = static_cast<float>(i * 2 + 10);
   }

   auto expr = a + b;

   if (expr.numDims() != 2)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Same-dim: numDims wrong (expected 2, got " << expr.numDims()
                << ")" << std::endl;
      n_failed++;
   }
   else if (expr.shape(0) != 3 || expr.shape(1) != 4)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Same-dim: shape wrong" << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Same-dim shape correct" << std::endl;
   }

   bool all_correct = true;
   for (index_t i = 0; i < a.size(); ++i)
   {
      float expected = a[i] + b[i];
      if (std::abs(expr[i] - expected) > 1e-6f)
      {
         std::cout << "\t" << ColorText::red("[ ✗ ]") << " Same-dim: element mismatch at index " << i << std::endl;
         all_correct = false;
         n_failed++;
         break;
      }
   }

   if (all_correct)
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Same-dim: all elements correct" << std::endl;
   }

   return n_failed;
}

int testTrailingSingletonBroadcasting()
{
   int n_failed = 0;

   Tensor<float, 2> a(3, 4);
   Tensor<float, 3> b(3, 4, 1);

   for (index_t i = 0; i < a.size(); ++i)
   {
      a[i] = static_cast<float>(i + 1);
      b[i] = static_cast<float>(i * 3 + 20);
   }

   auto expr = a + b;

   if (expr.numDims() != 3)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Trailing singleton: numDims wrong (expected 3, got "
                << expr.numDims() << ")" << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Trailing singleton: numDims correct" << std::endl;
   }

   if (expr.shape(0) != 3 || expr.shape(1) != 4 || expr.shape(2) != 1)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Trailing singleton: shape wrong" << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Trailing singleton: shape correct" << std::endl;
   }

   bool all_correct = true;
   for (index_t i = 0; i < a.size(); ++i)
   {
      float expected = a[i] + b[i];
      if (std::abs(expr[i] - expected) > 1e-6f)
      {
         std::cout << "\t" << ColorText::red("[ ✗ ]") << " Trailing singleton: element mismatch at index " << i
                   << std::endl;
         all_correct = false;
         n_failed++;
         break;
      }
   }

   if (all_correct)
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Trailing singleton: all elements correct" << std::endl;
   }

   return n_failed;
}

int testTrailingSingletonBroadcasting3D()
{
   int n_failed = 0;

   Tensor<float, 3> a(3, 4, 1);
   Tensor<float, 2> b(3, 4);

   for (index_t i = 0; i < a.size(); ++i)
   {
      a[i] = static_cast<float>(i + 1);
   }

   for (index_t i = 0; i < b.size(); ++i)
   {
      b[i] = static_cast<float>(i * 3 + 20);
   }

   auto expr = a + b;

   if (expr.numDims() != 3)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " 3D+2D: numDims wrong (expected 3, got " << expr.numDims() << ")"
                << std::endl;
      n_failed++;
   }
   else if (expr.shape(0) != 3 || expr.shape(1) != 4 || expr.shape(2) != 1)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " 3D+2D: shape wrong" << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " 3D+2D: shape correct" << std::endl;
   }

   bool all_correct = true;
   for (index_t i = 0; i < a.size(); ++i)
   {
      float expected = a[i] + b[i];
      if (std::abs(expr[i] - expected) > 1e-6f)
      {
         std::cout << "\t" << ColorText::red("[ ✗ ]") << " 3D+2D: element mismatch at index " << i << std::endl;
         all_correct = false;
         n_failed++;
         break;
      }
   }

   if (all_correct)
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " 3D+2D: all elements correct" << std::endl;
   }

   return n_failed;
}

int testMultipleTrailingSingletons()
{
   int n_failed = 0;

   Tensor<float, 3> a(3, 4, 1);
   Tensor<float, 4> b(3, 4, 1, 1);

   for (index_t i = 0; i < a.size(); ++i)
   {
      a[i] = static_cast<float>(i + 1);
   }

   for (index_t i = 0; i < b.size(); ++i)
   {
      b[i] = static_cast<float>(i * 2 + 30);
   }

   auto expr = a + b;

   if (expr.numDims() != 4)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Multiple singletons: numDims wrong" << std::endl;
      n_failed++;
   }
   else if (expr.shape(0) != 3 || expr.shape(1) != 4 || expr.shape(2) != 1 || expr.shape(3) != 1)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Multiple singletons: shape wrong" << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Multiple singletons: shape correct" << std::endl;
   }

   bool all_correct = true;
   for (index_t i = 0; i < a.size(); ++i)
   {
      float expected = a[i] + b[i];
      if (std::abs(expr[i] - expected) > 1e-6f)
      {
         std::cout << "\t" << ColorText::red("[ ✗ ]") << " Multiple singletons: element mismatch at index " << i
                   << std::endl;
         all_correct = false;
         n_failed++;
         break;
      }
   }

   if (all_correct)
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Multiple singletons: all elements correct" << std::endl;
   }

   return n_failed;
}

int testScalarBroadcasting()
{
   int n_failed = 0;

   Tensor<float, 2> a(3, 4);
   ScalarExpr<float> scalar(5.5f);

   for (index_t i = 0; i < a.size(); ++i)
   {
      a[i] = static_cast<float>(i + 1);
   }

   auto expr = a + scalar;

   if (expr.numDims() != 2)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Scalar: numDims wrong" << std::endl;
      n_failed++;
   }
   else if (expr.shape(0) != 3 || expr.shape(1) != 4)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Scalar: shape wrong" << std::endl;
      n_failed++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Scalar: shape correct" << std::endl;
   }

   bool all_correct = true;
   for (index_t i = 0; i < a.size(); ++i)
   {
      float expected = a[i] + 5.5f;
      if (std::abs(expr[i] - expected) > 1e-6f)
      {
         std::cout << "\t" << ColorText::red("[ ✗ ]") << " Scalar: element mismatch at index " << i << std::endl;
         all_correct = false;
         n_failed++;
         break;
      }
   }

   if (all_correct)
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Scalar: all elements correct" << std::endl;
   }

   return n_failed;
}

int testBroadcastingMultiDimensionalAccess()
{
   int n_failed = 0;

   Tensor<float, 2> a(3, 4);
   Tensor<float, 3> b(3, 4, 1);

   for (index_t i = 0; i < a.size(); ++i)
   {
      a[i] = static_cast<float>(i + 1);
      b[i] = static_cast<float>(i * 10 + 100);
   }

   auto expr = a + b;

   bool all_correct = true;
   for (int i = 0; i < 3; ++i)
   {
      for (int j = 0; j < 4; ++j)
      {
         float expected = a(i, j) + b(i, j, 0);
         float actual = expr(i, j, 0);
         if (std::abs(actual - expected) > 1e-6f)
         {
            std::cout << "\t" << ColorText::red("[ ✗ ]") << " Broadcasting access mismatch at (" << i << ", " << j
                      << ", 0): expected " << expected << ", got " << actual << std::endl;
            all_correct = false;
            n_failed++;
            break;
         }
      }
      if (!all_correct)
         break;
   }

   if (all_correct)
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Broadcasting multi-dimensional access: all elements correct"
                << std::endl;
   }

   return n_failed;
}

int testChainedBroadcastingExpression()
{
   int n_failed = 0;

   Tensor<float, 2> a(2, 3);
   Tensor<float, 3> b(2, 3, 1);
   Tensor<float, 3> c(2, 3, 1);

   for (index_t i = 0; i < a.size(); ++i)
   {
      a[i] = static_cast<float>(i + 1);
   }

   for (index_t i = 0; i < b.size(); ++i)
   {
      b[i] = static_cast<float>(i * 2 + 10);
   }

   for (index_t i = 0; i < c.size(); ++i)
   {
      c[i] = static_cast<float>(i * 3 + 20);
   }

   auto expr = (a + b) * c;

   bool all_correct = true;
   for (int i = 0; i < 2; ++i)
   {
      for (int j = 0; j < 3; ++j)
      {
         float expected = (a(i, j) + b(i, j, 0)) * c(i, j, 0);
         float actual = expr(i, j, 0);
         if (std::abs(actual - expected) > 1e-6f)
         {
            std::cout << "\t" << ColorText::red("[ ✗ ]") << " Chained broadcasting mismatch at (" << i << ", " << j
                      << ", 0): expected " << expected << ", got " << actual << std::endl;
            all_correct = false;
            n_failed++;
            break;
         }
      }
      if (!all_correct)
         break;
   }

   if (all_correct)
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Chained broadcasting expression: all elements correct"
                << std::endl;
   }

   return n_failed;
}

int testBinaryExprFancyIndexing_BothViews()
{
   int n_fails = 0;

   auto a = makeTensor<double>(5, 4, 3);
   auto b = makeTensor<double>(5, 4, 3);

   for (int i = 0; i < 5; ++i)
      for (int j = 0; j < 4; ++j)
         for (int k = 0; k < 3; ++k)
         {
            a(i, j, k) = i * 100 + j * 10 + k;
            b(i, j, k) = (i + 1) * 100 + (j + 1) * 10 + (k + 1);
         }

   auto expr = a + b;

   auto subexpr = expr.at(Range(1, 4), All(), 1);

   if (subexpr.numDims() != 2)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]")
                << " Fancy indexing on binary expression produced wrong number of dimensions. "
                << "Expected 2, got " << subexpr.numDims() << "." << std::endl;
      n_fails++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]")
                << " Fancy indexing on binary expression produced correct dimensions." << std::endl;
   }

   bool mismatch = false;
   for (int i = 0; i < 3; ++i)
   {
      for (int j = 0; j < 4; ++j)
      {
         double expected = a(i + 1, j, 1) + b(i + 1, j, 1);
         double actual = subexpr(i, j);
         if (std::abs(actual - expected) > 1e-10)
         {
            std::cout << "\t" << ColorText::red("[ ✗ ]") << " Mismatch at (" << i << ", " << j << "): "
                      << "expected " << expected << ", got " << actual << std::endl;
            mismatch = true;
            n_fails++;
            break;
         }
      }
      if (mismatch)
         break;
   }

   if (!mismatch)
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Values of fancy-indexed binary expression are correct."
                << std::endl;
   }

   return n_fails;
}

int testBinaryExprFancyIndexing_MixedScalarView()
{
   int n_fails = 0;

   auto a = makeTensor<double>(5, 4, 3);
   auto b = makeTensor<double>(5, 4, 3);

   for (int i = 0; i < 5; ++i)
      for (int j = 0; j < 4; ++j)
         for (int k = 0; k < 3; ++k)
         {
            a(i, j, k) = i * 100 + j * 10 + k;
            b(i, j, k) = (i + 1) * 1000 + (j + 1) * 100 + (k + 1) * 10;
         }

   auto expr = a + b;

   auto subexpr = expr.at(Range(1, 4), 2, 1);

   if (subexpr.numDims() != 1)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Mixed fancy indexing produced wrong dimensions. "
                << "Expected 1, got " << subexpr.numDims() << "." << std::endl;
      n_fails++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Mixed fancy indexing produced correct dimensions."
                << std::endl;
   }

   bool mismatch = false;
   for (int i = 0; i < 3; ++i)
   {
      double expected = a(i + 1, 2, 1) + b(i + 1, 2, 1);
      double actual = subexpr(i);
      if (std::abs(actual - expected) > 1e-10)
      {
         std::cout << "\t" << ColorText::red("[ ✗ ]") << " Mismatch at index " << i << ": "
                   << "expected " << expected << ", got " << actual << std::endl;
         mismatch = true;
         n_fails++;
         break;
      }
   }

   if (!mismatch)
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Values of mixed fancy-indexed expression are correct."
                << std::endl;
   }

   return n_fails;
}

int testBinaryExprScalarIndexing()
{
   int n_fails = 0;

   auto a = makeTensor<double>(5, 4, 3);
   auto b = makeTensor<double>(5, 4, 3);

   for (int i = 0; i < 5; ++i)
      for (int j = 0; j < 4; ++j)
         for (int k = 0; k < 3; ++k)
         {
            a(i, j, k) = i * 100 + j * 10 + k;
            b(i, j, k) = (i + 1) * 100 + (j + 1) * 10 + (k + 1);
         }

   auto expr = a + b;

   auto result = expr.at(2, 3, 1);

   double expected = a(2, 3, 1) + b(2, 3, 1);
   if (std::abs(result - expected) > 1e-10)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Scalar indexing produced wrong value. "
                << "Expected " << expected << ", got " << result << std::endl;
      n_fails++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Scalar indexing produced correct scalar value." << std::endl;
   }

   return n_fails;
}

int testBinaryExprFancyIndexing_ChainedExpressions()
{
   int n_fails = 0;

   auto a = makeTensor<double>(4, 5, 3);
   auto b = makeTensor<double>(4, 5, 3);
   auto c = makeTensor<double>(4, 5, 3);

   for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 5; ++j)
         for (int k = 0; k < 3; ++k)
         {
            a(i, j, k) = i + j + k;
            b(i, j, k) = i * j + k;
            c(i, j, k) = i - j + k * 2;
         }

   auto expr = (a + b) * c;

   auto subexpr = expr.at(Range(1, 3), 2, All());

   if (subexpr.numDims() != 2)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]")
                << " Fancy indexing on chained expression produced wrong dimensions. "
                << "Expected 2, got " << subexpr.numDims() << "." << std::endl;
      n_fails++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]")
                << " Fancy indexing on chained expression produced correct dimensions." << std::endl;
   }

   bool mismatch = false;
   for (int i = 0; i < 2; ++i)
   {
      for (int k = 0; k < 3; ++k)
      {
         double expected = (a(i + 1, 2, k) + b(i + 1, 2, k)) * c(i + 1, 2, k);
         double actual = subexpr(i, k);
         if (std::abs(actual - expected) > 1e-10)
         {
            std::cout << "\t" << ColorText::red("[ ✗ ]") << " Mismatch in chained expression at (" << i << ", " << k
                      << "): "
                      << "expected " << expected << ", got " << actual << std::endl;
            mismatch = true;
            n_fails++;
            break;
         }
      }
      if (mismatch)
         break;
   }

   if (!mismatch)
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Values of fancy-indexed chained expression are correct."
                << std::endl;
   }

   return n_fails;
}

int testBinaryExprFancyIndexing_Assignment()
{
   int n_fails = 0;

   auto a = makeTensor<double>(5, 4);
   auto b = makeTensor<double>(5, 4);

   for (int i = 0; i < 5; ++i)
      for (int j = 0; j < 4; ++j)
      {
         a(i, j) = i * 10 + j;
         b(i, j) = (i + 1) * 10 + (j + 1);
      }

   auto expr = a + b;
   auto subexpr = expr.at(Range(1, 4), All());

   auto result = makeTensor<double>(3, 4);

   for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 4; ++j)
         result(i, j) = subexpr(i, j);

   bool mismatch = false;
   for (int i = 0; i < 3; ++i)
   {
      for (int j = 0; j < 4; ++j)
      {
         double expected = a(i + 1, j) + b(i + 1, j);
         double actual = result(i, j);
         if (std::abs(actual - expected) > 1e-10)
         {
            std::cout << "\t" << ColorText::red("[ ✗ ]") << " Assignment mismatch at (" << i << ", " << j << "): "
                      << "expected " << expected << ", got " << actual << std::endl;
            mismatch = true;
            n_fails++;
            break;
         }
      }
      if (mismatch)
         break;
   }

   if (!mismatch)
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Assignment from fancy-indexed expression works correctly."
                << std::endl;
   }

   return n_fails;
}

int testBinaryExprFancyIndexing_DifferentDimensions()
{
   int n_fails = 0;

   auto a = makeTensor<double>(2, 3, 1);
   auto b = makeTensor<double>(2, 3);

   for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 3; ++j)
      {
         a(i, j, 0) = i * 10 + j;
         b(i, j) = (i + 1) * 100 + (j + 1) * 10;
      }

   auto expr = a + b;

   if (expr.numDims() != 3)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Different-dim expression has wrong numDims. "
                << "Expected 3, got " << expr.numDims() << "." << std::endl;
      n_fails++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Different-dim expression has correct numDims." << std::endl;
   }

   auto subexpr = expr.at(Range(0, 2), 1, All());

   if (subexpr.numDims() != 2)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Fancy indexing on different-dim expression has wrong dims. "
                << "Expected 2, got " << subexpr.numDims() << "." << std::endl;
      n_fails++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Fancy indexing on different-dim expression has correct dims."
                << std::endl;
   }

   bool mismatch = false;
   for (int i = 0; i < 2; ++i)
   {
      double expected = a(i, 1, 0) + b(i, 1);
      double actual = subexpr(i, 0);
      if (std::abs(actual - expected) > 1e-10)
      {
         std::cout << "\t" << ColorText::red("[ ✗ ]") << " Value mismatch at index " << i << ": "
                   << "expected " << expected << ", got " << actual << std::endl;
         mismatch = true;
         n_fails++;
         break;
      }
   }

   if (!mismatch)
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Values of fancy-indexed different-dim expression are correct."
                << std::endl;
   }

   return n_fails;
}

int testBinaryExprScalarIndexing_DifferentDimensions()
{
   int n_fails = 0;

   auto a = makeTensor<double>(3, 4, 1);
   auto b = makeTensor<double>(3, 4);

   for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 4; ++j)
      {
         a(i, j, 0) = i * 10.0 + j * 1.0;
         b(i, j) = (i + 1) * 100.0 + (j + 1) * 10.0;
      }

   auto expr = a + b;

   auto result = expr.at(1, 2, 0);

   double expected = a(1, 2, 0) + b(1, 2);
   if (std::abs(result - expected) > 1e-10)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Scalar indexing on different-dim expression gave wrong value. "
                << "Expected " << expected << ", got " << result << std::endl;
      n_fails++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Scalar indexing on different-dim expression is correct."
                << std::endl;
   }

   return n_fails;
}

int testBinaryExprSubview_MixedDimensionality()
{
   int n_fails = 0;

   auto a = makeTensor<double>(4, 3, 1);
   auto b = makeTensor<double>(4, 3);

   for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 3; ++j)
      {
         b(i, j) = i * 100.0 + j * 10.0;
         a(i, j, 0) = i * 100.0 + j * 10.0 + 1.0;
      }

   auto expr = a + b;

   auto subexpr = expr.at(1, Range(0, 3), All());

   if (subexpr.numDims() != 2)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Subview of different-dim expression has wrong dims. "
                << "Expected 2, got " << subexpr.numDims() << "." << std::endl;
      n_fails++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Subview of different-dim expression has correct dims."
                << std::endl;
   }

   bool mismatch = false;
   for (int j = 0; j < 3; ++j)
   {
      double expected = a(1, j, 0) + b(1, j);
      double actual = subexpr(j, 0);
      if (std::abs(actual - expected) > 1e-10)
      {
         std::cout << "\t" << ColorText::red("[ ✗ ]") << " Value mismatch at (" << j << ", 0): "
                   << "expected " << expected << ", got " << actual << std::endl;
         mismatch = true;
         n_fails++;
         break;
      }
   }

   if (!mismatch)
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Values of subview from different-dim expression are correct."
                << std::endl;
   }

   return n_fails;
}

int testBinaryExprFancyIndexing_AsymmetricCollapse()
{
   int n_fails = 0;

   auto a = makeTensor<double>(2, 3, 4, 1);
   auto b = makeTensor<double>(2, 3, 4);

   for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 3; ++j)
         for (int k = 0; k < 4; ++k)
         {
            a(i, j, k, 0) = i * 1000.0 + j * 100.0 + k * 10.0;
            b(i, j, k) = (i + 1) * 1000.0 + (j + 1) * 100.0 + (k + 1) * 10.0;
         }

   auto expr = a + b;

   auto subexpr = expr.at(Range(0, 2), 1, Range(1, 3), All());

   if (subexpr.numDims() != 3)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Asymmetric collapse has wrong dims. "
                << "Expected 3, got " << subexpr.numDims() << "." << std::endl;
      n_fails++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Asymmetric collapse has correct dims." << std::endl;
   }

   bool mismatch = false;
   for (int i = 0; i < 2; ++i)
   {
      for (int k = 0; k < 2; ++k)
      {
         double expected = a(i, 1, k + 1, 0) + b(i, 1, k + 1);
         double actual = subexpr(i, k, 0);
         if (std::abs(actual - expected) > 1e-10)
         {
            std::cout << "\t" << ColorText::red("[ ✗ ]") << " Value mismatch in asymmetric collapse at (" << i << ", "
                      << k << ", 0): "
                      << "expected " << expected << ", got " << actual << std::endl;
            mismatch = true;
            n_fails++;
            break;
         }
      }
      if (mismatch)
         break;
   }

   if (!mismatch)
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Values of asymmetrically collapsed expression are correct."
                << std::endl;
   }

   return n_fails;
}

int testEval()
{
   int n_failed = 0;

   // Test 1: Basic eval with addition
   {
      Tensor<float, 2> a(2, 3);
      Tensor<float, 2> b(2, 3);

      for (index_t i = 0; i < a.size(); ++i)
      {
         a[i] = static_cast<float>(i + 1);
         b[i] = static_cast<float>(i * 2);
      }

      auto expr = a + b;
      auto result = eval(expr);

      bool all_correct = true;
      for (index_t i = 0; i < result.size(); ++i)
      {
         float expected = a[i] + b[i];
         if (std::abs(result[i] - expected) > 1e-6f)
         {
            std::cout << "\t" << ColorText::red("[ ✗ ]") << " eval(a + b) mismatch at index " << i << ": expected "
                      << expected << ", got " << result[i] << std::endl;
            all_correct = false;
            n_failed++;
            break;
         }
      }

      if (all_correct)
      {
         std::cout << "\t" << ColorText::green("[ ✓ ]") << " eval(a + b) produces correct tensor" << std::endl;
      }
   }

   // Test 2: eval with scalar operators
   {
      Tensor<float, 2> a(2, 3);
      for (index_t i = 0; i < a.size(); ++i)
         a[i] = static_cast<float>(i);

      auto expr = a * 2.0f;
      auto result = eval(expr);

      bool all_correct = true;
      for (index_t i = 0; i < result.size(); ++i)
      {
         float expected = a[i] * 2.0f;
         if (std::abs(result[i] - expected) > 1e-6f)
         {
            std::cout << "\t" << ColorText::red("[ ✗ ]") << " eval(a * 2.0f) mismatch at index " << i << ": expected "
                      << expected << ", got " << result[i] << std::endl;
            all_correct = false;
            n_failed++;
            break;
         }
      }

      if (all_correct)
      {
         std::cout << "\t" << ColorText::green("[ ✓ ]") << " eval(a * 2.0f) produces correct tensor" << std::endl;
      }
   }

   // Test 3: eval with broadcasting (trailing singleton dimension)
   {
      Tensor<float, 2> a(3, 4);
      Tensor<float, 3> b(3, 4, 1);

      for (int i = 0; i < 3; ++i)
         for (int j = 0; j < 4; ++j)
         {
            a(i, j) = static_cast<float>(i * 10 + j);
            b(i, j, 0) = static_cast<float>(j * 2);
         }

      auto expr = a + b;
      auto result = eval(expr);

      bool all_correct = true;
      for (int i = 0; i < 3 && all_correct; ++i)
      {
         for (int j = 0; j < 4; ++j)
         {
            float expected = a(i, j) + b(i, j, 0);
            if (std::abs(result(i, j, 0) - expected) > 1e-6f)
            {
               std::cout << "\t" << ColorText::red("[ ✗ ]") << " eval with broadcasting mismatch at (" << i << ", " << j
                         << ", 0): "
                         << "expected " << expected << ", got " << result(i, j, 0) << std::endl;
               all_correct = false;
               n_failed++;
               break;
            }
         }
      }

      if (all_correct)
      {
         std::cout << "\t" << ColorText::green("[ ✓ ]") << " eval with broadcasting produces correct tensor"
                   << std::endl;
      }
   }

   // Test 4: eval with type promotion (int + float)
   {
      Tensor<int, 1> a(3);
      for (index_t i = 0; i < a.size(); ++i)
         a[i] = static_cast<int>(i + 1);

      float scalar = 2.5f;
      auto expr = a + scalar;
      auto result = eval(expr);

      // Check that result type is promoted to float
      static_assert(std::is_same_v<decltype(result)::value_type, float>, "Result should have float value_type");

      bool all_correct = true;
      for (index_t i = 0; i < result.size(); ++i)
      {
         float expected = static_cast<float>(a[i]) + scalar;
         if (std::abs(result[i] - expected) > 1e-6f)
         {
            std::cout << "\t" << ColorText::red("[ ✗ ]") << " eval(int_tensor + float) mismatch at index " << i
                      << ": expected " << expected << ", got " << result[i] << std::endl;
            all_correct = false;
            n_failed++;
            break;
         }
      }

      if (all_correct)
      {
         std::cout << "\t" << ColorText::green("[ ✓ ]") << " eval(int_tensor + float) produces correct float tensor"
                   << std::endl;
      }
   }

   return n_failed;
}

// Helper trait to check if a type is complex
template <typename T>
struct is_complex : std::false_type
{};

template <typename Real>
struct is_complex<std::complex<Real>> : std::true_type
{};

template <typename T>
inline constexpr bool is_complex_v = is_complex<T>::value;

// Helper for tolerance-based comparison
template <typename T>
inline bool isClose(const T &a, const T &b, double tol = 1e-6)
{
   if constexpr (std::is_integral_v<T>)
      return a == b;
   else if constexpr (is_complex_v<T>)
   {
      return isClose(a.real(), b.real(), tol) && isClose(a.imag(), b.imag(), tol);
   }
   else
      return std::abs(a - b) <= tol * std::max(std::abs(a), std::abs(b)) + tol;
}

// Template test for mixed-type binary operations
template <typename LhsType, typename RhsType>
int testMixedTypeBinaryOperations()
{
   int n_failed = 0;

   // Get a descriptive name for the types
   std::string lhs_name = (std::is_same_v<LhsType, int>)                    ? "int"
                          : (std::is_same_v<LhsType, float>)                ? "float"
                          : (std::is_same_v<LhsType, double>)               ? "double"
                          : (std::is_same_v<LhsType, std::complex<float>>)  ? "complex<float>"
                          : (std::is_same_v<LhsType, std::complex<double>>) ? "complex<double>"
                                                                            : "unknown";
   std::string rhs_name = (std::is_same_v<RhsType, int>)                    ? "int"
                          : (std::is_same_v<RhsType, float>)                ? "float"
                          : (std::is_same_v<RhsType, double>)               ? "double"
                          : (std::is_same_v<RhsType, std::complex<float>>)  ? "complex<float>"
                          : (std::is_same_v<RhsType, std::complex<double>>) ? "complex<double>"
                                                                            : "unknown";

   auto test_name = lhs_name + " + " + rhs_name;

   using common_t = std::common_type_t<LhsType, RhsType>;

   auto a = makeTensor<LhsType>(2, 3);
   auto b = makeTensor<RhsType>(2, 3);

   // Initialize tensors with appropriate values
   if constexpr (is_complex_v<LhsType>)
   {
      for (index_t i = 0; i < a.size(); ++i)
         a[i] = LhsType(i + 1, i + 2);
   }
   else
   {
      for (index_t i = 0; i < a.size(); ++i)
         a[i] = static_cast<LhsType>(i + 1);
   }

   if constexpr (is_complex_v<RhsType>)
   {
      for (index_t i = 0; i < b.size(); ++i)
         b[i] = RhsType(i * 2 + 10, i * 2 + 11);
   }
   else
   {
      for (index_t i = 0; i < b.size(); ++i)
         b[i] = static_cast<RhsType>(i * 2 + 10);
   }

   // Test addition
   {
      auto expr = a + b;
      bool all_correct = true;
      for (index_t i = 0; i < a.size(); ++i)
      {
         auto expected = static_cast<common_t>(a[i]) + static_cast<common_t>(b[i]);
         auto actual = expr[i];
         if (!isClose(actual, expected, 1e-5))
         {
            std::cout << "\t" << ColorText::red("[ ✗ ]") << " Addition (" << test_name << ") mismatch at index " << i
                      << std::endl;
            all_correct = false;
            n_failed++;
            break;
         }
      }
      if (all_correct)
      {
         std::cout << "\t" << ColorText::green("[ ✓ ]") << " Addition: " << test_name << std::endl;
      }
   }

   // Test subtraction
   {
      auto expr = a - b;
      bool all_correct = true;
      for (index_t i = 0; i < a.size(); ++i)
      {
         auto expected = static_cast<common_t>(a[i]) - static_cast<common_t>(b[i]);
         auto actual = expr[i];
         if (!isClose(actual, expected, 1e-5))
         {
            std::cout << "\t" << ColorText::red("[ ✗ ]") << " Subtraction (" << test_name << ") mismatch at index " << i
                      << std::endl;
            all_correct = false;
            n_failed++;
            break;
         }
      }
      if (all_correct)
      {
         std::cout << "\t" << ColorText::green("[ ✓ ]") << " Subtraction: " << test_name << std::endl;
      }
   }

   // Test multiplication
   {
      auto expr = a * b;
      bool all_correct = true;
      for (index_t i = 0; i < a.size(); ++i)
      {
         auto expected = static_cast<common_t>(a[i]) * static_cast<common_t>(b[i]);
         auto actual = expr[i];
         if (!isClose(actual, expected, 1e-5))
         {
            std::cout << "\t" << ColorText::red("[ ✗ ]") << " Multiplication (" << test_name << ") mismatch at index "
                      << i << std::endl;
            all_correct = false;
            n_failed++;
            break;
         }
      }
      if (all_correct)
      {
         std::cout << "\t" << ColorText::green("[ ✓ ]") << " Multiplication: " << test_name << std::endl;
      }
   }

   // Test division
   {
      auto expr = a / b;
      bool all_correct = true;
      for (index_t i = 0; i < a.size(); ++i)
      {
         auto expected = static_cast<common_t>(a[i]) / static_cast<common_t>(b[i]);
         auto actual = expr[i];
         if (!isClose(actual, expected, 1e-5))
         {
            std::cout << "\t" << ColorText::red("[ ✗ ]") << " Division (" << test_name << ") mismatch at index " << i
                      << std::endl;
            all_correct = false;
            n_failed++;
            break;
         }
      }
      if (all_correct)
      {
         std::cout << "\t" << ColorText::green("[ ✓ ]") << " Division: " << test_name << std::endl;
      }
   }

   return n_failed;
}

int main()

{
   int n_failed = 0;

   std::cout << "\n=== Basic Binary Operations ===" << std::endl;
   n_failed += testAdditionExpression();
   n_failed += testSubtractionExpression();
   n_failed += testMultiplicationExpression();
   n_failed += testDivisionExpression();

   std::cout << "\n=== Chained Expressions ===" << std::endl;
   n_failed += testChainedExpressions();

   std::cout << "\n=== Multi-dimensional Access ===" << std::endl;
   n_failed += testMultiDimensionalAccess();

   std::cout << "\n=== Expression Metadata ===" << std::endl;
   n_failed += testExpressionMetadata();

   std::cout << "\n=== ScalarExpr Tests ===" << std::endl;
   n_failed += testScalarExprWithTensor();
   n_failed += testScalarExprProperties();
   n_failed += testScalarExprFromIndexing();
   n_failed += testScalarOperators();

   std::cout << "\n=== Eval Tests ===" << std::endl;
   n_failed += testEval();

   std::cout << "\n=== Mixed-Type Binary Operations ===" << std::endl;
   n_failed += testMixedTypeBinaryOperations<int, int>();
   n_failed += testMixedTypeBinaryOperations<int, float>();
   n_failed += testMixedTypeBinaryOperations<float, int>();
   n_failed += testMixedTypeBinaryOperations<float, float>();
   n_failed += testMixedTypeBinaryOperations<float, double>();
   n_failed += testMixedTypeBinaryOperations<double, float>();
   n_failed += testMixedTypeBinaryOperations<double, double>();
   n_failed += testMixedTypeBinaryOperations<int, double>();
   n_failed += testMixedTypeBinaryOperations<double, int>();
   n_failed += testMixedTypeBinaryOperations<int, std::complex<float>>();
   n_failed += testMixedTypeBinaryOperations<std::complex<float>, int>();
   n_failed += testMixedTypeBinaryOperations<int, std::complex<double>>();
   n_failed += testMixedTypeBinaryOperations<std::complex<double>, int>();
   n_failed += testMixedTypeBinaryOperations<std::complex<float>, std::complex<float>>();
   n_failed += testMixedTypeBinaryOperations<std::complex<double>, std::complex<double>>();
   n_failed += testMixedTypeBinaryOperations<std::complex<float>, float>();
   n_failed += testMixedTypeBinaryOperations<float, std::complex<float>>();
   n_failed += testMixedTypeBinaryOperations<std::complex<double>, double>();
   n_failed += testMixedTypeBinaryOperations<double, std::complex<double>>();

   std::cout << "\n=== Subview Tests ===" << std::endl;
   n_failed += testExpressionOnSubview();
   n_failed += testChainedExpressionWithSubviews();
   n_failed += testSubviewOfExpression();
   n_failed += testAssignExpressionToSubview();

   std::cout << "\n=== Broadcasting Tests ===" << std::endl;
   n_failed += testSameDimensionalBroadcasting();
   n_failed += testTrailingSingletonBroadcasting();
   n_failed += testTrailingSingletonBroadcasting3D();
   n_failed += testMultipleTrailingSingletons();
   n_failed += testScalarBroadcasting();
   n_failed += testBroadcastingMultiDimensionalAccess();
   n_failed += testChainedBroadcastingExpression();

   std::cout << "\n=== Fancy Indexing Tests ===" << std::endl;
   n_failed += testBinaryExprFancyIndexing_BothViews();
   n_failed += testBinaryExprFancyIndexing_MixedScalarView();
   n_failed += testBinaryExprScalarIndexing();
   n_failed += testBinaryExprFancyIndexing_ChainedExpressions();
   n_failed += testBinaryExprFancyIndexing_Assignment();
   n_failed += testBinaryExprFancyIndexing_DifferentDimensions();
   n_failed += testBinaryExprScalarIndexing_DifferentDimensions();
   n_failed += testBinaryExprSubview_MixedDimensionality();
   n_failed += testBinaryExprFancyIndexing_AsymmetricCollapse();

   PRINT_RESULT(n_failed);
   return n_failed;
}
