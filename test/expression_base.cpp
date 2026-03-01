#include "TensorView/Expressions/ExpressionTraits.hpp"
#include "test.hpp"

using namespace tensor;
using namespace tensor::details;

int main()
{
   int n_failed = 0;

   // Test that TensorBase is now an expression
   {
      Tensor<float, 2> t(3, 4);

      // Check if Tensor is recognized as an expression
      static_assert(is_expression<decltype(t)>, "Tensor should be an expression");
      static_assert(Expression<decltype(t)>, "Tensor should satisfy Expression concept");

      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Tensor is recognized as an expression" << std::endl;
   }

   // Test that TensorView is an expression
   {
      float data[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
      TensorView<float, 2> view(data, 3, 4);

      static_assert(is_expression<decltype(view)>, "TensorView should be an expression");
      static_assert(Expression<decltype(view)>, "TensorView should satisfy Expression concept");

      std::cout << "\t" << ColorText::green("[ ✓ ]") << " TensorView is recognized as an expression" << std::endl;
   }

   // Test that StaticTensor is an expression
   {
      StaticTensor<double, 2, 3> st;

      static_assert(is_expression<decltype(st)>, "StaticTensor should be an expression");
      static_assert(Expression<decltype(st)>, "StaticTensor should satisfy Expression concept");

      std::cout << "\t" << ColorText::green("[ ✓ ]") << " StaticTensor is recognized as an expression" << std::endl;
   }

   // Test expression_value_type_t
   {
      Tensor<float, 2> t(3, 4);
      static_assert(std::is_same_v<expression_value_type_t<decltype(t)>, float>,
                    "expression_value_type_t should extract float");

      std::cout << "\t" << ColorText::green("[ ✓ ]") << " expression_value_type_t works correctly" << std::endl;
   }

   // Test expression_num_dims
   {
      Tensor<int, 3> t(2, 3, 4);
      static_assert(expression_num_dims<decltype(t)> == 3, "expression_num_dims should return 3");

      std::cout << "\t" << ColorText::green("[ ✓ ]") << " expression_num_dims works correctly" << std::endl;
   }

   // Test that ExpressionBase methods can be called
   {
      Tensor<float, 2> t(3, 4);
      for (index_t i = 0; i < t.size(); ++i)
      {
         t[i] = static_cast<float>(i);
      }

      // Access through ExpressionBase interface
      ExpressionBase<Tensor<float, 2>> &expr_base = t;

      if (expr_base.size() != 12)
      {
         std::cout << "\t" << ColorText::red("[ ✗ ]") << " ExpressionBase::size() failed" << std::endl;
         n_failed++;
      }
      else
      {
         std::cout << "\t" << ColorText::green("[ ✓ ]") << " ExpressionBase::size() works" << std::endl;
      }

      if (expr_base.shape(0) != 3 || expr_base.shape(1) != 4)
      {
         std::cout << "\t" << ColorText::red("[ ✗ ]") << " ExpressionBase::shape() failed" << std::endl;
         n_failed++;
      }
      else
      {
         std::cout << "\t" << ColorText::green("[ ✓ ]") << " ExpressionBase::shape() works" << std::endl;
      }

      if (expr_base[5] != 5.0f)
      {
         std::cout << "\t" << ColorText::red("[ ✗ ]") << " ExpressionBase::operator[] failed" << std::endl;
         n_failed++;
      }
      else
      {
         std::cout << "\t" << ColorText::green("[ ✓ ]") << " ExpressionBase::operator[] works" << std::endl;
      }

      // Test begin/end through ExpressionBase
      int count = 0;
      for (auto it = expr_base.begin(); it != expr_base.end(); ++it)
      {
         count++;
      }

      if (count != 12)
      {
         std::cout << "\t" << ColorText::red("[ ✗ ]") << " ExpressionBase iterators failed" << std::endl;
         n_failed++;
      }
      else
      {
         std::cout << "\t" << ColorText::green("[ ✓ ]") << " ExpressionBase iterators work" << std::endl;
      }
   }

   // Test multi-dimensional indexing through ExpressionBase
   {
      Tensor<float, 2> t(3, 4);
      for (index_t i = 0; i < t.size(); ++i)
      {
         t[i] = static_cast<float>(i);
      }

      ExpressionBase<Tensor<float, 2>> &expr_base = t;

      // Test at() with multi-dimensional indices
      if (expr_base.at(1, 2) != t(1, 2) || expr_base.at(1, 2) != 7.0f)
      {
         std::cout << "\t" << ColorText::red("[ ✗ ]") << " ExpressionBase::at(i, j) failed" << std::endl;
         n_failed++;
      }
      else
      {
         std::cout << "\t" << ColorText::green("[ ✓ ]") << " ExpressionBase::at(i, j) works" << std::endl;
      }

      // Test operator() with multi-dimensional indices
      if (expr_base(2, 1) != t(2, 1) || expr_base(2, 1) != 5.0f)
      {
         std::cout << "\t" << ColorText::red("[ ✗ ]") << " ExpressionBase::operator()(i, j) failed" << std::endl;
         n_failed++;
      }
      else
      {
         std::cout << "\t" << ColorText::green("[ ✓ ]") << " ExpressionBase::operator()(i, j) works" << std::endl;
      }

      // Test at() with std::array
      std::array<index_t, 2> indices = {2, 3};
      if (expr_base.at(indices) != t(2, 3) || expr_base.at(indices) != 11.0f)
      {
         std::cout << "\t" << ColorText::red("[ ✗ ]") << " ExpressionBase::at(std::array) failed" << std::endl;
         n_failed++;
      }
      else
      {
         std::cout << "\t" << ColorText::green("[ ✓ ]") << " ExpressionBase::at(std::array) works" << std::endl;
      }

      // Test const version
      const ExpressionBase<Tensor<float, 2>> &const_expr_base = t;
      if (const_expr_base.at(0, 1) != 3.0f)
      {
         std::cout << "\t" << ColorText::red("[ ✗ ]") << " const ExpressionBase::at(i, j) failed" << std::endl;
         n_failed++;
      }
      else
      {
         std::cout << "\t" << ColorText::green("[ ✓ ]") << " const ExpressionBase::at(i, j) works" << std::endl;
      }

      if (const_expr_base(1, 3) != 10.0f)
      {
         std::cout << "\t" << ColorText::red("[ ✗ ]") << " const ExpressionBase::operator()(i, j) failed" << std::endl;
         n_failed++;
      }
      else
      {
         std::cout << "\t" << ColorText::green("[ ✓ ]") << " const ExpressionBase::operator()(i, j) works" << std::endl;
      }
   }

   // Test multi-dimensional indexing with 3D tensor
   {
      Tensor<int, 3> t(2, 3, 4);
      for (index_t i = 0; i < t.size(); ++i)
      {
         t[i] = static_cast<int>(i);
      }

      ExpressionBase<Tensor<int, 3>> &expr_base = t;

      if (expr_base.at(1, 2, 1) != t(1, 2, 1))
      {
         std::cout << "\t" << ColorText::red("[ ✗ ]") << " ExpressionBase::at(i, j, k) for 3D tensor failed"
                   << std::endl;
         n_failed++;
      }
      else
      {
         std::cout << "\t" << ColorText::green("[ ✓ ]") << " ExpressionBase::at(i, j, k) for 3D tensor works"
                   << std::endl;
      }

      std::array<index_t, 3> indices = {1, 1, 2};
      if (expr_base.at(indices) != t(1, 1, 2))
      {
         std::cout << "\t" << ColorText::red("[ ✗ ]") << " ExpressionBase::at(std::array) for 3D tensor failed"
                   << std::endl;
         n_failed++;
      }
      else
      {
         std::cout << "\t" << ColorText::green("[ ✓ ]") << " ExpressionBase::at(std::array) for 3D tensor works"
                   << std::endl;
      }
   }

   PRINT_RESULT(n_failed);
   return n_failed;
}
